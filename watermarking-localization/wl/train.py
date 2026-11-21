"""YAML-driven training loop with AMP, cosine LR, EMA, wrong-message negatives, and watermark-safe TTA."""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

from wl import CELL, ROOT_DIR, load_config
from wl.data import build_dataloaders
from wl.evaluate import (
    evaluate_predictions,
    format_metrics_summary,
    load_train_ground_truth,
    save_probs,
    write_metrics,
)
from wl.losses import CompositeLoss
from wl.models import build_model
from wl.postproc import fit_postproc_oof


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelEMA:
    """Exponential moving average copy of model parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            src_v = msd[k]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(src_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(src_v)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _apply_wrong_message_negatives(
    batch: dict[str, torch.Tensor],
    prob: float,
) -> dict[str, torch.Tensor]:
    """Corrupt message bits with probability `prob` and zero target masks so the model learns message specificity."""
    if prob <= 0.0:
        return batch
    bsz = batch["mask128"].shape[0]
    neg_mask = torch.rand(bsz, device=batch["mask128"].device) < prob
    if not neg_mask.any():
        return batch
    out = dict(batch)
    bits_01 = batch["message_bits"].clone()
    bits_signed = batch["message_signed"].clone()
    mask128 = batch["mask128"].clone()
    rand_bits = (torch.rand_like(bits_01[neg_mask]) > 0.5).float() * batch["message_mask"][neg_mask]
    bits_01[neg_mask] = rand_bits
    bits_signed[neg_mask] = (2.0 * rand_bits - 1.0) * batch["message_mask"][neg_mask]
    mask128[neg_mask] = 0.0
    out["message_bits"] = bits_01
    out["message_signed"] = bits_signed
    out["mask128"] = mask128
    return out


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    shifts: Sequence[Sequence[int]] = ((0, 0),),
    max_batches: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """Run inference over a DataLoader using watermark-safe integer cell shifts."""
    model.eval()
    all_names: list[str] = []
    all_probs: list[np.ndarray] = []

    for b_idx, raw_batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        prob_sum = None
        for dy, dx in shifts:
            dy_i, dx_i = int(dy), int(dx)
            if dy_i == 0 and dx_i == 0:
                cur_batch = batch
            else:
                cur_batch = dict(batch)
                cur_batch["image"] = torch.roll(batch["image"], shifts=(dy_i * CELL, dx_i * CELL), dims=(-2, -1))
                cur_batch["rgb"] = torch.roll(batch["rgb"], shifts=(dy_i * CELL, dx_i * CELL), dims=(-2, -1))
            probs = torch.sigmoid(model(cur_batch)["logits"])
            if dy_i != 0 or dx_i != 0:
                probs = torch.roll(probs, shifts=(-dy_i, -dx_i), dims=(-2, -1))
            prob_sum = probs if prob_sum is None else prob_sum + probs

        mean_prob = (prob_sum / float(len(shifts))).detach().cpu().numpy()
        all_names.extend(list(raw_batch["image_name"]))
        all_probs.append(mean_prob)

    if not all_names:
        return [], np.zeros((0, 128, 128), dtype=np.float32)
    return all_names, np.concatenate(all_probs, axis=0).astype(np.float32)


def run_training(
    cfg: dict[str, Any],
    data_root: str | Path = ROOT_DIR / "data" / "release",
    synth_root: str | Path = ROOT_DIR / "data" / "synth",
    out_root: str | Path = ROOT_DIR / "output",
) -> dict[str, Any]:
    """Train a segmentation model from config, evaluate on val, and write test_probs.npz."""
    train_cfg = cfg.get("train", {})
    post_cfg = cfg.get("post", {})
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed)

    exp_id = str(cfg.get("id", "run"))
    out_dir = Path(out_root) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    device = torch.device(train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, test_loader = build_dataloaders(cfg, data_root=data_root, synth_root=synth_root)
    meta_by_name, masks_by_name = load_train_ground_truth(data_root)

    model = build_model(cfg).to(device)
    criterion = CompositeLoss(cfg.get("loss", {})).to(device)
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    max_batches = train_cfg.get("max_batches")
    steps_per_epoch = min(len(train_loader), int(max_batches)) if max_batches else max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = min(total_steps // 10, int(train_cfg.get("warmup_steps", 200)))

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.02 + 0.98 * 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_ema = bool(train_cfg.get("ema", True))
    ema = ModelEMA(model, decay=float(train_cfg.get("ema_decay", 0.995))) if use_ema else None

    wrong_msg_prob = float(train_cfg.get("wrong_msg_prob", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    tta_shifts = post_cfg.get("tta_shifts", [[0, 0]])

    best_score = -1.0
    best_val_names: list[str] = []
    best_val_probs = np.zeros((0, 128, 128), dtype=np.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0

        for b_idx, raw_batch in enumerate(train_loader):
            if max_batches is not None and b_idx >= int(max_batches):
                break
            batch = _move_batch(raw_batch, device)
            if wrong_msg_prob > 0.0:
                batch = _apply_wrong_message_negatives(batch, wrong_msg_prob)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch)
                loss, _ = criterion(outputs, batch)

            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            running_loss += float(loss.detach())
            n_steps += 1

        eval_model = ema.ema if ema is not None else model
        val_names, val_probs = predict_with_tta(
            eval_model, val_loader, device, shifts=[[0, 0]], max_batches=max_batches
        )
        metrics = evaluate_predictions(val_names, val_probs, meta_by_name, masks_by_name, threshold=0.5)
        avg_loss = running_loss / max(1, n_steps)
        print(f"[{exp_id}] epoch {epoch}/{epochs} loss={avg_loss:.4f} | {format_metrics_summary(metrics)}")

        if float(metrics["mean"]) > best_score:
            best_score = float(metrics["mean"])
            best_val_names, best_val_probs = val_names, val_probs
            torch.save({"model": eval_model.state_dict(), "epoch": epoch, "score": best_score}, out_dir / "best.pt")

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    final_model = ema.ema if ema is not None else model
    final_model.load_state_dict(ckpt["model"])

    # Recompute val probabilities with TTA if configured.
    if len(tta_shifts) > 1:
        best_val_names, best_val_probs = predict_with_tta(
            final_model, val_loader, device, shifts=tta_shifts, max_batches=max_batches
        )

    save_probs(out_dir / "val_probs.npz", best_val_names, best_val_probs)
    final_metrics = evaluate_predictions(best_val_names, best_val_probs, meta_by_name, masks_by_name, threshold=0.5)
    final_metrics["id"] = exp_id
    final_metrics["parent"] = cfg.get("parent", "—")
    final_metrics["data"] = "+".join(cfg.get("data", {}).get("train", ["real"]))
    final_metrics["val"] = cfg.get("data", {}).get("val", "f0")
    final_metrics["best_epoch"] = int(ckpt["epoch"])

    if post_cfg.get("threshold") == "oof":
        oof_fit = fit_postproc_oof(best_val_names, best_val_probs, meta_by_name, masks_by_name)
        final_metrics["oof_post_cfg"] = oof_fit["post_cfg"]
        final_metrics["oof_mean"] = oof_fit["oof_score"]
        final_metrics["oof_metrics"] = oof_fit["metrics"]
        print(f"[{exp_id}] OOF post-proc: {oof_fit['raw_score']:.4f} -> {oof_fit['oof_score']:.4f}")

    write_metrics(out_dir, final_metrics)

    if not bool(train_cfg.get("skip_test", False)):
        test_names, test_probs = predict_with_tta(
            final_model, test_loader, device, shifts=tta_shifts, max_batches=max_batches
        )
        if test_names:
            save_probs(out_dir / "test_probs.npz", test_names, test_probs)

    return final_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a watermark localization model from YAML config.")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--synth", default=str(ROOT_DIR / "data" / "synth"))
    ap.add_argument("--out", default=str(ROOT_DIR / "output"))
    ap.add_argument("overrides", nargs="*", help="Dotted key=value overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    run_training(cfg, data_root=args.data, synth_root=args.synth, out_root=args.out)


if __name__ == "__main__":
    main()
