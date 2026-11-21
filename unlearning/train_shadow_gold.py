"""Pillar 2 Shadow-Gold ensemble trainer: trains fresh ViTs excluding forget_images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ul import (
    add_common_io_args,
    aggregate_gold_ensemble,
    augment_batch,
    build_eval_alterations,
    build_model,
    infer_vit_dims,
    kd_loss,
    load_images_uint8,
    load_model,
    normalize_images,
    predict_logits,
    resolve_amp_dtype,
    resolve_device,
    resolve_task_paths,
    to_float01,
)


def _train_single_shadow(
    teacher: torch.nn.Module,
    retain_imgs: torch.Tensor,
    num_classes: int,
    patch_size: int,
    task_dir: Path | None,
    device: str,
    amp_dtype: torch.dtype | None,
    epochs: int,
    bs: int,
    lr: float,
    weight_decay: float,
    kd_weight: float,
    ce_weight: float,
    seed: int,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    shadow = build_model(num_classes=num_classes, task_dir=task_dir).to(device)
    if str(device).startswith("cuda") and hasattr(shadow, "set_grad_checkpointing"):
        shadow.set_grad_checkpointing(True)

    opt = torch.optim.AdamW(shadow.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    steps = max(1, epochs * math.ceil(len(retain_imgs) / bs))
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, i / steps)))
    )

    shadow.train()
    for step in range(steps):
        idx = torch.randint(0, len(retain_imgs), (bs,))
        r_aug = normalize_images(augment_batch(to_float01(retain_imgs[idx], device), clean_prob=0.25, patch_size=patch_size))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            zt = teacher(r_aug).float()
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            zs = shadow(r_aug).float()
        loss = kd_weight * kd_loss(F.softmax(zt / 2.0, dim=1), zs, temperature=2.0)
        if ce_weight > 0.0:
            loss = loss + ce_weight * F.cross_entropy(zs, zt.argmax(dim=1))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(shadow.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            print(f"  [shadow seed={seed}] step {step + 1}/{steps} loss={loss.item():.4f}", flush=True)

    if hasattr(shadow, "set_grad_checkpointing"):
        shadow.set_grad_checkpointing(False)
    shadow.eval()
    return shadow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_io_args(parser)
    parser.add_argument("--retain", type=Path, required=True, help="retain_fit.npy uint8 proxy images")
    parser.add_argument("--out", type=Path, required=True, help="Output shadow_gold.pt path")
    parser.add_argument("--num-models", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp, device)
    task_dir, model_path, forget_path = resolve_task_paths(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    teacher = load_model(model_path, device=device, task_dir=task_dir)
    forget_imgs = load_images_uint8(forget_path)
    retain_imgs = load_images_uint8(args.retain)
    if args.max_train_samples is not None and args.max_train_samples < len(retain_imgs):
        retain_imgs = retain_imgs[: args.max_train_samples]

    num_classes, _, patch_size = infer_vit_dims(teacher, forget_imgs)
    z_orig_clean = predict_logits(teacher, forget_imgs, device=device, batch_size=128)
    y_forget_ref = z_orig_clean.argmax(dim=1)

    eval_views = build_eval_alterations(patch_size=patch_size)
    alt_view_list = [eval_views["crop85"], eval_views["blur5"], eval_views["rot10"]]

    def _composite_alt_logits(model: torch.nn.Module) -> torch.Tensor:
        acc = torch.zeros(len(forget_imgs), num_classes, dtype=torch.float32)
        for vfn in alt_view_list:
            acc = acc + predict_logits(model, forget_imgs, device=device, batch_size=128, view_fn=vfn)
        return acc / float(len(alt_view_list))

    clean_logits_list: list[torch.Tensor] = []
    alt_logits_list: list[torch.Tensor] = []
    per_model_acc: list[float] = []

    for m in range(args.num_models):
        print(f"Training Shadow-Gold model {m + 1}/{args.num_models} on {len(retain_imgs)} retain images")
        shadow = _train_single_shadow(
            teacher=teacher,
            retain_imgs=retain_imgs,
            num_classes=num_classes,
            patch_size=patch_size,
            task_dir=task_dir,
            device=device,
            amp_dtype=amp_dtype,
            epochs=args.epochs,
            bs=args.bs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            kd_weight=args.kd_weight,
            ce_weight=args.ce_weight,
            seed=args.seed + m * 101,
        )
        z_clean_m = predict_logits(shadow, forget_imgs, device=device, batch_size=128)
        z_alt_m = _composite_alt_logits(shadow)
        clean_logits_list.append(z_clean_m)
        alt_logits_list.append(z_alt_m)
        acc_m = float((z_clean_m.argmax(1) == y_forget_ref).float().mean().item())
        per_model_acc.append(acc_m)
        print(f"  [shadow {m + 1}] forget clean acc vs orig={acc_m:.3f}")

    clean_agg = aggregate_gold_ensemble(clean_logits_list, y_forget_ref, temperature=2.0)
    alt_agg = aggregate_gold_ensemble(alt_logits_list, y_forget_ref, temperature=2.0)

    payload = {
        "num_models": args.num_models,
        "y_forget_ref": y_forget_ref,
        "clean_mean_L_gold": float(clean_agg["mean_L_gold"]),
        "clean_std_L_gold": float(clean_agg["std_L_gold"]),
        "clean_p_gold": clean_agg["p_gold"].float().cpu(),
        "clean_per_sample_ce": clean_agg["mean_per_sample_ce"].float().cpu(),
        "alt_mean_L_gold": float(alt_agg["mean_L_gold"]),
        "alt_std_L_gold": float(alt_agg["std_L_gold"]),
        "alt_p_gold": alt_agg["p_gold"].float().cpu(),
        "alt_per_sample_ce": alt_agg["mean_per_sample_ce"].float().cpu(),
        "per_model_clean_acc": per_model_acc,
    }
    torch.save(payload, args.out)

    summary = {
        "num_models": args.num_models,
        "retain_samples": int(len(retain_imgs)),
        "clean_mean_L_gold": payload["clean_mean_L_gold"],
        "clean_std_L_gold": payload["clean_std_L_gold"],
        "alt_mean_L_gold": payload["alt_mean_L_gold"],
        "alt_std_L_gold": payload["alt_std_L_gold"],
        "per_model_clean_acc": per_model_acc,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved Shadow-Gold reference to {args.out}: clean CE={summary['clean_mean_L_gold']:.3f}±{summary['clean_std_L_gold']:.3f}")


if __name__ == "__main__":
    main()
