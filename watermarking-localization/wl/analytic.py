"""Training-free watermark localizers (T1 agreement/matched-filter, T2 rectangle search, T3 TTO, E01)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter
import yaml

from wl import CELL, GRID, IMG_SIZE, MAX_ASPECT, NBITS, ROOT_DIR, load_config
from wl.evaluate import (
    evaluate_predictions,
    format_metrics_summary,
    load_probs,
    load_train_ground_truth,
    save_probs,
    write_metrics,
)
from wl.features import FeatureCache, compute_decoder_agreement, compute_matched_filter, load_kit_models
from wl.postproc import fit_postproc_oof


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _bit_acc(logits: np.ndarray, bits: np.ndarray) -> float:
    pred = (logits > 0).astype(np.int32).reshape(-1)
    tgt = np.asarray(bits, dtype=np.int32).reshape(-1)
    return float((pred == tgt).mean())


def run_e01(
    data_root: str | Path = ROOT_DIR / "data" / "release",
    device: str = "cpu",
    limit: int | None = None,
    out_dir: str | Path = ROOT_DIR / "output" / "E01_box_decode",
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """E01: verify that decoding the watermarked bounding box beats full-image and shifted-box decoding."""
    import torch

    train_dir = Path(data_root) / "train"
    meta = json.loads((train_dir / "metadata.json").read_text(encoding="utf-8"))
    if methods:
        allowed = set(methods)
        meta = [m for m in meta if m["method"] in allowed]
    if limit is not None:
        meta = meta[:limit]

    with np.load(train_dir / "masks_128.npz", allow_pickle=False) as z:
        masks = {str(n): mk.astype(np.uint8) for n, mk in zip(z["names"].tolist(), z["masks"])}

    wm_models = load_kit_models(sorted({m["method"] for m in meta}), device=device)
    rows: list[dict[str, Any]] = []

    for row in meta:
        name = row["image_name"]
        method = row["method"]
        kind = "full" if row["full_watermark"] else "patch"
        bits = np.array([int(c) for c in row["message"]], dtype=np.uint8)
        img = np.asarray(Image.open(train_dir / "images" / f"{name}.png").convert("RGB"), dtype=np.uint8)
        wm = wm_models[method]

        full_t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0)
        acc_full = _bit_acc(wm.logits(full_t)[0].detach().cpu().numpy(), bits)

        m128 = masks[name]
        ys, xs = np.nonzero(m128 > 0)
        y0, y1 = int(ys.min()) * CELL, (int(ys.max()) + 1) * CELL
        x0, x1 = int(xs.min()) * CELL, (int(xs.max()) + 1) * CELL
        # Keep minimum 16x16 crop so resizers stay numerically stable.
        if y1 - y0 < 16:
            y1 = min(IMG_SIZE, y0 + 16)
            y0 = max(0, y1 - 16)
        if x1 - x0 < 16:
            x1 = min(IMG_SIZE, x0 + 16)
            x0 = max(0, x1 - 16)

        crop = img[y0:y1, x0:x1]
        crop_t = torch.from_numpy(crop).permute(2, 0, 1)[None].float().div(255.0)
        acc_box = _bit_acc(wm.logits(crop_t)[0].detach().cpu().numpy(), bits)

        # Shift by 32 px along y/x inside image bounds (for full, shift from full box -> center crop offset).
        h, w = y1 - y0, x1 - x0
        sy0 = (y0 + 32) if (y0 + 32 + h <= IMG_SIZE) else max(0, y0 - 32)
        sx0 = (x0 + 32) if (x0 + 32 + w <= IMG_SIZE) else max(0, x0 - 32)
        if sy0 == y0 and sx0 == x0:
            sy0 = min(IMG_SIZE - 16, 32)
            sx0 = min(IMG_SIZE - 16, 32)
            sh_crop = img[sy0 : min(IMG_SIZE, sy0 + max(16, h - 64)), sx0 : min(IMG_SIZE, sx0 + max(16, w - 64))]
        else:
            sh_crop = img[sy0 : sy0 + h, sx0 : sx0 + w]
        sh_t = torch.from_numpy(sh_crop).permute(2, 0, 1)[None].float().div(255.0)
        acc_shifted = _bit_acc(wm.logits(sh_t)[0].detach().cpu().numpy(), bits)

        rows.append(
            {
                "image_name": name,
                "method": method,
                "kind": kind,
                "acc_full": acc_full,
                "acc_box": acc_box,
                "acc_shifted": acc_shifted,
            }
        )

    summary: dict[str, Any] = {}
    for method in sorted({r["method"] for r in rows}):
        for kind in ("full", "patch"):
            sub = [r for r in rows if r["method"] == method and r["kind"] == kind]
            if not sub:
                continue
            summary[f"{method}_{kind}"] = {
                "n": len(sub),
                "acc_full": round(float(np.mean([r["acc_full"] for r in sub])), 4),
                "acc_box": round(float(np.mean([r["acc_box"] for r in sub])), 4),
                "acc_shifted": round(float(np.mean([r["acc_shifted"] for r in sub])), 4),
            }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result = {"summary": summary, "rows": rows}
    (out_path / "e01_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("E01 Summary:", json.dumps(summary, indent=2))
    return result


def predict_analytic_t1(
    img_uint8: np.ndarray,
    method: str,
    message_bits: np.ndarray,
    wm_model: Any = None,
    cached_feats: dict[str, Any] | None = None,
    use_matched: bool = False,
    w_match: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Training-free T1 probability map [128, 128] and predicted p(kind == 'full')."""
    feats = cached_feats or compute_decoder_agreement(img_uint8, method, message_bits, wm_model=wm_model)
    agree = np.asarray(feats["agreement128"], dtype=np.float32)
    bit_acc_full = float(feats.get("bit_acc_full", 0.5))
    pos_frac = float((agree > 0.5).mean())
    p_full = float(_sigmoid(12.0 * (bit_acc_full - 0.72) + 4.0 * (pos_frac - 0.5)))

    z_agree = (agree - float(agree.mean())) / (float(agree.std()) + 1e-6)
    z_comb = z_agree
    if use_matched:
        matched = (
            np.asarray(feats["matched128"], dtype=np.float32)
            if "matched128" in feats
            else compute_matched_filter(img_uint8, method, message_bits, wm_model=wm_model)
        )
        z_match = (matched - float(matched.mean())) / (float(matched.std()) + 1e-6)
        z_comb = (1.0 - w_match) * z_agree + w_match * z_match

    # Shift probability baseline toward full-with-holes or clean-with-patch based on whole-image decode.
    bias = 1.8 * p_full - 1.8 * (1.0 - p_full)
    prob = _sigmoid(1.5 * z_comb + bias)
    return np.asarray(prob, dtype=np.float32), p_full


def _generate_candidate_boxes(method: str, step: int = 16) -> list[tuple[int, int, int, int]]:
    max_aspect = MAX_ASPECT[method]
    boxes: list[tuple[int, int, int, int]] = []
    # Thin strips (up to 16:1) and small patches (< 64 px) must be reachable:
    # median patch area is ~4 % (~25x25 cells), minimum ~0.3 % (~7x7 cells).
    sizes = [6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    for h in sizes:
        for w in sizes:
            if max(h, w) / min(h, w) > max_aspect:
                continue
            if h * w > 0.55 * GRID * GRID:
                continue
            for y0 in range(0, GRID - h + 1, step):
                for x0 in range(0, GRID - w + 1, step):
                    boxes.append((y0, y0 + h, x0, x0 + w))
    return boxes


def _score_boxes_batched(
    img_uint8: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    signed_msg: np.ndarray,
    wm_model: Any,
    batch_size: int = 32,
) -> list[float]:
    import torch

    if not boxes:
        return []
    res = int(getattr(wm_model, "resolution", 256))
    base = torch.from_numpy(img_uint8).permute(2, 0, 1).float().div(255.0)
    signed_t = torch.from_numpy(signed_msg).float()
    crops: list[torch.Tensor] = []
    for y0, y1, x0, x1 in boxes:
        crop = base[:, y0 * CELL : y1 * CELL, x0 * CELL : x1 * CELL].unsqueeze(0)
        crop_rs = torch.nn.functional.interpolate(
            crop, size=(res, res), mode="bilinear", align_corners=False, antialias=True
        )[0]
        crops.append(crop_rs)

    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch_t = torch.stack(crops[i : i + batch_size], dim=0)
            logits = wm_model.logits(batch_t).detach().cpu()
            s = (torch.tanh(logits * 0.5) * signed_t[None, :]).mean(dim=1)
            scores.extend(s.tolist())
    return scores


def rectangle_search_single(
    img_uint8: np.ndarray,
    method: str,
    message_bits: np.ndarray,
    wm_model: Any,
    cached_feats: dict[str, Any] | None = None,
) -> np.ndarray:
    """T2: coarse-to-fine bounding box search maximizing message log-likelihood."""
    t1_prob, p_full = predict_analytic_t1(img_uint8, method, message_bits, wm_model=wm_model, cached_feats=cached_feats)
    if wm_model is None or p_full >= 0.75:
        return t1_prob

    nbits = NBITS[method]
    bits = np.asarray(message_bits, dtype=np.float32).reshape(-1)[:nbits]
    signed_msg = (2.0 * bits - 1.0).astype(np.float32)

    coarse_boxes = _generate_candidate_boxes(method, step=16)
    coarse_scores = _score_boxes_batched(img_uint8, coarse_boxes, signed_msg, wm_model)
    top_idx = np.argsort(coarse_scores)[::-1][:5]

    fine_boxes_set: set[tuple[int, int, int, int]] = set()
    max_aspect = MAX_ASPECT[method]
    for idx in top_idx:
        y0, y1, x0, x1 = coarse_boxes[int(idx)]
        h, w = y1 - y0, x1 - x0
        for dy in (-8, -4, 0, 4, 8):
            for dx in (-8, -4, 0, 4, 8):
                for dh in (-4, 0, 4):
                    for dw in (-4, 0, 4):
                        nh = int(np.clip(h + dh, 4, GRID))
                        nw = int(np.clip(w + dw, 4, GRID))
                        if max(nh, nw) / min(nh, nw) > max_aspect:
                            continue
                        ny0 = int(np.clip(y0 + dy, 0, GRID - nh))
                        nx0 = int(np.clip(x0 + dx, 0, GRID - nw))
                        fine_boxes_set.add((ny0, ny0 + nh, nx0, nx0 + nw))

    fine_boxes = list(fine_boxes_set)
    fine_scores = _score_boxes_batched(img_uint8, fine_boxes, signed_msg, wm_model)
    best_i = int(np.argmax(fine_scores))
    by0, by1, bx0, bx1 = fine_boxes[best_i]

    box_map = np.full((GRID, GRID), 0.05, dtype=np.float32)
    box_map[by0:by1, bx0:bx1] = 0.90
    return (0.65 * box_map + 0.35 * t1_prob).astype(np.float32)


def tto_refine_single(
    img_uint8: np.ndarray,
    init_prob128: np.ndarray,
    method: str,
    message_bits: np.ndarray,
    wm_model: Any,
    steps: int = 20,
    lr: float = 0.12,
) -> np.ndarray:
    """T3: refine a 128x128 probability map via differentiable PixelSeal / TrustMark decoders."""
    if wm_model is None or method not in ("pixelseal", "trustmark"):
        return np.asarray(init_prob128, dtype=np.float32)

    import torch
    import torch.nn.functional as F

    device = torch.device(getattr(wm_model, "device", "cpu"))
    nbits = NBITS[method]
    bits_t = torch.tensor(message_bits[:nbits], dtype=torch.float32, device=device).unsqueeze(0)
    img_t = torch.from_numpy(img_uint8).permute(2, 0, 1)[None].float().div(255.0).to(device)
    blurred_np = np.asarray(Image.fromarray(img_uint8).filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.uint8)
    blur_t = torch.from_numpy(blurred_np).permute(2, 0, 1)[None].float().div(255.0).to(device)

    p0 = np.clip(init_prob128.astype(np.float32), 0.03, 0.97)
    logit_init = np.log(p0 / (1.0 - p0))
    mask_logits = torch.tensor(logit_init[None, None], dtype=torch.float32, device=device, requires_grad=True)
    prior_t = torch.tensor(p0[None, None], dtype=torch.float32, device=device)

    opt = torch.optim.Adam([mask_logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        m128 = torch.sigmoid(mask_logits)
        m512 = F.interpolate(m128, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        comp = m512 * img_t + (1.0 - m512) * blur_t
        if method == "pixelseal":
            logits = wm_model.det(wm_model._resize(comp))[:, 1:]
        else:
            logits = wm_model.net(wm_model._prep(comp))
        loss_msg = F.binary_cross_entropy_with_logits(logits, bits_t)
        loss_prior = F.mse_loss(m128, prior_t)
        tv = (m128[:, :, 1:, :] - m128[:, :, :-1, :]).abs().mean() + (m128[:, :, :, 1:] - m128[:, :, :, :-1]).abs().mean()
        loss = loss_msg + 1.5 * loss_prior + 0.15 * tv
        loss.backward()
        opt.step()

    with torch.no_grad():
        return torch.sigmoid(mask_logits[0, 0]).detach().cpu().numpy().astype(np.float32)


def run_analytic(
    cfg: dict[str, Any],
    data_root: str | Path = ROOT_DIR / "data" / "release",
    out_root: str | Path = ROOT_DIR / "output",
    device: str = "cpu",
) -> dict[str, Any]:
    """Run training-free localization pipeline and save val_probs.npz, test_probs.npz, metrics.json."""
    exp_id = str(cfg.get("id", "E100_analytic"))
    out_dir = Path(out_root) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})
    post_cfg = cfg.get("post", {})
    mode = str(mcfg.get("mode", "t1"))
    use_kit = bool(mcfg.get("use_kit", True))
    use_matched = bool(mcfg.get("use_matched", False))
    cache_dir = dcfg.get("cache_dir", str(ROOT_DIR / "output" / "cache" / "features"))
    cache = FeatureCache(cache_dir) if cache_dir not in (None, "none", "null", False) else None

    if mode == "e01":
        return run_e01(data_root=data_root, device=device, limit=mcfg.get("limit"), out_dir=out_dir)

    meta_by_name, masks_by_name = load_train_ground_truth(data_root)
    val_spec = str(dcfg.get("val", "f0"))
    if val_spec == "all510":
        val_names = sorted(meta_by_name.keys())
    else:
        fold_idx = int(val_spec[1:])
        fjson = json.loads((ROOT_DIR / "splits" / "folds.json").read_text(encoding="utf-8"))
        val_names = list(fjson["folds"][fold_idx])

    limit = mcfg.get("limit")
    if limit is not None:
        val_names = val_names[: int(limit)]

    methods_used = sorted({meta_by_name[n]["method"] for n in val_names})
    wm_models = load_kit_models(methods_used, device=device) if use_kit else {}

    def _infer_split(split: str, meta_rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
        names_out: list[str] = []
        probs_out: list[np.ndarray] = []
        split_dir = Path(data_root) / split
        for row in meta_rows:
            n = row["image_name"]
            method = row["method"]
            bits = np.array([int(c) for c in row["message"]], dtype=np.uint8)
            img = np.asarray(Image.open(split_dir / "images" / f"{n}.png").convert("RGB"), dtype=np.uint8)
            cached = cache.load(split, n) if cache is not None else None
            wm = wm_models.get(method)
            if mode == "t2_rect":
                p = rectangle_search_single(img, method, bits, wm, cached_feats=cached)
            elif mode == "t3_tto":
                base_p, _ = predict_analytic_t1(img, method, bits, wm, cached_feats=cached, use_matched=use_matched)
                p = tto_refine_single(img, base_p, method, bits, wm)
            else:
                p, _ = predict_analytic_t1(img, method, bits, wm, cached_feats=cached, use_matched=use_matched)
            names_out.append(n)
            probs_out.append(p)
        return names_out, np.stack(probs_out, axis=0).astype(np.float32)

    val_metas = [meta_by_name[n] for n in val_names]
    v_names, v_probs = _infer_split("train", val_metas)
    save_probs(out_dir / "val_probs.npz", v_names, v_probs)

    metrics = evaluate_predictions(v_names, v_probs, meta_by_name, masks_by_name, threshold=0.5)
    metrics.update({"id": exp_id, "parent": cfg.get("parent", "—"), "data": "none", "val": val_spec})
    if post_cfg.get("threshold") == "oof":
        oof = fit_postproc_oof(v_names, v_probs, meta_by_name, masks_by_name)
        metrics["oof_post_cfg"] = oof["post_cfg"]
        metrics["oof_mean"] = oof["oof_score"]
        metrics["oof_metrics"] = oof["metrics"]
    write_metrics(out_dir, metrics)
    print(f"[{exp_id}] analytic ({mode}): {format_metrics_summary(metrics)}")

    if not bool(cfg.get("train", {}).get("skip_test", False)):
        test_meta = json.loads((Path(data_root) / "test" / "metadata.json").read_text(encoding="utf-8"))
        if limit is not None:
            test_meta = test_meta[: int(limit)]
        t_names, t_probs = _infer_split("test", test_meta)
        save_probs(out_dir / "test_probs.npz", t_names, t_probs)

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run training-free watermark localization or E01 verification.")
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--mode", default=None, choices=["t1", "t2_rect", "t3_tto", "e01"])
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--out", default=str(ROOT_DIR / "output"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tto-init", default=None, help="Optional val_probs.npz to refine with T3 TTO")
    args = ap.parse_args()

    if args.mode == "e01" and not args.config:
        run_e01(data_root=args.data, device=args.device, limit=args.limit, out_dir=Path(args.out) / "E01_box_decode")
        return

    if args.tto_init:
        names, probs = load_probs(args.tto_init)
        meta_by_name, masks_by_name = load_train_ground_truth(args.data)
        wm_models = load_kit_models(sorted({meta_by_name[n]["method"] for n in names}), device=args.device)
        refined = []
        for n, p in zip(names[: args.limit] if args.limit else names, probs):
            m = meta_by_name[n]
            img = np.asarray(Image.open(Path(args.data) / "train" / "images" / f"{n}.png").convert("RGB"), dtype=np.uint8)
            bits = np.array([int(c) for c in m["message"]], dtype=np.uint8)
            refined.append(tto_refine_single(img, p, m["method"], bits, wm_models[m["method"]]))
        sub_names = names[: len(refined)]
        ref_arr = np.stack(refined, axis=0)
        out_dir = Path(args.out) / "T3_tto_refined"
        save_probs(out_dir / "val_probs.npz", sub_names, ref_arr)
        metrics = evaluate_predictions(sub_names, ref_arr, meta_by_name, masks_by_name)
        write_metrics(out_dir, metrics)
        print("T3 TTO:", format_metrics_summary(metrics))
        return

    if not args.config:
        raise ValueError("Provide --config or --mode e01")
    overrides = []
    if args.mode:
        overrides.append(f"model.mode={args.mode}")
    if args.limit is not None:
        overrides.append(f"model.limit={args.limit}")
    cfg = load_config(args.config, overrides=overrides)
    run_analytic(cfg, data_root=args.data, out_root=args.out, device=args.device)


if __name__ == "__main__":
    main()
