"""Pillar 0 forensic diagnostic suite for model.pt and forget_images.npz."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ul import (
    add_common_io_args,
    build_eval_alterations,
    classify_vit_param,
    extract_features_and_logits,
    infer_vit_depth,
    infer_vit_dims,
    jpeg_compress_uint8,
    kl_from_logits,
    load_images_uint8,
    load_model,
    predict_logits,
    resolve_device,
    resolve_task_paths,
)


def _quantiles(x: torch.Tensor) -> dict[str, float]:
    xf = x.float()
    qs = torch.quantile(xf, torch.tensor([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]))
    return {
        "mean": float(xf.mean().item()),
        "std": float(xf.std(correction=0).item()),
        "min": float(qs[0].item()),
        "p10": float(qs[1].item()),
        "p25": float(qs[2].item()),
        "median": float(qs[3].item()),
        "p75": float(qs[4].item()),
        "p90": float(qs[5].item()),
        "max": float(qs[6].item()),
    }


def _save_contact_sheet(
    images_uint8: torch.Tensor,
    y_orig: torch.Tensor,
    out_path: Path,
    num_classes_to_show: int,
    max_per_class: int = 8,
) -> None:
    counts = torch.bincount(y_orig)
    top_classes = torch.argsort(counts, descending=True)[:num_classes_to_show].tolist()
    top_classes = [c for c in top_classes if counts[c].item() > 0]
    if not top_classes:
        return
    h, w = int(images_uint8.shape[1]), int(images_uint8.shape[2])
    canvas = np.zeros((len(top_classes) * h, max_per_class * w, 3), dtype=np.uint8)
    for row, cls_id in enumerate(top_classes):
        idx = (y_orig == cls_id).nonzero(as_tuple=True)[0][:max_per_class]
        for col, sample_idx in enumerate(idx.tolist()):
            canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = images_uint8[sample_idx].numpy()
    Image.fromarray(canvas).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_io_args(parser)
    parser.add_argument("--out-dir", type=Path, default=Path("output/diagnose"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-grids", type=int, default=0, help="Save contact sheet for top N classes")
    args = parser.parse_args()

    device = resolve_device(args.device)
    task_dir, model_path, forget_path = resolve_task_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path, device=device, task_dir=task_dir)
    forget_imgs = load_images_uint8(forget_path)
    num_classes, img_size, patch_size = infer_vit_dims(model, forget_imgs)

    feats_clean, z_clean = extract_features_and_logits(
        model, forget_imgs, device=device, batch_size=args.batch_size
    )
    top2_vals, top2_idx = z_clean.topk(2, dim=1)
    y_orig = top2_idx[:, 0]
    y_second = top2_idx[:, 1]
    margins = top2_vals[:, 0] - top2_vals[:, 1]
    ce_clean = F.cross_entropy(z_clean, y_orig, reduction="none")
    p_clean_t2 = F.softmax(z_clean / 2.0, dim=1)
    p1_clean = p_clean_t2.gather(1, y_orig[:, None]).squeeze(1)

    counts = torch.bincount(y_orig, minlength=num_classes)
    probs_cls = counts.float() / max(1, len(y_orig))
    nonzero = probs_cls > 0
    cls_entropy = float(-(probs_cls[nonzero] * probs_cls[nonzero].log()).sum().item())
    norm_entropy = cls_entropy / math.log(max(2, num_classes))

    # Block coherence of top-2 classes (detects superclass/WordNet grouping)
    same_block5 = float(((y_orig // 5) == (y_second // 5)).float().mean().item())
    same_block10 = float(((y_orig // 10) == (y_second // 10)).float().mean().item())
    near_5 = float(((y_orig - y_second).abs() < 5).float().mean().item())

    # Parameter & LayerScale anatomy
    depth = infer_vit_depth(name for name, _ in model.named_parameters())
    layerscale_norms: dict[str, float] = {}
    group_param_norms: dict[str, float] = {}
    group_param_counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        grp = classify_vit_param(name, depth=depth)
        sq_norm = float(param.detach().float().square().sum().item())
        group_param_norms[grp] = group_param_norms.get(grp, 0.0) + sq_norm
        group_param_counts[grp] = group_param_counts.get(grp, 0) + param.numel()
        if grp == "layerscale":
            layerscale_norms[name] = float(param.detach().float().norm(p=2).item())
    group_l2_norms = {k: math.sqrt(v) for k, v in sorted(group_param_norms.items())}

    # Corruption fragility across 8 tensor views + JPEG 60
    eval_views = build_eval_alterations(patch_size=patch_size)
    corruption_stats: dict[str, dict[str, float]] = {}
    for view_name, view_fn in eval_views.items():
        z_v = predict_logits(model, forget_imgs, device=device, batch_size=args.batch_size, view_fn=view_fn)
        ce_v = F.cross_entropy(z_v, y_orig, reduction="none")
        kl_v = kl_from_logits(z_clean, z_v, temperature=2.0)
        p1_v = F.softmax(z_v / 2.0, dim=1).gather(1, y_orig[:, None]).squeeze(1)
        corruption_stats[view_name] = {
            "ce_mean": float(ce_v.mean().item()),
            "ce_std": float(ce_v.std(correction=0).item()),
            "acc_vs_orig": float((z_v.argmax(1) == y_orig).float().mean().item()),
            "kl_from_clean_t2": float(kl_v.mean().item()),
            "top1_prob_t2": float(p1_v.mean().item()),
        }

    jpeg_imgs = jpeg_compress_uint8(forget_imgs, quality=60)
    z_jpeg = predict_logits(model, jpeg_imgs, device=device, batch_size=args.batch_size)
    ce_jpeg = F.cross_entropy(z_jpeg, y_orig, reduction="none")
    corruption_stats["jpeg60"] = {
        "ce_mean": float(ce_jpeg.mean().item()),
        "ce_std": float(ce_jpeg.std(correction=0).item()),
        "acc_vs_orig": float((z_jpeg.argmax(1) == y_orig).float().mean().item()),
        "kl_from_clean_t2": float(kl_from_logits(z_clean, z_jpeg, temperature=2.0).mean().item()),
        "top1_prob_t2": float(F.softmax(z_jpeg / 2.0, dim=1).gather(1, y_orig[:, None]).squeeze(1).mean().item()),
    }

    # Near-duplicate detection in penultimate feature space
    f_hat = F.normalize(feats_clean, dim=-1)
    sim = f_hat @ f_hat.T
    triu_mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    upper_sims = sim[triu_mask]
    near_dup_95 = int((upper_sims > 0.95).sum().item()) if upper_sims.numel() > 0 else 0
    near_dup_90 = int((upper_sims > 0.90).sum().item()) if upper_sims.numel() > 0 else 0

    report = {
        "num_forget_images": int(len(forget_imgs)),
        "img_size": img_size,
        "patch_size": patch_size,
        "num_classes": num_classes,
        "vit_depth": depth,
        "class_histogram": {
            "counts": counts.tolist(),
            "active_classes": int((counts > 0).sum().item()),
            "min_count": int(counts.min().item()),
            "median_count": float(counts.float().median().item()),
            "max_count": int(counts.max().item()),
            "normalized_entropy": float(norm_entropy),
        },
        "clean_ce": _quantiles(ce_clean),
        "clean_margin": _quantiles(margins),
        "clean_top1_prob_t2_mean": float(p1_clean.mean().item()),
        "runner_up_coherence": {
            "same_block_5": same_block5,
            "same_block_10": same_block10,
            "abs_diff_lt_5": near_5,
        },
        "near_duplicates": {
            "pairs_cos_gt_0_95": near_dup_95,
            "pairs_cos_gt_0_90": near_dup_90,
        },
        "group_l2_norms": group_l2_norms,
        "group_param_counts": dict(sorted(group_param_counts.items())),
        "layerscale_norms": layerscale_norms,
        "corruption_fragility": corruption_stats,
    }

    out_json = args.out_dir / "diagnostics.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    if args.save_grids > 0:
        _save_contact_sheet(
            forget_imgs, y_orig, args.out_dir / "contact_sheet.png", num_classes_to_show=args.save_grids
        )

    print(
        f"Diagnosed {len(forget_imgs)} images ({num_classes} classes, depth={depth}): "
        f"clean CE={report['clean_ce']['mean']:.4f}, margin={report['clean_margin']['mean']:.3f}, "
        f"active_classes={report['class_histogram']['active_classes']}/{num_classes}, "
        f"near_dups(>0.95)={near_dup_95}"
    )
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
