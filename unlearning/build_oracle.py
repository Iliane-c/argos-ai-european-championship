"""Pillar 2 oracle builder: per-sample memorization scores, OOF non-memorizing oracle, and gold targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ul import (
    add_common_io_args,
    build_hybrid_gold_targets,
    compute_class_feature_stats,
    compute_memorization_scores,
    compute_oof_embedding_oracle,
    extract_features_and_logits,
    infer_vit_dims,
    load_images_uint8,
    load_model,
    resolve_device,
    resolve_task_paths,
)


def _load_external_embeddings(path: Path, expected_rows: int) -> torch.Tensor:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        emb = torch.from_numpy(arr.copy()).float()
    else:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            for key in ("embeddings", "features", "feats"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        emb = torch.as_tensor(loaded, dtype=torch.float32)
    if emb.ndim != 2 or emb.shape[0] != expected_rows:
        raise ValueError(f"Expected external embeddings [{expected_rows}, D], got {tuple(emb.shape)}")
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_io_args(parser)
    parser.add_argument("--out", type=Path, required=True, help="Output oracle_targets.pt path")
    parser.add_argument("--shadow-gold", type=Path, default=None, help="Optional shadow_gold.pt")
    parser.add_argument("--ext-embeddings", type=Path, default=None, help="Optional frozen DINO/SigLIP embeddings")
    parser.add_argument("--num-weight-samples", type=int, default=4)
    parser.add_argument("--weight-noise-std", type=float, default=0.02)
    parser.add_argument("--prune-ratio", type=float, default=0.30)
    parser.add_argument("--patch-mask-ratio", type=float, default=0.35)
    parser.add_argument("--no-input-grad", action="store_true")
    parser.add_argument("--oof-splits", type=int, default=5)
    parser.add_argument("--delta-min", type=float, default=2.0)
    parser.add_argument("--delta-max", type=float, default=16.0)
    parser.add_argument("--gamma-mem", type=float, default=1.5)
    parser.add_argument("--soften", type=float, default=0.10)
    parser.add_argument("--w-shift", type=float, default=0.55)
    parser.add_argument("--w-shadow", type=float, default=0.25)
    parser.add_argument("--w-oof", type=float, default=0.10)
    parser.add_argument("--w-corrupt", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    task_dir, model_path, forget_path = resolve_task_paths(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    teacher = load_model(model_path, device=device, task_dir=task_dir)
    forget_imgs = load_images_uint8(forget_path)
    num_classes, _, patch_size = infer_vit_dims(teacher, forget_imgs)

    feats_clean, z_clean = extract_features_and_logits(
        teacher, forget_imgs, device=device, batch_size=args.batch_size
    )
    mem = compute_memorization_scores(
        model=teacher,
        images_uint8=forget_imgs,
        device=device,
        z_clean=z_clean,
        feats_clean=feats_clean,
        num_weight_samples=args.num_weight_samples,
        weight_noise_std=args.weight_noise_std,
        prune_ratio=args.prune_ratio,
        patch_mask_ratio=args.patch_mask_ratio,
        compute_input_grad=not args.no_input_grad,
        batch_size=args.batch_size,
        patch_size=patch_size,
        seed=args.seed,
    )
    y_orig = mem["y_orig"]

    oof_feats = F.normalize(feats_clean, dim=-1)
    if args.ext_embeddings is not None:
        ext = F.normalize(_load_external_embeddings(args.ext_embeddings, len(forget_imgs)), dim=-1)
        oof_feats = F.normalize(torch.cat([oof_feats, ext], dim=-1), dim=-1)

    oof = compute_oof_embedding_oracle(
        embeddings=oof_feats,
        labels=y_orig,
        num_classes=num_classes,
        n_splits=args.oof_splits,
        seed=args.seed,
    )

    shadow_probs = None
    shadow_ce_mean = None
    shadow_ce_std = None
    shadow_ce = None
    alt_ce_mean = None
    alt_ce_std = None
    alt_sorted_ce = None
    if args.shadow_gold is not None:
        sg = torch.load(args.shadow_gold, map_location="cpu", weights_only=True)
        shadow_probs = sg["clean_p_gold"]
        shadow_ce_mean = float(sg["clean_mean_L_gold"])
        shadow_ce_std = float(sg["clean_std_L_gold"])
        shadow_ce = sg["clean_per_sample_ce"]
        alt_ce_mean = float(sg["alt_mean_L_gold"])
        alt_ce_std = float(sg["alt_std_L_gold"])
        alt_sorted_ce, _ = torch.sort(sg["alt_per_sample_ce"].float())

    targets = build_hybrid_gold_targets(
        z_teacher_clean=z_clean,
        y_all=y_orig,
        mem_scores=mem["mem_scores"],
        num_classes=num_classes,
        shadow_probs=shadow_probs,
        shadow_ce_mean=shadow_ce_mean,
        shadow_ce_std=shadow_ce_std,
        shadow_per_sample_ce=shadow_ce,
        oof_probs=oof["p_oof"],
        corrupt_probs=mem["corrupt_probs"],
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        gamma_mem=args.gamma_mem,
        soften=args.soften,
        w_shift=args.w_shift,
        w_shadow=args.w_shadow,
        w_oof=args.w_oof,
        w_corrupt=args.w_corrupt,
    )
    class_means, class_inv_vars = compute_class_feature_stats(feats_clean, y_orig, num_classes=num_classes)

    clean_ce_mean = float(targets["target_ce_mean"])
    clean_ce_std = float(targets["target_ce_std"])
    if alt_ce_mean is None:
        alt_ce_mean = clean_ce_mean * 1.15
        alt_ce_std = clean_ce_std * 1.10
        alt_sorted_ce = targets["target_sorted_ce"] * 1.15

    payload = {
        "y_orig": y_orig,
        "z_clean": z_clean,
        "feats_clean": feats_clean,
        "class_means": class_means,
        "class_inv_vars": class_inv_vars,
        **mem,
        **oof,
        **targets,
        "alt_target_ce_mean": float(alt_ce_mean),
        "alt_target_ce_std": float(alt_ce_std),
        "alt_target_sorted_ce": alt_sorted_ce,
    }
    torch.save(payload, args.out)

    summary = {
        "num_samples": int(len(forget_imgs)),
        "rum_strata_counts": [int((mem["rum_strata"] == s).sum().item()) for s in (0, 1, 2)],
        "delta_mean": float(targets["delta_i"].mean().item()),
        "delta_min": float(targets["delta_i"].min().item()),
        "delta_max": float(targets["delta_i"].max().item()),
        "target_ce_mean": clean_ce_mean,
        "target_ce_std": clean_ce_std,
        "alt_target_ce_mean": float(alt_ce_mean),
        "alt_target_ce_std": float(alt_ce_std),
        "oof_ce_mean": float(oof["ce_oof"].mean().item()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Built oracle targets for {len(forget_imgs)} images: strata={summary['rum_strata_counts']}, "
        f"delta={summary['delta_min']:.1f}..{summary['delta_max']:.1f}, "
        f"target CE={clean_ce_mean:.3f}±{clean_ce_std:.3f}"
    )


if __name__ == "__main__":
    main()
