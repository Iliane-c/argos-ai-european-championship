"""Pillar 4 layerwise gamma scaling, multi-paradigm model souping, and local competition scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from hybrid_unlearn import evaluate_unlearning
from ul import (
    add_common_io_args,
    build_model,
    infer_vit_dims,
    layerwise_interpolate,
    load_images_uint8,
    load_model,
    parse_group_gammas,
    predict_logits,
    resolve_device,
    resolve_task_paths,
    soup_checkpoints,
    validate_and_save_submission,
)


def _score_metrics(metrics: dict[str, float | bool], has_shadow: bool) -> float:
    if has_shadow and "final_score" in metrics:
        return float(metrics["final_score"])
    ce_m = float(metrics["forget_clean_ce_mean"])
    ce_s = float(metrics["forget_clean_ce_std"])
    ret_sim = float(metrics.get("retain_kl_similarity", 1.0))
    ce_score = math.exp(-0.5 * ((ce_m - 1.3) / 0.8) ** 2) * (0.5 + 0.5 * min(1.5, ce_s / max(0.2, ce_m)))
    return ce_score * ret_sim


def _candidate_weight_vectors(num_models: int, seed: int = 0) -> list[list[float]]:
    if num_models == 1:
        return [[1.0]]
    vecs: list[list[float]] = [[1.0 / num_models] * num_models]
    for i in range(num_models):
        v = [0.0] * num_models
        v[i] = 1.0
        vecs.append(v)
    if num_models == 2:
        for w0 in (0.25, 0.75):
            vecs.append([w0, 1.0 - w0])
    else:
        gen = torch.Generator().manual_seed(seed)
        for _ in range(8):
            raw = torch.rand(num_models, generator=gen)
            raw = raw / raw.sum()
            vecs.append([float(x) for x in raw.tolist()])
    return vecs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_io_args(parser)
    parser.add_argument("--models", type=Path, nargs="+", required=True, help="One or more candidate .pt files")
    parser.add_argument("--weights", type=str, default=None, help="Comma-separated soup weights")
    parser.add_argument("--group-gammas", type=str, default="", help="e.g. late_mlp=1.4,head=0.6")
    parser.add_argument("--default-gamma", type=float, default=1.0)
    parser.add_argument("--search-weights", action="store_true")
    parser.add_argument("--search-gammas", action="store_true")
    parser.add_argument("--retain-audit", type=Path, default=None)
    parser.add_argument("--shadow-gold", type=Path, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    task_dir, model_path, forget_path = resolve_task_paths(args)

    teacher = load_model(model_path, device=device, task_dir=task_dir)
    forget_imgs = load_images_uint8(forget_path)
    num_classes, img_size, _ = infer_vit_dims(teacher, forget_imgs)
    sd_0 = {k: v.detach().cpu().float().clone() for k, v in teacher.state_dict().items()}

    cand_sds = [torch.load(p, map_location="cpu", weights_only=True) for p in args.models]
    retain_audit = load_images_uint8(args.retain_audit) if args.retain_audit is not None else None
    shadow_gold = (
        torch.load(args.shadow_gold, map_location="cpu", weights_only=True) if args.shadow_gold is not None else None
    )

    z0 = predict_logits(teacher, forget_imgs, device=device, batch_size=args.eval_batch_size)
    y_all = z0.argmax(dim=1)
    all_idx = torch.arange(len(forget_imgs), dtype=torch.long)
    eval_model = build_model(num_classes=num_classes, task_dir=task_dir).to(device).eval()

    def _eval_sd(sd: dict[str, torch.Tensor]) -> dict[str, float | bool]:
        eval_model.load_state_dict(sd, strict=True)
        return evaluate_unlearning(
            teacher=teacher,
            student=eval_model,
            forget_imgs=forget_imgs,
            eval_indices=all_idx,
            y_all=y_all,
            z_orig_clean=z0,
            retain_audit=retain_audit,
            device=device,
            eval_bs=args.eval_batch_size,
            shadow_gold=shadow_gold,
        )

    if args.weights:
        weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    else:
        weights = [1.0 / len(cand_sds)] * len(cand_sds)

    group_gammas = parse_group_gammas(args.group_gammas)

    if args.search_weights and len(cand_sds) > 1:
        best_w = weights
        best_sc = -1.0
        for w_vec in _candidate_weight_vectors(len(cand_sds), seed=args.seed):
            sd_try = soup_checkpoints(sd_0, cand_sds, weights=w_vec, group_gammas=group_gammas, default_gamma=args.default_gamma)
            m_try = _eval_sd(sd_try)
            sc = _score_metrics(m_try, has_shadow=shadow_gold is not None)
            if sc > best_sc:
                best_sc = sc
                best_w = w_vec
        weights = best_w
        print(f"Selected soup weights={weights} (score={best_sc:.4f})")

    base_soup_sd = soup_checkpoints(sd_0, cand_sds, weights=weights)

    if args.search_gammas:
        search_groups = ("late_mlp", "mid_mlp", "late_attn", "head", "norm")
        gamma_grid = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
        best_gammas = dict(group_gammas)
        best_sd_g = layerwise_interpolate(sd_0, base_soup_sd, best_gammas, default_gamma=args.default_gamma)
        best_sc = _score_metrics(_eval_sd(best_sd_g), has_shadow=shadow_gold is not None)
        for grp in search_groups:
            cur_best_val = best_gammas.get(grp, args.default_gamma)
            for g in gamma_grid:
                trial = dict(best_gammas)
                trial[grp] = g
                sd_try = layerwise_interpolate(sd_0, base_soup_sd, trial, default_gamma=args.default_gamma)
                sc = _score_metrics(_eval_sd(sd_try), has_shadow=shadow_gold is not None)
                if sc > best_sc + 1e-6:
                    best_sc = sc
                    cur_best_val = g
            best_gammas[grp] = cur_best_val
        group_gammas = best_gammas
        print(f"Coordinate-searched group gammas={group_gammas} (score={best_sc:.4f})")

    final_sd = layerwise_interpolate(sd_0, base_soup_sd, group_gammas=group_gammas, default_gamma=args.default_gamma)
    final_metrics = _eval_sd(final_sd)
    final_metrics["soup_weights"] = weights
    final_metrics["group_gammas"] = group_gammas
    final_metrics["default_gamma"] = args.default_gamma

    if args.out is not None:
        factory = lambda: build_model(num_classes=num_classes, task_dir=task_dir)
        size_bytes = validate_and_save_submission(
            final_sd,
            args.out,
            reference_state=sd_0,
            model_factory=factory,
            verify_forward_shape=(1, 3, img_size, img_size),
        )
        final_metrics["checkpoint"] = str(args.out)
        final_metrics["checkpoint_bytes"] = size_bytes

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(final_metrics, indent=2) + "\n")

    print(
        f"Soup eval: clean CE={final_metrics['forget_clean_ce_mean']:.3f}/{final_metrics['forget_clean_ce_std']:.3f} "
        f"acc={final_metrics['forget_clean_acc']:.3f} "
        f"retain_kl={final_metrics.get('retain_mean_kl', float('nan')):.4f} "
        f"final_score={final_metrics.get('final_score', 'N/A')}"
    )


if __name__ == "__main__":
    main()
