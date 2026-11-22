"""Pillar 4 layerwise gamma scaling, multi-paradigm model souping, and local competition scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hybrid_unlearn import evaluate_unlearning
from ul import (
    add_common_io_args,
    build_model,
    compute_retain_score,
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
    parser.add_argument("--search-subset", type=int, default=512, help="Max forget images used during fast weight/gamma search")
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
    r_orig = (
        predict_logits(teacher, retain_audit, device=device, batch_size=args.eval_batch_size)
        if retain_audit is not None and len(retain_audit) > 0
        else None
    )
    eval_model = build_model(num_classes=num_classes, task_dir=task_dir).to(device).eval()

    # Deterministic subset for fast intermediate weight/gamma trials (avoids 50x full 7-view evals).
    if args.search_subset > 0 and len(forget_imgs) > args.search_subset and shadow_gold is None:
        sub_gen = torch.Generator().manual_seed(args.seed + 7)
        sub_idx = torch.randperm(len(forget_imgs), generator=sub_gen)[: args.search_subset]
    else:
        sub_idx = all_idx
    sub_imgs = forget_imgs[sub_idx]
    sub_y = y_all[sub_idx]

    def _eval_fast(sd: dict[str, torch.Tensor]) -> dict[str, float | bool]:
        if shadow_gold is not None:
            return _eval_sd(sd)
        eval_model.load_state_dict(sd, strict=True)
        zs = predict_logits(eval_model, sub_imgs, device=device, batch_size=args.eval_batch_size)
        ce = F.cross_entropy(zs, sub_y, reduction="none")
        m: dict[str, float | bool] = {
            "forget_clean_ce_mean": float(ce.mean().item()),
            "forget_clean_ce_std": float(ce.std(correction=0).item()),
            "forget_clean_acc": float((zs.argmax(1) == sub_y).float().mean().item()),
        }
        if retain_audit is not None and r_orig is not None:
            r_sub = predict_logits(eval_model, retain_audit, device=device, batch_size=args.eval_batch_size)
            m.update(compute_retain_score(r_orig, r_sub, labels_ctrl=None, temperature=2.0))
        return m

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
        w_cands = _candidate_weight_vectors(len(cand_sds), seed=args.seed)
        for idx_w, w_vec in enumerate(w_cands, start=1):
            sd_try = soup_checkpoints(sd_0, cand_sds, weights=w_vec, group_gammas=group_gammas, default_gamma=args.default_gamma)
            m_try = _eval_fast(sd_try)
            sc = _score_metrics(m_try, has_shadow=shadow_gold is not None)
            del sd_try
            if sc > best_sc:
                best_sc = sc
                best_w = w_vec
            if idx_w == 1 or idx_w == len(w_cands) or idx_w % 5 == 0:
                print(
                    f"  [weight search {idx_w}/{len(w_cands)}] CE={m_try['forget_clean_ce_mean']:.3f} "
                    f"ret_kl={m_try.get('retain_mean_kl', 0.0):.4f} best_sc={best_sc:.4f}",
                    flush=True,
                )
        weights = best_w
        print(f"Selected soup weights={weights} (score={best_sc:.4f})", flush=True)

    base_soup_sd = soup_checkpoints(sd_0, cand_sds, weights=weights)
    del cand_sds

    if args.search_gammas:
        search_groups = ("late_mlp", "mid_mlp", "late_attn", "head", "norm")
        gamma_grid = (0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
        best_gammas = dict(group_gammas)
        best_sd_g = layerwise_interpolate(sd_0, base_soup_sd, best_gammas, default_gamma=args.default_gamma)
        best_sc = _score_metrics(_eval_fast(best_sd_g), has_shadow=shadow_gold is not None)
        del best_sd_g
        for grp in search_groups:
            cur_best_val = best_gammas.get(grp, args.default_gamma)
            for g in gamma_grid:
                trial = dict(best_gammas)
                trial[grp] = g
                sd_try = layerwise_interpolate(sd_0, base_soup_sd, trial, default_gamma=args.default_gamma)
                m_try = _eval_fast(sd_try)
                sc = _score_metrics(m_try, has_shadow=shadow_gold is not None)
                del sd_try
                if sc > best_sc + 1e-6:
                    best_sc = sc
                    cur_best_val = g
            best_gammas[grp] = cur_best_val
            print(f"  [gamma search] {grp}={cur_best_val} (best_sc={best_sc:.4f})", flush=True)
        group_gammas = best_gammas
        print(f"Coordinate-searched group gammas={group_gammas} (score={best_sc:.4f})", flush=True)

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
