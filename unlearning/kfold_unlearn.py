"""Pillar 1 Tier C: K-fold cross-fitted in-distribution task-vector unlearning + TIES/Fisher merge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hybrid_unlearn import add_hybrid_args, config_from_args, evaluate_unlearning, run_hybrid_unlearning
from ul import (
    add_common_io_args,
    augment_batch,
    build_model,
    compute_diagonal_fisher,
    delta_cosine_matrix,
    ewc_fisher_penalty,
    feature_anchor_loss,
    forward_features_and_logits,
    infer_vit_dims,
    kd_loss,
    load_images_uint8,
    load_model,
    merge_task_vectors,
    normalize_images,
    project_out_common_direction,
    resolve_amp_dtype,
    resolve_device,
    resolve_task_paths,
    select_trainable_params,
    state_dict_delta,
    to_float01,
    validate_and_save_submission,
)


def _stratified_folds(labels: torch.Tensor, k_folds: int, seed: int) -> list[torch.Tensor]:
    if k_folds < 2:
        raise ValueError(f"--folds must be >= 2, got {k_folds}")
    gen = torch.Generator().manual_seed(seed)
    folds: list[list[int]] = [[] for _ in range(k_folds)]
    num_classes = int(labels.max().item()) + 1
    offset = 0
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(len(idx), generator=gen)].tolist()
        for pos, sample_id in enumerate(perm):
            folds[(offset + pos) % k_folds].append(sample_id)
        offset += len(perm)
    return [torch.tensor(sorted(f), dtype=torch.long) for f in folds if f]


def _retain_polish_model(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    retain_imgs: torch.Tensor,
    device: str,
    amp_dtype: torch.dtype | None,
    patch_size: int,
    epochs: int,
    bs: int,
    lr: float,
    scope: str,
    train_blocks: int,
    include_head: bool,
    include_norm: bool,
    w_ewc: float,
    w_feat_retain: float,
    fisher_retain: dict[str, torch.Tensor] | None,
    orig_params: dict[str, torch.Tensor],
) -> None:
    if epochs <= 0:
        return
    select_trainable_params(
        student, scope=scope, train_blocks=train_blocks, include_head=include_head, include_norm=include_norm
    )
    opt_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    steps = max(1, epochs * math.ceil(len(retain_imgs) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    student.train()
    for _ in range(steps):
        pick = torch.randint(len(retain_imgs), (bs,))
        r_aug = normalize_images(augment_batch(to_float01(retain_imgs[pick], device), clean_prob=0.35, patch_size=patch_size))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            rt_feat, zt_r = forward_features_and_logits(teacher, r_aug)
            rt_feat, zt_r = rt_feat.float(), zt_r.float()
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            fs_r, zs_r = forward_features_and_logits(student, r_aug)
            fs_r, zs_r = fs_r.float(), zs_r.float()
        loss = kd_loss(F.softmax(zt_r / 2.0, dim=1), zs_r, temperature=2.0)
        if w_feat_retain > 0.0:
            loss = loss + w_feat_retain * feature_anchor_loss(fs_r, rt_feat)
        if w_ewc > 0.0 and fisher_retain is not None:
            loss = loss + w_ewc * ewc_fisher_penalty(student, orig_params, fisher_retain)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
    student.eval()


def _selection_score(metrics: dict[str, float | bool], has_shadow: bool) -> float:
    if has_shadow and "final_score" in metrics:
        return float(metrics["final_score"])
    ce_m = float(metrics["forget_clean_ce_mean"])
    ce_s = float(metrics["forget_clean_ce_std"])
    ret_sim = float(metrics.get("retain_kl_similarity", 1.0))
    # Heuristic proxy when no shadow_gold is passed: encourage gold-like skewed CE while keeping retain drift low
    ce_score = math.exp(-0.5 * ((ce_m - 1.3) / 0.8) ** 2) * (0.5 + 0.5 * min(1.5, ce_s / max(0.2, ce_m)))
    return ce_score * ret_sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_io_args(parser)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--retain", type=Path, default=None, help="Optional extra retain_fit.npy proxy")
    parser.add_argument("--retain-audit", type=Path, default=None, help="Optional retain_audit.npy")
    parser.add_argument("--oracle", type=Path, default=None, help="Optional oracle_targets.pt")
    parser.add_argument("--shadow-gold", type=Path, default=None, help="Optional shadow_gold.pt")
    parser.add_argument("--merge-method", choices=["sum", "mean", "ties", "fisher"], default="ties")
    parser.add_argument("--alphas", type=str, default="0.6,0.8,1.0,1.2,1.5")
    parser.add_argument("--trim-ratio", type=float, default=0.20)
    parser.add_argument("--project-common", type=float, default=0.0)
    parser.add_argument("--merged-polish-epochs", type=int, default=1)
    parser.add_argument("--save-folds", action="store_true")
    add_hybrid_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp, device)
    task_dir, model_path, forget_path = resolve_task_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_model(model_path, device=device, task_dir=task_dir)
    forget_imgs = load_images_uint8(forget_path)
    num_classes, img_size, patch_size = infer_vit_dims(teacher, forget_imgs)
    sd_0 = {k: v.detach().cpu().float().clone() for k, v in teacher.state_dict().items()}

    oracle_data = torch.load(args.oracle, map_location="cpu", weights_only=True) if args.oracle is not None else None
    shadow_gold = (
        torch.load(args.shadow_gold, map_location="cpu", weights_only=True) if args.shadow_gold is not None else None
    )
    extra_retain = load_images_uint8(args.retain) if args.retain is not None else None
    retain_audit = load_images_uint8(args.retain_audit) if args.retain_audit is not None else None

    if oracle_data is not None and "y_orig" in oracle_data:
        y_all = oracle_data["y_orig"].long().cpu()
        z0 = oracle_data["z_clean"].float().cpu()
    else:
        _, z0 = extract_features_and_logits(teacher, forget_imgs, device=device, batch_size=args.eval_bs)
        y_all = z0.argmax(dim=1)

    folds = _stratified_folds(y_all, k_folds=args.folds, seed=args.seed)
    base_cfg = config_from_args(args)
    # Individual fold runs skip trailing polish unless explicitly wanted; merged model gets polish
    fold_cfg = config_from_args(args)
    fold_cfg.polish_epochs = 0

    deltas: list[dict[str, torch.Tensor]] = []
    fold_fishers: list[dict[str, torch.Tensor]] = []
    fold_metrics: list[dict] = []
    all_idx = torch.arange(len(forget_imgs), dtype=torch.long)

    for k, fold_idx in enumerate(folds):
        in_mask = torch.ones(len(forget_imgs), dtype=torch.bool)
        in_mask[fold_idx] = False
        other_idx = all_idx[in_mask]
        in_dist_retain = forget_imgs[other_idx]
        if extra_retain is not None:
            combined_retain = torch.cat([in_dist_retain, extra_retain], dim=0)
        else:
            combined_retain = in_dist_retain

        print(f"\n=== K-Fold {k + 1}/{len(folds)}: unlearning {len(fold_idx)} imgs, retain={len(combined_retain)} imgs ===")
        fold_cfg.seed = args.seed + k * 31
        student = load_model(model_path, device=device, task_dir=task_dir)
        student, m_fold = run_hybrid_unlearning(
            cfg=fold_cfg,
            teacher=teacher,
            student=student,
            forget_imgs=forget_imgs,
            retain_fit=combined_retain,
            retain_audit=retain_audit if retain_audit is not None else in_dist_retain[: min(64, len(in_dist_retain))],
            oracle_data=oracle_data,
            forget_indices=fold_idx,
            eval_indices=fold_idx,
            device=device,
            shadow_gold=None,
        )
        sd_k = {name: val.detach().cpu().float() for name, val in student.state_dict().items()}
        deltas.append(state_dict_delta(sd_k, sd_0))
        fold_metrics.append(m_fold)

        if args.merge_method == "fisher":
            trainable_set = set(
                select_trainable_params(
                    student,
                    scope=base_cfg.scope,
                    train_blocks=base_cfg.train_blocks,
                    include_head=base_cfg.include_head,
                    include_norm=base_cfg.include_norm,
                )
            )
            f_k = compute_diagonal_fisher(
                student,
                forget_imgs[fold_idx],
                device=device,
                mode="empirical",
                labels=y_all[fold_idx],
                batch_size=base_cfg.fisher_batch_size,
                max_samples=min(base_cfg.fisher_samples, len(fold_idx)),
                param_names=trainable_set,
                per_sample=base_cfg.fisher_per_sample,
            )
            fold_fishers.append(f_k)

        if args.save_folds:
            factory = lambda: build_model(num_classes=num_classes, task_dir=task_dir)
            validate_and_save_submission(sd_k, args.out_dir / f"fold_{k}.pt", reference_state=sd_0, model_factory=factory)

    cos_mat = delta_cosine_matrix(deltas)
    print("\nPairwise fold delta cosine matrix:")
    for row in cos_mat.tolist():
        print("  " + "  ".join(f"{v:+.3f}" for v in row))

    proc_deltas = project_out_common_direction(deltas, strength=args.project_common)

    polish_retain = extra_retain if extra_retain is not None else forget_imgs
    polish_fisher = None
    if args.merged_polish_epochs > 0 and base_cfg.w_ewc > 0.0:
        tmp_model = load_model(model_path, device=device, task_dir=task_dir)
        allowed = set(
            select_trainable_params(
                tmp_model,
                scope=base_cfg.scope,
                train_blocks=base_cfg.train_blocks,
                include_head=base_cfg.include_head,
                include_norm=base_cfg.include_norm,
            )
        )
        polish_fisher = compute_diagonal_fisher(
            tmp_model,
            polish_retain,
            device=device,
            mode="predictive",
            batch_size=base_cfg.fisher_batch_size,
            max_samples=min(base_cfg.fisher_samples, len(polish_retain)),
            param_names=allowed,
            per_sample=base_cfg.fisher_per_sample,
        )
        orig_allowed = {n: sd_0[n] for n in allowed}
    else:
        orig_allowed = {}

    alphas = [float(s.strip()) for s in args.alphas.split(",") if s.strip()]
    best_score = -1.0
    best_sd: dict[str, torch.Tensor] | None = None
    best_alpha = alphas[0]
    alpha_results: list[dict] = []

    for alpha in alphas:
        merged_sd = merge_task_vectors(
            sd_0,
            proc_deltas,
            method=args.merge_method,
            alpha=alpha,
            trim_ratio=args.trim_ratio,
            fishers=fold_fishers if args.merge_method == "fisher" else None,
        )
        cand = build_model(num_classes=num_classes, task_dir=task_dir)
        cand.load_state_dict(merged_sd, strict=True)
        cand = cand.to(device).eval()

        if args.merged_polish_epochs > 0:
            _retain_polish_model(
                teacher=teacher,
                student=cand,
                retain_imgs=polish_retain,
                device=device,
                amp_dtype=amp_dtype,
                patch_size=patch_size,
                epochs=args.merged_polish_epochs,
                bs=base_cfg.bs,
                lr=base_cfg.polish_lr,
                scope=base_cfg.scope,
                train_blocks=base_cfg.train_blocks,
                include_head=base_cfg.include_head,
                include_norm=base_cfg.include_norm,
                w_ewc=base_cfg.w_ewc,
                w_feat_retain=base_cfg.w_feat_retain,
                fisher_retain=polish_fisher,
                orig_params=orig_allowed,
            )

        ev = evaluate_unlearning(
            teacher=teacher,
            student=cand,
            forget_imgs=forget_imgs,
            eval_indices=all_idx,
            y_all=y_all,
            z_orig_clean=z0,
            retain_audit=retain_audit,
            device=device,
            eval_bs=args.eval_bs,
            shadow_gold=shadow_gold,
        )
        sel_sc = _selection_score(ev, has_shadow=shadow_gold is not None)
        alpha_results.append({"alpha": alpha, "selection_score": sel_sc, **ev})
        print(
            f"[merge={args.merge_method} alpha={alpha:.2f}] clean CE={ev['forget_clean_ce_mean']:.3f}/"
            f"{ev['forget_clean_ce_std']:.3f} acc={ev['forget_clean_acc']:.3f} "
            f"ret_kl={ev.get('retain_mean_kl', float('nan')):.4f} sel={sel_sc:.4f}"
        )
        if sel_sc > best_score:
            best_score = sel_sc
            best_alpha = alpha
            best_sd = {k: v.detach().cpu().float() for k, v in cand.state_dict().items()}

    assert best_sd is not None
    factory = lambda: build_model(num_classes=num_classes, task_dir=task_dir)
    out_ckpt = args.out_dir / "merged_best.pt"
    size_bytes = validate_and_save_submission(
        best_sd,
        out_ckpt,
        reference_state=sd_0,
        model_factory=factory,
        verify_forward_shape=(1, 3, img_size, img_size),
    )
    summary = {
        "folds": len(folds),
        "merge_method": args.merge_method,
        "best_alpha": best_alpha,
        "best_selection_score": best_score,
        "delta_cosine_matrix": cos_mat.tolist(),
        "fold_metrics": fold_metrics,
        "alpha_sweep": alpha_results,
        "checkpoint": str(out_ckpt),
        "checkpoint_bytes": size_bytes,
    }
    (args.out_dir / "kfold_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved best K-fold merged checkpoint to {out_ckpt} (alpha={best_alpha}, {size_bytes:,} bytes)")


if __name__ == "__main__":
    main()
