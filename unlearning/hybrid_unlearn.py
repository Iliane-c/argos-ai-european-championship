"""Flagship Pillar 3 & 4 hybrid unlearning trainer (also reusable by kfold_unlearn.py)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ul import (
    ActivationSubspaceProjector,
    MemoryBankMomentLoss,
    add_common_io_args,
    apply_grad_mask,
    apply_magnitude_pruning,
    apply_ssd_dampening,
    augment_batch,
    build_active_head_mask,
    build_eval_alterations,
    build_hybrid_gold_targets,
    build_model,
    build_salun_mask,
    compute_class_feature_stats,
    compute_competition_score,
    compute_diagonal_fisher,
    compute_forget_score,
    compute_margin_power_deltas,
    compute_retain_score,
    ewc_fisher_penalty,
    extract_features_and_logits,
    feature_anchor_loss,
    forward_features_and_logits,
    infer_vit_dims,
    inject_lora,
    kd_loss,
    kl_from_logits,
    load_images_uint8,
    load_model,
    local_smoothness_loss,
    merge_and_unload_lora,
    normalize_images,
    npo_loss,
    predict_logits,
    resolve_amp_dtype,
    resolve_device,
    resolve_task_paths,
    scar_feature_push_loss,
    select_trainable_params,
    softmax_temperature,
    to_float01,
    validate_and_save_submission,
)


@dataclass
class HybridUnlearnConfig:
    epochs: int = 8
    bs: int = 32
    eval_bs: int = 128
    lr: float = 3e-5
    head_lr_scale: float = 0.5
    weight_decay: float = 0.0
    scope: str = "late_mlp"
    train_blocks: int = 6
    include_head: bool = True
    include_norm: bool = True
    freeze_inactive_head: bool = True
    grad_checkpointing: bool = True
    amp: str = "bf16"
    seed: int = 0
    # Warm-starts
    ssd_warmstart: bool = False
    ssd_alpha: float = 10.0
    ssd_lambda: float = 1.0
    prune_warmstart: bool = False
    prune_sparsity: float = 0.15
    # Saliency & parameter-space constraints
    salun_ratio: float = 1.0
    salun_mode: str = "relative_ssd"
    fisher_samples: int = 512
    fisher_batch_size: int = 16
    fisher_per_sample: bool = True
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_targets: tuple[str, ...] = ("mlp.fc1", "mlp.fc2", "attn.proj")
    ogp_rank: int = 0
    ogp_var: float = 0.95
    ogp_samples: int = 512
    # Adaptive targets & loss weights
    delta_mode: str = "oracle"
    delta: float = 12.0
    margin_power: float | None = None
    fragility_boost: float = 0.0
    delta_clamp_min: float = 2.0
    delta_clamp_max: float = 36.0
    delta_min: float = 2.0
    delta_max: float = 16.0
    soften: float = 0.0
    view_ratio: bool = False
    clean_target_blend: float = 0.0
    target_ce_mean: float | None = None
    target_ce_std: float | None = None
    w_kd_t2: float = 1.0
    w_kd_t1: float = 1.0
    w_moment: float = 0.0
    moment_mode: str = "mean_std"
    moment_bank_size: int = 512
    w_npo: float = 0.0
    npo_beta: float = 0.2
    npo_min_mem: float = 0.70
    w_smooth: float = 0.0
    w_scar: float = 0.0
    scar_margin: float = 1.0
    w_retain: float = 1.0
    w_feat_retain: float = 0.10
    w_head_gaussian: float = 0.0
    w_ewc: float = 0.0
    # Curriculum & SCRUB polish
    rum_curriculum: bool = False
    polish_epochs: int = 0
    polish_lr: float = 5e-6


def _true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    zy = logits.gather(1, labels[:, None]).squeeze(1)
    other_max = logits.scatter(1, labels[:, None], -1e9).max(dim=1).values
    return zy - other_max


def _snapshot_merged_state(student: nn.Module, has_lora: bool) -> dict[str, torch.Tensor]:
    if not has_lora:
        return {k: v.detach().cpu().float().clone() for k, v in student.state_dict().items()}
    clone = copy.deepcopy(student)
    merge_and_unload_lora(clone)
    return {k: v.detach().cpu().float().clone() for k, v in clone.state_dict().items()}


def evaluate_unlearning(
    teacher: nn.Module,
    student: nn.Module,
    forget_imgs: torch.Tensor,
    eval_indices: torch.Tensor,
    y_all: torch.Tensor,
    z_orig_clean: torch.Tensor,
    retain_audit: torch.Tensor | None,
    device: str,
    eval_bs: int = 128,
    shadow_gold: dict | None = None,
    retain_audit_active: torch.Tensor | None = None,
    retain_audit_inactive: torch.Tensor | None = None,
) -> dict[str, float | bool]:
    """Evaluate clean forget, altered forget, retain audit, and optional Shadow-Gold score."""
    ev_imgs = forget_imgs[eval_indices.cpu()]
    ev_y = y_all[eval_indices.cpu()].cpu()
    zo_clean = z_orig_clean[eval_indices.cpu()].cpu()
    zs_clean = predict_logits(student, ev_imgs, device=device, batch_size=eval_bs)

    ce_sub_clean = F.cross_entropy(zs_clean, ev_y, reduction="none")
    ce_orig_clean = F.cross_entropy(zo_clean, ev_y, reduction="none")
    acc_clean = float((zs_clean.argmax(1) == ev_y).float().mean().item())
    kl_clean = float(kl_from_logits(zo_clean, zs_clean, temperature=2.0).mean().item())

    _, _, patch_size = infer_vit_dims(teacher, forget_imgs)
    views = build_eval_alterations(patch_size=patch_size)
    alt_fns = [views["crop85"], views["blur5"], views["rot10"]]
    zo_alt = sum(predict_logits(teacher, ev_imgs, device=device, batch_size=eval_bs, view_fn=fn) for fn in alt_fns) / len(alt_fns)
    zs_alt = sum(predict_logits(student, ev_imgs, device=device, batch_size=eval_bs, view_fn=fn) for fn in alt_fns) / len(alt_fns)
    assert isinstance(zo_alt, torch.Tensor) and isinstance(zs_alt, torch.Tensor)
    ce_sub_alt = F.cross_entropy(zs_alt, ev_y, reduction="none")
    ce_orig_alt = F.cross_entropy(zo_alt, ev_y, reduction="none")
    acc_alt = float((zs_alt.argmax(1) == ev_y).float().mean().item())

    metrics: dict[str, float | bool] = {
        "forget_clean_ce_mean": float(ce_sub_clean.mean().item()),
        "forget_clean_ce_std": float(ce_sub_clean.std(correction=0).item()),
        "forget_clean_acc": acc_clean,
        "forget_clean_kl_orig_sub_t2": kl_clean,
        "forget_alt_ce_mean": float(ce_sub_alt.mean().item()),
        "forget_alt_ce_std": float(ce_sub_alt.std(correction=0).item()),
        "forget_alt_acc": acc_alt,
        "forget_alt_kl_orig_sub_t2": float(kl_from_logits(zo_alt, zs_alt, temperature=2.0).mean().item()),
    }

    retain_metrics = None
    if retain_audit is not None and len(retain_audit) > 0:
        r_orig = predict_logits(teacher, retain_audit, device=device, batch_size=eval_bs)
        r_sub = predict_logits(student, retain_audit, device=device, batch_size=eval_bs)
        retain_metrics = compute_retain_score(r_orig, r_sub, labels_ctrl=None, temperature=2.0)
        metrics.update(retain_metrics)

    for split_prefix, split_imgs in (("active45", retain_audit_active), ("inactive55", retain_audit_inactive)):
        if split_imgs is not None and len(split_imgs) > 0:
            s_orig = predict_logits(teacher, split_imgs, device=device, batch_size=eval_bs)
            s_sub = predict_logits(student, split_imgs, device=device, batch_size=eval_bs)
            sm = compute_retain_score(s_orig, s_sub, labels_ctrl=None, temperature=2.0)
            metrics[f"{split_prefix}_retain_score"] = sm["retain_score"]
            metrics[f"{split_prefix}_retain_acc"] = sm["retain_acc_sub"]
            metrics[f"{split_prefix}_retain_kl"] = sm["retain_mean_kl"]

    if shadow_gold is not None and len(eval_indices) == len(forget_imgs):
        fg_metrics = compute_forget_score(
            clean_ce_sub=ce_sub_clean,
            clean_ce_orig=ce_orig_clean,
            clean_gold_mean=float(shadow_gold["clean_mean_L_gold"]),
            clean_gold_std=float(shadow_gold["clean_std_L_gold"]),
            clean_p_gold=shadow_gold["clean_p_gold"],
            clean_p_sub=softmax_temperature(zs_clean, 2.0),
            clean_p_orig=softmax_temperature(zo_clean, 2.0),
            alt_ce_sub=ce_sub_alt,
            alt_ce_orig=ce_orig_alt,
            alt_gold_mean=float(shadow_gold["alt_mean_L_gold"]),
            alt_gold_std=float(shadow_gold["alt_std_L_gold"]),
            alt_p_gold=shadow_gold["alt_p_gold"],
            alt_p_sub=softmax_temperature(zs_alt, 2.0),
            alt_p_orig=softmax_temperature(zo_alt, 2.0),
        )
        ret_in = retain_metrics or {
            "retain_score": 1.0,
            "retain_accuracy_score": 1.0,
            "retain_kl_similarity": 1.0,
            "retain_mean_kl": 0.0,
            "retain_acc_sub": 1.0,
            "retain_acc_orig": 1.0,
        }
        comp = compute_competition_score(fg_metrics, ret_in)
        metrics.update(comp)

    return metrics


def run_hybrid_unlearning(
    cfg: HybridUnlearnConfig,
    teacher: nn.Module,
    student: nn.Module,
    forget_imgs: torch.Tensor,
    retain_fit: torch.Tensor,
    retain_audit: torch.Tensor | None = None,
    oracle_data: dict | None = None,
    forget_indices: torch.Tensor | None = None,
    eval_indices: torch.Tensor | None = None,
    device: str = "cpu",
    shadow_gold: dict | None = None,
    save_epoch_dir: Path | None = None,
    model_factory=None,
    active_classes: torch.Tensor | None = None,
    retain_audit_active: torch.Tensor | None = None,
    retain_audit_inactive: torch.Tensor | None = None,
) -> tuple[nn.Module, dict]:
    """Execute the full hybrid unlearning pipeline on student starting from teacher."""
    torch.manual_seed(cfg.seed)
    amp_dtype = resolve_amp_dtype(cfg.amp, device)
    num_classes, _, patch_size = infer_vit_dims(teacher, forget_imgs)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tr_idx = forget_indices.cpu() if forget_indices is not None else torch.arange(len(forget_imgs))
    ev_idx = eval_indices.cpu() if eval_indices is not None else tr_idx

    if oracle_data is not None and "z_clean" in oracle_data and "feats_clean" in oracle_data:
        z0 = oracle_data["z_clean"].float().cpu()
        feats0 = oracle_data["feats_clean"].float().cpu()
    else:
        feats0, z0 = extract_features_and_logits(teacher, forget_imgs, device=device, batch_size=cfg.eval_bs)

    y_all = oracle_data["y_orig"].long().cpu() if (oracle_data is not None and "y_orig" in oracle_data) else z0.argmax(1)
    m_clean = _true_margin(z0, y_all).clamp_min(1e-3)

    if oracle_data is not None and "delta_i" in oracle_data:
        mem_scores = oracle_data["mem_scores"].float().cpu()
        rum_strata = oracle_data["rum_strata"].long().cpu()
        delta_i = oracle_data["delta_i"].float().cpu()
        prior = oracle_data["prior"].float().cpu()
        target_probs_t2 = oracle_data["target_probs_t2"].float().cpu()
        target_ce_mean = float(oracle_data["target_ce_mean"])
        target_ce_std = float(oracle_data["target_ce_std"])
        target_sorted_ce = oracle_data["target_sorted_ce"].float().cpu()
        class_means = oracle_data["class_means"].float().cpu()
        class_inv_vars = oracle_data["class_inv_vars"].float().cpu()
    else:
        # Lightweight standalone fallback scaled by normalized clean margin fragility
        norm_rank = torch.argsort(torch.argsort(-m_clean, stable=True)).float() / float(max(1, len(m_clean) - 1))
        mem_scores = norm_rank
        rum_strata = torch.where(mem_scores >= 0.67, 2, torch.where(mem_scores >= 0.33, 1, 0))
        built = build_hybrid_gold_targets(
            z_teacher_clean=z0,
            y_all=y_all,
            mem_scores=mem_scores,
            num_classes=num_classes,
            delta_min=min(cfg.delta_min, cfg.delta),
            delta_max=max(cfg.delta_max, cfg.delta),
            soften=cfg.soften,
        )
        delta_i = built["delta_i"]
        prior = built["prior"]
        target_probs_t2 = built["target_probs_t2"]
        target_ce_mean = float(built["target_ce_mean"])
        target_ce_std = float(built["target_ce_std"])
        target_sorted_ce = built["target_sorted_ce"]
        class_means, class_inv_vars = compute_class_feature_stats(feats0, y_all, num_classes=num_classes)

    if cfg.margin_power is not None or cfg.delta_mode == "margin_power":
        eff_mp = float(cfg.margin_power) if cfg.margin_power is not None else -2.2
        delta_i = compute_margin_power_deltas(
            z_teacher_clean=z0,
            y_all=y_all,
            mem_scores=mem_scores,
            base_delta=cfg.delta,
            margin_power=eff_mp,
            fragility_boost=cfg.fragility_boost,
            delta_clamp_min=cfg.delta_clamp_min,
            delta_clamp_max=cfg.delta_clamp_max,
        )
    if cfg.target_ce_mean is not None:
        target_ce_mean = float(cfg.target_ce_mean)
    if cfg.target_ce_std is not None:
        target_ce_std = float(cfg.target_ce_std)

    resolved_active_classes = active_classes.long().cpu() if active_classes is not None else y_all.unique().cpu()

    # Select surgical trainable subset BEFORE optional Fisher / SSD / pruning / LoRA
    trainable_names = select_trainable_params(
        student,
        scope=cfg.scope,
        train_blocks=cfg.train_blocks,
        include_head=cfg.include_head,
        include_norm=cfg.include_norm,
    )
    allowed_set = set(trainable_names)

    need_fisher_retain = cfg.w_ewc > 0.0 or cfg.ssd_warmstart or (cfg.salun_ratio < 1.0 and cfg.salun_mode == "relative_ssd")
    need_fisher_forget = cfg.ssd_warmstart or cfg.salun_ratio < 1.0

    if hasattr(student, "set_grad_checkpointing"):
        student.set_grad_checkpointing(False)

    fisher_retain = None
    if need_fisher_retain:
        fisher_retain = compute_diagonal_fisher(
            student,
            retain_fit,
            device=device,
            mode="predictive",
            batch_size=cfg.fisher_batch_size,
            max_samples=min(cfg.fisher_samples, len(retain_fit)),
            param_names=allowed_set,
            per_sample=cfg.fisher_per_sample,
        )

    fisher_forget = None
    if need_fisher_forget:
        f_sub = forget_imgs[tr_idx]
        f_lbl = y_all[tr_idx]
        fisher_forget = compute_diagonal_fisher(
            student,
            f_sub,
            device=device,
            mode="empirical",
            labels=f_lbl,
            batch_size=cfg.fisher_batch_size,
            max_samples=min(cfg.fisher_samples, len(f_sub)),
            param_names=allowed_set,
            per_sample=cfg.fisher_per_sample,
        )

    warmstart_info: dict[str, dict[str, float]] = {}
    if cfg.ssd_warmstart:
        assert fisher_forget is not None and fisher_retain is not None
        warmstart_info["ssd"] = apply_ssd_dampening(
            student,
            fisher_forget=fisher_forget,
            fisher_retain=fisher_retain,
            alpha=cfg.ssd_alpha,
            damp_lambda=cfg.ssd_lambda,
            allowed_names=allowed_set,
        )

    if cfg.prune_warmstart:
        warmstart_info["prune"] = apply_magnitude_pruning(
            student, sparsity=cfg.prune_sparsity, allowed_names=allowed_set
        )

    orig_params = {
        n: p.detach().cpu().float().clone()
        for n, p in teacher.named_parameters()
        if n in allowed_set
    }

    salun_mask = None
    has_lora = cfg.lora_rank > 0
    if has_lora:
        lora_replaced = inject_lora(
            student,
            target_patterns=cfg.lora_targets,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            train_blocks=cfg.train_blocks,
            freeze_base=True,
        )
        trainable_names = [n for n, p in student.named_parameters() if p.requires_grad]
    else:
        lora_replaced = []
        if cfg.salun_ratio < 1.0:
            assert fisher_forget is not None
            salun_mask = build_salun_mask(
                fisher_forget=fisher_forget,
                fisher_retain=fisher_retain,
                keep_ratio=cfg.salun_ratio,
                mode=cfg.salun_mode,
                allowed_names=allowed_set,
            )

    ogp: ActivationSubspaceProjector | None = None
    if cfg.ogp_rank > 0:
        ogp_targets = (
            lora_replaced
            if has_lora
            else [
                m_name
                for m_name, mod in student.named_modules()
                if isinstance(mod, nn.Linear) and f"{m_name}.weight" in allowed_set
            ]
        )
        if ogp_targets:
            ogp = ActivationSubspaceProjector()
            ogp.fit(
                student,
                retain_fit,
                device=device,
                target_linear_names=ogp_targets,
                variance_threshold=cfg.ogp_var,
                max_rank=cfg.ogp_rank,
                max_samples=min(cfg.ogp_samples, len(retain_fit)),
            )

    # Gaussian feature replay stats for head stabilization
    head_trainable = any(p.requires_grad for p in student.head.parameters()) if hasattr(student, "head") else False
    retain_gauss_classes = torch.empty(0, dtype=torch.long)
    retain_gauss_means = torch.empty(0)
    retain_gauss_stds = torch.empty(0)
    if cfg.w_head_gaussian > 0.0 and head_trainable:
        r_sample = retain_fit[: min(len(retain_fit), 2048)]
        r_feats, r_logits = extract_features_and_logits(teacher, r_sample, device=device, batch_size=cfg.eval_bs)
        r_y = r_logits.argmax(1)
        counts_r = torch.bincount(r_y, minlength=num_classes)
        retain_gauss_classes = (counts_r > 0).nonzero(as_tuple=True)[0]
        r_means, r_inv_vars = compute_class_feature_stats(r_feats, r_y, num_classes=num_classes)
        retain_gauss_means = r_means.to(device)
        retain_gauss_stds = r_inv_vars.clamp_min(1e-6).rsqrt().to(device)

    if cfg.grad_checkpointing and str(device).startswith("cuda") and hasattr(student, "set_grad_checkpointing"):
        student.set_grad_checkpointing(True)

    # Move lookup tensors to device
    y_dev = y_all.to(device)
    m_clean_dev = m_clean.to(device)
    delta_dev = delta_i.to(device)
    prior_dev = prior.to(device)
    target_t2_dev = target_probs_t2.to(device)
    mem_dev = mem_scores.to(device)
    class_means_dev = class_means.to(device)
    class_inv_vars_dev = class_inv_vars.to(device)

    hard_mask = rum_strata[tr_idx] >= 1
    hard_tr_idx = tr_idx[hard_mask] if hard_mask.any() else tr_idx

    moment_criterion = MemoryBankMomentLoss(bank_size=cfg.moment_bank_size).to(device)
    head_mask = build_active_head_mask(student, resolved_active_classes) if (cfg.freeze_inactive_head and not has_lora) else None
    backbone_params = [p for n, p in student.named_parameters() if p.requires_grad and not n.startswith("head.")]
    head_params = [p for n, p in student.named_parameters() if p.requires_grad and n.startswith("head.")]
    opt_groups: list[dict] = []
    if backbone_params:
        opt_groups.append({"params": backbone_params, "lr": cfg.lr})
    if head_params:
        opt_groups.append({"params": head_params, "lr": cfg.lr * cfg.head_lr_scale})
    opt_params = backbone_params + head_params
    opt = torch.optim.AdamW(opt_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    steps_per_epoch = max(1, math.ceil(len(tr_idx) / cfg.bs))
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warm = max(1, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, i / total_steps)))
    )

    student.train()
    moment_criterion.train()
    t0 = time.time()
    ref_state = {k: v.detach().cpu().float() for k, v in teacher.state_dict().items()}
    named_params = dict(student.named_parameters())

    def _step_optimize(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if head_mask is not None:
            apply_grad_mask(student, head_mask)
        if salun_mask is not None:
            apply_grad_mask(student, salun_mask)
        if ogp is not None:
            ogp.project_gradients(student)
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if head_mask is not None and cfg.weight_decay > 0.0:
            with torch.no_grad():
                for name, mask in head_mask.items():
                    if name in orig_params and name in named_params:
                        p = named_params[name]
                        keep = mask.to(device=p.device, dtype=torch.bool)
                        p.data.copy_(torch.where(keep, p.data, orig_params[name].to(device=p.device, dtype=p.dtype)))

    for step in range(total_steps):
        if cfg.rum_curriculum and step < int(0.30 * total_steps):
            pool_idx = hard_tr_idx
        else:
            pool_idx = tr_idx

        pick = pool_idx[torch.randint(len(pool_idx), (cfg.bs,))]
        pick_dev = pick.to(device)
        r_pick = torch.randint(len(retain_fit), (cfg.bs,))

        f01 = to_float01(forget_imgs[pick], device)
        v_aug = normalize_images(
            augment_batch(
                f01,
                clean_prob=0.25,
                crop_scale=(0.75, 1.0),
                max_rot_deg=10.0,
                erase_prob=0.15,
                patch_size=patch_size,
            )
        )
        r_aug = normalize_images(augment_batch(to_float01(retain_fit[r_pick], device), clean_prob=0.25, patch_size=patch_size))

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            _, zt_v = forward_features_and_logits(teacher, v_aug)
            rt_feat, zt_r = forward_features_and_logits(teacher, r_aug)
            zt_v, rt_feat, zt_r = zt_v.float(), rt_feat.float(), zt_r.float()

        y_b = y_dev[pick_dev]
        d_b = delta_dev[pick_dev]
        if cfg.view_ratio:
            d_b = d_b * (_true_margin(zt_v, y_b) / m_clean_dev[pick_dev]).clamp(0.0, 1.0)
        shifted_v = zt_v.scatter_add(1, y_b[:, None], -d_b[:, None])
        q_view_t2 = F.softmax(shifted_v / 2.0, dim=1)
        q_view_t1 = F.softmax(shifted_v, dim=1)
        if cfg.soften > 0.0:
            q_view_t2 = (1.0 - cfg.soften) * q_view_t2 + cfg.soften * prior_dev[y_b]
            q_view_t1 = (1.0 - cfg.soften) * q_view_t1 + cfg.soften * prior_dev[y_b]
        if cfg.clean_target_blend > 0.0:
            blend = float(cfg.clean_target_blend)
            q_t2 = (1.0 - blend) * q_view_t2 + blend * target_t2_dev[pick_dev]
        else:
            q_t2 = q_view_t2

        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            cat_in = torch.cat([v_aug, r_aug], dim=0)
            cat_feat, cat_logits = forward_features_and_logits(student, cat_in)
            fs_v, fs_r = cat_feat.float().split(cfg.bs)
            zs_v, zs_r = cat_logits.float().split(cfg.bs)

        loss = cfg.w_kd_t2 * kd_loss(q_t2, zs_v, temperature=2.0)
        if cfg.w_kd_t1 > 0.0:
            loss = loss + cfg.w_kd_t1 * kd_loss(q_view_t1, zs_v, temperature=1.0)
        if cfg.w_moment > 0.0:
            loss = loss + cfg.w_moment * moment_criterion(
                zs_v, y_b, target_ce_mean, target_ce_std, target_sorted_ce=target_sorted_ce, mode=cfg.moment_mode
            )
        if cfg.w_npo > 0.0:
            npo_w = (mem_dev[pick_dev] >= cfg.npo_min_mem).float()
            if npo_w.any():
                loss = loss + cfg.w_npo * npo_loss(zs_v, zt_v, y_b, beta=cfg.npo_beta, sample_weights=npo_w)
        if cfg.w_smooth > 0.0:
            v_clean = normalize_images(f01)
            with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                _, zs_clean_b = forward_features_and_logits(student, v_clean)
            loss = loss + cfg.w_smooth * local_smoothness_loss(zs_clean_b.float(), zs_v, temperature=2.0)
        if cfg.w_scar > 0.0:
            scar_w = (mem_dev[pick_dev] >= cfg.npo_min_mem).float()
            if scar_w.any():
                loss = loss + cfg.w_scar * scar_feature_push_loss(
                    fs_v, y_b, class_means_dev, class_inv_vars_dev, margin=cfg.scar_margin, sample_weights=scar_w
                )
        if cfg.w_retain > 0.0:
            loss = loss + cfg.w_retain * kd_loss(F.softmax(zt_r / 2.0, dim=1), zs_r, temperature=2.0)
        if cfg.w_feat_retain > 0.0:
            loss = loss + cfg.w_feat_retain * feature_anchor_loss(fs_r, rt_feat)
        if cfg.w_head_gaussian > 0.0 and len(retain_gauss_classes) > 0:
            c_pick = retain_gauss_classes[torch.randint(len(retain_gauss_classes), (cfg.bs,))].to(device)
            h_synth = retain_gauss_means[c_pick] + torch.randn_like(retain_gauss_means[c_pick]) * retain_gauss_stds[c_pick]
            with torch.no_grad():
                hz_t = teacher.head(h_synth).float()
            hz_s = student.head(h_synth).float()
            loss = loss + cfg.w_head_gaussian * kd_loss(F.softmax(hz_t / 2.0, dim=1), hz_s, temperature=2.0)
        if cfg.w_ewc > 0.0 and fisher_retain is not None:
            loss = loss + cfg.w_ewc * ewc_fisher_penalty(student, orig_params, fisher_retain)

        _step_optimize(loss, opt)
        sched.step()

        is_epoch_end = (step + 1) % steps_per_epoch == 0 or step == total_steps - 1
        if step % 50 == 0 or is_epoch_end:
            print(f"step {step + 1}/{total_steps} loss={loss.item():.4f} ({time.time() - t0:.1f}s)", flush=True)

        if save_epoch_dir is not None and is_epoch_end:
            ep_idx = (step + 1 + steps_per_epoch - 1) // steps_per_epoch
            ep_sd = _snapshot_merged_state(student, has_lora=has_lora)
            validate_and_save_submission(
                ep_sd, save_epoch_dir / f"epoch_{ep_idx:02d}.pt", reference_state=ref_state, model_factory=model_factory
            )

    # SCRUB retain-only polish phase to minimize control KL drift without touching forget images
    if cfg.polish_epochs > 0:
        polish_steps = max(1, cfg.polish_epochs * math.ceil(len(retain_fit) / cfg.bs))
        polish_groups: list[dict] = []
        if backbone_params:
            polish_groups.append({"params": backbone_params, "lr": cfg.polish_lr})
        if head_params:
            polish_groups.append({"params": head_params, "lr": cfg.polish_lr * cfg.head_lr_scale})
        polish_opt = torch.optim.AdamW(polish_groups, lr=cfg.polish_lr, weight_decay=0.0)
        polish_sched = torch.optim.lr_scheduler.CosineAnnealingLR(polish_opt, T_max=polish_steps)
        print(f"Running {cfg.polish_epochs} SCRUB retain-polish epoch(s) ({polish_steps} steps, lr={cfg.polish_lr})")
        for p_step in range(polish_steps):
            r_pick = torch.randint(len(retain_fit), (cfg.bs,))
            r_aug = normalize_images(augment_batch(to_float01(retain_fit[r_pick], device), clean_prob=0.35, patch_size=patch_size))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                rt_feat, zt_r = forward_features_and_logits(teacher, r_aug)
                rt_feat, zt_r = rt_feat.float(), zt_r.float()
            with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                fs_r, zs_r = forward_features_and_logits(student, r_aug)
                fs_r, zs_r = fs_r.float(), zs_r.float()
            p_loss = max(1.0, cfg.w_retain) * kd_loss(F.softmax(zt_r / 2.0, dim=1), zs_r, temperature=2.0)
            if cfg.w_feat_retain > 0.0:
                p_loss = p_loss + cfg.w_feat_retain * feature_anchor_loss(fs_r, rt_feat)
            if cfg.w_ewc > 0.0 and fisher_retain is not None:
                p_loss = p_loss + cfg.w_ewc * ewc_fisher_penalty(student, orig_params, fisher_retain)
            _step_optimize(p_loss, polish_opt)
            polish_sched.step()

    if hasattr(student, "set_grad_checkpointing"):
        student.set_grad_checkpointing(False)
    if has_lora:
        merge_and_unload_lora(student)
    student.eval()

    metrics = evaluate_unlearning(
        teacher=teacher,
        student=student,
        forget_imgs=forget_imgs,
        eval_indices=ev_idx,
        y_all=y_all,
        z_orig_clean=z0,
        retain_audit=retain_audit,
        device=device,
        eval_bs=cfg.eval_bs,
        shadow_gold=shadow_gold,
        retain_audit_active=retain_audit_active,
        retain_audit_inactive=retain_audit_inactive,
    )
    metrics["trainable_param_tensors"] = len(trainable_names)
    metrics["warmstart"] = warmstart_info
    metrics["config"] = {
        k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()
    }
    return student, metrics


def add_hybrid_args(parser: argparse.ArgumentParser) -> None:
    """Register hybrid unlearning hyperparameters cleanly for CLI entrypoints."""
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--eval-bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr-scale", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--scope",
        choices=["all", "last_k_blocks", "mlp_only", "late_mlp", "late_mlp_attn_proj", "head_only"],
        default="late_mlp",
    )
    parser.add_argument("--train-blocks", type=int, default=6)
    parser.add_argument("--no-head", action="store_true")
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--no-freeze-inactive-head", action="store_true")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    # Warm-starts
    parser.add_argument("--ssd-warmstart", action="store_true")
    parser.add_argument("--ssd-alpha", type=float, default=10.0)
    parser.add_argument("--ssd-lambda", type=float, default=1.0)
    parser.add_argument("--prune-warmstart", action="store_true")
    parser.add_argument("--prune-sparsity", type=float, default=0.15)
    # Saliency & LoRA & OGP
    parser.add_argument("--salun-ratio", type=float, default=1.0)
    parser.add_argument("--salun-mode", choices=["salun", "relative_ssd"], default="relative_ssd")
    parser.add_argument("--fisher-samples", type=int, default=512)
    parser.add_argument("--fisher-batch-size", type=int, default=16)
    parser.add_argument("--no-per-sample-fisher", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--ogp-rank", type=int, default=0)
    parser.add_argument("--ogp-var", type=float, default=0.95)
    parser.add_argument("--ogp-samples", type=int, default=512)
    # Loss weights
    parser.add_argument("--delta-mode", choices=["oracle", "margin_power"], default="oracle")
    parser.add_argument("--delta", type=float, default=12.0)
    parser.add_argument("--margin-power", type=float, default=None)
    parser.add_argument("--fragility-boost", type=float, default=0.0)
    parser.add_argument("--delta-clamp-min", type=float, default=2.0)
    parser.add_argument("--delta-clamp-max", type=float, default=36.0)
    parser.add_argument("--delta-min", type=float, default=2.0)
    parser.add_argument("--delta-max", type=float, default=16.0)
    parser.add_argument("--soften", type=float, default=0.0)
    parser.add_argument("--view-ratio", dest="view_ratio", action="store_true")
    parser.add_argument("--no-view-ratio", dest="view_ratio", action="store_false")
    parser.set_defaults(view_ratio=False)
    parser.add_argument("--clean-target-blend", type=float, default=0.0)
    parser.add_argument("--target-ce-mean", type=float, default=None)
    parser.add_argument("--target-ce-std", type=float, default=None)
    parser.add_argument("--w-kd-t2", type=float, default=1.0)
    parser.add_argument("--w-kd-t1", type=float, default=1.0)
    parser.add_argument("--w-moment", type=float, default=0.0)
    parser.add_argument("--moment-mode", choices=["mean_std", "wasserstein"], default="mean_std")
    parser.add_argument("--moment-bank-size", type=int, default=512)
    parser.add_argument("--w-npo", type=float, default=0.0)
    parser.add_argument("--npo-beta", type=float, default=0.2)
    parser.add_argument("--npo-min-mem", type=float, default=0.70)
    parser.add_argument("--w-smooth", type=float, default=0.0)
    parser.add_argument("--w-scar", type=float, default=0.0)
    parser.add_argument("--scar-margin", type=float, default=1.0)
    parser.add_argument("--w-retain", type=float, default=1.0)
    parser.add_argument("--w-feat-retain", type=float, default=0.10)
    parser.add_argument("--w-head-gaussian", type=float, default=0.0)
    parser.add_argument("--w-ewc", type=float, default=0.0)
    parser.add_argument("--rum-curriculum", action="store_true")
    parser.add_argument("--polish-epochs", type=int, default=0)
    parser.add_argument("--polish-lr", type=float, default=5e-6)


def config_from_args(args: argparse.Namespace) -> HybridUnlearnConfig:
    return HybridUnlearnConfig(
        epochs=args.epochs,
        bs=args.bs,
        eval_bs=args.eval_bs,
        lr=args.lr,
        head_lr_scale=args.head_lr_scale,
        weight_decay=args.weight_decay,
        scope=args.scope,
        train_blocks=args.train_blocks,
        include_head=not args.no_head,
        include_norm=not args.no_norm,
        freeze_inactive_head=not args.no_freeze_inactive_head,
        grad_checkpointing=not args.no_grad_checkpointing,
        amp=args.amp,
        seed=args.seed,
        ssd_warmstart=args.ssd_warmstart,
        ssd_alpha=args.ssd_alpha,
        ssd_lambda=args.ssd_lambda,
        prune_warmstart=args.prune_warmstart,
        prune_sparsity=args.prune_sparsity,
        salun_ratio=args.salun_ratio,
        salun_mode=args.salun_mode,
        fisher_samples=args.fisher_samples,
        fisher_batch_size=args.fisher_batch_size,
        fisher_per_sample=not args.no_per_sample_fisher,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        ogp_rank=args.ogp_rank,
        ogp_var=args.ogp_var,
        ogp_samples=args.ogp_samples,
        delta_mode=args.delta_mode,
        delta=args.delta,
        margin_power=args.margin_power,
        fragility_boost=args.fragility_boost,
        delta_clamp_min=args.delta_clamp_min,
        delta_clamp_max=args.delta_clamp_max,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        soften=args.soften,
        view_ratio=args.view_ratio,
        clean_target_blend=args.clean_target_blend,
        target_ce_mean=args.target_ce_mean,
        target_ce_std=args.target_ce_std,
        w_kd_t2=args.w_kd_t2,
        w_kd_t1=args.w_kd_t1,
        w_moment=args.w_moment,
        moment_mode=args.moment_mode,
        moment_bank_size=args.moment_bank_size,
        w_npo=args.w_npo,
        npo_beta=args.npo_beta,
        npo_min_mem=args.npo_min_mem,
        w_smooth=args.w_smooth,
        w_scar=args.w_scar,
        scar_margin=args.scar_margin,
        w_retain=args.w_retain,
        w_feat_retain=args.w_feat_retain,
        w_head_gaussian=args.w_head_gaussian,
        w_ewc=args.w_ewc,
        rum_curriculum=args.rum_curriculum,
        polish_epochs=args.polish_epochs,
        polish_lr=args.polish_lr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_io_args(parser)
    parser.add_argument("--retain", type=Path, required=True, help="uint8 .npy/.npz retain proxy images")
    parser.add_argument("--retain-audit", type=Path, default=None, help="Optional uint8 retain audit images")
    parser.add_argument("--retain-audit-active", type=Path, default=None)
    parser.add_argument("--retain-audit-inactive", type=Path, default=None)
    parser.add_argument("--retain-audit-count", type=int, default=500, help="Used if --retain-audit is omitted")
    parser.add_argument("--oracle", type=Path, default=None, help="Optional oracle_targets.pt")
    parser.add_argument("--shadow-gold", type=Path, default=None, help="Optional shadow_gold.pt")
    parser.add_argument("--out", type=Path, required=True, help="Output unlearned .pt checkpoint")
    parser.add_argument("--save-every-epoch", action="store_true")
    add_hybrid_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    task_dir, model_path, forget_path = resolve_task_paths(args)
    teacher = load_model(model_path, device=device, task_dir=task_dir)
    student = load_model(model_path, device=device, task_dir=task_dir)
    forget_imgs = load_images_uint8(forget_path)

    retain_all = load_images_uint8(args.retain)
    if args.retain_audit is not None:
        retain_fit = retain_all
        retain_audit = load_images_uint8(args.retain_audit)
    elif args.retain_audit_count > 0 and len(retain_all) > args.retain_audit_count:
        retain_fit = retain_all[: -args.retain_audit_count]
        retain_audit = retain_all[-args.retain_audit_count :]
    else:
        retain_fit = retain_all
        retain_audit = None

    active_path = args.retain_audit_active
    inactive_path = args.retain_audit_inactive
    if args.retain_audit is not None:
        if active_path is None and (args.retain_audit.parent / "retain_audit_active.npy").is_file():
            active_path = args.retain_audit.parent / "retain_audit_active.npy"
        if inactive_path is None and (args.retain_audit.parent / "retain_audit_inactive.npy").is_file():
            inactive_path = args.retain_audit.parent / "retain_audit_inactive.npy"
    retain_audit_active = load_images_uint8(active_path) if active_path is not None else None
    retain_audit_inactive = load_images_uint8(inactive_path) if inactive_path is not None else None

    oracle_data = torch.load(args.oracle, map_location="cpu", weights_only=True) if args.oracle is not None else None
    shadow_gold = (
        torch.load(args.shadow_gold, map_location="cpu", weights_only=True) if args.shadow_gold is not None else None
    )

    num_classes, img_size, _ = infer_vit_dims(teacher, forget_imgs)
    factory = lambda: build_model(num_classes=num_classes, task_dir=task_dir)
    save_epoch_dir = args.out.parent / f"{args.out.stem}_epochs" if args.save_every_epoch else None

    cfg = config_from_args(args)
    student, metrics = run_hybrid_unlearning(
        cfg=cfg,
        teacher=teacher,
        student=student,
        forget_imgs=forget_imgs,
        retain_fit=retain_fit,
        retain_audit=retain_audit,
        oracle_data=oracle_data,
        device=device,
        shadow_gold=shadow_gold,
        save_epoch_dir=save_epoch_dir,
        model_factory=factory,
        retain_audit_active=retain_audit_active,
        retain_audit_inactive=retain_audit_inactive,
    )

    ref_state = {k: v.detach().cpu().float() for k, v in teacher.state_dict().items()}
    size_bytes = validate_and_save_submission(
        student,
        args.out,
        reference_state=ref_state,
        model_factory=factory,
        verify_forward_shape=(1, 3, img_size, img_size),
    )
    metrics["checkpoint_bytes"] = size_bytes
    metrics_path = args.out.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"Saved {args.out} ({size_bytes:,} bytes): "
        f"clean CE={metrics['forget_clean_ce_mean']:.3f}/{metrics['forget_clean_ce_std']:.3f}, "
        f"acc={metrics['forget_clean_acc']:.3f}, "
        f"retain_kl={metrics.get('retain_mean_kl', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
