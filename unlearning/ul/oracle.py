"""Per-sample memorization scoring, OOF non-memorizing oracle, and calibrated gold targets."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentations import build_eval_alterations, patch_mask
from .model_io import extract_features_and_logits, normalize_images, predict_logits, preprocess, to_float01


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    n = len(values)
    if n <= 1:
        return torch.zeros_like(values, dtype=torch.float32)
    ranks = torch.empty(n, dtype=torch.float32, device=values.device)
    order = torch.argsort(values.float(), stable=True)
    ranks[order] = torch.arange(n, dtype=torch.float32, device=values.device) / float(n - 1)
    return ranks


def compute_class_confusion_prior(
    z_teacher: torch.Tensor,
    y_all: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute [C, C] per-class T=1 teacher runner-up confusion distribution (true class masked)."""
    z = z_teacher.float()
    y = y_all.long()
    masked = z.scatter(1, y[:, None], -1e9)
    alt = F.softmax(masked, dim=1)
    prior = torch.zeros(num_classes, num_classes, dtype=torch.float32, device=z.device)
    prior.index_add_(0, y, alt)
    row_sums = prior.sum(dim=1, keepdim=True)
    # Fallback for classes absent in forget set: uniform off-diagonal distribution.
    uniform_off = (1.0 - torch.eye(num_classes, dtype=torch.float32, device=z.device)) / max(1, num_classes - 1)
    has_mass = row_sums > 1e-12
    safe_prior = torch.where(has_mass, prior / row_sums.clamp_min(1e-12), uniform_off)
    return safe_prior


def compute_class_feature_stats(
    features: torch.Tensor,
    y_all: torch.Tensor,
    num_classes: int,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class feature centroids [C, D] and clamped inverse variances [C, D]."""
    feats = features.float()
    y = y_all.long()
    d = feats.shape[1]
    global_mean = feats.mean(dim=0)
    global_var = feats.var(dim=0, unbiased=False).clamp_min(eps)

    means = global_mean.unsqueeze(0).repeat(num_classes, 1)
    inv_vars = (1.0 / global_var).unsqueeze(0).repeat(num_classes, 1)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        cls_feats = feats[idx]
        means[c] = cls_feats.mean(dim=0)
        if idx.numel() > 1:
            cls_var = cls_feats.var(dim=0, unbiased=False).clamp_min(eps)
            inv_vars[c] = 1.0 / (0.5 * cls_var + 0.5 * global_var)
    return means, inv_vars


def compute_memorization_scores(
    model: nn.Module,
    images_uint8: torch.Tensor,
    device: str,
    z_clean: torch.Tensor | None = None,
    feats_clean: torch.Tensor | None = None,
    num_weight_samples: int = 4,
    weight_noise_std: float = 0.02,
    prune_ratio: float = 0.30,
    patch_mask_ratio: float = 0.35,
    compute_input_grad: bool = True,
    batch_size: int = 32,
    patch_size: int = 16,
    seed: int = 0,
    use_fp16_on_cuda: bool = True,
) -> dict[str, torch.Tensor]:
    """Score per-sample memorization from augmentation, weight perturbation, input grad, and outlierness."""
    if z_clean is None or feats_clean is None:
        feats_clean, z_clean = extract_features_and_logits(
            model, images_uint8, device, batch_size=batch_size, use_fp16_on_cuda=use_fp16_on_cuda
        )
    else:
        z_clean = z_clean.float().cpu()
        feats_clean = feats_clean.float().cpu()

    y_orig = z_clean.argmax(dim=1)
    p_clean = F.softmax(z_clean / 2.0, dim=1)
    p_y_clean = p_clean.gather(1, y_orig[:, None]).squeeze(1)

    # 1. Augmentation & patch-mask fragility
    eval_views = build_eval_alterations(patch_size=patch_size)
    gen_mask = torch.Generator(device=device).manual_seed(seed + 11)
    corrupt_fns = [
        eval_views["blur5"],
        eval_views["crop85"],
        eval_views["downup50"],
        eval_views["noise025"],
        lambda x: patch_mask(x, mask_ratio=patch_mask_ratio, patch_size=patch_size, generator=gen_mask),
    ]
    corrupt_probs_acc = torch.zeros_like(p_clean)
    for view_fn in corrupt_fns:
        z_v = predict_logits(
            model, images_uint8, device, batch_size=batch_size, view_fn=view_fn, use_fp16_on_cuda=use_fp16_on_cuda
        )
        corrupt_probs_acc = corrupt_probs_acc + F.softmax(z_v / 2.0, dim=1)
    corrupt_probs = corrupt_probs_acc / float(len(corrupt_fns))
    fragility_aug = (p_y_clean - corrupt_probs.gather(1, y_orig[:, None]).squeeze(1)).clamp_min(0.0)

    # 2. Weight-noise & magnitude-pruning fragility (snapshot and restore exact model weights)
    fragility_weight = torch.zeros(len(images_uint8), dtype=torch.float32)
    weight_mc_probs = torch.zeros_like(p_clean)
    if num_weight_samples > 0:
        perturb_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.ndim >= 2 and ("blocks." in name or name.startswith("head."))
        ]
        snapshot = {name: param.data.detach().clone() for name, param in perturb_params}
        rng_gen = torch.Generator(device=device).manual_seed(seed + 29)
        try:
            for sample_idx in range(num_weight_samples):
                with torch.no_grad():
                    for name, param in perturb_params:
                        base_w = snapshot[name]
                        param.data.copy_(base_w)
                        if sample_idx % 2 == 0 and prune_ratio > 0.0:
                            flat = base_w.abs().reshape(-1)
                            k_drop = max(1, int(round(prune_ratio * flat.numel())))
                            thresh = torch.kthvalue(flat, k_drop).values
                            param.data.mul_((base_w.abs() > thresh).to(base_w.dtype))
                        noise_scale = weight_noise_std * base_w.float().std().clamp_min(1e-5)
                        noise = torch.randn(base_w.shape, device=base_w.device, dtype=torch.float32, generator=rng_gen)
                        param.data.add_((noise * noise_scale).to(base_w.dtype))

                z_mc = predict_logits(
                    model, images_uint8, device, batch_size=batch_size, use_fp16_on_cuda=use_fp16_on_cuda
                )
                fragility_weight = fragility_weight + F.cross_entropy(z_mc, y_orig, reduction="none")
                weight_mc_probs = weight_mc_probs + F.softmax(z_mc / 2.0, dim=1)
        finally:
            with torch.no_grad():
                for name, param in perturb_params:
                    param.data.copy_(snapshot[name])

        fragility_weight = fragility_weight / float(num_weight_samples)
        weight_mc_probs = weight_mc_probs / float(num_weight_samples)
    else:
        weight_mc_probs = p_clean.clone()

    # 3. Input-gradient sensitivity (MU-Mis)
    input_grad_norm = torch.zeros(len(images_uint8), dtype=torch.float32)
    if compute_input_grad:
        was_training = model.training
        orig_req_grad = {n: p.requires_grad for n, p in model.named_parameters()}
        model.eval()
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        grads_out: list[torch.Tensor] = []
        try:
            for start in range(0, len(images_uint8), batch_size):
                end = min(len(images_uint8), start + batch_size)
                x = preprocess(images_uint8[start:end], device).requires_grad_(True)
                y_batch = y_orig[start:end].to(device)
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=use_fp16_on_cuda and str(device).startswith("cuda")
                ):
                    scores = model(x).float()
                zy_sum = scores.gather(1, y_batch[:, None]).sum()
                (gx,) = torch.autograd.grad(zy_sum, x, create_graph=False)
                norms = gx.detach().float().flatten(1).norm(p=2, dim=1).cpu()
                grads_out.append(norms)
        finally:
            for n, p in model.named_parameters():
                p.requires_grad_(orig_req_grad[n])
            model.train(was_training)
        input_grad_norm = torch.cat(grads_out, dim=0)

    # 4. Leave-one-out feature outlierness
    f_hat = F.normalize(feats_clean, dim=-1)
    num_classes = int(z_clean.shape[1])
    class_sums = torch.zeros(num_classes, f_hat.shape[1], dtype=torch.float32)
    class_sums.index_add_(0, y_orig, f_hat)
    class_counts = torch.bincount(y_orig, minlength=num_classes)
    feat_outlier = torch.ones(len(images_uint8), dtype=torch.float32)
    for idx in range(len(images_uint8)):
        c = int(y_orig[idx].item())
        cnt = int(class_counts[c].item())
        if cnt > 1:
            loo_proto = F.normalize((class_sums[c] - f_hat[idx]) / float(cnt - 1), dim=0)
            feat_outlier[idx] = (1.0 - torch.dot(f_hat[idx], loo_proto)).clamp(0.0, 2.0)

    signals = [
        _rank_normalize(fragility_aug),
        _rank_normalize(feat_outlier),
    ]
    if num_weight_samples > 0:
        signals.append(_rank_normalize(fragility_weight))
    if compute_input_grad:
        signals.append(_rank_normalize(input_grad_norm))

    mem_scores = torch.stack(signals, dim=0).mean(dim=0).clamp(0.0, 1.0)
    rum_strata = torch.where(
        mem_scores >= 0.67,
        torch.full_like(y_orig, 2),
        torch.where(mem_scores >= 0.33, torch.ones_like(y_orig), torch.zeros_like(y_orig)),
    )

    return {
        "y_orig": y_orig,
        "mem_scores": mem_scores,
        "rum_strata": rum_strata,
        "fragility_aug": fragility_aug,
        "fragility_weight": fragility_weight,
        "input_grad_norm": input_grad_norm,
        "feat_outlier": feat_outlier,
        "corrupt_probs": corrupt_probs,
        "weight_mc_probs": weight_mc_probs,
    }


def _stratified_folds(labels: torch.Tensor, n_splits: int, seed: int) -> list[torch.Tensor]:
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    gen = torch.Generator().manual_seed(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    num_classes = int(labels.max().item()) + 1
    offset = 0
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(len(idx), generator=gen)].tolist()
        for pos, sample_id in enumerate(perm):
            folds[(offset + pos) % n_splits].append(sample_id)
        offset += len(perm)
    return [torch.tensor(sorted(f), dtype=torch.long) for f in folds if f]


def compute_oof_embedding_oracle(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    n_splits: int = 5,
    ridge_alpha: float = 10.0,
    ridge_scale: float = 8.0,
    knn_k: int = 15,
    knn_temp: float = 0.1,
    temperature: float = 2.0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Stratified K-fold OOF linear Ridge + cosine k-NN oracle on frozen embeddings."""
    x = F.normalize(embeddings.float().cpu(), dim=-1)
    y = labels.long().cpu()
    n, d = x.shape
    folds = _stratified_folds(y, n_splits=min(n_splits, n), seed=seed)

    oof_probs = torch.zeros(n, num_classes, dtype=torch.float32)
    oof_logits = torch.zeros(n, num_classes, dtype=torch.float32)
    all_idx = torch.arange(n, dtype=torch.long)

    for va_idx in folds:
        mask = torch.ones(n, dtype=torch.bool)
        mask[va_idx] = False
        tr_idx = all_idx[mask]

        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_va = x[va_idx]
        y_onehot = F.one_hot(y_tr, num_classes=num_classes).float()

        # Primal when D <= N_tr, dual otherwise for fast, well-conditioned Ridge solve.
        if d <= len(tr_idx):
            gram = x_tr.T @ x_tr + ridge_alpha * torch.eye(d, dtype=torch.float32)
            w = torch.linalg.solve(gram, x_tr.T @ y_onehot)
            ridge_scores = x_va @ w
        else:
            gram = x_tr @ x_tr.T + ridge_alpha * torch.eye(len(tr_idx), dtype=torch.float32)
            dual = torch.linalg.solve(gram, y_onehot)
            ridge_scores = (x_va @ x_tr.T) @ dual

        p_ridge = F.softmax((ridge_scores * ridge_scale) / temperature, dim=1)

        k_eff = min(knn_k, len(tr_idx))
        sim = x_va @ x_tr.T
        top_sim, top_pos = sim.topk(k_eff, dim=1)
        weights = F.softmax(top_sim / knn_temp, dim=1)
        neigh_labels = y_tr[top_pos]
        p_knn = torch.zeros(len(va_idx), num_classes, dtype=torch.float32)
        p_knn.scatter_add_(1, neigh_labels, weights)
        p_knn_t = F.softmax(p_knn.clamp_min(1e-8).log() / temperature, dim=1)

        p_blend = (0.5 * p_ridge + 0.5 * p_knn_t).clamp_min(1e-8)
        p_blend = p_blend / p_blend.sum(dim=1, keepdim=True)
        oof_probs[va_idx] = p_blend
        oof_logits[va_idx] = p_blend.log() * temperature

    ce_oof = F.cross_entropy(oof_logits, y, reduction="none")
    return {
        "p_oof": oof_probs,
        "logits_oof": oof_logits,
        "ce_oof": ce_oof,
    }


def build_hybrid_gold_targets(
    z_teacher_clean: torch.Tensor,
    y_all: torch.Tensor,
    mem_scores: torch.Tensor,
    num_classes: int,
    shadow_probs: torch.Tensor | None = None,
    shadow_ce_mean: float | None = None,
    shadow_ce_std: float | None = None,
    shadow_per_sample_ce: torch.Tensor | None = None,
    oof_probs: torch.Tensor | None = None,
    corrupt_probs: torch.Tensor | None = None,
    delta_min: float = 2.0,
    delta_max: float = 16.0,
    gamma_mem: float = 1.5,
    soften: float = 0.10,
    w_shift: float = 0.55,
    w_shadow: float = 0.25,
    w_oof: float = 0.10,
    w_corrupt: float = 0.10,
) -> dict[str, torch.Tensor | float]:
    """Construct calibrated per-sample delta_i, T=2 target distributions, and T=1 moment targets."""
    z = z_teacher_clean.float().cpu()
    y = y_all.long().cpu()
    m = mem_scores.float().cpu().clamp(0.0, 1.0)

    delta_i = delta_min + (delta_max - delta_min) * (m ** gamma_mem)
    t_clean = z.scatter_add(1, y[:, None], -delta_i[:, None])
    prior = compute_class_confusion_prior(z, y, num_classes=num_classes)

    def _mix_prior(probs: torch.Tensor) -> torch.Tensor:
        return (1.0 - soften) * probs + soften * prior[y] if soften > 0.0 else probs

    p_shift_t2 = _mix_prior(F.softmax(t_clean / 2.0, dim=1))
    p_shift_t1 = _mix_prior(F.softmax(t_clean, dim=1))

    components: list[tuple[float, torch.Tensor]] = [(w_shift, p_shift_t2)]
    if shadow_probs is not None:
        components.append((w_shadow, shadow_probs.float().cpu()))
    if oof_probs is not None:
        components.append((w_oof, oof_probs.float().cpu()))
    if corrupt_probs is not None:
        components.append((w_corrupt, corrupt_probs.float().cpu()))

    total_w = sum(w for w, _ in components if w > 0.0)
    if total_w <= 0.0:
        raise ValueError("Total oracle blend weight must be positive")
    target_probs_t2 = sum((w / total_w) * p for w, p in components if w > 0.0)
    assert isinstance(target_probs_t2, torch.Tensor)
    target_probs_t2 = target_probs_t2.clamp_min(1e-12)
    target_probs_t2 = target_probs_t2 / target_probs_t2.sum(dim=1, keepdim=True)

    # Sharpen T=2 target probabilities back to T=1 scale for consistent CE moment estimates.
    sharpened_t1 = F.softmax(target_probs_t2.log() * 2.0, dim=1)
    est_ce = -sharpened_t1.gather(1, y[:, None]).squeeze(1).clamp_min(1e-8).log()

    if shadow_per_sample_ce is not None:
        ref_ce = shadow_per_sample_ce.float().cpu()
    else:
        ref_ce = est_ce

    target_ce_mean = float(shadow_ce_mean) if shadow_ce_mean is not None else float(ref_ce.mean().item())
    target_ce_std = (
        float(shadow_ce_std) if shadow_ce_std is not None else float(ref_ce.std(correction=0).item())
    )
    target_sorted_ce, _ = torch.sort(ref_ce)

    return {
        "delta_i": delta_i,
        "prior": prior,
        "target_probs_t2": target_probs_t2,
        "target_probs_t1": p_shift_t1,
        "est_per_sample_ce": est_ce,
        "target_ce_mean": target_ce_mean,
        "target_ce_std": target_ce_std,
        "target_sorted_ce": target_sorted_ce,
    }
