"""Hybrid unlearning objectives: KD, moment matching, NPO, smoothness, SCAR, and Fisher EWC."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(
    target_probs: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 2.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward KL divergence KL(target_probs || softmax(student_logits / T)) * T^2."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    log_q = F.log_softmax(student_logits.float() / temperature, dim=1)
    per_sample_kl = F.kl_div(log_q, target_probs.float(), reduction="none").sum(dim=1) * (temperature**2)
    if sample_weights is not None:
        w = sample_weights.float().to(per_sample_kl.device)
        return (per_sample_kl * w).sum() / w.sum().clamp_min(1e-8)
    return per_sample_kl.mean()


def moment_matching_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mean: float,
    target_std: float,
    huber_beta: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Smooth L1 loss matching per-sample T=1 cross-entropy mean and std to target moments."""
    ce = F.cross_entropy(student_logits.float(), labels.long(), reduction="none")
    return _moment_loss_from_ce(ce, target_mean, target_std, huber_beta=huber_beta, eps=eps)


def _moment_loss_from_ce(
    ce: torch.Tensor,
    target_mean: float,
    target_std: float,
    huber_beta: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    mu = ce.mean()
    sigma = ce.var(unbiased=False).clamp_min(eps).sqrt()
    t_mu = torch.as_tensor(target_mean, device=ce.device, dtype=ce.dtype)
    t_sigma = torch.as_tensor(target_std, device=ce.device, dtype=ce.dtype)
    return F.smooth_l1_loss(mu, t_mu, beta=huber_beta) + F.smooth_l1_loss(sigma, t_sigma, beta=huber_beta)


def wasserstein_1d_ce_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    target_sorted_ce: torch.Tensor,
    huber_beta: float = 0.25,
) -> torch.Tensor:
    """1D Wasserstein quantile matching on sorted T=1 cross-entropy losses."""
    ce = F.cross_entropy(student_logits.float(), labels.long(), reduction="none")
    return _wasserstein_from_ce(ce, target_sorted_ce, huber_beta=huber_beta)


def _wasserstein_from_ce(
    ce: torch.Tensor,
    target_sorted_ce: torch.Tensor,
    huber_beta: float = 0.25,
) -> torch.Tensor:
    ce_sorted, _ = torch.sort(ce)
    target = target_sorted_ce.float().to(ce.device)
    if len(target) != len(ce_sorted):
        if len(ce_sorted) == 1:
            target = target.mean().unsqueeze(0)
        else:
            target = F.interpolate(
                target.view(1, 1, -1),
                size=len(ce_sorted),
                mode="linear",
                align_corners=True,
            ).view(-1)
    return F.smooth_l1_loss(ce_sorted, target, beta=huber_beta)


class MemoryBankMomentLoss(nn.Module):
    """Differentiable distribution matcher combining live batch CE with a detached ring buffer."""

    def __init__(self, bank_size: int = 512, huber_beta: float = 0.25) -> None:
        super().__init__()
        if bank_size < 0:
            raise ValueError(f"bank_size must be non-negative, got {bank_size}")
        self.bank_size = int(bank_size)
        self.huber_beta = float(huber_beta)
        self.register_buffer("bank", torch.empty(0, dtype=torch.float32))

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        target_mean: float,
        target_std: float,
        target_sorted_ce: torch.Tensor | None = None,
        mode: str = "mean_std",
    ) -> torch.Tensor:
        if mode not in {"mean_std", "wasserstein"}:
            raise ValueError(f"Unknown moment matching mode '{mode}'")
        if mode == "wasserstein" and target_sorted_ce is None:
            raise ValueError("mode='wasserstein' requires target_sorted_ce")

        ce_live = F.cross_entropy(student_logits.float(), labels.long(), reduction="none")
        if self.bank_size > 0 and self.bank.numel() > 0:
            bank_vals = self.bank.to(device=ce_live.device, dtype=ce_live.dtype).detach()
            combined = torch.cat([ce_live, bank_vals], dim=0)
            # Rescale so batch-size vs bank-size does not attenuate live sample gradients.
            grad_scale = min(8.0, float(len(combined)) / float(max(1, len(ce_live))))
        else:
            combined = ce_live
            grad_scale = 1.0

        if mode == "mean_std":
            loss = _moment_loss_from_ce(combined, target_mean, target_std, huber_beta=self.huber_beta)
        else:
            assert target_sorted_ce is not None
            loss = _wasserstein_from_ce(combined, target_sorted_ce, huber_beta=self.huber_beta)

        if self.bank_size > 0 and self.training:
            with torch.no_grad():
                updated = torch.cat([self.bank.to(ce_live.device), ce_live.detach()], dim=0)
                self.bank = updated[-self.bank_size :].detach().cpu()

        return loss * grad_scale


def npo_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.2,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative Preference Optimization loss (Zhang et al., COLM 2024): bounded forgetting."""
    if beta <= 0:
        raise ValueError(f"NPO beta must be positive, got {beta}")
    y = labels.long()
    log_p_s = F.log_softmax(student_logits.float(), dim=1).gather(1, y[:, None]).squeeze(1)
    log_p_t = F.log_softmax(teacher_logits.detach().float(), dim=1).gather(1, y[:, None]).squeeze(1)
    losses = (2.0 / beta) * F.softplus(beta * (log_p_s - log_p_t))
    if sample_weights is not None:
        w = sample_weights.float().to(losses.device)
        denom = w.sum()
        if denom <= 1e-8:
            return losses.new_zeros(())
        return (losses * w).sum() / denom
    return losses.mean()


def local_smoothness_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Symmetric KL consistency between two views of the same sample (MU-Mis/JiT smoothness)."""
    p_a = F.softmax(logits_a.float() / temperature, dim=1)
    p_b = F.softmax(logits_b.float() / temperature, dim=1)
    log_a = F.log_softmax(logits_a.float() / temperature, dim=1)
    log_b = F.log_softmax(logits_b.float() / temperature, dim=1)
    kl_ab = (p_a * (log_a - log_b)).sum(dim=1)
    kl_ba = (p_b * (log_b - log_a)).sum(dim=1)
    return 0.5 * (kl_ab + kl_ba).mean() * (temperature**2)


def scar_feature_push_loss(
    student_feats: torch.Tensor,
    labels: torch.Tensor,
    class_means: torch.Tensor,
    class_inv_vars: torch.Tensor | None = None,
    margin: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push penultimate features toward nearest incorrect class prototype (SCAR-style)."""
    feats = student_feats.float()
    means = class_means.float().to(feats.device)
    diff = feats.unsqueeze(1) - means.unsqueeze(0)
    if class_inv_vars is not None:
        inv_vars = class_inv_vars.float().to(feats.device)
        dist = (diff.square() * inv_vars.unsqueeze(0)).mean(dim=-1)
    else:
        dist = diff.square().mean(dim=-1)

    y = labels.long()
    one_hot = F.one_hot(y, num_classes=means.shape[0]).bool()
    wrong_dist = dist.masked_fill(one_hot, float("inf")).min(dim=1).values
    true_dist = dist.gather(1, y[:, None]).squeeze(1)
    losses = F.relu(wrong_dist - true_dist + margin)

    if sample_weights is not None:
        w = sample_weights.float().to(losses.device)
        denom = w.sum()
        if denom <= 1e-8:
            return losses.new_zeros(())
        return (losses * w).sum() / denom
    return losses.mean()


def feature_anchor_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    """Anchor penultimate retain representations via normalized MSE + cosine similarity."""
    s = F.normalize(student_feats.float(), dim=-1)
    t = F.normalize(teacher_feats.detach().float(), dim=-1)
    cos_term = (1.0 - (s * t).sum(dim=-1)).mean()
    mse_term = F.mse_loss(s, t)
    return 0.5 * cos_term + 0.5 * mse_term


def ewc_fisher_penalty(
    model: nn.Module,
    orig_params: Mapping[str, torch.Tensor],
    fisher_retain: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Quadratic Fisher trust-region penalty 0.5 * sum(F_r * (theta - theta_0)^2)."""
    total: torch.Tensor | None = None
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in fisher_retain or name not in orig_params:
            continue
        p0 = orig_params[name].to(device=param.device, dtype=torch.float32)
        f_r = fisher_retain[name].to(device=param.device, dtype=torch.float32)
        term = 0.5 * (f_r * (param.float() - p0).square()).sum()
        total = term if total is None else total + term
    if total is None:
        first = next(model.parameters())
        return first.new_zeros((), dtype=torch.float32)
    return total
