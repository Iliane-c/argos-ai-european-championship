"""Exact Task 3 evaluation metrics matching the Unlearning.pdf specification."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_temperature(logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Compute softmax probabilities at the given temperature (default T=2)."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return F.softmax(logits.float() / temperature, dim=-1)


def kl_divergence(p_target: torch.Tensor, q_pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample KL(p_target || q_pred) in nats, clamped non-negative."""
    p = p_target.float().clamp_min(eps)
    q = q_pred.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl = (p * (p.log() - q.log())).sum(dim=-1)
    return kl.clamp_min(0.0)


def kl_from_logits(
    logits_target: torch.Tensor, logits_pred: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    """Numerically exact per-sample KL(softmax(z_target/T) || softmax(z_pred/T))."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    log_p = F.log_softmax(logits_target.float() / temperature, dim=-1)
    log_q = F.log_softmax(logits_pred.float() / temperature, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).clamp_min(0.0)


def per_sample_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample T=1 cross-entropy loss L_m(x)."""
    return F.cross_entropy(logits.float(), labels.long(), reduction="none")


def aggregate_gold_ensemble(
    gold_logits_list: list[torch.Tensor],
    labels: torch.Tensor,
    temperature: float = 2.0,
    correction: int = 0,
) -> dict[str, torch.Tensor | float]:
    """Compute gold reference statistics across independently retrained models.

    Matches Unlearning.pdf: mean(L_gold) and std(L_gold) are averaged across gold models,
    while p_gold(x) is the average of each gold model's softmax(logits / 2).
    """
    if not gold_logits_list:
        raise ValueError("gold_logits_list must contain at least one gold model's logits")
    means: list[float] = []
    stds: list[float] = []
    ce_per_model: list[torch.Tensor] = []
    probs_sum = torch.zeros_like(gold_logits_list[0], dtype=torch.float32)

    for z_gold in gold_logits_list:
        ce = per_sample_ce(z_gold, labels)
        ce_per_model.append(ce)
        means.append(float(ce.mean().item()))
        stds.append(float(ce.std(correction=correction).item()))
        probs_sum = probs_sum + softmax_temperature(z_gold, temperature=temperature)

    n_models = len(gold_logits_list)
    return {
        "mean_L_gold": float(sum(means) / n_models),
        "std_L_gold": float(sum(stds) / n_models),
        "p_gold": probs_sum / n_models,
        "mean_per_sample_ce": torch.stack(ce_per_model, dim=0).mean(dim=0),
    }


def compute_mean_std_score(
    ce_sub: torch.Tensor,
    ce_orig: torch.Tensor,
    gold_ce_mean: float,
    gold_ce_std: float,
    correction: int = 0,
) -> dict[str, float]:
    """Compute Loss-distribution score: 1 - clip(d_sub / d_orig, 0, 1)."""
    sub_mean = float(ce_sub.float().mean().item())
    sub_std = float(ce_sub.float().std(correction=correction).item())
    orig_mean = float(ce_orig.float().mean().item())
    orig_std = float(ce_orig.float().std(correction=correction).item())

    d_sub = abs(sub_mean - gold_ce_mean) + abs(sub_std - gold_ce_std)
    d_orig = abs(orig_mean - gold_ce_mean) + abs(orig_std - gold_ce_std)

    if d_orig <= 1e-12:
        score = 1.0 if d_sub <= 1e-12 else 0.0
    else:
        ratio = max(0.0, min(1.0, d_sub / d_orig))
        score = 1.0 - ratio

    return {
        "mean_std_score": float(score),
        "d_sub": float(d_sub),
        "d_orig": float(d_orig),
        "sub_ce_mean": sub_mean,
        "sub_ce_std": sub_std,
        "orig_ce_mean": orig_mean,
        "orig_ce_std": orig_std,
    }


def compute_prediction_score(
    p_gold: torch.Tensor,
    p_sub: torch.Tensor,
    p_orig: torch.Tensor,
) -> dict[str, float]:
    """Compute Prediction score: 1 - clip(mean KL(p_gold || p_sub) / mean KL(p_gold || p_orig), 0, 1)."""
    kl_sub = float(kl_divergence(p_gold, p_sub).mean().item())
    kl_orig = float(kl_divergence(p_gold, p_orig).mean().item())

    if kl_orig <= 1e-12:
        score = 1.0 if kl_sub <= 1e-12 else 0.0
    else:
        ratio = max(0.0, min(1.0, kl_sub / kl_orig))
        score = 1.0 - ratio

    return {
        "prediction_score": float(score),
        "kl_gold_sub": kl_sub,
        "kl_gold_orig": kl_orig,
    }


def compute_forget_score(
    clean_ce_sub: torch.Tensor,
    clean_ce_orig: torch.Tensor,
    clean_gold_mean: float,
    clean_gold_std: float,
    clean_p_gold: torch.Tensor,
    clean_p_sub: torch.Tensor,
    clean_p_orig: torch.Tensor,
    alt_ce_sub: torch.Tensor,
    alt_ce_orig: torch.Tensor,
    alt_gold_mean: float,
    alt_gold_std: float,
    alt_p_gold: torch.Tensor,
    alt_p_sub: torch.Tensor,
    alt_p_orig: torch.Tensor,
    correction: int = 0,
) -> dict[str, float]:
    """Compute Forget quality equally across clean and altered forget images."""
    clean_ms = compute_mean_std_score(
        clean_ce_sub, clean_ce_orig, clean_gold_mean, clean_gold_std, correction=correction
    )
    alt_ms = compute_mean_std_score(
        alt_ce_sub, alt_ce_orig, alt_gold_mean, alt_gold_std, correction=correction
    )
    clean_pred = compute_prediction_score(clean_p_gold, clean_p_sub, clean_p_orig)
    alt_pred = compute_prediction_score(alt_p_gold, alt_p_sub, alt_p_orig)

    mean_std_score = 0.5 * (clean_ms["mean_std_score"] + alt_ms["mean_std_score"])
    prediction_score = 0.5 * (clean_pred["prediction_score"] + alt_pred["prediction_score"])
    forget_score = mean_std_score * prediction_score

    return {
        "forget_score": float(forget_score),
        "mean_std_score": float(mean_std_score),
        "prediction_score": float(prediction_score),
        "clean_mean_std_score": clean_ms["mean_std_score"],
        "alt_mean_std_score": alt_ms["mean_std_score"],
        "clean_prediction_score": clean_pred["prediction_score"],
        "alt_prediction_score": alt_pred["prediction_score"],
        "clean_sub_ce_mean": clean_ms["sub_ce_mean"],
        "clean_sub_ce_std": clean_ms["sub_ce_std"],
        "alt_sub_ce_mean": alt_ms["sub_ce_mean"],
        "alt_sub_ce_std": alt_ms["sub_ce_std"],
        "clean_kl_gold_sub": clean_pred["kl_gold_sub"],
        "alt_kl_gold_sub": alt_pred["kl_gold_sub"],
    }


def compute_retain_score(
    logits_orig_ctrl: torch.Tensor,
    logits_sub_ctrl: torch.Tensor,
    labels_ctrl: torch.Tensor | None = None,
    temperature: float = 2.0,
) -> dict[str, float]:
    """Compute Retain quality on a control set (agreement proxy when ground-truth labels absent)."""
    if labels_ctrl is None:
        labels_ref = logits_orig_ctrl.argmax(dim=-1)
        acc_orig = 1.0
    else:
        labels_ref = labels_ctrl.long()
        acc_orig = float((logits_orig_ctrl.argmax(dim=-1) == labels_ref).float().mean().item())

    acc_sub = float((logits_sub_ctrl.argmax(dim=-1) == labels_ref).float().mean().item())
    if acc_orig <= 1e-12:
        retain_accuracy_score = 1.0 if acc_sub <= 1e-12 else 0.0
    else:
        retain_accuracy_score = max(0.0, min(1.0, acc_sub / acc_orig))

    mean_kl = float(kl_from_logits(logits_orig_ctrl, logits_sub_ctrl, temperature=temperature).mean().item())
    retain_kl_similarity = 1.0 / (1.0 + mean_kl)
    retain_score = retain_accuracy_score * retain_kl_similarity

    return {
        "retain_score": float(retain_score),
        "retain_accuracy_score": float(retain_accuracy_score),
        "retain_kl_similarity": float(retain_kl_similarity),
        "retain_mean_kl": float(mean_kl),
        "retain_acc_sub": float(acc_sub),
        "retain_acc_orig": float(acc_orig),
    }


def compute_competition_score(
    forget_metrics: dict[str, float],
    retain_metrics: dict[str, float],
    val_acc: float | None = None,
    utility_gate: float = 0.70,
) -> dict[str, float | bool]:
    """Combine forget and retain scores multiplicatively with the 70% validation gate."""
    raw_forget = float(forget_metrics["forget_score"])
    raw_retain = float(retain_metrics["retain_score"])
    raw_final = raw_forget * raw_retain

    passed_gate = True if val_acc is None else bool(val_acc >= utility_gate)
    forget_score = raw_forget if passed_gate else 0.0
    retain_score = raw_retain if passed_gate else 0.0
    final_score = raw_final if passed_gate else 0.0

    result: dict[str, float | bool] = {
        **forget_metrics,
        **retain_metrics,
        "raw_forget_score": raw_forget,
        "raw_retain_score": raw_retain,
        "raw_final_score": raw_final,
        "forget_score": forget_score,
        "retain_score": retain_score,
        "final_score": final_score,
        "passed_utility_gate": passed_gate,
    }
    if val_acc is not None:
        result["val_acc"] = float(val_acc)
    return result
