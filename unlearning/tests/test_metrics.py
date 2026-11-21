"""Unit tests for Task 3 competition evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from ul.metrics import (
    aggregate_gold_ensemble,
    compute_competition_score,
    compute_forget_score,
    compute_mean_std_score,
    compute_prediction_score,
    compute_retain_score,
    kl_divergence,
    kl_from_logits,
    per_sample_ce,
    softmax_temperature,
)


def test_kl_from_logits_matches_kl_divergence() -> None:
    torch.manual_seed(0)
    z_a = torch.randn(16, 10)
    z_b = torch.randn(16, 10)
    p_a = softmax_temperature(z_a, 2.0)
    p_b = softmax_temperature(z_b, 2.0)
    kl_a = kl_from_logits(z_a, z_b, temperature=2.0)
    kl_b = kl_divergence(p_a, p_b)
    torch.testing.assert_close(kl_a, kl_b, atol=1e-5, rtol=1e-4)
    assert torch.all(kl_a >= 0.0)


def test_mean_std_and_prediction_score_endpoints() -> None:
    torch.manual_seed(1)
    y = torch.arange(20) % 5
    z_orig = torch.zeros(20, 5)
    z_orig.scatter_(1, y[:, None], 10.0)
    z_gold = torch.randn(20, 5)

    ce_orig = per_sample_ce(z_orig, y)
    ce_gold = per_sample_ce(z_gold, y)
    gold_mean = float(ce_gold.mean().item())
    gold_std = float(ce_gold.std(correction=0).item())

    p_orig = softmax_temperature(z_orig, 2.0)
    p_gold = softmax_temperature(z_gold, 2.0)

    # sub == orig -> scores 0
    ms_orig = compute_mean_std_score(ce_orig, ce_orig, gold_mean, gold_std)
    pr_orig = compute_prediction_score(p_gold, p_orig, p_orig)
    assert ms_orig["mean_std_score"] == pytest.approx(0.0, abs=1e-6)
    assert pr_orig["prediction_score"] == pytest.approx(0.0, abs=1e-6)

    # sub == gold -> scores 1
    ms_gold = compute_mean_std_score(ce_gold, ce_orig, gold_mean, gold_std)
    pr_gold = compute_prediction_score(p_gold, p_gold, p_orig)
    assert ms_gold["mean_std_score"] == pytest.approx(1.0, abs=1e-6)
    assert pr_gold["prediction_score"] == pytest.approx(1.0, abs=1e-6)


def test_aggregate_gold_ensemble_averages_probs_not_logits() -> None:
    y = torch.tensor([0, 1])
    z1 = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    z2 = torch.tensor([[0.0, 4.0], [4.0, 0.0]])
    agg = aggregate_gold_ensemble([z1, z2], y, temperature=2.0)
    # Softmax([4, 0]/2) = [e^2/(1+e^2), 1/(1+e^2)]; averaged with flipped model gives [0.5, 0.5]
    expected_p = torch.full((2, 2), 0.5)
    torch.testing.assert_close(agg["p_gold"], expected_p, atol=1e-6, rtol=1e-6)
    ce1 = per_sample_ce(z1, y)
    ce2 = per_sample_ce(z2, y)
    expected_mean = 0.5 * (ce1.mean().item() + ce2.mean().item())
    assert agg["mean_L_gold"] == pytest.approx(expected_mean, abs=1e-6)


def test_competition_score_and_utility_gate() -> None:
    torch.manual_seed(2)
    y = torch.arange(16) % 4
    z_orig = torch.zeros(16, 4).scatter(1, y[:, None], 8.0)
    z_gold = torch.randn(16, 4)
    ce_orig = per_sample_ce(z_orig, y)
    ce_gold = per_sample_ce(z_gold, y)
    p_orig = softmax_temperature(z_orig, 2.0)
    p_gold = softmax_temperature(z_gold, 2.0)

    fg = compute_forget_score(
        clean_ce_sub=ce_gold,
        clean_ce_orig=ce_orig,
        clean_gold_mean=float(ce_gold.mean()),
        clean_gold_std=float(ce_gold.std(correction=0)),
        clean_p_gold=p_gold,
        clean_p_sub=p_gold,
        clean_p_orig=p_orig,
        alt_ce_sub=ce_gold,
        alt_ce_orig=ce_orig,
        alt_gold_mean=float(ce_gold.mean()),
        alt_gold_std=float(ce_gold.std(correction=0)),
        alt_p_gold=p_gold,
        alt_p_sub=p_gold,
        alt_p_orig=p_orig,
    )
    ret = compute_retain_score(z_orig, z_orig, labels_ctrl=y)
    assert fg["forget_score"] == pytest.approx(1.0, abs=1e-6)
    assert ret["retain_score"] == pytest.approx(1.0, abs=1e-6)

    # Utility gate passed (>= 0.70)
    passed = compute_competition_score(fg, ret, val_acc=0.72)
    assert passed["passed_utility_gate"] is True
    assert passed["final_score"] == pytest.approx(1.0, abs=1e-6)

    # Utility gate failed (< 0.70) zeroes both forget_score and retain_score
    disqualified = compute_competition_score(fg, ret, val_acc=0.69)
    assert disqualified["passed_utility_gate"] is False
    assert disqualified["forget_score"] == 0.0
    assert disqualified["retain_score"] == 0.0
    assert disqualified["final_score"] == 0.0
