"""Unit tests for all hybrid unlearning loss modules."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ul.losses import (
    MemoryBankMomentLoss,
    ewc_fisher_penalty,
    feature_anchor_loss,
    kd_loss,
    local_smoothness_loss,
    moment_matching_loss,
    npo_loss,
    scar_feature_push_loss,
    wasserstein_1d_ce_loss,
)


def test_kd_moment_and_wasserstein_losses_backward() -> None:
    torch.manual_seed(0)
    logits = torch.randn(12, 10, requires_grad=True)
    target_probs = F.softmax(torch.randn(12, 10) / 2.0, dim=1)
    labels = torch.arange(12) % 10

    l_kd = kd_loss(target_probs, logits, temperature=2.0)
    l_mom = moment_matching_loss(logits, labels, target_mean=1.5, target_std=1.2)
    sorted_target, _ = torch.sort(torch.linspace(0.05, 4.0, 20))
    l_was = wasserstein_1d_ce_loss(logits, labels, sorted_target)
    total = l_kd + l_mom + l_was
    total.backward()

    assert torch.isfinite(total)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.norm().item() > 0.0


def test_memory_bank_moment_loss_updates_and_flows() -> None:
    bank_loss = MemoryBankMomentLoss(bank_size=16)
    sorted_t = torch.linspace(0.1, 3.0, 16)
    for step in range(3):
        logits = torch.randn(6, 10, requires_grad=True)
        labels = torch.arange(6) % 10
        loss = bank_loss(logits, labels, target_mean=1.2, target_std=0.8, target_sorted_ce=sorted_t, mode="wasserstein" if step == 2 else "mean_std")
        loss.backward()
        assert torch.isfinite(loss)
        assert logits.grad is not None and logits.grad.norm().item() > 0.0
    assert bank_loss.bank.numel() == 16


def test_npo_exact_formula_and_dampening() -> None:
    beta = 0.25
    labels = torch.tensor([0, 2])
    z_teacher = torch.tensor([[5.0, 1.0, 0.0], [0.0, 1.0, 5.0]])
    z_student = torch.tensor([[2.0, 1.0, 0.0], [-10.0, 2.0, -10.0]], requires_grad=True)

    loss_val = npo_loss(z_student, z_teacher, labels, beta=beta)

    with torch.no_grad():
        pt = F.softmax(z_teacher, dim=1).gather(1, labels[:, None]).squeeze(1)
        ps = F.softmax(z_student, dim=1).gather(1, labels[:, None]).squeeze(1)
        expected = ((2.0 / beta) * torch.log(1.0 + (ps / pt) ** beta)).mean()

    assert loss_val.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-5)

    # Check self-dampening: NPO stops pushing once p_s << p_t (unlike unbounded GradAscent)
    z_mid = torch.tensor([[1.2, 1.0, 0.5], [0.5, 1.0, 1.2]], requires_grad=True)
    z_low = torch.tensor([[-8.0, 3.0, 3.0], [3.0, 3.0, -8.0]], requires_grad=True)
    z_ga = z_low.clone().detach().requires_grad_(True)
    npo_loss(z_mid, z_teacher, labels, beta=1.0).backward()
    npo_loss(z_low, z_teacher, labels, beta=1.0).backward()
    (-F.log_softmax(z_ga, dim=1).gather(1, labels[:, None]).mean()).backward()
    assert z_low.grad.norm().item() < 0.01 * z_mid.grad.norm().item()
    assert z_low.grad.norm().item() < 0.01 * z_ga.grad.norm().item()


def test_smoothness_scar_feature_and_ewc_losses(tiny_model) -> None:
    z_a = torch.randn(8, 10, requires_grad=True)
    z_b = torch.randn(8, 10, requires_grad=True)
    l_sm = local_smoothness_loss(z_a, z_b, temperature=2.0)
    assert l_sm.item() >= 0.0
    l_sm.backward()
    assert z_a.grad.norm().item() > 0.0 and z_b.grad.norm().item() > 0.0

    feats = torch.randn(8, 32, requires_grad=True)
    t_feats = torch.randn(8, 32)
    means = torch.randn(10, 32)
    inv_vars = torch.ones(10, 32)
    labels = torch.arange(8) % 10
    l_scar = scar_feature_push_loss(feats, labels, means, inv_vars, margin=1.0)
    l_anc = feature_anchor_loss(feats, t_feats)
    (l_scar + l_anc).backward()
    assert torch.isfinite(l_scar) and torch.isfinite(l_anc)
    assert feats.grad.norm().item() > 0.0

    orig_params = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    fisher = {n: torch.ones_like(p) for n, p in tiny_model.named_parameters()}
    assert ewc_fisher_penalty(tiny_model, orig_params, fisher).item() == pytest.approx(0.0, abs=1e-8)

    with torch.no_grad():
        tiny_model.head.weight.add_(0.1)
    ewc_val = ewc_fisher_penalty(tiny_model, orig_params, fisher)
    assert ewc_val.item() > 0.0
    ewc_val.backward()
    assert tiny_model.head.weight.grad is not None
