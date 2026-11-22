"""Unit tests for oracle scoring/target synthesis and task-vector merging/souping."""

from __future__ import annotations

import pytest
import torch

from ul.merging import (
    delta_cosine_matrix,
    layerwise_interpolate,
    merge_task_vectors,
    parse_group_gammas,
    project_out_common_direction,
    soup_checkpoints,
    state_dict_delta,
)
from ul.oracle import (
    build_hybrid_gold_targets,
    compute_class_confusion_prior,
    compute_margin_power_deltas,
    compute_memorization_scores,
    compute_oof_embedding_oracle,
)


def test_memorization_scores_restores_exact_weights(tiny_model) -> None:
    before = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}
    imgs = torch.randint(0, 256, (16, 32, 32, 3), dtype=torch.uint8)

    out = compute_memorization_scores(
        tiny_model, imgs, device="cpu", num_weight_samples=2, batch_size=8, patch_size=8
    )

    # Original weights must remain bit-identical after MC pruning/noise probing
    for k, ref_v in before.items():
        torch.testing.assert_close(tiny_model.state_dict()[k], ref_v, atol=0.0, rtol=0.0)

    mem = out["mem_scores"]
    strata = out["rum_strata"]
    assert mem.shape == (16,)
    assert torch.isfinite(mem).all()
    assert (mem >= 0.0).all() and (mem <= 1.0).all()
    assert set(strata.unique().tolist()).issubset({0, 1, 2})
    assert len(strata.unique()) >= 2


def test_oof_oracle_and_hybrid_targets() -> None:
    torch.manual_seed(3)
    feats = torch.randn(30, 32)
    labels = torch.arange(30) % 10
    oof = compute_oof_embedding_oracle(feats, labels, num_classes=10, n_splits=3, knn_k=5)
    p_oof = oof["p_oof"]
    assert p_oof.shape == (30, 10)
    assert torch.isfinite(p_oof).all()
    torch.testing.assert_close(p_oof.sum(dim=1), torch.ones(30), atol=1e-5, rtol=1e-5)

    z_clean = torch.zeros(30, 10).scatter(1, labels[:, None], 10.0) + 0.1 * torch.randn(30, 10)
    prior = compute_class_confusion_prior(z_clean, labels, num_classes=10)
    assert prior.shape == (10, 10)
    torch.testing.assert_close(prior.sum(dim=1), torch.ones(10), atol=1e-5, rtol=1e-5)
    assert torch.all(torch.diag(prior) < 1e-3)

    mem = torch.linspace(0.0, 1.0, 30)
    targets = build_hybrid_gold_targets(
        z_teacher_clean=z_clean,
        y_all=labels,
        mem_scores=mem,
        num_classes=10,
        oof_probs=p_oof,
        delta_min=2.0,
        delta_max=14.0,
        gamma_mem=1.5,
        soften=0.10,
    )
    delta_i = targets["delta_i"]
    assert delta_i[0].item() == pytest.approx(2.0, abs=1e-5)
    assert delta_i[-1].item() == pytest.approx(14.0, abs=1e-5)
    assert torch.all(delta_i[1:] >= delta_i[:-1])
    torch.testing.assert_close(targets["target_probs_t2"].sum(dim=1), torch.ones(30), atol=1e-5, rtol=1e-5)


def test_task_vector_merging_and_layerwise_soup(tiny_model) -> None:
    sd_0 = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}
    deltas = []
    fishers = []
    for k in range(3):
        d = {name: torch.zeros_like(val) for name, val in sd_0.items()}
        d["blocks.3.mlp.fc1.weight"].fill_(0.1 * (k + 1))
        d["head.weight"].fill_(0.05 * (1 if k % 2 == 0 else -1))
        deltas.append(d)
        fishers.append({name: torch.full_like(val, float(k + 1)) for name, val in sd_0.items()})

    cos_mat = delta_cosine_matrix(deltas)
    assert cos_mat.shape == (3, 3)
    torch.testing.assert_close(torch.diag(cos_mat), torch.ones(3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cos_mat, cos_mat.T, atol=1e-6, rtol=1e-6)

    for method in ("sum", "mean", "ties", "fisher"):
        merged = merge_task_vectors(sd_0, deltas, method=method, alpha=1.0, trim_ratio=0.2, fishers=fishers)
        assert set(merged.keys()) == set(sd_0.keys())
        assert all(torch.isfinite(v).all() for v in merged.values())

    proj = project_out_common_direction(deltas, strength=0.5)
    assert len(proj) == 3

    gammas = parse_group_gammas("late_mlp=2.0,head=0.0")
    interp = layerwise_interpolate(sd_0, merge_task_vectors(sd_0, deltas[:1], method="sum"), group_gammas=gammas, default_gamma=1.0)
    torch.testing.assert_close(interp["head.weight"], sd_0["head.weight"], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        interp["blocks.3.mlp.fc1.weight"],
        sd_0["blocks.3.mlp.fc1.weight"] + 0.2,
        atol=1e-6,
        rtol=1e-6,
    )

    cand_sds = [{n: sd_0[n] + d[n] for n in sd_0} for d in deltas]
    souped = soup_checkpoints(sd_0, cand_sds, weights=[0.5, 0.5, 0.0], group_gammas={"head": 0.0})
    torch.testing.assert_close(souped["head.weight"], sd_0["head.weight"], atol=1e-7, rtol=1e-7)
    delta_back = state_dict_delta(souped, sd_0)
    assert delta_back["blocks.3.mlp.fc1.weight"].mean().item() == pytest.approx(0.15, abs=1e-5)


def test_margin_power_deltas_and_fragility_boost() -> None:
    num_classes = 10
    labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    margins = torch.tensor([2.0, 7.5, 10.0, 15.0, 22.0], dtype=torch.float32)
    z_clean = torch.zeros(len(labels), num_classes, dtype=torch.float32)
    z_clean.scatter_(1, labels[:, None], margins[:, None])

    deltas_plain = compute_margin_power_deltas(
        z_teacher_clean=z_clean,
        y_all=labels,
        mem_scores=None,
        base_delta=12.0,
        margin_power=-2.0,
        fragility_boost=0.0,
        delta_clamp_min=2.0,
        delta_clamp_max=36.0,
    )
    assert deltas_plain.shape == (5,)
    assert torch.all(deltas_plain[:-1] >= deltas_plain[1:])
    assert deltas_plain.min().item() >= 2.0
    assert deltas_plain.max().item() <= 36.0
    assert deltas_plain[2].item() == pytest.approx(12.0, abs=1e-5)

    mem_scores = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    deltas_tilted = compute_margin_power_deltas(
        z_teacher_clean=z_clean,
        y_all=labels,
        mem_scores=mem_scores,
        base_delta=12.0,
        margin_power=-2.0,
        fragility_boost=0.25,
        delta_clamp_min=2.0,
        delta_clamp_max=36.0,
    )
    assert deltas_tilted[2].item() == pytest.approx(12.0, abs=1e-5)
    assert deltas_tilted[4].item() > deltas_plain[4].item()
    assert deltas_tilted[1].item() < deltas_plain[1].item()

    targets = build_hybrid_gold_targets(
        z_teacher_clean=z_clean,
        y_all=labels,
        mem_scores=mem_scores,
        num_classes=num_classes,
        delta_mode="margin_power",
        base_delta=12.0,
        margin_power=-2.0,
        fragility_boost=0.25,
        soften=0.0,
        w_shift=1.0,
    )
    torch.testing.assert_close(targets["delta_i"], deltas_tilted, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(targets["target_probs_t2"].sum(dim=1), torch.ones(5), atol=1e-5, rtol=1e-5)
