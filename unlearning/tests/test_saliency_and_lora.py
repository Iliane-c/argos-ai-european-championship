"""Unit tests for ViT saliency, SSD, pruning, mergeable LoRA, and OGP gradient projection."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from hybrid_unlearn import HybridUnlearnConfig, run_hybrid_unlearning
from ul.saliency import (
    ActivationSubspaceProjector,
    apply_grad_mask,
    apply_magnitude_pruning,
    apply_ssd_dampening,
    build_active_head_mask,
    build_salun_mask,
    classify_vit_param,
    compute_diagonal_fisher,
    inject_lora,
    merge_and_unload_lora,
    select_trainable_params,
)


def test_classify_vit_param_depth_4_and_12() -> None:
    assert classify_vit_param("patch_embed.proj.weight", depth=12) == "patch_embed"
    assert classify_vit_param("pos_embed", depth=12) == "pos_embed"
    assert classify_vit_param("blocks.0.attn.qkv.weight", depth=12) == "early_attn"
    assert classify_vit_param("blocks.5.mlp.fc1.weight", depth=12) == "mid_mlp"
    assert classify_vit_param("blocks.11.mlp.fc2.bias", depth=12) == "late_mlp"
    assert classify_vit_param("blocks.2.ls1.gamma", depth=4) == "layerscale"
    assert classify_vit_param("blocks.3.mlp.fc1.weight", depth=4) == "late_mlp"
    assert classify_vit_param("norm.weight", depth=4) == "norm"
    assert classify_vit_param("head.weight", depth=4) == "head"


def test_select_trainable_params_scopes(tiny_model) -> None:
    late_mlp = select_trainable_params(tiny_model, scope="late_mlp", train_blocks=2, include_head=True, include_norm=False)
    assert all((".mlp." in n and n.startswith(("blocks.2.", "blocks.3."))) or n.startswith("head.") for n in late_mlp)

    head_only = select_trainable_params(tiny_model, scope="head_only", include_norm=False)
    assert head_only == ["head.bias", "head.weight"]

    all_p = select_trainable_params(tiny_model, scope="all")
    assert len(all_p) == len(list(tiny_model.named_parameters()))


def test_fisher_salun_ssd_and_pruning(tiny_model) -> None:
    imgs = torch.randint(0, 256, (8, 32, 32, 3), dtype=torch.uint8)
    labels = torch.arange(8) % 10
    allowed = set(select_trainable_params(tiny_model, scope="late_mlp", train_blocks=2))

    f_emp = compute_diagonal_fisher(
        tiny_model, imgs, device="cpu", mode="empirical", labels=labels, param_names=allowed, per_sample=True
    )
    f_pred = compute_diagonal_fisher(
        tiny_model, imgs, device="cpu", mode="predictive", param_names=allowed, per_sample=False, batch_size=4
    )
    for f_dict in (f_emp, f_pred):
        assert set(f_dict.keys()) == allowed
        for v in f_dict.values():
            assert torch.isfinite(v).all() and (v >= 0.0).all()

    mask_salun = build_salun_mask(f_emp, keep_ratio=0.5, mode="salun", allowed_names=allowed)
    mask_rel = build_salun_mask(f_emp, fisher_retain=f_pred, keep_ratio=0.4, mode="relative_ssd", allowed_names=allowed)
    frac_salun = torch.cat([m.reshape(-1) for m in mask_salun.values()]).mean().item()
    frac_rel = torch.cat([m.reshape(-1) for m in mask_rel.values()]).mean().item()
    assert frac_salun == pytest.approx(0.5, abs=0.02)
    assert frac_rel == pytest.approx(0.4, abs=0.02)

    # Check apply_grad_mask zeroes masked grads
    for n, p in tiny_model.named_parameters():
        if n in allowed:
            p.grad = torch.ones_like(p)
    apply_grad_mask(tiny_model, mask_salun)
    for n, m in mask_salun.items():
        g = dict(tiny_model.named_parameters())[n].grad
        assert torch.all(g[m == 0] == 0.0)

    # Check SSD dampening scales only weights where F_f > alpha * F_r
    m_ssd = copy.deepcopy(tiny_model)
    ff_syn = {n: torch.zeros_like(p) for n, p in m_ssd.named_parameters() if n in allowed}
    fr_syn = {n: torch.ones_like(p) for n, p in m_ssd.named_parameters() if n in allowed}
    target_name = "blocks.3.mlp.fc1.weight"
    ff_syn[target_name].fill_(50.0)
    stats = apply_ssd_dampening(m_ssd, ff_syn, fr_syn, alpha=10.0, damp_lambda=1.0, min_scale=0.1, allowed_names=allowed)
    assert stats["dampened_fraction"] > 0.0
    w_before = dict(tiny_model.named_parameters())[target_name].data
    w_after = dict(m_ssd.named_parameters())[target_name].data
    torch.testing.assert_close(w_after, w_before * 0.1, atol=1e-6, rtol=1e-5)

    # Check magnitude pruning sparsity
    m_prune = copy.deepcopy(tiny_model)
    p_stats = apply_magnitude_pruning(m_prune, sparsity=0.25, allowed_names=allowed)
    assert p_stats["pruned_fraction"] == pytest.approx(0.25, abs=0.02)


def test_lora_inject_train_and_exact_merge(tiny_model) -> None:
    ref_state = {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out_before = tiny_model(x)

    replaced = inject_lora(tiny_model, target_patterns=("mlp.fc1", "mlp.fc2", "attn.proj"), rank=4, alpha=8.0, train_blocks=2)
    assert len(replaced) == 6
    with torch.no_grad():
        out_zero_init = tiny_model(x)
    torch.testing.assert_close(out_before, out_zero_init, atol=1e-6, rtol=1e-6)

    # Run a few gradient steps on LoRA weights
    opt = torch.optim.SGD([p for p in tiny_model.parameters() if p.requires_grad], lr=0.05)
    for _ in range(3):
        opt.zero_grad()
        loss = F.cross_entropy(tiny_model(x), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        opt.step()

    with torch.no_grad():
        out_lora_trained = tiny_model(x)

    merged_count = merge_and_unload_lora(tiny_model)
    assert merged_count == 6
    with torch.no_grad():
        out_merged = tiny_model(x)
    torch.testing.assert_close(out_lora_trained, out_merged, atol=1e-5, rtol=1e-5)

    # State dict must match original schema strictly
    assert set(tiny_model.state_dict().keys()) == set(ref_state.keys())
    for k, ref_v in ref_state.items():
        assert tiny_model.state_dict()[k].shape == ref_v.shape


def test_orthogonal_gradient_projection(tiny_model) -> None:
    retain_imgs = torch.randint(0, 256, (12, 32, 32, 3), dtype=torch.uint8)
    target_layers = ["blocks.3.mlp.fc1", "blocks.3.mlp.fc2"]
    select_trainable_params(tiny_model, scope="late_mlp", train_blocks=1)

    proj = ActivationSubspaceProjector()
    ranks = proj.fit(tiny_model, retain_imgs, device="cpu", target_linear_names=target_layers, variance_threshold=0.95, max_rank=16)
    assert all(r > 0 for r in ranks.values())

    x = torch.randn(4, 3, 32, 32)
    loss = tiny_model(x).sum()
    loss.backward()
    proj.project_gradients(tiny_model)

    for layer_name in target_layers:
        grad = tiny_model.get_submodule(layer_name).weight.grad
        v = proj.basis[layer_name]
        assert grad is not None
        residual = (grad @ v).norm().item()
        assert residual < 1e-4


def test_build_active_head_mask_freezes_inactive_classes(tiny_model) -> None:
    active = [1, 3, 7]
    mask = build_active_head_mask(tiny_model, active)
    assert set(mask.keys()) == {"head.weight", "head.bias"}
    inactive = [c for c in range(10) if c not in active]
    assert torch.all(mask["head.weight"][inactive] == 0.0)
    assert torch.all(mask["head.weight"][active] == 1.0)
    assert torch.all(mask["head.bias"][inactive] == 0.0)
    assert torch.all(mask["head.bias"][active] == 1.0)

    teacher = copy.deepcopy(tiny_model).eval()
    student = copy.deepcopy(tiny_model)
    w0 = teacher.head.weight.detach().clone()
    b0 = teacher.head.bias.detach().clone()

    torch.manual_seed(9)
    forget_imgs = torch.randint(0, 256, (12, 32, 32, 3), dtype=torch.uint8)
    retain_imgs = torch.randint(0, 256, (12, 32, 32, 3), dtype=torch.uint8)
    cfg = HybridUnlearnConfig(
        epochs=2,
        bs=6,
        eval_bs=12,
        lr=1e-2,
        head_lr_scale=1.0,
        scope="head_only",
        include_norm=False,
        freeze_inactive_head=True,
        amp="fp32",
        w_retain=0.0,
        w_feat_retain=0.0,
        w_ewc=0.0,
        polish_epochs=0,
    )
    trained, _ = run_hybrid_unlearning(
        cfg=cfg,
        teacher=teacher,
        student=student,
        forget_imgs=forget_imgs,
        retain_fit=retain_imgs,
        active_classes=torch.tensor(active, dtype=torch.long),
        device="cpu",
    )
    torch.testing.assert_close(trained.head.weight[inactive], w0[inactive], atol=0.0, rtol=0.0)
    torch.testing.assert_close(trained.head.bias[inactive], b0[inactive], atol=0.0, rtol=0.0)
    assert (trained.head.weight[active] - w0[active]).abs().max().item() > 1e-5
