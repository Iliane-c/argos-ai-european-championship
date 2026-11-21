"""Task-vector arithmetic, TIES/Fisher merging, layerwise interpolation, and model souping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .saliency import classify_vit_param, infer_vit_depth


def state_dict_delta(
    sd_u: Mapping[str, torch.Tensor],
    sd_0: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute parameter-wise delta theta_u - theta_0 on CPU float32."""
    return {
        k: (sd_u[k].detach().cpu().float() - sd_0[k].detach().cpu().float())
        for k in sd_0
    }


def delta_cosine_matrix(deltas: Sequence[Mapping[str, torch.Tensor]]) -> torch.Tensor:
    """Compute [K, K] pairwise cosine similarity across flattened weight deltas."""
    k = len(deltas)
    if k == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    keys = list(deltas[0].keys())
    flat = [
        torch.cat([d[name].detach().cpu().float().reshape(-1) for name in keys])
        for d in deltas
    ]
    mat = torch.stack(flat, dim=0)
    norms = mat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    normed = mat / norms
    sim = normed @ normed.T
    zero_mask = (mat.norm(p=2, dim=1) <= 1e-12)
    if zero_mask.any():
        sim[zero_mask, :] = 0.0
        sim[:, zero_mask] = 0.0
    return sim.clamp(-1.0, 1.0)


def project_out_common_direction(
    deltas: Sequence[Mapping[str, torch.Tensor]],
    strength: float = 0.0,
) -> list[dict[str, torch.Tensor]]:
    """Subtract a fraction of the mean delta vector from each delta to suppress shared drift."""
    if strength <= 0.0 or len(deltas) <= 1:
        return [{k: v.detach().cpu().float().clone() for k, v in d.items()} for d in deltas]

    keys = list(deltas[0].keys())
    out: list[dict[str, torch.Tensor]] = [{} for _ in deltas]
    for name in keys:
        stacked = torch.stack([d[name].detach().cpu().float() for d in deltas], dim=0)
        mean_dir = stacked.mean(dim=0, keepdim=True)
        adjusted = stacked - strength * mean_dir
        for i in range(len(deltas)):
            out[i][name] = adjusted[i]
    return out


def _ties_merge_tensor(stacked: torch.Tensor, trim_ratio: float) -> torch.Tensor:
    """Trim low-magnitude coordinates, elect dominant sign by magnitude, average agreeing values."""
    k = stacked.shape[0]
    flat = stacked.reshape(k, -1)
    if trim_ratio > 0.0 and flat.shape[1] > 0:
        n_keep = max(1, int(round((1.0 - trim_ratio) * flat.shape[1])))
        trimmed = torch.zeros_like(flat)
        for i in range(k):
            abs_row = flat[i].abs()
            if n_keep >= flat.shape[1]:
                trimmed[i] = flat[i]
            else:
                top_idx = torch.topk(abs_row, k=n_keep, largest=True).indices
                trimmed[i, top_idx] = flat[i, top_idx]
    else:
        trimmed = flat

    elected_sign = torch.sign(trimmed.sum(dim=0, keepdim=True))
    agree = (torch.sign(trimmed) == elected_sign) & (elected_sign != 0)
    denom = agree.sum(dim=0).float().clamp_min(1.0)
    merged_flat = (trimmed * agree.float()).sum(dim=0) / denom
    return merged_flat.view(stacked.shape[1:])


def merge_task_vectors(
    sd_0: Mapping[str, torch.Tensor],
    deltas: Sequence[Mapping[str, torch.Tensor]],
    method: str = "sum",
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    trim_ratio: float = 0.2,
    fishers: Sequence[Mapping[str, torch.Tensor]] | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Merge K unlearning task vectors into base state_dict sd_0."""
    if not deltas:
        raise ValueError("deltas must be non-empty")
    if method not in {"sum", "mean", "ties", "fisher"}:
        raise ValueError(f"Unknown merge method '{method}'")
    if method == "fisher" and (fishers is None or len(fishers) != len(deltas)):
        raise ValueError("method='fisher' requires one Fisher dict per delta")

    k = len(deltas)
    if weights is None:
        w = torch.ones(k, dtype=torch.float32)
        if method == "mean":
            w = w / float(k)
    else:
        if len(weights) != k:
            raise ValueError(f"Expected {k} weights, got {len(weights)}")
        w = torch.tensor(weights, dtype=torch.float32)

    merged: dict[str, torch.Tensor] = {}
    for name, base_val in sd_0.items():
        base = base_val.detach().cpu().float()
        if not torch.is_floating_point(base_val):
            merged[name] = base_val.detach().cpu().clone()
            continue

        stacked = torch.stack([d[name].detach().cpu().float() for d in deltas], dim=0)
        view_shape = (k,) + (1,) * base.ndim
        weighted = stacked * w.view(view_shape)

        if method in {"sum", "mean"}:
            delta_out = weighted.sum(dim=0)
        elif method == "ties":
            delta_out = _ties_merge_tensor(weighted, trim_ratio=trim_ratio)
        else:
            assert fishers is not None
            f_stack = torch.stack(
                [
                    (f[name].detach().cpu().float().clamp_min(0.0) if name in f else torch.ones_like(base))
                    for f in fishers
                ],
                dim=0,
            ) * w.abs().view(view_shape)
            f_sum = f_stack.sum(dim=0)
            delta_out = torch.where(
                f_sum > eps,
                (stacked * f_stack).sum(dim=0) / f_sum.clamp_min(eps),
                weighted.mean(dim=0),
            )
        merged[name] = (base + alpha * delta_out).contiguous()
    return merged


def parse_group_gammas(spec: str) -> dict[str, float]:
    """Parse 'late_mlp=1.4,head=0.5' into a {group: gamma} mapping."""
    if not spec or not spec.strip():
        return {}
    out: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid group gamma entry '{item}'; expected group=value")
        key, val = item.split("=", 1)
        out[key.strip()] = float(val.strip())
    return out


def layerwise_interpolate(
    sd_0: Mapping[str, torch.Tensor],
    sd_u: Mapping[str, torch.Tensor],
    group_gammas: Mapping[str, float] | None = None,
    default_gamma: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Interpolate theta_0 + gamma_group * (theta_u - theta_0) per ViT component group."""
    gammas = dict(group_gammas or {})
    depth = infer_vit_depth(sd_0.keys())
    out: dict[str, torch.Tensor] = {}
    for name, v0 in sd_0.items():
        if not torch.is_floating_point(v0):
            out[name] = v0.detach().cpu().clone()
            continue
        group = classify_vit_param(name, depth=depth)
        g = float(gammas.get(group, gammas.get(name, default_gamma)))
        b0 = v0.detach().cpu().float()
        bu = sd_u[name].detach().cpu().float()
        out[name] = (b0 + g * (bu - b0)).contiguous()
    return out


def soup_checkpoints(
    sd_0: Mapping[str, torch.Tensor],
    checkpoint_sds: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
    group_gammas: Mapping[str, float] | None = None,
    default_gamma: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combine multiple unlearned checkpoints in delta space with optional layerwise gammas."""
    if not checkpoint_sds:
        raise ValueError("checkpoint_sds must be non-empty")
    k = len(checkpoint_sds)
    if weights is None:
        w = [1.0 / float(k)] * k
    else:
        if len(weights) != k:
            raise ValueError(f"Expected {k} soup weights, got {len(weights)}")
        w = [float(val) for val in weights]

    deltas = [state_dict_delta(sd, sd_0) for sd in checkpoint_sds]
    blended = merge_task_vectors(sd_0, deltas, method="sum", weights=w, alpha=1.0)
    if group_gammas or default_gamma != 1.0:
        return layerwise_interpolate(sd_0, blended, group_gammas=group_gammas, default_gamma=default_gamma)
    return blended
