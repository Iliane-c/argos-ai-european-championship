"""ViT memorization localization, SalUn/SSD saliency, mergeable LoRA, and gradient projection."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_io import preprocess

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def infer_vit_depth(param_names: Iterable[str]) -> int:
    """Infer transformer block count from parameter names (defaults to 12)."""
    max_idx = -1
    for name in param_names:
        match = _BLOCK_RE.match(name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1 if max_idx >= 0 else 12


def classify_vit_param(name: str, depth: int = 12) -> str:
    """Categorize a ViT state_dict parameter into semantic layer groups."""
    if name.startswith("patch_embed"):
        return "patch_embed"
    if name in {"cls_token", "pos_embed", "reg_token"} or name.startswith(("cls_token.", "pos_embed.")):
        return "pos_embed"
    if name.startswith("head."):
        return "head"
    if name.startswith(("norm.", "fc_norm.")):
        return "norm"

    match = _BLOCK_RE.match(name)
    if not match:
        return "other"

    idx = int(match.group(1))
    rest = name[match.end() :]
    if rest.startswith(("ls1.", "ls2.", "gamma_1", "gamma_2")):
        return "layerscale"
    if rest.startswith(("norm1.", "norm2.")):
        return "norm"

    third = max(1, depth // 3)
    stage = "early" if idx < third else ("mid" if idx < 2 * third else "late")
    if rest.startswith("attn."):
        return f"{stage}_attn"
    if rest.startswith("mlp."):
        return f"{stage}_mlp"
    return "other"


def select_trainable_params(
    model: nn.Module,
    scope: str = "late_mlp",
    train_blocks: int = 6,
    include_head: bool = True,
    include_norm: bool = True,
) -> list[str]:
    """Enable requires_grad on a surgical subset of ViT parameters and return their names."""
    all_names = [name for name, _ in model.named_parameters()]
    depth = infer_vit_depth(all_names)
    cutoff = max(0, depth - train_blocks)

    selected: list[str] = []
    for name, param in model.named_parameters():
        enable = False
        if scope == "all":
            enable = True
        elif scope == "head_only":
            enable = name.startswith("head.") or (include_norm and name.startswith(("norm.", "fc_norm.")))
        else:
            match = _BLOCK_RE.match(name)
            in_tail = match is not None and int(match.group(1)) >= cutoff
            if scope == "last_k_blocks":
                enable = in_tail
            elif scope == "mlp_only":
                enable = ".mlp." in name
            elif scope == "late_mlp":
                enable = in_tail and ".mlp." in name
            elif scope == "late_mlp_attn_proj":
                enable = in_tail and (".mlp." in name or ".attn.proj." in name)
            else:
                raise ValueError(f"Unknown parameter scope '{scope}'")

            if not enable and include_head and name.startswith("head."):
                enable = True
            if not enable and include_norm and name.startswith(("norm.", "fc_norm.")):
                enable = True

        param.requires_grad_(enable)
        if enable:
            selected.append(name)

    if not selected:
        raise ValueError(f"Parameter selection scope='{scope}' matched zero parameters")
    return sorted(selected)


def compute_diagonal_fisher(
    model: nn.Module,
    images_uint8: torch.Tensor,
    device: str,
    mode: str = "empirical",
    labels: torch.Tensor | None = None,
    batch_size: int = 16,
    max_samples: int | None = None,
    temperature: float = 2.0,
    param_names: set[str] | None = None,
    per_sample: bool = True,
    use_fp16_on_cuda: bool = True,
) -> dict[str, torch.Tensor]:
    """Estimate diagonal Fisher information on CPU float32 for targeted parameters."""
    if mode not in {"empirical", "predictive"}:
        raise ValueError(f"Unknown Fisher mode '{mode}'; expected 'empirical' or 'predictive'")

    if max_samples is not None and max_samples < len(images_uint8):
        images_uint8 = images_uint8[:max_samples]
        if labels is not None:
            labels = labels[:max_samples]

    target_params = {
        name: param
        for name, param in model.named_parameters()
        if param_names is None or name in param_names
    }
    if not target_params:
        raise ValueError("No matching parameters found for Fisher computation")

    orig_req_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    was_training = model.training
    fisher = {name: torch.zeros_like(p, device="cpu", dtype=torch.float32) for name, p in target_params.items()}
    count = 0

    try:
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad_(name in target_params)

        step_bs = 1 if per_sample else batch_size
        n_total = len(images_uint8)
        for start in range(0, n_total, step_bs):
            end = min(n_total, start + step_bs)
            x_norm = preprocess(images_uint8[start:end], device)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16_on_cuda and str(device).startswith("cuda")):
                scores = model(x_norm).float()
            if mode == "empirical":
                if labels is None:
                    y = scores.detach().argmax(dim=1)
                else:
                    y = labels[start:end].to(device, non_blocking=True).long()
                loss = F.cross_entropy(scores, y, reduction="mean")
            else:
                probs = F.softmax(scores.detach() / temperature, dim=1)
                sampled = torch.distributions.Categorical(probs=probs).sample()
                loss = F.cross_entropy(scores / temperature, sampled, reduction="mean")
            loss.backward()

            weight = end - start
            count += weight
            for name, param in target_params.items():
                if param.grad is not None:
                    fisher[name].add_(param.grad.detach().float().cpu().square(), alpha=float(weight))
    finally:
        model.zero_grad(set_to_none=True)
        for name, param in model.named_parameters():
            param.requires_grad_(orig_req_grad[name])
        model.train(was_training)

    denom = float(max(1, count))
    for name in fisher:
        fisher[name].div_(denom)
    return fisher


def build_salun_mask(
    fisher_forget: Mapping[str, torch.Tensor],
    fisher_retain: Mapping[str, torch.Tensor] | None = None,
    keep_ratio: float = 0.5,
    mode: str = "salun",
    damp_tau: float = 1e-6,
    allowed_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build binary gradient masks keeping top keep_ratio forget-salient entries."""
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must lie in (0, 1], got {keep_ratio}")
    if mode not in {"salun", "relative_ssd"}:
        raise ValueError(f"Unknown SalUn mask mode '{mode}'")
    if mode == "relative_ssd" and fisher_retain is None:
        raise ValueError("mode='relative_ssd' requires fisher_retain")

    scores: dict[str, torch.Tensor] = {}
    flat_scores: list[torch.Tensor] = []
    for name, f_forget in fisher_forget.items():
        if allowed_names is not None and name not in allowed_names:
            continue
        if mode == "salun":
            s = f_forget.detach().float().abs()
        else:
            assert fisher_retain is not None
            f_ret = fisher_retain[name].detach().float().abs()
            s = f_forget.detach().float().abs() / (f_ret + damp_tau)
        scores[name] = s
        flat_scores.append(s.reshape(-1))

    if not flat_scores:
        raise ValueError("No tensors matched allowed_names for SalUn mask")

    all_vals = torch.cat(flat_scores)
    if keep_ratio >= 1.0:
        return {name: torch.ones_like(val, dtype=torch.float32) for name, val in scores.items()}

    k_drop = max(0, int(round((1.0 - keep_ratio) * all_vals.numel())))
    if k_drop == 0:
        return {name: torch.ones_like(val, dtype=torch.float32) for name, val in scores.items()}
    threshold = torch.kthvalue(all_vals, k_drop).values
    return {name: (val > threshold).float() for name, val in scores.items()}


def apply_grad_mask(model: nn.Module, mask: Mapping[str, torch.Tensor]) -> None:
    """Zero out non-salient parameter gradients in-place before optimizer.step()."""
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask:
            param.grad.mul_(mask[name].to(device=param.grad.device, dtype=param.grad.dtype))


@torch.no_grad()
def apply_ssd_dampening(
    model: nn.Module,
    fisher_forget: Mapping[str, torch.Tensor],
    fisher_retain: Mapping[str, torch.Tensor],
    alpha: float = 10.0,
    damp_lambda: float = 1.0,
    min_scale: float = 0.1,
    eps: float = 1e-12,
    allowed_names: set[str] | None = None,
) -> dict[str, float]:
    """Apply Selective Synaptic Dampening (Foster et al., AAAI 2024) in-place."""
    total_checked = 0
    total_dampened = 0
    scale_sum = 0.0

    for name, param in model.named_parameters():
        if allowed_names is not None and name not in allowed_names:
            continue
        if name not in fisher_forget or name not in fisher_retain:
            continue
        f_f = fisher_forget[name].to(device=param.device, dtype=torch.float32)
        f_r = fisher_retain[name].to(device=param.device, dtype=torch.float32)
        mask = f_f > (alpha * f_r)
        scale = torch.ones_like(f_f)
        if mask.any():
            beta = (damp_lambda * f_r[mask]) / (f_f[mask] + eps)
            scale[mask] = beta.clamp(min=min_scale, max=1.0)
            param.data.mul_(scale.to(dtype=param.dtype))
            total_dampened += int(mask.sum().item())
        total_checked += param.numel()
        scale_sum += float(scale.sum().item())

    return {
        "dampened_fraction": float(total_dampened / max(1, total_checked)),
        "mean_scale": float(scale_sum / max(1, total_checked)),
        "total_checked": float(total_checked),
    }


@torch.no_grad()
def apply_magnitude_pruning(
    model: nn.Module,
    sparsity: float = 0.15,
    allowed_names: set[str] | None = None,
) -> dict[str, float]:
    """Prune smallest-magnitude 2D+ weights globally across allowed_names (L1-sparse init)."""
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must lie in [0, 1), got {sparsity}")
    targets = [
        (name, param)
        for name, param in model.named_parameters()
        if param.ndim >= 2 and (allowed_names is None or name in allowed_names)
    ]
    if not targets or sparsity == 0.0:
        return {"pruned_fraction": 0.0, "total_params": float(sum(p.numel() for _, p in targets))}

    flat_abs = torch.cat([p.detach().float().abs().reshape(-1) for _, p in targets])
    k_prune = max(1, int(round(sparsity * flat_abs.numel())))
    threshold = torch.kthvalue(flat_abs, k_prune).values
    pruned = 0
    total = 0
    for _, param in targets:
        keep = param.data.abs() > threshold
        pruned += int((~keep).sum().item())
        total += param.numel()
        param.data.mul_(keep.to(dtype=param.dtype))
    return {"pruned_fraction": float(pruned / max(1, total)), "total_params": float(total)}


class LoRALinear(nn.Module):
    """Frozen Linear layer augmented with a mergeable low-rank residual BA."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        dev = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out + delta

    @torch.no_grad()
    def merged_linear(self, requires_grad: bool = False) -> nn.Linear:
        """Fold BA into a plain nn.Linear matching the original state_dict schema."""
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=torch.float32,
        )
        delta_w = (self.lora_B.detach().float() @ self.lora_A.detach().float()) * self.scaling
        merged.weight.copy_(self.base.weight.detach().float() + delta_w)
        if self.base.bias is not None:
            assert merged.bias is not None
            merged.bias.copy_(self.base.bias.detach().float())
        merged.requires_grad_(requires_grad)
        return merged


def _set_submodule(root: nn.Module, module_path: str, replacement: nn.Module) -> None:
    parts = module_path.split(".")
    parent = root
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, parts[-1], replacement)


def inject_lora(
    model: nn.Module,
    target_patterns: Sequence[str] = ("mlp.fc1", "mlp.fc2", "attn.proj"),
    rank: int = 16,
    alpha: float = 32.0,
    train_blocks: int = 6,
    freeze_base: bool = True,
) -> list[str]:
    """Replace matching nn.Linear modules in the tail blocks with mergeable LoRALinear."""
    depth = infer_vit_depth(name for name, _ in model.named_parameters())
    cutoff = max(0, depth - train_blocks)
    if freeze_base:
        for param in model.parameters():
            param.requires_grad_(False)

    candidates: list[tuple[str, nn.Linear]] = []
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        match = _BLOCK_RE.match(mod_name)
        in_tail = match is not None and int(match.group(1)) >= cutoff
        if in_tail and any(pat in mod_name for pat in target_patterns):
            candidates.append((mod_name, mod))

    if not candidates:
        raise ValueError(f"inject_lora matched zero Linear layers for patterns={target_patterns}")

    replaced: list[str] = []
    for mod_name, linear in candidates:
        _set_submodule(model, mod_name, LoRALinear(linear, rank=rank, alpha=alpha))
        replaced.append(mod_name)
    return replaced


def merge_and_unload_lora(model: nn.Module, requires_grad: bool = False) -> int:
    """Merge all LoRALinear modules back into plain nn.Linear layers in-place."""
    lora_names = [name for name, mod in model.named_modules() if isinstance(mod, LoRALinear)]
    for mod_name in lora_names:
        lora_mod = model.get_submodule(mod_name)
        assert isinstance(lora_mod, LoRALinear)
        _set_submodule(model, mod_name, lora_mod.merged_linear(requires_grad=requires_grad))
    return len(lora_names)


class ActivationSubspaceProjector:
    """Projects gradients orthogonal to the top retain-set input activation subspace (PGU/OGP)."""

    def __init__(self) -> None:
        self.basis: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def fit(
        self,
        model: nn.Module,
        images_uint8: torch.Tensor,
        device: str,
        target_linear_names: Sequence[str] | None = None,
        variance_threshold: float = 0.95,
        max_rank: int = 64,
        batch_size: int = 32,
        max_samples: int = 512,
        use_fp16_on_cuda: bool = True,
    ) -> dict[str, int]:
        """Collect input covariance X^T X on retain images and extract top-k eigenvectors."""
        if max_samples < len(images_uint8):
            images_uint8 = images_uint8[:max_samples]

        target_set = set(target_linear_names) if target_linear_names is not None else None
        tracked: dict[str, nn.Linear] = {}
        for mod_name, mod in model.named_modules():
            base_mod = mod.base if isinstance(mod, LoRALinear) else mod
            if isinstance(base_mod, nn.Linear) and (target_set is None or mod_name in target_set):
                tracked[mod_name] = base_mod
        if not tracked:
            raise ValueError("ActivationSubspaceProjector.fit found no matching Linear modules")

        covs = {
            name: torch.zeros(mod.in_features, mod.in_features, dtype=torch.float64, device="cpu")
            for name, mod in tracked.items()
        }
        handles = []

        def make_hook(layer_name: str):
            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(dtype=torch.float64, device="cpu")
                covs[layer_name].add_(x.T @ x)

            return hook

        for mod_name, mod in tracked.items():
            handles.append(mod.register_forward_pre_hook(make_hook(mod_name)))

        was_training = model.training
        model.eval()
        try:
            for batch in images_uint8.split(batch_size):
                x_norm = preprocess(batch, device)
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=use_fp16_on_cuda and str(device).startswith("cuda")
                ):
                    model(x_norm)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

        self.basis.clear()
        ranks: dict[str, int] = {}
        for mod_name, cov in covs.items():
            evals, evecs = torch.linalg.eigh(cov)
            evals = evals.flip(0).clamp_min(0.0)
            evecs = evecs.flip(1)
            total_var = evals.sum()
            if total_var <= 1e-12:
                k = 0
            else:
                cum_ratio = torch.cumsum(evals, dim=0) / total_var
                above = (cum_ratio >= variance_threshold).nonzero(as_tuple=True)[0]
                k = int(above[0].item()) + 1 if above.numel() > 0 else len(evals)
                k = min(k, max_rank, len(evals) - 1 if len(evals) > 1 else len(evals))
            self.basis[mod_name] = evecs[:, :k].float().contiguous()
            ranks[mod_name] = k
        return ranks

    @torch.no_grad()
    def project_gradients(self, model: nn.Module) -> None:
        """Project Linear/LoRA gradients onto the orthogonal complement of retain subspace."""
        for mod_name, basis_cpu in self.basis.items():
            if basis_cpu.shape[1] == 0:
                continue
            mod = model.get_submodule(mod_name)
            if isinstance(mod, LoRALinear):
                grad = mod.lora_A.grad
            elif isinstance(mod, nn.Linear):
                grad = mod.weight.grad
            else:
                continue
            if grad is None:
                continue
            v = basis_cpu.to(device=grad.device, dtype=grad.dtype)
            grad.sub_((grad @ v) @ v.T)
