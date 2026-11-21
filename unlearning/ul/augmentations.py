"""Differentiable GPU-friendly augmentations and deterministic altered evaluation views."""

from __future__ import annotations

import io
import math
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def jpeg_compress_uint8(
    images_uint8: torch.Tensor,
    quality: int | tuple[int, int] = 70,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Apply in-memory PIL JPEG compression round-trip to uint8 [B, H, W, 3] images."""
    if images_uint8.dtype != torch.uint8:
        raise ValueError(f"jpeg_compress_uint8 expects uint8, got {images_uint8.dtype}")
    arr = images_uint8.detach().cpu().numpy()
    out: list[np.ndarray] = []
    if isinstance(quality, tuple):
        lo, hi = int(quality[0]), int(quality[1])
        gen = rng if rng is not None else np.random.default_rng(0)
        qualities = gen.integers(lo, hi + 1, size=len(arr)).tolist()
    else:
        qualities = [int(quality)] * len(arr)

    for img_np, q in zip(arr, qualities):
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="JPEG", quality=int(q))
        buf.seek(0)
        with Image.open(buf) as decoded:
            out.append(np.asarray(decoded.convert("RGB"), dtype=np.uint8))
    return torch.from_numpy(np.stack(out, axis=0))


def gaussian_blur(x01: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply separable binomial Gaussian-like blur to [B, 3, H, W] tensors."""
    if kernel_size == 3:
        weights_1d = torch.tensor([1.0, 2.0, 1.0], device=x01.device, dtype=x01.dtype)
    elif kernel_size == 5:
        weights_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x01.device, dtype=x01.dtype)
    else:
        raise ValueError(f"Unsupported kernel_size={kernel_size}; expected 3 or 5")
    kernel = (weights_1d[:, None] * weights_1d[None, :])
    kernel = kernel / kernel.sum()
    pad = kernel_size // 2
    channels = x01.shape[1]
    weight = kernel.view(1, 1, kernel_size, kernel_size).expand(channels, 1, kernel_size, kernel_size)
    padded = F.pad(x01, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, weight, groups=channels).clamp(0.0, 1.0)


def downscale_upscale(x01: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Downsample then upsample back to original resolution to simulate compression/resizing."""
    if not (0.1 <= scale <= 1.0):
        raise ValueError(f"scale must lie in [0.1, 1.0], got {scale}")
    _, _, h, w = x01.shape
    low_h = max(4, int(round(h * scale)))
    low_w = max(4, int(round(w * scale)))
    low = F.interpolate(x01, size=(low_h, low_w), mode="bilinear", align_corners=False)
    return F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def patch_mask(
    x01: torch.Tensor,
    mask_ratio: float = 0.30,
    patch_size: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Zero out random ViT patches per image (fills with 0.5 mid-gray in [0, 1] space)."""
    if mask_ratio <= 0.0:
        return x01
    b, _, h, w = x01.shape
    eff_p = patch_size if (h % patch_size == 0 and w % patch_size == 0 and h // patch_size >= 2) else math.gcd(h, w)
    if eff_p < 4 or h // eff_p < 2:
        eff_p = max(2, min(h, w) // 4)
    gh, gw = max(1, h // eff_p), max(1, w // eff_p)
    num_patches = gh * gw
    num_keep = max(1, int(round(num_patches * (1.0 - mask_ratio))))

    gen_dev = generator.device if generator is not None else x01.device
    scores = torch.rand(b, num_patches, device=gen_dev, generator=generator).to(x01.device)
    keep_idx = scores.argsort(dim=1)[:, :num_keep]
    mask = torch.zeros(b, num_patches, device=x01.device, dtype=x01.dtype)
    mask.scatter_(1, keep_idx, 1.0)
    mask = mask.view(b, 1, gh, gw)
    mask_full = F.interpolate(mask, size=(h, w), mode="nearest")
    return x01 * mask_full + 0.5 * (1.0 - mask_full)


def _apply_affine(
    x01: torch.Tensor,
    crop_scale: tuple[float, float],
    max_rot_deg: float,
    hflip_prob: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    b = x01.shape[0]
    dev, dt = x01.device, x01.dtype
    s = torch.empty(b, device=dev, dtype=dt).uniform_(crop_scale[0], crop_scale[1], generator=generator)
    flip_rand = torch.rand(b, device=dev, generator=generator)
    flip = torch.where(flip_rand < hflip_prob, -1.0, 1.0).to(dt)
    rad = torch.empty(b, device=dev, dtype=dt).uniform_(
        -max_rot_deg, max_rot_deg, generator=generator
    ) * (math.pi / 180.0)
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)

    tx = (torch.rand(b, device=dev, dtype=dt, generator=generator) * 2.0 - 1.0) * (1.0 - s)
    ty = (torch.rand(b, device=dev, dtype=dt, generator=generator) * 2.0 - 1.0) * (1.0 - s)

    theta = torch.zeros(b, 2, 3, device=dev, dtype=dt)
    theta[:, 0, 0] = flip * s * cos_a
    theta[:, 0, 1] = -s * sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = flip * s * sin_a
    theta[:, 1, 1] = s * cos_a
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, x01.shape, align_corners=False)
    return F.grid_sample(x01, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def augment_batch(
    x01: torch.Tensor,
    clean_prob: float = 0.25,
    crop_scale: tuple[float, float] = (0.65, 1.0),
    max_rot_deg: float = 15.0,
    hflip_prob: float = 0.5,
    blur_prob: float = 0.3,
    downup_prob: float = 0.2,
    color_prob: float = 0.5,
    noise_prob: float = 0.3,
    noise_std: float = 0.03,
    erase_prob: float = 0.25,
    patch_size: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply diverse training-time photometric/geometric/masking alterations on [0, 1] tensors."""
    b = x01.shape[0]
    if b == 0:
        return x01.clone()
    if clean_prob >= 1.0:
        return x01.clone()

    dev, dt = x01.device, x01.dtype
    v = _apply_affine(x01, crop_scale=crop_scale, max_rot_deg=max_rot_deg, hflip_prob=hflip_prob, generator=generator)

    blur_mask = torch.rand(b, device=dev, generator=generator) < blur_prob
    if blur_mask.any():
        v_blur = gaussian_blur(v[blur_mask], kernel_size=5)
        v = v.clone()
        v[blur_mask] = v_blur

    du_mask = torch.rand(b, device=dev, generator=generator) < downup_prob
    if du_mask.any():
        v_du = downscale_upscale(v[du_mask], scale=0.5)
        v = v.clone()
        v[du_mask] = v_du

    col_mask = (torch.rand(b, device=dev, generator=generator) < color_prob)[:, None, None, None]
    if col_mask.any():
        bright = torch.empty(b, 1, 1, 1, device=dev, dtype=dt).uniform_(0.78, 1.22, generator=generator)
        contrast = torch.empty(b, 1, 1, 1, device=dev, dtype=dt).uniform_(0.75, 1.25, generator=generator)
        sat = torch.empty(b, 1, 1, 1, device=dev, dtype=dt).uniform_(0.70, 1.30, generator=generator)
        mean_all = v.mean(dim=(1, 2, 3), keepdim=True)
        v_bc = (v - mean_all) * contrast + mean_all
        gray = v_bc.mean(dim=1, keepdim=True)
        v_col = (gray + (v_bc - gray) * sat) * bright
        v = torch.where(col_mask, v_col, v)

    noise_mask = (torch.rand(b, device=dev, generator=generator) < noise_prob)[:, None, None, None]
    if noise_mask.any():
        noise = torch.randn(v.shape, device=dev, dtype=dt, generator=generator) * noise_std
        v = torch.where(noise_mask, v + noise, v)

    erase_mask = torch.rand(b, device=dev, generator=generator) < erase_prob
    if erase_mask.any():
        v_erased = patch_mask(v[erase_mask], mask_ratio=0.25, patch_size=patch_size, generator=generator)
        v = v.clone()
        v[erase_mask] = v_erased

    clean_mask = (torch.rand(b, device=dev, generator=generator) < clean_prob)[:, None, None, None]
    return torch.where(clean_mask, x01, v.clamp(0.0, 1.0))


def build_eval_alterations(patch_size: int = 16) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Deterministic [B, 3, H, W] -> [B, 3, H, W] views in [0, 1] for local alteration audits."""

    def hflip(x: torch.Tensor) -> torch.Tensor:
        return x.flip(-1)

    def crop85(x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        margin_h = max(1, int(round(h * 0.075)))
        margin_w = max(1, int(round(w * 0.075)))
        cropped = x[:, :, margin_h : h - margin_h, margin_w : w - margin_w]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    def rot10(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        angle = math.radians(10.0)
        theta = torch.tensor(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0]],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0).expand(b, -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False).clamp(0.0, 1.0)

    def blur5(x: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(x, kernel_size=5)

    def downup50(x: torch.Tensor) -> torch.Tensor:
        return downscale_upscale(x, scale=0.5)

    def color_shift(x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return (((x - m) * 1.15 + m) * 0.90).clamp(0.0, 1.0)

    def noise025(x: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device=x.device).manual_seed(1337)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen) * 0.025
        return (x + noise).clamp(0.0, 1.0)

    def patch20(x: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device=x.device).manual_seed(2026)
        return patch_mask(x, mask_ratio=0.20, patch_size=patch_size, generator=gen)

    return {
        "hflip": hflip,
        "crop85": crop85,
        "rot10": rot10,
        "blur5": blur5,
        "downup50": downup50,
        "color_shift": color_shift,
        "noise025": noise025,
        "patch20": patch20,
    }
