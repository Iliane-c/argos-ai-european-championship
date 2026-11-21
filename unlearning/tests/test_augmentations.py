"""Unit tests for training augmentations and deterministic altered views."""

from __future__ import annotations

import torch

from ul.augmentations import (
    augment_batch,
    build_eval_alterations,
    downscale_upscale,
    gaussian_blur,
    jpeg_compress_uint8,
    patch_mask,
)


def test_augment_batch_clean_identity_and_alteration() -> None:
    torch.manual_seed(0)
    x = torch.rand(6, 3, 32, 32)
    clean_out = augment_batch(x, clean_prob=1.0, patch_size=8)
    torch.testing.assert_close(clean_out, x, atol=0.0, rtol=0.0)

    aug_out = augment_batch(x, clean_prob=0.0, patch_size=8, erase_prob=1.0, blur_prob=1.0, noise_prob=1.0)
    assert aug_out.shape == x.shape
    assert aug_out.min().item() >= 0.0
    assert aug_out.max().item() <= 1.0
    assert not torch.allclose(aug_out, x)


def test_jpeg_and_primitive_transforms() -> None:
    imgs_u8 = torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8)
    compressed = jpeg_compress_uint8(imgs_u8, quality=(45, 75))
    assert compressed.shape == imgs_u8.shape
    assert compressed.dtype == torch.uint8

    x01 = imgs_u8.permute(0, 3, 1, 2).float() / 255.0
    for out in (
        gaussian_blur(x01, 3),
        gaussian_blur(x01, 5),
        downscale_upscale(x01, 0.5),
        patch_mask(x01, mask_ratio=0.35, patch_size=8),
    ):
        assert out.shape == x01.shape
        assert torch.isfinite(out).all()
        assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_build_eval_alterations_deterministic() -> None:
    views = build_eval_alterations(patch_size=8)
    assert len(views) == 8
    x = torch.rand(4, 3, 32, 32)
    for name, fn in views.items():
        y1 = fn(x)
        y2 = fn(x)
        assert y1.shape == x.shape, name
        assert y1.min().item() >= 0.0 and y1.max().item() <= 1.0, name
        torch.testing.assert_close(y1, y2)
        assert not torch.allclose(y1, x), name
