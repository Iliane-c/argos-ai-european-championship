"""Watermark-safe augmentations, clean-image transforms, and real-to-real copy-paste."""
from __future__ import annotations

import io
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from wl import CELL, GRID
from wl.evaluate import downsample_mask

try:
    import pilgram as _pilgram
except ImportError:
    _pilgram = None


_PILGRAM_FILTERS = (
    "aden",
    "brooklyn",
    "clarendon",
    "gingham",
    "hudson",
    "juno",
    "lark",
    "lofi",
    "moon",
    "reyes",
    "slumber",
    "valencia",
)


def _jpeg_roundtrip(img_uint8: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def augment_clean_before_embed(
    img_uint8: np.ndarray,
    rng: np.random.Generator,
    use_pilgram: bool = True,
) -> np.ndarray:
    """Apply aggressive geometric & photometric transforms BEFORE watermark embedding."""
    arr = np.asarray(img_uint8, dtype=np.uint8)
    if rng.random() < 0.5:
        arr = np.ascontiguousarray(arr[:, ::-1, :])
    if rng.random() < 0.25:
        arr = np.ascontiguousarray(arr[::-1, :, :])
    k = int(rng.integers(0, 4))
    if k:
        arr = np.ascontiguousarray(np.rot90(arr, k=k, axes=(0, 1)))

    pil = Image.fromarray(arr)
    if use_pilgram and _pilgram is not None and rng.random() < 0.25:
        fname = str(rng.choice(_PILGRAM_FILTERS))
        pil = getattr(_pilgram, fname)(pil).convert("RGB")
    else:
        if rng.random() < 0.7:
            pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(0.85, 1.15)))
            pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(0.85, 1.18)))
            pil = ImageEnhance.Color(pil).enhance(float(rng.uniform(0.82, 1.20)))
        if rng.random() < 0.15:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.9))))

    out = np.asarray(pil, dtype=np.uint8)
    if rng.random() < 0.35:
        out = _jpeg_roundtrip(out, int(rng.integers(75, 96)))
    return out


def augment_mask512(mask512: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip, rotate 90deg, and integer-cell-shift a binary 512x512 mask."""
    m = np.asarray(mask512, dtype=np.uint8)
    if rng.random() < 0.5:
        m = np.ascontiguousarray(m[:, ::-1])
    if rng.random() < 0.5:
        m = np.ascontiguousarray(m[::-1, :])
    k = int(rng.integers(0, 4))
    if k:
        m = np.ascontiguousarray(np.rot90(m, k=k))
    dy = int(rng.integers(-16, 17)) * CELL
    dx = int(rng.integers(-16, 17)) * CELL
    if dy != 0 or dx != 0:
        m = np.roll(m, shift=(dy, dx), axis=(0, 1))
    return m


def integer_cell_shift(
    img: np.ndarray,
    mask128: np.ndarray,
    dy_cells: int,
    dx_cells: int,
    mask512: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift image by 4*dy, 4*dx pixels and mask128 by dy, dx cells, preserving 4x4 cell alignment."""
    dy_px = int(dy_cells) * CELL
    dx_px = int(dx_cells) * CELL
    if img.ndim == 3 and img.shape[0] <= 16 and img.shape[1] == img.shape[2]:
        img_s = np.roll(img, shift=(dy_px, dx_px), axis=(1, 2))
    else:
        img_s = np.roll(img, shift=(dy_px, dx_px), axis=(0, 1))
    m128_s = np.roll(mask128, shift=(int(dy_cells), int(dx_cells)), axis=(0, 1))
    if mask512 is None:
        return img_s, m128_s
    m512_s = np.roll(mask512, shift=(dy_px, dx_px), axis=(0, 1))
    return img_s, m128_s, m512_s


def copy_paste_watermark_safe(
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Paste the minority region of item_b onto item_a (same method) at 4px-aligned coordinates."""
    if item_a["method"] != item_b["method"]:
        raise ValueError(f"Copy-paste requires matching methods: {item_a['method']} vs {item_b['method']}")
    if "mask" not in item_a or "mask" not in item_b:
        raise ValueError("Copy-paste requires 512x512 pixel masks (`mask`) on both items")

    img_a = np.array(item_a["image"], dtype=np.uint8, copy=True)
    mask_a = np.array(item_a["mask"], dtype=np.uint8, copy=True)
    img_b = np.asarray(item_b["image"], dtype=np.uint8)
    mask_b = np.asarray(item_b["mask"], dtype=np.uint8)

    b_is_full = bool(item_b.get("full_watermark", item_b.get("kind") == "full"))
    region_b = (mask_b == 0) if b_is_full else (mask_b == 1)
    if not region_b.any():
        return dict(item_a)

    ys, xs = np.nonzero(region_b)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    # Snap bounding box and paste offset to the 4-pixel cell grid.
    y0 = (y0 // CELL) * CELL
    x0 = (x0 // CELL) * CELL
    h, w = y1 - y0, x1 - x0
    max_dy = (img_a.shape[0] - h) // CELL
    max_dx = (img_a.shape[1] - w) // CELL
    ty0 = int(rng.integers(0, max(1, max_dy + 1))) * CELL
    tx0 = int(rng.integers(0, max(1, max_dx + 1))) * CELL

    sub_region = region_b[y0:y1, x0:x1]
    dst_img = img_a[ty0 : ty0 + h, tx0 : tx0 + w]
    src_img = img_b[y0:y1, x0:x1]
    dst_img[sub_region] = src_img[sub_region]

    dst_mask = mask_a[ty0 : ty0 + h, tx0 : tx0 + w]
    src_mask = mask_b[y0:y1, x0:x1]
    dst_mask[sub_region] = src_mask[sub_region]

    out = dict(item_a)
    out["image"] = img_a
    out["mask"] = mask_a
    out["mask128"] = downsample_mask(mask_a)
    return out


def build_train_augment(
    cfg_aug: dict[str, Any] | None,
) -> Callable[[np.ndarray, np.ndarray, dict[str, Any], np.random.Generator], tuple[np.ndarray, np.ndarray]]:
    """Build watermark-safe training augmentation hook returning (img_uint8, mask128)."""
    cfg = cfg_aug or {}
    do_shift = bool(cfg.get("integer_shift", True))
    max_shift = int(cfg.get("max_shift_cells", 8))
    post_flip = bool(cfg.get("post_flip", False))
    post_jitter = bool(cfg.get("post_jitter", False))
    post_jpeg = bool(cfg.get("post_jpeg", False))

    def _augment(
        img_uint8: np.ndarray,
        mask128: np.ndarray,
        meta: dict[str, Any],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        img = np.asarray(img_uint8, dtype=np.uint8)
        m128 = np.asarray(mask128, dtype=np.uint8)

        if do_shift and max_shift > 0:
            dy = int(rng.integers(-max_shift, max_shift + 1))
            dx = int(rng.integers(-max_shift, max_shift + 1))
            if dy != 0 or dx != 0:
                img, m128 = integer_cell_shift(img, m128, dy, dx)

        if post_flip:
            if rng.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                m128 = np.ascontiguousarray(m128[:, ::-1])
            if rng.random() < 0.25:
                img = np.ascontiguousarray(img[::-1, :, :])
                m128 = np.ascontiguousarray(m128[::-1, :])

        if post_jitter and rng.random() < 0.5:
            pil = Image.fromarray(img)
            pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(0.92, 1.08)))
            pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(0.92, 1.08)))
            img = np.asarray(pil, dtype=np.uint8)

        if post_jpeg and rng.random() < 0.25:
            img = _jpeg_roundtrip(img, int(rng.integers(82, 97)))

        return img, m128

    return _augment
