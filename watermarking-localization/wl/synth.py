"""Synthetic watermarked dataset generator and mask library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter

from wl import CELL, GRID, IMG_SIZE, MAX_ASPECT, METHODS, NBITS, ROOT_DIR
from wl.augment import augment_clean_before_embed, augment_mask512
from wl.evaluate import downsample_mask
from wl.features import load_kit_models


def _random_blob_mask(
    height: int,
    width: int,
    target_frac: float,
    rng: np.random.Generator,
    n_modes: int = 3,
) -> np.ndarray:
    """Generate a smooth organic binary mask inside (height, width) via multi-scale noise."""
    acc = np.zeros((height, width), dtype=np.float32)
    for idx in range(n_modes):
        grid_sz = int(rng.integers(5, 14 + idx * 4))
        coarse = rng.normal(0.0, 1.0, size=(grid_sz, grid_sz)).astype(np.float32)
        up = np.asarray(
            Image.fromarray(coarse, mode="F").resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        acc += up / float(idx + 1)
    # Bias toward center or random anchor so objects stay coherent rather than speckly.
    cy = float(rng.uniform(0.2, 0.8)) * height
    cx = float(rng.uniform(0.2, 0.8)) * width
    yy, xx = np.ogrid[:height, :width]
    dist2 = ((yy - cy) / max(1.0, height * 0.5)) ** 2 + ((xx - cx) / max(1.0, width * 0.5)) ** 2
    field = acc - 1.2 * dist2
    thr = float(np.quantile(field, np.clip(1.0 - target_frac, 0.02, 0.99)))
    mask = (field >= thr).astype(np.uint8)
    if not mask.any():
        mask[height // 2, width // 2] = 1
    return mask


def _sample_box(
    area_frac: float,
    max_aspect: float,
    rng: np.random.Generator,
    touch_border_prob: float = 0.35,
) -> tuple[int, int, int, int]:
    """Sample a 4px-aligned bounding box [y0, y1, x0, x1] in 512x512 respecting aspect limit."""
    total_area = float(IMG_SIZE * IMG_SIZE) * float(np.clip(area_frac, 0.002, 0.65))
    log_max = float(np.log(max(1.05, min(max_aspect, 15.5))))
    aspect = float(np.exp(rng.uniform(0.0, log_max)))
    if rng.random() < 0.5:
        h_f = np.sqrt(total_area * aspect)
        w_f = total_area / max(1.0, h_f)
    else:
        w_f = np.sqrt(total_area * aspect)
        h_f = total_area / max(1.0, w_f)

    # Round to multiples of CELL=4 and ensure embedders receive at least 24x24 crops.
    h = int(np.clip(round(h_f / CELL) * CELL, 24, IMG_SIZE))
    w = int(np.clip(round(w_f / CELL) * CELL, 24, IMG_SIZE))
    # Re-enforce aspect ceiling after clamping to 512 so TrustMark never triggers center-crop.
    if max(h, w) / max(1, min(h, w)) > max_aspect:
        if h > w:
            h = int(max(24, (int(w * max_aspect) // CELL) * CELL))
        else:
            w = int(max(24, (int(h * max_aspect) // CELL) * CELL))

    max_y = (IMG_SIZE - h) // CELL
    max_x = (IMG_SIZE - w) // CELL
    y0 = int(rng.integers(0, max(1, max_y + 1))) * CELL
    x0 = int(rng.integers(0, max(1, max_x + 1))) * CELL
    if rng.random() < touch_border_prob:
        side = int(rng.integers(0, 4))
        if side == 0:
            y0 = 0
        elif side == 1:
            y0 = max_y * CELL
        elif side == 2:
            x0 = 0
        else:
            x0 = max_x * CELL
    return y0, y0 + h, x0, x0 + w


class MaskLibrary:
    """Real training-fold mask library + procedural mask generator for synthetic samples."""

    def __init__(
        self,
        data_root: str | Path = ROOT_DIR / "data" / "release",
        folds_path: str | Path = ROOT_DIR / "splits" / "folds.json",
        train_folds: Sequence[int] = (1, 2, 3, 4),
    ):
        self.full_holes: list[np.ndarray] = []
        self.patch_irregular_crops: list[np.ndarray] = []

        train_dir = Path(data_root) / "train"
        if not (train_dir / "metadata.json").is_file():
            return

        allowed_names: set[str] | None = None
        if Path(folds_path).is_file():
            fjson = json.loads(Path(folds_path).read_text(encoding="utf-8"))
            allowed_names = {n for k in train_folds for n in fjson["folds"][int(k)]}

        meta = json.loads((train_dir / "metadata.json").read_text(encoding="utf-8"))
        for row in meta:
            name = row["image_name"]
            if allowed_names is not None and name not in allowed_names:
                continue
            mask_path = train_dir / "masks" / f"{name}.png"
            if not mask_path.is_file():
                continue
            m512 = (np.asarray(Image.open(mask_path)) > 127).astype(np.uint8)
            if row["full_watermark"]:
                holes = 1 - m512
                if holes.any():
                    self.full_holes.append(holes)
            else:
                ys, xs = np.nonzero(m512)
                if not ys.size:
                    continue
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                crop = m512[y0:y1, x0:x1]
                fill = float(crop.mean())
                if fill < 0.96:
                    self.patch_irregular_crops.append(crop)

    def sample_target_mask(
        self,
        method: str,
        kind: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Sample a 512x512 minority target mask (1=patch for `patch`, 1=clean holes for `full`)."""
        max_aspect = min(1.95, MAX_ASPECT[method]) if method == "trustmark" else MAX_ASPECT[method]
        if kind == "patch":
            area_frac = float(np.exp(rng.uniform(np.log(0.004), np.log(0.42))))
            y0, y1, x0, x1 = _sample_box(area_frac, max_aspect, rng, touch_border_prob=0.25)
            h, w = y1 - y0, x1 - x0
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            if rng.random() < 0.5:
                mask[y0:y1, x0:x1] = 1
            elif self.patch_irregular_crops and rng.random() < 0.6:
                crop = self.patch_irregular_crops[int(rng.integers(0, len(self.patch_irregular_crops)))]
                if rng.random() < 0.5:
                    crop = crop[:, ::-1]
                if rng.random() < 0.5:
                    crop = crop[::-1, :]
                resized = Image.fromarray(crop * 255).resize((w, h), Image.Resampling.NEAREST)
                sub = (np.asarray(resized) > 127).astype(np.uint8)
                if not sub.any():
                    sub = _random_blob_mask(h, w, target_frac=0.55, rng=rng)
                mask[y0:y1, x0:x1] = sub
            else:
                mask[y0:y1, x0:x1] = _random_blob_mask(h, w, target_frac=float(rng.uniform(0.35, 0.78)), rng=rng)
            return mask, (y0, y1, x0, x1)

        # kind == "full": sample clean hole mask (1 = hole, 0 = watermarked).
        if self.full_holes and rng.random() < 0.55:
            hole = self.full_holes[int(rng.integers(0, len(self.full_holes)))]
            hole = augment_mask512(hole, rng)
            if hole.any() and float(hole.mean()) < 0.49:
                ys, xs = np.nonzero(hole)
                return hole, (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)

        hole = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        n_holes = int(rng.choice([1, 1, 2, 3]))
        total_frac = float(np.exp(rng.uniform(np.log(0.01), np.log(0.42))))
        for _ in range(n_holes):
            y0, y1, x0, x1 = _sample_box(total_frac / n_holes, max_aspect=4.0, rng=rng, touch_border_prob=0.6)
            sub = _random_blob_mask(y1 - y0, x1 - x0, target_frac=float(rng.uniform(0.4, 0.85)), rng=rng)
            hole[y0:y1, x0:x1] |= sub
        if not hole.any():
            hole[:32, :32] = 1
        ys, xs = np.nonzero(hole)
        return hole, (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)


def _scrub_patch_clean(img_uint8: np.ndarray, mask512: np.ndarray) -> np.ndarray:
    """Erase the watermarked patch from a `patch` image using heavily smoothed outside context."""
    out = np.array(img_uint8, dtype=np.float32, copy=True)
    blurred = np.asarray(
        Image.fromarray(img_uint8).filter(ImageFilter.GaussianBlur(radius=18)),
        dtype=np.float32,
    )
    m = mask512 > 0
    out[m] = blurred[m]
    return np.clip(out, 0, 255).astype(np.uint8)


class CleanImagePool:
    """Pool of clean 512x512 images from external directory and/or scrubbed real patch images."""

    def __init__(
        self,
        data_root: str | Path = ROOT_DIR / "data" / "release",
        folds_path: str | Path = ROOT_DIR / "splits" / "folds.json",
        train_folds: Sequence[int] = (1, 2, 3, 4),
        clean_dir: str | Path | None = None,
        external_prob: float = 0.7,
    ):
        self.external_prob = float(external_prob)
        self.external_paths: list[Path] = []
        self.real_patch_entries: list[tuple[Path, Path]] = []

        if clean_dir is not None and Path(clean_dir).is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                self.external_paths.extend(sorted(Path(clean_dir).rglob(ext)))

        train_dir = Path(data_root) / "train"
        if (train_dir / "metadata.json").is_file():
            allowed_names: set[str] | None = None
            if Path(folds_path).is_file():
                fjson = json.loads(Path(folds_path).read_text(encoding="utf-8"))
                allowed_names = {n for k in train_folds for n in fjson["folds"][int(k)]}
            meta = json.loads((train_dir / "metadata.json").read_text(encoding="utf-8"))
            for row in meta:
                name = row["image_name"]
                if row["full_watermark"]:
                    continue
                if allowed_names is not None and name not in allowed_names:
                    continue
                img_p = train_dir / "images" / f"{name}.png"
                msk_p = train_dir / "masks" / f"{name}.png"
                if img_p.is_file() and msk_p.is_file():
                    self.real_patch_entries.append((img_p, msk_p))

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if self.external_paths and (not self.real_patch_entries or rng.random() < self.external_prob):
            p = self.external_paths[int(rng.integers(0, len(self.external_paths)))]
            pil = Image.open(p).convert("RGB")
            # Random square crop keeps the aspect ratio; resizing a whole 4:3 photo would stretch it.
            w, h = pil.size
            side = max(1, int(min(w, h) * float(rng.uniform(0.6, 1.0))))
            x0 = int(rng.integers(0, w - side + 1))
            y0 = int(rng.integers(0, h - side + 1))
            pil = pil.crop((x0, y0, x0 + side, y0 + side)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            return np.asarray(pil, dtype=np.uint8), f"ext:{p.name}"

        if self.real_patch_entries:
            img_p, msk_p = self.real_patch_entries[int(rng.integers(0, len(self.real_patch_entries)))]
            img = np.asarray(Image.open(img_p).convert("RGB"), dtype=np.uint8)
            msk = (np.asarray(Image.open(msk_p)) > 127).astype(np.uint8)
            return _scrub_patch_clean(img, msk), f"scrub:{img_p.stem}"

        # Procedural textured canvas fallback if no images are supplied.
        base = rng.normal(128.0, 35.0, size=(16, 16, 3)).astype(np.float32)
        up = np.stack(
            [
                np.asarray(Image.fromarray(base[:, :, c], mode="F").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC))
                for c in range(3)
            ],
            axis=-1,
        )
        return np.clip(up, 0, 255).astype(np.uint8), "procedural"


class _MockWatermarker:
    """Fast deterministic embedder used for CPU unit tests when kit weights are absent."""

    def __init__(self, method: str):
        self.name = method
        self.nbits = NBITS[method]

    def embed(self, img_uint8: np.ndarray, bits: Sequence[int]) -> np.ndarray:
        h, w = img_uint8.shape[:2]
        seed = int(sum((i + 1) * int(b) for i, b in enumerate(bits))) % 9973
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        pattern = np.sin(0.9 * xx + 0.7 * yy + seed * 0.01) * 4.0
        out = img_uint8.astype(np.float32) + pattern[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)


def _feather_alpha(mask_crop: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return mask_crop.astype(np.float32)
    pil = Image.fromarray((mask_crop * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=float(feather_px))
    )
    return (np.asarray(pil, dtype=np.float32) / 255.0) * mask_crop.astype(np.float32)


HOLE_FILLS = ("restore", "paste", "inpaint")


def _hole_fill(
    mode: str,
    clean_aug: np.ndarray,
    wm_full: np.ndarray,
    hole: np.ndarray,
    alt_img: np.ndarray | None,
) -> np.ndarray:
    """Content placed in the clean holes of a `full` image, one per real edit type.

    restore: the original unwatermarked pixels (alteration; seamless, the hardest case).
    paste:   pixels of another clean image (object addition).
    inpaint: OpenCV inpainting from the watermarked surroundings (object removal).
    """
    if mode == "paste" and alt_img is not None:
        return alt_img
    if mode == "inpaint":
        try:
            import cv2

            return cv2.inpaint(wm_full, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        except ImportError:
            pass
    return clean_aug


def synthesize_one(
    clean_img: np.ndarray,
    method: str,
    kind: str,
    wm_model: Any,
    mask_lib: MaskLibrary,
    rng: np.random.Generator,
    source: str = "synth",
    alt_img: np.ndarray | None = None,
    fill_probs: Sequence[float] = (1.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Synthesize a single watermarked training example matching the task data generation.

    `fill_probs` weights the `full`-image hole fills in HOLE_FILLS order; `alt_img` feeds "paste".
    """
    clean_aug = augment_clean_before_embed(clean_img, rng)
    bits = rng.integers(0, 2, size=NBITS[method], dtype=np.uint8)
    target_mask, (y0, y1, x0, x1) = mask_lib.sample_target_mask(method, kind, rng)

    if kind == "full":
        wm_full = wm_model.embed(clean_aug, bits)
        mask512 = (1 - target_mask).astype(np.uint8)
        probs = np.asarray(fill_probs, dtype=np.float64)
        mode = HOLE_FILLS[int(rng.choice(len(HOLE_FILLS), p=probs / probs.sum()))]
        fill = _hole_fill(mode, clean_aug, wm_full, target_mask, alt_img)
        img_out = np.where(mask512[:, :, None] == 1, wm_full, fill)
        source = f"{source}|{mode}"
    else:
        crop = clean_aug[y0:y1, x0:x1]
        wm_crop = wm_model.embed(crop, bits)
        sub_mask = target_mask[y0:y1, x0:x1]
        feather = 2 if method == "trustmark" and rng.random() < 0.5 else 0
        alpha = _feather_alpha(sub_mask, feather)[:, :, None]
        img_out = np.array(clean_aug, dtype=np.uint8, copy=True)
        blended = alpha * wm_crop.astype(np.float32) + (1.0 - alpha) * crop.astype(np.float32)
        img_out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        mask512 = target_mask

    mask128 = downsample_mask(mask512)
    return {
        "image": img_out,
        "mask128": mask128,
        "method": method,
        "message": "".join(str(int(b)) for b in bits.tolist()),
        "kind": kind,
        "source": source,
    }


def generate_shards(
    data_root: str | Path = ROOT_DIR / "data" / "release",
    out_dir: str | Path = ROOT_DIR / "data" / "synth" / "v1",
    num_per_method: int = 200,
    shard_size: int = 100,
    train_folds: Sequence[int] = (1, 2, 3, 4),
    seed: int = 0,
    device: str = "cpu",
    clean_dir: str | Path | None = None,
    methods: Sequence[str] = METHODS,
    mock_embed: bool = False,
    shard_prefix: str = "shard",
    external_prob: float = 0.7,
    fill_probs: Sequence[float] = (1.0, 0.0, 0.0),
) -> list[Path]:
    """Generate compressed .npz shards following the standard training shard schema."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    folds_path = ROOT_DIR / "splits" / "folds.json"
    mask_lib = MaskLibrary(data_root=data_root, folds_path=folds_path, train_folds=train_folds)
    clean_pool = CleanImagePool(
        data_root=data_root,
        folds_path=folds_path,
        train_folds=train_folds,
        clean_dir=clean_dir,
        external_prob=external_prob,
    )
    if clean_dir is not None and not clean_pool.external_paths:
        raise FileNotFoundError(f"--clean-dir {clean_dir} has no .png/.jpg/.jpeg/.webp images")
    needs_alt = float(fill_probs[1]) > 0.0

    if mock_embed:
        wm_models = {m: _MockWatermarker(m) for m in methods}
    else:
        wm_models = load_kit_models(methods, device=device)

    specs: list[tuple[str, str]] = []
    for m in methods:
        for i in range(num_per_method):
            specs.append((m, "full" if i % 2 == 0 else "patch"))
    rng.shuffle(specs)

    shard_paths: list[Path] = []
    buf: list[dict[str, Any]] = []
    shard_idx = 0

    def _flush() -> None:
        nonlocal shard_idx
        if not buf:
            return
        sp = out_path / f"{shard_prefix}_{shard_idx:04d}.npz"
        np.savez_compressed(
            sp,
            image=np.stack([b["image"] for b in buf], axis=0).astype(np.uint8),
            mask128=np.stack([b["mask128"] for b in buf], axis=0).astype(np.uint8),
            method=np.asarray([b["method"] for b in buf], dtype=str),
            message=np.asarray([b["message"] for b in buf], dtype=str),
            kind=np.asarray([b["kind"] for b in buf], dtype=str),
            source=np.asarray([b["source"] for b in buf], dtype=str),
        )
        shard_paths.append(sp)
        shard_idx += 1
        buf.clear()

    for method, kind in specs:
        clean_img, src = clean_pool.sample(rng)
        alt_img = clean_pool.sample(rng)[0] if kind == "full" and needs_alt else None
        sample = synthesize_one(
            clean_img, method, kind, wm_models[method], mask_lib, rng,
            source=src, alt_img=alt_img, fill_probs=fill_probs,
        )
        buf.append(sample)
        if len(buf) >= shard_size:
            _flush()
    _flush()
    return shard_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic watermark localization shards.")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--out", default=str(ROOT_DIR / "data" / "synth" / "v1"))
    ap.add_argument("--num-per-method", type=int, default=200)
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--train-folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clean-dir", default=None)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--shard-prefix", default="shard", help="Filename prefix for shard files")
    ap.add_argument("--mock-embed", action="store_true", help="Use mock embedder for offline tests")
    ap.add_argument("--external-prob", type=float, default=0.7,
                    help="Share of clean bases drawn from --clean-dir (rest: scrubbed real patch images)")
    ap.add_argument("--fill-probs", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                    metavar=("RESTORE", "PASTE", "INPAINT"), help="Hole fill mix for `full` images")
    args = ap.parse_args()

    shards = generate_shards(
        data_root=args.data,
        out_dir=args.out,
        num_per_method=args.num_per_method,
        shard_size=args.shard_size,
        train_folds=args.train_folds,
        seed=args.seed,
        device=args.device,
        clean_dir=args.clean_dir,
        methods=args.methods,
        mock_embed=args.mock_embed,
        shard_prefix=args.shard_prefix,
        external_prob=args.external_prob,
        fill_probs=args.fill_probs,
    )
    print(f"Wrote {len(shards)} shards to {args.out}")


if __name__ == "__main__":
    main()
