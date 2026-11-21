"""Forensic high-pass maps, native-scale views, decoder agreement, and matched filters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from wl import CELL, GRID, IMG_SIZE, NATIVE_RES, NBITS, ROOT_DIR, ensure_kit_on_path


def _conv3x3_single(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Fast 3x3 correlation using NumPy edge padding (avoids extra C dependencies)."""
    p = np.pad(plane, 1, mode="edge")
    h, w = plane.shape
    out = np.zeros((h, w), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            coeff = float(kernel[dy, dx])
            if coeff != 0.0:
                out += coeff * p[dy : dy + h, dx : dx + w]
    return out


_KB_KERNEL = np.array(
    [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]], dtype=np.float32
) / 4.0
_LAPLACIAN_KERNEL = np.array(
    [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32
) / 4.0
_DIAG_KERNEL = np.array(
    [[-1.0, 0.0, -1.0], [0.0, 4.0, 0.0], [-1.0, 0.0, -1.0]], dtype=np.float32
) / 4.0


def compute_srm(img_uint8: np.ndarray) -> np.ndarray:
    """Compute 3 forensic SRM-style high-pass residual channels in [-1, 1] float32 [3, H, W]."""
    rgb = np.asarray(img_uint8, dtype=np.float32)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = (rgb[:, :, 0] - rgb[:, :, 1]) * 0.5 + (rgb[:, :, 2] - rgb[:, :, 1]) * 0.5

    ch0 = _conv3x3_single(luma, _KB_KERNEL)
    ch1 = _conv3x3_single(luma, _LAPLACIAN_KERNEL)
    ch2 = _conv3x3_single(chroma, _DIAG_KERNEL)
    stacked = np.stack([ch0, ch1, ch2], axis=0) / 16.0
    return np.clip(stacked, -1.0, 1.0).astype(np.float32)


def compute_native_view(img_uint8: np.ndarray, method: str) -> np.ndarray:
    """Rescale to method processing resolution and back; return normalized residual [3, H, W]."""
    res = int(NATIVE_RES[method])
    h, w = img_uint8.shape[:2]
    resample = Image.Resampling.BICUBIC if method == "stegastamp" else Image.Resampling.BILINEAR
    pil_img = Image.fromarray(img_uint8)
    roundtrip = pil_img.resize((res, res), resample).resize((w, h), resample)
    diff = (np.asarray(img_uint8, dtype=np.float32) - np.asarray(roundtrip, dtype=np.float32)) / 32.0
    return np.clip(diff.transpose(2, 0, 1), -1.0, 1.0).astype(np.float32)


def _resize_map(arr2d: np.ndarray, size: int = GRID) -> np.ndarray:
    pil = Image.fromarray(arr2d.astype(np.float32), mode="F")
    return np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def _fallback_agreement_map(img_uint8: np.ndarray) -> dict[str, Any]:
    """Offline heuristic residual map when kit watermark models are unavailable."""
    srm = compute_srm(img_uint8)
    energy = np.abs(srm).mean(axis=0)
    cell_energy = energy.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3))
    std = float(cell_energy.std()) + 1e-6
    z = (cell_energy - float(cell_energy.mean())) / std
    prob = 1.0 / (1.0 + np.exp(-z))
    return {"agreement128": prob.astype(np.float32), "bit_acc_full": 0.5, "margin_full": 0.0}


def _sliding_window_agreement(
    img_uint8: np.ndarray,
    signed_msg: np.ndarray,
    wm_model: Any,
    sizes: Sequence[int] = (256, 128),
    stride: int = 64,
    batch_size: int = 32,
) -> np.ndarray:
    """Evaluate decoder agreement over a grid of crops and project scores onto 128x128."""
    import torch

    h, w = img_uint8.shape[:2]
    acc_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    crops: list[torch.Tensor] = []
    boxes: list[tuple[int, int, int, int, float]] = []
    base = torch.from_numpy(img_uint8).permute(2, 0, 1).float().div(255.0)
    res = int(getattr(wm_model, "resolution", 256))

    for sz in sizes:
        step = max(32, stride if sz >= 128 else stride // 2)
        ys = list(range(0, max(1, h - sz + 1), step))
        xs = list(range(0, max(1, w - sz + 1), step))
        if ys[-1] != h - sz:
            ys.append(max(0, h - sz))
        if xs[-1] != w - sz:
            xs.append(max(0, w - sz))
        weight = 1.0 if sz == 128 else 0.7
        for y0 in ys:
            for x0 in xs:
                crop = base[:, y0 : y0 + sz, x0 : x0 + sz].unsqueeze(0)
                crop_rs = torch.nn.functional.interpolate(
                    crop, size=(res, res), mode="bilinear", align_corners=False, antialias=True
                )[0]
                crops.append(crop_rs)
                boxes.append((y0, y0 + sz, x0, x0 + sz, weight))

    if not crops:
        return np.full((GRID, GRID), 0.5, dtype=np.float32)

    signed_t = torch.from_numpy(signed_msg).float()
    all_margins: list[float] = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch_t = torch.stack(crops[i : i + batch_size], dim=0)
            logits = wm_model.logits(batch_t).detach().cpu()
            # tanh keeps very large single-bit logits from dominating the average.
            margins = (torch.tanh(logits * 0.5) * signed_t[None, :]).mean(dim=1)
            all_margins.extend(margins.tolist())

    for (y0, y1, x0, x1, wt), margin in zip(boxes, all_margins):
        acc_map[y0:y1, x0:x1] += wt * float(margin)
        weight_map[y0:y1, x0:x1] += wt

    merged = acc_map / np.maximum(weight_map, 1e-6)
    return merged.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3)).astype(np.float32)


def _occlusion_agreement_stegastamp(
    img_uint8: np.ndarray,
    signed_msg: np.ndarray,
    wm_model: Any,
    full_margin: float,
    patch_size: int = 64,
    stride: int = 64,
) -> np.ndarray:
    """Measure drop in StegaStamp decoder margin when each grid patch is blurred (for `full` holes)."""
    import torch

    h, w = img_uint8.shape[:2]
    blurred = np.asarray(
        Image.fromarray(img_uint8).resize((64, 64), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    occ_imgs: list[torch.Tensor] = []
    boxes: list[tuple[int, int, int, int]] = []
    for y0 in range(0, h, stride):
        y1 = min(h, y0 + patch_size)
        for x0 in range(0, w, stride):
            x1 = min(w, x0 + patch_size)
            canvas = img_uint8.copy()
            canvas[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]
            occ_imgs.append(torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0))
            boxes.append((y0, y1, x0, x1))

    signed_t = torch.from_numpy(signed_msg).float()
    with torch.no_grad():
        batch_t = torch.stack(occ_imgs, dim=0)
        logits = wm_model.logits(batch_t).detach().cpu()
        occ_margins = (torch.tanh(logits * 0.5) * signed_t[None, :]).mean(dim=1).tolist()

    drop_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    for (y0, y1, x0, x1), m in zip(boxes, occ_margins):
        drop_map[y0:y1, x0:x1] += float(full_margin - m)
        count_map[y0:y1, x0:x1] += 1.0
    avg_drop = drop_map / np.maximum(count_map, 1.0)
    return avg_drop.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3)).astype(np.float32)


def compute_decoder_agreement(
    img_uint8: np.ndarray,
    method: str,
    message_bits: np.ndarray,
    wm_model: Any = None,
) -> dict[str, Any]:
    """Compute 128x128 spatial agreement map and whole-image decode stats."""
    nbits = NBITS[method]
    bits = np.asarray(message_bits, dtype=np.float32).reshape(-1)[:nbits]
    if wm_model is None:
        return _fallback_agreement_map(img_uint8)

    import torch

    signed_msg = (2.0 * bits - 1.0).astype(np.float32)
    signed_t = torch.from_numpy(signed_msg).float()
    img_t = torch.from_numpy(img_uint8).permute(2, 0, 1)[None].float().div(255.0)

    with torch.no_grad():
        full_logits = wm_model.logits(img_t)[0].detach().cpu()
        bit_acc_full = float(((full_logits > 0).float() == torch.from_numpy(bits)).float().mean())
        margin_full = float((torch.tanh(full_logits * 0.5) * signed_t).mean())

        if method in ("pixelseal", "trustmark"):
            feats = wm_model.features(img_t)
            per_pos = feats["per_position_logits"][0].detach().cpu()  # [nbits, Hf, Wf]
            spatial_raw = (torch.tanh(per_pos * 0.5) * signed_t[:, None, None]).mean(dim=0).numpy()
            spatial_128 = _resize_map(spatial_raw, GRID)
            win_128 = _sliding_window_agreement(img_uint8, signed_msg, wm_model, sizes=(256, 128), stride=64)
            combined = 0.55 * spatial_128 + 0.45 * win_128
        elif method == "stegastamp":
            win_128 = _sliding_window_agreement(img_uint8, signed_msg, wm_model, sizes=(256, 160), stride=64)
            occ_128 = _occlusion_agreement_stegastamp(img_uint8, signed_msg, wm_model, margin_full)
            occ_norm = occ_128 / (float(np.abs(occ_128).max()) + 1e-6)
            combined = 0.5 * win_128 + 0.5 * occ_norm * max(0.0, margin_full)
        else:
            raise ValueError(f"Unknown watermarking method: {method}")

    prob_128 = 1.0 / (1.0 + np.exp(-np.clip(combined * 4.0, -10.0, 10.0)))
    return {
        "agreement128": prob_128.astype(np.float32),
        "bit_acc_full": bit_acc_full,
        "margin_full": margin_full,
    }


def compute_matched_filter(
    img_uint8: np.ndarray,
    method: str,
    message_bits: np.ndarray,
    wm_model: Any = None,
) -> np.ndarray:
    """Compare local image high-pass residual against re-embedded watermark residual on 128x128."""
    if wm_model is None:
        return np.full((GRID, GRID), 0.5, dtype=np.float32)

    nbits = NBITS[method]
    bits = np.asarray(message_bits, dtype=np.uint8).reshape(-1)[:nbits]
    re_wm = wm_model.embed(img_uint8, bits)
    srm_img = compute_srm(img_uint8)
    srm_rewm = compute_srm(re_wm)
    template = srm_rewm - srm_img

    # Cell-wise normalized correlation over 4x4 pixel cells.
    c, h, w = srm_img.shape
    a = srm_img.reshape(c, GRID, CELL, GRID, CELL).transpose(1, 3, 0, 2, 4).reshape(GRID, GRID, -1)
    b = template.reshape(c, GRID, CELL, GRID, CELL).transpose(1, 3, 0, 2, 4).reshape(GRID, GRID, -1)
    num = np.sum(a * b, axis=-1)
    denom = np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1)) + 1e-6
    corr = num / denom

    # Smooth over 3x3 cell neighborhood to suppress single-cell noise.
    smooth = _conv3x3_single(corr, np.ones((3, 3), dtype=np.float32) / 9.0)
    prob = 1.0 / (1.0 + np.exp(-np.clip(smooth * 6.0, -10.0, 10.0)))
    return prob.astype(np.float32)


class FeatureCache:
    """Disk cache for per-image precomputed message-aware maps."""

    def __init__(self, cache_dir: str | Path = ROOT_DIR / "output" / "cache" / "features"):
        self.cache_dir = Path(cache_dir)

    def path_for(self, split: str, image_name: str) -> Path:
        return self.cache_dir / split / f"{image_name}.npz"

    def exists(self, split: str, image_name: str) -> bool:
        return self.path_for(split, image_name).is_file()

    def load(self, split: str, image_name: str) -> dict[str, Any] | None:
        p = self.path_for(split, image_name)
        if not p.is_file():
            return None
        with np.load(p, allow_pickle=False) as z:
            return {
                "agreement128": z["agreement128"].astype(np.float32),
                "matched128": z["matched128"].astype(np.float32),
                "bit_acc_full": float(z["bit_acc_full"]),
                "margin_full": float(z["margin_full"]),
            }

    def save(self, split: str, image_name: str, feats: dict[str, Any]) -> Path:
        p = self.path_for(split, image_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            agreement128=np.asarray(feats["agreement128"], dtype=np.float16),
            matched128=np.asarray(feats["matched128"], dtype=np.float16),
            bit_acc_full=np.float32(feats.get("bit_acc_full", 0.5)),
            margin_full=np.float32(feats.get("margin_full", 0.0)),
        )
        return p


def load_kit_models(
    methods: Sequence[str],
    models_dir: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load starter-kit watermark models by method name."""
    ensure_kit_on_path()
    from wmloc.watermarks import load_watermark

    mdir = Path(models_dir) if models_dir is not None else ROOT_DIR / "kit" / "models"
    return {m: load_watermark(m, models_dir=mdir, device=device) for m in methods}


def precompute_feature_cache(
    data_root: str | Path,
    split: str,
    cache_dir: str | Path,
    device: str = "cpu",
    methods: Sequence[str] | None = None,
    limit: int | None = None,
    use_kit: bool = True,
) -> int:
    """Compute and store agreement128 + matched128 features for a split."""
    split_dir = Path(data_root) / split
    meta = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    if methods:
        allowed = set(methods)
        meta = [m for m in meta if m["method"] in allowed]
    if limit is not None:
        meta = meta[:limit]

    used_methods = sorted({m["method"] for m in meta})
    wm_models = load_kit_models(used_methods, device=device) if use_kit else {}
    cache = FeatureCache(cache_dir)
    written = 0

    for row in meta:
        name = row["image_name"]
        if cache.exists(split, name):
            continue
        method = row["method"]
        img = np.asarray(Image.open(split_dir / "images" / f"{name}.png").convert("RGB"), dtype=np.uint8)
        bits = np.array([int(c) for c in row["message"]], dtype=np.uint8)
        wm = wm_models.get(method)
        agree = compute_decoder_agreement(img, method, bits, wm_model=wm)
        matched = compute_matched_filter(img, method, bits, wm_model=wm)
        agree["matched128"] = matched
        cache.save(split, name, agree)
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute message-aware feature maps.")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--cache", default=str(ROOT_DIR / "output" / "cache" / "features"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offline-fallback", action="store_true", help="Skip kit models and write SRM heuristic")
    args = ap.parse_args()

    n = precompute_feature_cache(
        data_root=args.data,
        split=args.split,
        cache_dir=args.cache,
        device=args.device,
        methods=args.methods,
        limit=args.limit,
        use_kit=not args.offline_fallback,
    )
    print(f"Cached {n} feature files into {args.cache}/{args.split}")


if __name__ == "__main__":
    main()
