"""A-contrario watermark localization: decode once, test the true key against random keys.

Post-hoc decoders (PixelSeal, TrustMark, StegaStamp) output one message per input, not a map.
We turn them into calibrated spatial detectors:

1. Decode once (GPU, `extract`): for every image we store the *raw* decoder logits of
   - sliding windows (several sizes, resized to the method's native resolution),
   - the per-position logits of the extractor head (PixelSeal, TrustMark),
   - occlusion probes (StegaStamp: each 64 px patch blurred, whole image decoded),
   - the whole image.
   Testing any message is then a dot product: no re-decoding.

2. Per-image empirical null (CPU, `maps`): the agreement map with the true message is compared,
   cell by cell, with the agreement maps of K random messages ("random keys", as in Fernandez et al.,
   Three Bricks, WIFS 2023, where empirical false-positive rates are checked with random keys).
   Decoder biases on textured content affect true and random keys alike and cancel out.
   Output per cell: a calibrated z-score and an empirical p-value; per image: a global z-score.

Nothing is fitted on the real training labels.

    python -m wl.acontrario extract --split train --device cuda      # GPU, once
    python -m wl.acontrario extract --split test  --device cuda
    python -m wl.acontrario maps --keys 256                           # CPU, minutes
    python -m wl.scan --evidence acontrario --id-full E130_ac_full --id-patch E131_ac_patch

Caches: output/cache/raw/{split}/{name}.npz and output/cache/acontrario/{split}/{name}.npz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from wl import CELL, GRID, IMG_SIZE, NBITS, ROOT_DIR

RAW_DIR = ROOT_DIR / "output" / "cache" / "raw"
AC_DIR = ROOT_DIR / "output" / "cache" / "acontrario"
WINDOW_SIZES = {"pixelseal": (256, 128), "trustmark": (256, 128), "stegastamp": (256, 160)}
WINDOW_STRIDE = 64
OCC_PATCH = 32  # occlusion probes at native scale (whole image decoded), all methods
PP_SHIFTS = 4    # per-position maps at 4x4 sub-cell offsets -> 16 px resolution instead of 64 px


# ----------------------------------------------------------------------------- 1. extraction (GPU)


def _window_boxes(sizes: Sequence[int], stride: int, size: int = IMG_SIZE) -> list[tuple[int, int, int, int]]:
    boxes = []
    for sz in sizes:
        pos = list(range(0, size - sz + 1, stride))
        if pos[-1] != size - sz:
            pos.append(size - sz)
        boxes += [(y, y + sz, x, x + sz) for y in pos for x in pos]
    return boxes


def _decode_batches(wm_model: Any, tensors: list, batch_size: int = 64) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            out.append(wm_model.logits(torch.stack(tensors[i : i + batch_size])).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def extract_raw(img_uint8: np.ndarray, method: str, wm_model: Any) -> dict[str, np.ndarray]:
    """Raw decoder logits for windows, per-position head, occlusion probes and the whole image."""
    import torch
    import torch.nn.functional as F

    nbits = NBITS[method]
    res = int(getattr(wm_model, "resolution", 256))
    base = torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1).float().div(255.0)
    out: dict[str, np.ndarray] = {}

    out["full_logits"] = _decode_batches(wm_model, [base])[0, :nbits].astype(np.float16)

    boxes = _window_boxes(WINDOW_SIZES[method], WINDOW_STRIDE)
    crops = [
        F.interpolate(base[None, :, y0:y1, x0:x1], size=(res, res), mode="bilinear", align_corners=False, antialias=True)[0]
        for y0, y1, x0, x1 in boxes
    ]
    out["win_logits"] = _decode_batches(wm_model, crops)[:, :nbits].astype(np.float16)
    out["win_boxes"] = np.asarray(boxes, dtype=np.int16)

    if method in ("pixelseal", "trustmark") and hasattr(wm_model, "features"):
        # Shift-and-interleave: the extractor's per-position head has a stride of IMG_SIZE / Hf px.
        # Decoding the whole image (native scale) at PP_SHIFTS x PP_SHIFTS sub-stride offsets and
        # interleaving the results gives a map PP_SHIFTS times finer, still at the native scale.
        with torch.no_grad():
            pp0 = wm_model.features(base[None])["per_position_logits"][0].float().cpu().numpy()[:nbits]
            hf, wf = pp0.shape[-2:]
            step_y, step_x = IMG_SIZE // hf, IMG_SIZE // wf
            hr = np.zeros((nbits, hf * PP_SHIFTS, wf * PP_SHIFTS), dtype=np.float32)
            for sy in range(PP_SHIFTS):
                for sx in range(PP_SHIFTS):
                    dy, dx = sy * step_y // PP_SHIFTS, sx * step_x // PP_SHIFTS
                    shifted = torch.roll(base, shifts=(dy, dx), dims=(1, 2))
                    pp = wm_model.features(shifted[None])["per_position_logits"][0].float().cpu().numpy()[:nbits]
                    # cell (i, j) of the shifted image is centred on original pixel i*step + step/2 - dy
                    rows = ((np.arange(hf) * step_y + step_y // 2 - dy) % IMG_SIZE) // (step_y // PP_SHIFTS)
                    cols = ((np.arange(wf) * step_x + step_x // 2 - dx) % IMG_SIZE) // (step_x // PP_SHIFTS)
                    hr[:, rows[:, None], cols[None, :]] = pp
        out["pp_logits"] = pp0.astype(np.float16)  # [nbits, Hf, Wf]
        out["pp_hr_logits"] = hr.astype(np.float16)  # [nbits, Hf*S, Wf*S]

    if OCC_PATCH:  # occlusion probes, every method
        blurred = np.asarray(
            Image.fromarray(img_uint8).resize((64, 64), Image.Resampling.BILINEAR).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        occ_boxes, occ_imgs = [], []
        for y0 in range(0, IMG_SIZE, OCC_PATCH):
            for x0 in range(0, IMG_SIZE, OCC_PATCH):
                canvas = img_uint8.copy()
                canvas[y0 : y0 + OCC_PATCH, x0 : x0 + OCC_PATCH] = blurred[y0 : y0 + OCC_PATCH, x0 : x0 + OCC_PATCH]
                occ_imgs.append(torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0))
                occ_boxes.append((y0, y0 + OCC_PATCH, x0, x0 + OCC_PATCH))
        out["occ_logits"] = _decode_batches(wm_model, occ_imgs)[:, :nbits].astype(np.float16)
        out["occ_boxes"] = np.asarray(occ_boxes, dtype=np.int16)
    return out


def run_extract(data_root: Path, split: str, device: str, limit: int | None, methods: Sequence[str] | None) -> int:
    from wl.features import load_kit_models

    split_dir = data_root / split
    meta = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    if methods:
        meta = [m for m in meta if m["method"] in set(methods)]
    if limit:
        meta = meta[:limit]
    models = load_kit_models(sorted({m["method"] for m in meta}), device=device)
    out_dir = RAW_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, row in enumerate(meta):
        path = out_dir / f"{row['image_name']}.npz"
        if path.is_file():
            continue
        img = np.asarray(Image.open(split_dir / "images" / f"{row['image_name']}.png").convert("RGB"), dtype=np.uint8)
        np.savez_compressed(path, **extract_raw(img, row["method"], models[row["method"]]))
        written += 1
        if written % 50 == 0:
            print(f"[extract {split}] {i + 1}/{len(meta)}", flush=True)
    return written


# ----------------------------------------------------------------------------- 2. random-key null (CPU)


def _projection(boxes: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """[GRID*GRID, n_boxes] row-normalized matrix averaging box scores onto the 128 x 128 cells."""
    w = np.ones(len(boxes)) if weights is None else weights
    proj = np.zeros((GRID, GRID, len(boxes)), dtype=np.float32)
    for j, (y0, y1, x0, x1) in enumerate(np.asarray(boxes, dtype=int)):
        proj[y0 // CELL : -(-y1 // CELL), x0 // CELL : -(-x1 // CELL), j] = w[j]
    proj = proj.reshape(GRID * GRID, -1)
    return proj / np.maximum(proj.sum(axis=1, keepdims=True), 1e-6)


_PROJ_CACHE: dict[tuple, np.ndarray] = {}


def _cached_projection(boxes: np.ndarray, small_weight_first: bool) -> np.ndarray:
    key = (boxes.tobytes(), small_weight_first)
    if key not in _PROJ_CACHE:
        sizes = (boxes[:, 1] - boxes[:, 0]).astype(float)
        # Smaller windows localize better: weight 1 for the smallest size, 0.7 for larger ones.
        weights = np.where(sizes == sizes.min(), 1.0, 0.7) if small_weight_first else None
        _PROJ_CACHE[key] = _projection(boxes, weights)
    return _PROJ_CACHE[key]


def _zscore(true_map: np.ndarray, null_maps: np.ndarray) -> np.ndarray:
    mu = null_maps.mean(axis=0)
    sd = null_maps.std(axis=0) + 1e-6
    return (true_map - mu) / sd


def _resize(arr: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize((GRID, GRID), Image.Resampling.BILINEAR))


def acontrario_maps(raw: dict[str, np.ndarray], bits: np.ndarray, n_keys: int, seed: int) -> dict[str, Any]:
    """Calibrated z-map / p-map of the true key against `n_keys` random keys, plus a global z-score."""
    nbits = raw["full_logits"].shape[0]
    true = 2.0 * bits[:nbits].astype(np.float32) - 1.0
    rng = np.random.default_rng(seed)
    keys = np.vstack([true[None], rng.choice([-1.0, 1.0], size=(n_keys, nbits)).astype(np.float32)])  # [K+1, nbits]

    def agree(logits: np.ndarray) -> np.ndarray:  # logits [..., nbits] -> [..., K+1]
        return np.tanh(0.5 * logits.astype(np.float32)) @ keys.T / nbits

    z_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []  # [K+1, GRID*GRID] score maps, to compute empirical p-values

    win = agree(raw["win_logits"])  # [n_win, K+1]
    win_maps = (_cached_projection(raw["win_boxes"], True) @ win).T  # [K+1, GRID*GRID]
    z_parts.append(_zscore(win_maps[0], win_maps[1:]))
    score_parts.append(win_maps)

    pp_key = "pp_hr_logits" if "pp_hr_logits" in raw else "pp_logits"
    if pp_key in raw:
        pp = np.moveaxis(raw[pp_key].astype(np.float32), 0, -1)  # [H, W, nbits]
        pp_maps = agree(pp)  # [Hf, Wf, K+1]
        pp_up = np.stack([_resize(pp_maps[:, :, k]).reshape(-1) for k in range(pp_maps.shape[-1])])
        z_parts.append(_zscore(pp_up[0], pp_up[1:]))
        score_parts.append(pp_up)

    full = agree(raw["full_logits"][None])[0]  # [K+1]
    if "occ_logits" in raw:
        drop = full[None, :] - agree(raw["occ_logits"])  # margin lost when a patch is blurred
        occ_maps = (_projection(raw["occ_boxes"]) @ drop).T
        z_parts.append(_zscore(occ_maps[0], occ_maps[1:]))
        score_parts.append(occ_maps)

    z = sum(z_parts) / np.sqrt(len(z_parts))
    combined = sum((s - s[1:].mean(0)) / (s[1:].std(0) + 1e-6) for s in score_parts) / np.sqrt(len(score_parts))
    p_emp = (1.0 + (combined[1:] >= combined[0][None]).sum(axis=0)) / (1.0 + n_keys)
    z_global = float((full[0] - full[1:].mean()) / (full[1:].std() + 1e-6))
    return {
        "z128": z.reshape(GRID, GRID).astype(np.float32),
        "p128": p_emp.reshape(GRID, GRID).astype(np.float32),
        "z_global": z_global,
    }


def load_ac(split: str, name: str, ac_dir: Path = AC_DIR) -> dict[str, Any] | None:
    p = ac_dir / split / f"{name}.npz"
    if not p.is_file():
        return None
    with np.load(p) as z:
        return {"z128": z["z128"].astype(np.float32), "p128": z["p128"].astype(np.float32), "z_global": float(z["z_global"])}


def run_maps(data_root: Path, n_keys: int, seed: int, limit: int | None) -> None:
    for split in ("train", "test"):
        meta = json.loads((data_root / split / "metadata.json").read_text(encoding="utf-8"))
        if limit:
            meta = meta[:limit]
        out_dir = AC_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        done = missing = 0
        for i, row in enumerate(meta):
            raw_path = RAW_DIR / split / f"{row['image_name']}.npz"
            if not raw_path.is_file():
                missing += 1
                continue
            with np.load(raw_path) as z:
                raw = {k: z[k] for k in z.files}
            bits = np.array([int(c) for c in row["message"]], dtype=np.uint8)
            res = acontrario_maps(raw, bits, n_keys, seed + i)
            np.savez_compressed(
                out_dir / f"{row['image_name']}.npz",
                z128=res["z128"].astype(np.float16),
                p128=res["p128"].astype(np.float16),
                z_global=np.float32(res["z_global"]),
            )
            done += 1
        print(f"[maps {split}] wrote {done}, missing raw cache for {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="GPU: cache raw decoder logits")
    e.add_argument("--split", choices=["train", "test"], required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--methods", nargs="*", default=None)
    e.add_argument("--limit", type=int, default=None)
    m = sub.add_parser("maps", help="CPU: random-key null -> calibrated z / p maps")
    m.add_argument("--keys", type=int, default=256)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--limit", type=int, default=None)
    for p in (e, m):
        p.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    args = ap.parse_args()
    if args.cmd == "extract":
        n = run_extract(Path(args.data), args.split, args.device, args.limit, args.methods)
        print(f"Extracted {n} raw caches into {RAW_DIR / args.split}")
    else:
        run_maps(Path(args.data), args.keys, args.seed, args.limit)


if __name__ == "__main__":
    main()
