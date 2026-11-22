"""Differential re-synthesis matched filter with a random-message null (training-free, pixel-level).

Why: `wl.diagnose` shows the biggest loss is the holes of PixelSeal / TrustMark `full` images
(small-region IoU 0.16-0.18 vs 0.68 for StegaStamp). Decoders see ~32 px and blur the holes away.

Idea (informed detection + the random-key a-contrario test of Three Bricks):
  * the embedders are deterministic given (image, message), and the image is almost unchanged by the
    mark, so re-embedding the observed image y with the known message m reproduces the residual
    r_m = E(y, m) - y that was added where the image is watermarked;
  * correlating the high-pass of y with r_m locally detects the mark, but r_m also carries a
    content-shaped part (JND / texture modulation) that correlates with y everywhere, holes included;
  * re-embedding with K random messages gives templates sharing that content part but not the message
    part, so z = (c_m - mean_k c_k) / sigma_null cancels the content bias and is calibrated by the null.
At native scale this gives evidence on the 4-px grid, i.e. sharp hole boundaries for `full` images.

    python -m wl.resynth --split train --device cuda        # all 510 train images, prints in/out stats
    python -m wl.resynth --split test --device cuda
Writes output/resynth/{split}.npz with names + z128 (float16), one file per split (inode friendly).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from wl import CELL, GRID, NBITS, ROOT_DIR

OUT_DIR = ROOT_DIR / "output" / "resynth"


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur on the last two axes (reflect padding)."""
    pad = k // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(pad, pad), (pad, pad)], mode="reflect")
    c = np.cumsum(np.cumsum(xp, axis=-1), axis=-2)
    c = np.pad(c, [(0, 0)] * (x.ndim - 2) + [(1, 0), (1, 0)])
    h, w = x.shape[-2:]
    s = c[..., k:k + h, k:k + w] - c[..., :h, k:k + w] - c[..., k:k + h, :w] + c[..., :h, :w]
    return s / (k * k)


def _highpass(x: np.ndarray, k: int = 5) -> np.ndarray:
    return x - _box_blur(x, k)


def _cell_sum(x: np.ndarray) -> np.ndarray:
    """[..., 512, 512] -> [..., 128, 128] sums over 4x4 pixel cells."""
    s = x.shape[:-2]
    return x.reshape(*s, GRID, CELL, GRID, CELL).sum(axis=(-3, -1))


def resynth_z(img: np.ndarray, bits: np.ndarray, wm_model, n_keys: int, seed: int, win: int = 5) -> np.ndarray:
    """Per-cell z of the true-message template correlation against K random-message templates."""
    nb = len(bits)
    rng = np.random.default_rng(seed)
    msgs = [bits] + [rng.integers(0, 2, size=nb, dtype=np.uint8) for _ in range(n_keys)]
    y = img.astype(np.float32).transpose(2, 0, 1)                     # [3, H, W]
    hy = _highpass(y)
    num, den_t = [], []
    for m in msgs:
        t = wm_model.embed(img, m).astype(np.float32).transpose(2, 0, 1) - y
        ht = _highpass(t)
        num.append(_cell_sum((hy * ht).sum(0)))
        den_t.append(_cell_sum((ht * ht).sum(0)))
    num = np.stack(num)                                                 # [K+1, 128, 128]
    den_t = np.stack(den_t)
    den_y = _cell_sum((hy * hy).sum(0))
    num_w = _box_blur(num, win)
    corr = num_w / np.sqrt(_box_blur(den_t, win) * _box_blur(den_y[None], win) + 1e-6)
    null = corr[1:]
    mu = null.mean(0)
    sigma = float(np.std(null - mu[None])) + 1e-6                      # pooled null scale
    return ((corr[0] - mu) / sigma).astype(np.float32)


def candidate_box(prob: np.ndarray, margin_cells: int = 4, min_cells: int = 8) -> tuple[int, int, int, int] | None:
    """Bounding box (in 512-px coordinates) of the largest p>=0.5 component, grown by a margin."""
    from scipy import ndimage

    lab, n = ndimage.label(prob >= 0.5)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < 2:
        return None
    ys, xs = np.nonzero(lab == k)
    y0, y1 = max(0, ys.min() - margin_cells), min(GRID, ys.max() + 1 + margin_cells)
    x0, x1 = max(0, xs.min() - margin_cells), min(GRID, xs.max() + 1 + margin_cells)
    if (y1 - y0) < min_cells:
        c = (y0 + y1) // 2; y0, y1 = max(0, c - min_cells // 2), min(GRID, c + min_cells // 2)
    if (x1 - x0) < min_cells:
        c = (x0 + x1) // 2; x0, x1 = max(0, c - min_cells // 2), min(GRID, c + min_cells // 2)
    return y0 * CELL, y1 * CELL, x0 * CELL, x1 * CELL


def resynth_box_z(img: np.ndarray, bits: np.ndarray, wm_model, box, n_keys: int, seed: int, win: int = 3) -> np.ndarray:
    """Re-synthesis inside a candidate box at the box's own scale (how patches were embedded).

    The crop is re-embedded as a whole, like the task's patch generator does, so the template matches
    the mark at native scale; the z map is written back into the box, zeros elsewhere."""
    y0, y1, x0, x1 = box
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    h, w = crop.shape[:2]
    ph, pw = (-h) % CELL, (-w) % CELL                 # keep cell alignment
    rng = np.random.default_rng(seed)
    msgs = [bits] + [rng.integers(0, 2, size=len(bits), dtype=np.uint8) for _ in range(n_keys)]
    y = crop.astype(np.float32).transpose(2, 0, 1)
    hy = _highpass(y)
    nums, dens = [], []
    for m in msgs:
        t = wm_model.embed(crop, m).astype(np.float32).transpose(2, 0, 1) - y
        ht = _highpass(t)
        prod = np.pad((hy * ht).sum(0), ((0, ph), (0, pw)))
        tt = np.pad((ht * ht).sum(0), ((0, ph), (0, pw)))
        gh, gw = prod.shape[0] // CELL, prod.shape[1] // CELL
        nums.append(prod.reshape(gh, CELL, gw, CELL).sum(axis=(1, 3)))
        dens.append(tt.reshape(gh, CELL, gw, CELL).sum(axis=(1, 3)))
    yy = np.pad((hy * hy).sum(0), ((0, ph), (0, pw)))
    gh, gw = yy.shape[0] // CELL, yy.shape[1] // CELL
    den_y = yy.reshape(gh, CELL, gw, CELL).sum(axis=(1, 3))
    num, den = np.stack(nums), np.stack(dens)
    corr = _box_blur(num, win) / np.sqrt(_box_blur(den, win) * _box_blur(den_y[None], win) + 1e-6)
    null = corr[1:]
    mu = null.mean(0)
    z_box = (corr[0] - mu) / (float(np.std(null - mu[None])) + 1e-6)
    out = np.zeros((GRID, GRID), dtype=np.float32)
    cy, cx = y0 // CELL, x0 // CELL
    out[cy:cy + gh, cx:cx + gw] = z_box[: GRID - cy, : GRID - cx]
    return out


def _stats(names, zs, data_root: Path) -> None:
    from wl.evaluate import load_train_ground_truth

    meta, masks = load_train_ground_truth(data_root)
    rows = {}
    for n, z in zip(names, zs):
        m = meta[n]
        g = masks[n].astype(bool)
        if g.all() or (~g).all():
            continue
        zin, zout = z[g], z[~g]
        # AUC of z for watermarked vs clean cells (rank statistic, subsampled)
        rng = np.random.default_rng(0)
        a = rng.choice(zin, size=min(2000, zin.size)); b = rng.choice(zout, size=min(2000, zout.size))
        auc = float((a[:, None] > b[None, :]).mean())
        rows.setdefault((m["method"], m["kind"]), []).append((float(zin.mean()), float(zout.mean()), auc))
    print("method      kind   n   z(wm)   z(clean)   cell-AUC")
    for (meth, kind), v in sorted(rows.items()):
        v = np.array(v)
        print(f"{meth:10s}  {kind:5s} {len(v):3d}  {v[:, 0].mean():+6.2f}   {v[:, 1].mean():+6.2f}     {v[:, 2].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--keys", type=int, default=8)
    ap.add_argument("--win", type=int, default=5, help="Correlation window in cells (4 px each)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boxes-from", default=None,
                    help="Run dir: for images it predicts as patches, re-synthesise inside its main component box "
                         "(native patch scale) instead of the whole image. Output goes to output/resynth_box/")
    ap.add_argument("--kind", default=None, help="kind_probs.npz; with --boxes-from, images with p_full>=0.5 use the whole image")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    args = ap.parse_args()

    from wl.features import load_kit_models

    data_root = Path(args.data)
    split_dir = data_root / args.split
    meta = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    if args.methods:
        meta = [m for m in meta if m["method"] in set(args.methods)]
    if args.limit:
        meta = meta[: args.limit]
    models = load_kit_models(sorted({m["method"] for m in meta}), device=args.device)
    boxes_probs, p_full = {}, {}
    if args.boxes_from:
        from wl.evaluate import load_probs

        bn, bp = load_probs(Path(args.boxes_from) / ("val_probs.npz" if args.split == "train" else "test_probs.npz"))
        boxes_probs = {str(n): p.astype(np.float32) for n, p in zip(bn, bp)}
        if args.kind:
            kz = np.load(args.kind)
            key_n, key_p = ("train_names", "train_p_full_oof") if args.split == "train" else ("test_names", "test_p_full")
            p_full = {str(n): float(p) for n, p in zip(kz[key_n], kz[key_p])}
        meta = [m for m in meta if m["image_name"] in boxes_probs]
    names, zs = [], []
    for i, row in enumerate(meta):
        n = row["image_name"]
        img = np.asarray(Image.open(split_dir / "images" / f"{n}.png").convert("RGB"), dtype=np.uint8)
        bits = np.array([int(c) for c in row["message"]], dtype=np.uint8)[: NBITS[row["method"]]]
        seed = int(hashlib.md5(n.encode()).hexdigest()[:8], 16)
        box = None
        if args.boxes_from and p_full.get(n, 0.0) < 0.5:
            box = candidate_box(boxes_probs[n])
        if box is not None:
            zs.append(resynth_box_z(img, bits, models[row["method"]], box, args.keys, seed))
        elif args.boxes_from:
            zs.append(np.zeros((GRID, GRID), dtype=np.float32))   # full images: covered by the whole-image run
        else:
            zs.append(resynth_z(img, bits, models[row["method"]], args.keys, seed, args.win))
        names.append(n)
        if (i + 1) % 50 == 0:
            print(f"[resynth {args.split}] {i + 1}/{len(meta)}", flush=True)
    out_dir = OUT_DIR.parent / "resynth_box" if args.boxes_from else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if not (args.limit or args.methods) else "_partial"
    out = out_dir / f"{args.split}{suffix}.npz"
    np.savez_compressed(out, names=np.array(names), z128=np.stack(zs).astype(np.float16))
    print(f"Wrote {out} ({len(names)} images)")
    if args.split == "train":
        _stats(names, zs, data_root)


if __name__ == "__main__":
    main()
