"""Do the true masks follow object contours? (CPU, ~5 min, decides whether object proposals are worth it)

For each training image, the ground-truth 128x128 mask is upsampled to 512x512 and its contour is taken.
Statistic: mean image-gradient magnitude ON the true contour divided by the mean on the same contour
shifted by 12 px in 8 directions (same shape and length, but moved off any object edge it might follow).
The gradient is max-pooled over 5x5 px to absorb the 4-px quantisation of the 128 grid.

  ratio ~ 1   -> masks are placed independently of image content (object proposals are useless);
  ratio >> 1  -> mask contours sit on object edges (SAM / superpixel proposals + decoder test is promising).

Reported per method x kind, and for patches split into near-rectangles (bbox fill >= 0.9) vs other shapes.
Sign test: share of images with ratio > 1 (0.5 under the null).

    python -m wl.edge_align [--split train]
"""
from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np
from PIL import Image, ImageFilter

from wl import METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import load_train_ground_truth

SHIFT = 12
BORDER = 8


def grad_mag(path) -> np.ndarray:
    g = np.asarray(Image.open(path).convert("L").filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    m = np.hypot(gx, gy)
    out = m.copy()
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            out = np.maximum(out, np.roll(np.roll(m, dy, 0), dx, 1))
    return out


def contour(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    c = np.zeros_like(m)
    c[:-1, :] |= m[:-1, :] != m[1:, :]
    c[:, :-1] |= m[:, :-1] != m[:, 1:]
    c[:BORDER, :] = c[-BORDER:, :] = c[:, :BORDER] = c[:, -BORDER:] = False
    return c


def ratio(gm: np.ndarray, mask512: np.ndarray) -> float | None:
    c = contour(mask512)
    if c.sum() < 40:
        return None
    on = gm[c].mean()
    off = []
    for dy, dx in [(SHIFT, 0), (-SHIFT, 0), (0, SHIFT), (0, -SHIFT),
                   (SHIFT, SHIFT), (SHIFT, -SHIFT), (-SHIFT, SHIFT), (-SHIFT, -SHIFT)]:
        cs = contour(np.roll(np.roll(mask512, dy, 0), dx, 1))
        if cs.sum() >= 40:
            off.append(gm[cs].mean())
    return float(on / (np.mean(off) + 1e-6)) if off else None


def bbox_fill(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    return float(mask.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))


def sign_p(k: int, n: int) -> float:
    """Two-sided exact sign test."""
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(label: str, r: list[float]) -> None:
    if not r:
        print(f"  {label:22s} n=0")
        return
    a = np.array(r)
    k = int((a > 1).sum())
    print(f"  {label:22s} n={len(a):3d}  median ratio {np.median(a):.3f}  mean {a.mean():.3f}  "
          f"ratio>1: {k}/{len(a)} ({k / len(a):.0%})  sign p={sign_p(k, len(a)):.2g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    root = ROOT_DIR / "data" / "release"
    meta, masks = load_train_ground_truth(root)
    groups: dict[str, list[float]] = {}
    per_image = []
    for name, m in meta.items():
        mask = masks[name]
        # the region of interest is the minority class: holes for `full`, the patch for `patch`
        roi = (1 - mask) if m["kind"] == "full" else mask
        mask512 = np.kron(roi, np.ones((4, 4), dtype=np.uint8))
        r = ratio(grad_mag(root / args.split / "images" / f"{name}.png"), mask512)
        if r is None:
            continue
        key = f"{METHOD_SHORT[m['method']]} {m['kind']}"
        groups.setdefault(key, []).append(r)
        groups.setdefault(f"ALL {m['kind']}", []).append(r)
        if m["kind"] == "patch":
            shape = "rect" if bbox_fill(roi) >= 0.9 else "object-like"
            groups.setdefault(f"patch {shape}", []).append(r)
            groups.setdefault(f"{METHOD_SHORT[m['method']]} patch {shape}", []).append(r)
        per_image.append({"image": name, "method": m["method"], "kind": m["kind"], "ratio": r,
                          "fill": bbox_fill(roi)})
    print(f"Contour gradient ratio (true contour / contour shifted {SHIFT}px), {len(per_image)} images")
    for key in ["ALL full", "ALL patch", "patch rect", "patch object-like"] + \
               [f"{METHOD_SHORT[mt]} {k}" for mt in METHODS for k in ("full", "patch")] + \
               [f"{METHOD_SHORT[mt]} patch {s}" for mt in METHODS for s in ("rect", "object-like")]:
        summarize(key, groups.get(key, []))
    out = ROOT_DIR / "output" / "edge_align.json"
    try:
        out.write_text(json.dumps(per_image), encoding="utf-8")
        print(f"per-image ratios: {out}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
