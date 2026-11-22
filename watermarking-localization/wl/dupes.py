"""Dataset forensics: are some images built from the same base photo (train/test or test/test)? (CPU)

Every image is reduced to a 32x32 grayscale thumbnail (blurred, mean/std normalised); pairs are compared by
correlation. Watermarks and holes barely change such a thumbnail, so a correlation > 0.98 means the same
base photo. If such pairs exist, the difference of the two full-resolution images is non-zero exactly where
their watermarks / holes differ, which is direct localisation evidence.

    python -m wl.dupes
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from wl import ROOT_DIR


def thumb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").filter(ImageFilter.GaussianBlur(2)).resize((32, 32), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def main() -> None:
    root = ROOT_DIR / "data" / "release"
    items = []
    for split in ("train", "test"):
        meta = json.loads((root / split / "metadata.json").read_text(encoding="utf-8"))
        items += [(split, m["image_name"], m["method"]) for m in meta]
    T = np.stack([thumb(root / s / "images" / f"{n}.png") for s, n, _ in items])
    C = (T @ T.T) / T.shape[1]
    np.fill_diagonal(C, -1)
    iu = np.triu_indices(len(items), 1)
    vals = C[iu]
    for thr in (0.99, 0.98, 0.95, 0.90):
        print(f"pairs with thumbnail correlation > {thr}: {int((vals > thr).sum())}")
    order = np.argsort(-vals)[:15]
    print("top pairs:")
    for k in order:
        i, j = iu[0][k], iu[1][k]
        print(f"  {vals[k]:.4f}  {items[i][0]}/{items[i][1]} ({items[i][2]})  <->  {items[j][0]}/{items[j][1]} ({items[j][2]})")


if __name__ == "__main__":
    main()
