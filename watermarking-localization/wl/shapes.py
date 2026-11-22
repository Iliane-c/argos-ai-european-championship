"""Shape statistics of the ground-truth small regions (CPU, seconds).

For every train image: the small region (holes for `full`, the patch for `patch`), its number of connected
components and its fill ratio inside its bounding box (1.0 = exact axis-aligned rectangle).
If a slice is made of single rectangles, the Bayes decision should be searched over rectangles only.

    python -m wl.shapes
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from wl import METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import load_train_ground_truth


def main() -> None:
    meta, masks = load_train_ground_truth(ROOT_DIR / "data" / "release")
    rows: dict = {}
    for n, g in masks.items():
        kind = meta[n]["kind"]
        small = (g == 0) if kind == "full" else (g == 1)
        if small.sum() == 0:
            continue
        lab, k = ndimage.label(small)
        ys, xs = np.nonzero(small)
        fill = small.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        big = np.bincount(lab.ravel())[1:].max() / small.sum()
        rows.setdefault((meta[n]["method"], kind), []).append((k, fill, big, small.mean()))
    print("slice        n  comps(med)  1-comp%  bbox-fill(med)  fill>0.95%  largest-comp share  area(med)")
    for m in METHODS:
        for kd in ("full", "patch"):
            v = np.array(rows.get((m, kd), []))
            if len(v):
                print(f"{METHOD_SHORT[m]} {kd:5s}  {len(v):3d}  {np.median(v[:,0]):6.0f}     {100*(v[:,0]==1).mean():5.1f}    "
                      f"{np.median(v[:,1]):10.3f}      {100*(v[:,1]>0.95).mean():6.1f}       {np.median(v[:,2]):8.3f}        {np.median(v[:,3]):.3f}")


if __name__ == "__main__":
    main()
