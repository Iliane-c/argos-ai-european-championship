"""Is the loss in the *ranking* of cells or in the *size* of the predicted region? (CPU, seconds)

For each validation image the run's probabilities define a ranking of cells. We score:
  as-is      the run's own 0.5-threshold mask
  eiou       expected-IoU top-k decision on the probabilities (no oracle)
  oracle-n   top-k with k = true number of watermarked cells (oracle SIZE, same ranking)
  oracle-k   best k for this image (upper bound of any size rule on this ranking)
If oracle-n >> eiou, a good size estimate is the lever; if oracle-k ~ eiou, only better evidence helps.

    python -m wl.oracle_k output/E151_verify_e143
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wl import METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch
from wl.honest_eval import eiou_decide


def topk_scores(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """mIoU (both classes) of the top-k mask for every k = 0..N, vectorised."""
    order = np.argsort(-p.reshape(-1))
    gs = g.reshape(-1)[order].astype(np.float64)
    n, pos = gs.size, gs.sum()
    k = np.arange(n + 1)
    tp = np.concatenate([[0.0], np.cumsum(gs)])
    fp, fn = k - tp, pos - tp
    tn = n - k - fn
    iou1 = np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1e-9), 1.0)
    iou0 = np.where(tn + fp + fn > 0, tn / np.maximum(tn + fp + fn, 1e-9), 1.0)
    return 0.5 * (iou1 + iou0)


def main() -> None:
    run = Path(sys.argv[1])
    meta, masks = load_train_ground_truth(ROOT_DIR / "data" / "release")
    names, probs = load_probs(run / "val_probs.npz")
    rows = {}
    for n, p in zip(names, probs):
        n = str(n)
        g = masks[n].astype(np.uint8)
        s = topk_scores(p, g)
        as_is = miou_batch((p >= 0.5).astype(np.uint8)[None], g[None])[0]
        e = miou_batch(eiou_decide(p)[None], g[None])[0]
        key = (meta[n]["method"], meta[n]["kind"])
        rows.setdefault(key, []).append((as_is, e, s[int(g.sum())], s.max()))
    allv = np.array([v for vs in rows.values() for v in vs])
    print(f"{run.name}: {len(allv)} images")
    print("slice        n   as-is    eiou  oracle-n  oracle-k")
    for m in METHODS:
        for kd in ("full", "patch"):
            v = np.array(rows.get((m, kd), []))
            if len(v):
                print(f"{METHOD_SHORT[m]} {kd:5s}   {len(v):3d}  {v[:,0].mean():.4f}  {v[:,1].mean():.4f}  {v[:,2].mean():.4f}   {v[:,3].mean():.4f}")
    print(f"ALL          {len(allv):3d}  {allv[:,0].mean():.4f}  {allv[:,1].mean():.4f}  {allv[:,2].mean():.4f}   {allv[:,3].mean():.4f}")


if __name__ == "__main__":
    main()
