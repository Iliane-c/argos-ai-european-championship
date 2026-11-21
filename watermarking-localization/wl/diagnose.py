"""Where does a run lose points? Breaks validation scores down by method x kind and target size.

    python -m wl.diagnose output/E104_w3_msg_cond [output/E103_w2_wam_multitask ...]

For each run (its val_probs.npz, threshold 0.5) prints, per method x kind:
  score     official per-image mIoU
  tgtIoU    IoU of the small region (holes for `full`, patch for `patch`): the half that decides the score
  kindErr   share of images whose predicted map has the wrong kind (majority class flipped)
  empty     share of images with no predicted small region
then the small-region IoU by target size (tiny < 1 %, small 1-5 %, medium 5-15 %, large > 15 %).
CPU only, needs no models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wl import METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch

SIZE_BINS = ((0.0, 0.01, "tiny<1%"), (0.01, 0.05, "small1-5%"), (0.05, 0.15, "med5-15%"), (0.15, 1.01, "large>15%"))


def target_iou(pred: np.ndarray, gt: np.ndarray, kind: str) -> float:
    t_pred = pred == (0 if kind == "full" else 1)
    t_gt = gt == (0 if kind == "full" else 1)
    union = np.logical_or(t_pred, t_gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(t_pred, t_gt).sum() / union)


def diagnose(run_dir: Path, data_root: Path) -> None:
    meta, masks = load_train_ground_truth(data_root)
    names, probs = load_probs(run_dir / "val_probs.npz")
    preds = (probs >= 0.5).astype(np.uint8)
    gts = np.stack([masks[n] for n in names]).astype(np.uint8)
    scores = miou_batch(preds, gts)
    kinds = [meta[n]["kind"] for n in names]
    methods = [meta[n]["method"] for n in names]
    t_iou = np.array([target_iou(p, g, k) for p, g, k in zip(preds, gts, kinds)])
    kind_err = np.array([(p.mean() >= 0.5) != (k == "full") for p, k in zip(preds, kinds)])
    empty = np.array([(p == (0 if k == "full" else 1)).sum() == 0 for p, k in zip(preds, kinds)])
    t_frac = np.array([(g == (0 if k == "full" else 1)).mean() for g, k in zip(gts, kinds)])

    print(f"\n=== {run_dir.name}: {len(names)} val images, mean {scores.mean():.4f}")
    print(f"{'slice':10s} {'n':>4s} {'score':>7s} {'tgtIoU':>7s} {'kindErr':>8s} {'empty':>6s}")
    for m in METHODS:
        for k in ("full", "patch"):
            idx = np.array([mm == m and kk == k for mm, kk in zip(methods, kinds)])
            if idx.any():
                print(f"{METHOD_SHORT[m] + ' ' + k:10s} {int(idx.sum()):4d} {scores[idx].mean():7.4f} "
                      f"{t_iou[idx].mean():7.4f} {kind_err[idx].mean():8.2%} {empty[idx].mean():6.2%}")
    print("small-region IoU by target size: " + " | ".join(
        f"{label} {t_iou[(t_frac >= lo) & (t_frac < hi)].mean():.3f} (n={int(((t_frac >= lo) & (t_frac < hi)).sum())})"
        for lo, hi, label in SIZE_BINS if ((t_frac >= lo) & (t_frac < hi)).any()))
    fixed = np.where(kind_err, np.nan, scores)
    print(f"images with a kind error: {int(kind_err.sum())}; mean score on the rest: {np.nanmean(fixed):.4f}")
    worst = np.argsort(scores)[:8]
    print("worst: " + ", ".join(f"{names[i]}({METHOD_SHORT[methods[i]]} {kinds[i]} {scores[i]:.2f})" for i in worst))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        diagnose(Path(arg), ROOT_DIR / "data" / "release")


if __name__ == "__main__":
    main()
