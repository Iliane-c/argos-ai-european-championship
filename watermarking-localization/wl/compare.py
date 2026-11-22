"""Paired comparison of runs on the same validation images: which differences are real?

Two runs scored on the same images give paired per-image mIoU differences. The paired bootstrap CI of
the mean difference and the exact sign test on wins/losses are much tighter than comparing two means
with their fold std, and they show when a gain rests on two or three images only. Every run is
compared with the first one (threshold 0.5, as in metrics.json), overall and per method x kind.

    python -m wl.compare output/E158_verify_e157g output/E190_gate_e158 output/E161_tta
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np

from wl import METHOD_SHORT, METHODS, ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch


def per_image_scores(run: Path, masks_by_name: dict[str, np.ndarray]) -> dict[str, float]:
    names, probs = load_probs(run / "val_probs.npz")
    gts = np.stack([masks_by_name[n] for n in names]).astype(np.uint8)
    return dict(zip(names, miou_batch((probs >= 0.5).astype(np.uint8), gts)))


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial test of wins against losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)


def paired_summary(d: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float, int, int, float]:
    """Mean difference, its 95 % paired-bootstrap CI, wins, losses and the sign-test p-value."""
    boots = d[np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    wins, losses = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return float(d.mean()), float(lo), float(hi), wins, losses, sign_test(wins, losses)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="Run directories with val_probs.npz; the first one is the reference")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    args = ap.parse_args()

    meta_by_name, masks_by_name = load_train_ground_truth(args.data)
    scores = [per_image_scores(Path(r), masks_by_name) for r in args.runs]
    shared = sorted(set.intersection(*[set(s) for s in scores]))
    methods = np.array([meta_by_name[n]["method"] for n in shared])
    kinds = np.array([meta_by_name[n]["kind"] for n in shared])
    ref = np.array([scores[0][n] for n in shared])
    print(f"{len(shared)} shared validation images; reference {Path(args.runs[0]).name} = {ref.mean():.4f}")
    for run, s in zip(args.runs[1:], scores[1:]):
        x = np.array([s[n] for n in shared])
        mean, lo, hi, wins, losses, p = paired_summary(x - ref)
        slices = " ".join(f"{METHOD_SHORT[m]}{k[0]}={(x - ref)[(methods == m) & (kinds == k)].mean():+.3f}"
                          for m in METHODS for k in ("full", "patch"))
        print(f"{Path(run).name:28s} {x.mean():.4f}  d={mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"wins/losses={wins}/{losses} sign p={p:.3f} | {slices}")


if __name__ == "__main__":
    main()
