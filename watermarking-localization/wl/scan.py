"""Training-free localization from message evidence: self-normalized z-maps + rectangle scan statistic.

Nothing here is learned from the real training labels. Every image is processed on its own:

1. Evidence map: the cached decoder-agreement map (agreement with the *known* message, from
   `wl.features`) is turned back into its raw score and robustly standardized *within the image*
   (median / MAD). Clean and watermarked cells form two populations; the majority sets the
   reference, so no global calibration is needed.
2. `full` hypothesis (whole image watermarked, find the holes): holes are cells whose evidence is
   far below the image's own median (fixed z threshold, `--hole-z`).
3. `patch` hypothesis (small watermarked patch): the patch is the rectangle that maximizes the
   scan statistic sum(z) / sqrt(area) (the Gaussian scan statistic used for cluster detection),
   searched coarse-to-fine under each method's aspect-ratio limit. Inside the rectangle, cells keep
   their own evidence so object-shaped patches can be recovered by the threshold.

Outputs two standard runs, one per hypothesis, so `wl.kind_hybrid` can choose per image:

    python -m wl.scan --id-full E125_scan_full --id-patch E126_scan_patch [--use-matched]
    python -m wl.kind_hybrid --id E127_scan_hybrid --full-src E125_scan_full --patch-src E126_scan_patch
    python -m wl.honest_eval output/E125_scan_full output/E126_scan_patch output/E127_scan_hybrid

Requires the feature cache (`python -m wl.features --split train|test --device cuda`). CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wl import GRID, MAX_ASPECT, ROOT_DIR
from wl.evaluate import evaluate_predictions, format_metrics_summary, load_train_ground_truth, save_probs, write_metrics
from wl.features import FeatureCache

COARSE = 4  # coarse search grid: 128 / 4 = 32 x 32 blocks of 16 px
COARSE_SIZES = (1, 2, 3, 4, 6, 8, 11, 16, 22, 32)
MAX_PATCH_AREA = 0.5  # largest patch in train covers 44 % of the image


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _smooth3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    return sum(p[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)) / 9.0


def robust_z(raw: np.ndarray) -> np.ndarray:
    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med))) * 1.4826
    return (raw - med) / (mad + 1e-6)


def evidence_map(feats: dict[str, Any], use_matched: bool) -> np.ndarray:
    """Within-image standardized evidence that a cell carries the known message (128 x 128)."""
    agree = np.clip(np.asarray(feats["agreement128"], dtype=np.float64), 1e-4, 1 - 1e-4)
    z = robust_z(np.log(agree / (1 - agree)))
    if use_matched and "matched128" in feats:
        m = np.clip(np.asarray(feats["matched128"], dtype=np.float64), 1e-4, 1 - 1e-4)
        z = (z + robust_z(np.log(m / (1 - m)))) / np.sqrt(2.0)
    return _smooth3(z)


def _integral(a: np.ndarray) -> np.ndarray:
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    return ii


def _rect_sum(ii: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def _scan_score(ii: np.ndarray, r: tuple[int, int, int, int]) -> float:
    y0, y1, x0, x1 = r
    return _rect_sum(ii, y0, y1, x0, x1) / np.sqrt((y1 - y0) * (x1 - x0))


def _valid(r: tuple[int, int, int, int], max_aspect: float, n: int) -> bool:
    y0, y1, x0, x1 = r
    h, w = y1 - y0, x1 - x0
    return 0 <= y0 < y1 <= n and 0 <= x0 < x1 <= n and max(h, w) / min(h, w) <= max_aspect and h * w <= MAX_PATCH_AREA * n * n


def scan_rectangle(z: np.ndarray, max_aspect: float) -> tuple[tuple[int, int, int, int], float]:
    """Rectangle maximizing sum(z)/sqrt(area): exhaustive on a 32x32 grid, then edge refinement at 128."""
    n = z.shape[0]
    zc = z.reshape(n // COARSE, COARSE, n // COARSE, COARSE).mean(axis=(1, 3)) * COARSE  # keeps sum/sqrt(area) scale
    m = zc.shape[0]
    iic = _integral(zc)
    best, best_s = (0, 1, 0, 1), -np.inf
    for h in COARSE_SIZES:
        for w in COARSE_SIZES:
            if max(h, w) / min(h, w) > max_aspect or h * w > MAX_PATCH_AREA * m * m:
                continue
            sums = iic[h:, w:] - iic[:-h, w:] - iic[h:, :-w] + iic[:-h, :-w]
            s = sums / np.sqrt(h * w)
            iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
            if s[iy, ix] > best_s:
                best_s, best = float(s[iy, ix]), (iy, iy + h, ix, ix + w)

    ii = _integral(z)
    r = tuple(v * COARSE for v in best)
    s = _scan_score(ii, r)
    for _ in range(4):  # coordinate ascent on the four edges, 1-cell precision
        improved = False
        for edge in range(4):
            for delta in range(-8, 9):
                if delta == 0:
                    continue
                cand = list(r)
                cand[edge] += delta
                cand = tuple(cand)
                if _valid(cand, max_aspect, n):
                    cs = _scan_score(ii, cand)
                    if cs > s + 1e-9:
                        r, s, improved = cand, cs, True
        if not improved:
            break
    return r, s


def full_map(z: np.ndarray, hole_z: float) -> np.ndarray:
    """p(watermarked) under the `full` hypothesis: holes are strong negative outliers."""
    return _sigmoid(z + hole_z).astype(np.float32)


def patch_map(z: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    """p(watermarked) under the `patch` hypothesis: scan rectangle, cells keep their own evidence."""
    (y0, y1, x0, x1), score = scan_rectangle(z, MAX_ASPECT[method])
    rect = np.zeros_like(z)
    rect[y0:y1, x0:x1] = 1.0
    inside_ref = float(np.median(z[y0:y1, x0:x1]))
    # Inside: 0.55..1.0 depending on whether the cell looks like the rectangle's watermarked bulk.
    # Outside: at most 0.45, so the threshold decides between the box and its object-shaped core.
    p = 0.55 * rect + 0.45 * _sigmoid(z - 0.5 * inside_ref)
    return np.clip(p, 0.0, 1.0).astype(np.float32), {"rect": [int(y0), int(y1), int(x0), int(x1)], "scan_score": round(float(score), 3)}


def run_split(split: str, rows: list[dict[str, Any]], cache: FeatureCache, use_matched: bool, hole_z: float):
    names, full_probs, patch_probs, info, missing = [], [], [], {}, 0
    for row in rows:
        n = row["image_name"]
        feats = cache.load(split, n)
        if feats is None:
            missing += 1
            continue
        z = evidence_map(feats, use_matched)
        pm, meta = patch_map(z, row["method"])
        names.append(n)
        full_probs.append(full_map(z, hole_z))
        patch_probs.append(pm)
        info[n] = meta
    if missing > 0.05 * len(rows):
        raise SystemExit(f"ERROR: {missing}/{len(rows)} {split} images have no cached features. Run wl.features first.")
    return names, np.stack(full_probs), np.stack(patch_probs), info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id-full", default="E125_scan_full")
    ap.add_argument("--id-patch", default="E126_scan_patch")
    ap.add_argument("--use-matched", action="store_true", help="Average agreement and matched-filter evidence")
    ap.add_argument("--hole-z", type=float, default=2.0, help="Fixed z threshold for holes (not tuned on labels)")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--cache", default=str(ROOT_DIR / "output" / "cache" / "features"))
    ap.add_argument("--out", default=str(ROOT_DIR / "output"))
    ap.add_argument("--limit", type=int, default=None, help="Debug: only the first N images per split")
    args = ap.parse_args()

    cache = FeatureCache(args.cache)
    meta_by_name, masks_by_name = load_train_ground_truth(args.data)
    train_rows = [meta_by_name[n] for n in sorted(meta_by_name)]
    test_rows = json.loads((Path(args.data) / "test" / "metadata.json").read_text(encoding="utf-8"))
    if args.limit:
        train_rows, test_rows = train_rows[: args.limit], test_rows[: args.limit]

    tr = run_split("train", train_rows, cache, args.use_matched, args.hole_z)
    te = run_split("test", test_rows, cache, args.use_matched, args.hole_z)

    params = {"use_matched": args.use_matched, "hole_z": args.hole_z, "trained_on_labels": False}
    for exp_id, k in ((args.id_full, 1), (args.id_patch, 2)):
        out_dir = Path(args.out) / exp_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save_probs(out_dir / "val_probs.npz", tr[0], tr[k])
        save_probs(out_dir / "test_probs.npz", te[0], te[k])
        metrics = evaluate_predictions(tr[0], tr[k], meta_by_name, masks_by_name, threshold=0.5)
        metrics.update({"id": exp_id, "parent": "none", "data": "none", "val": "all510", **params})
        write_metrics(out_dir, metrics)
        print(f"[{exp_id}] {format_metrics_summary(metrics)}")
    (Path(args.out) / args.id_patch / "rects.json").write_text(
        json.dumps({"train": tr[3], "test": te[3]}, indent=1) + "\n", encoding="utf-8"
    )
    print("Note: each map is only meaningful for its own kind; combine them with wl.kind_hybrid,")
    print("then score with wl.honest_eval (cross-fitted thresholds).")


if __name__ == "__main__":
    main()
