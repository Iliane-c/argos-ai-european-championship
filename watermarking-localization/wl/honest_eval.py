"""Honest cross-fitted evaluation of a run's post-processing (guards against threshold overfitting).

`wl.postproc.fit_postproc_oof` tunes thresholds and clean-up parameters on the same images it then
scores, so its "OOF post-proc" number is optimistic. This module cross-fits instead: for each of
the 5 fixed folds, post-processing is tuned on the other 4 folds and applied to the held-out fold.
The reported score therefore never uses an image's own label to choose its threshold.

Only valid for runs whose `val_probs.npz` covers all 510 train images *without* having trained on
them (training-free methods, synthetic-only models, or merged 5-fold out-of-fold predictions).

    python -m wl.honest_eval output/E125_scan_full output/E126_scan_patch output/E122_hybrid

Writes `honest.json` next to each run's `val_probs.npz` and prints a comparison table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wl import METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch
from wl.postproc import apply_postproc, fit_postproc_oof


def cross_fitted_scores(
    names: list[str],
    probs: np.ndarray,
    meta_by_name: dict[str, dict[str, Any]],
    masks_by_name: dict[str, np.ndarray],
    folds: list[list[str]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Per-image mIoU where each fold's post-processing is fitted on the other folds only."""
    idx = {n: i for i, n in enumerate(names)}
    scores = np.full(len(names), np.nan)
    fold_cfgs: list[dict[str, Any]] = []
    for k, fold in enumerate(folds):
        val_i = np.array([idx[n] for n in fold if n in idx], dtype=int)
        tr_i = np.array([idx[n] for f, other in enumerate(folds) if f != k for n in other if n in idx], dtype=int)
        if len(val_i) == 0 or len(tr_i) == 0:
            continue
        tr_names = [names[i] for i in tr_i]
        fit = fit_postproc_oof(tr_names, probs[tr_i], meta_by_name, masks_by_name)
        cfg = fit["post_cfg"]
        fold_cfgs.append({"fold": k, "post_cfg": cfg, "fit_score_on_train_folds": fit["oof_score"]})
        val_names = [names[i] for i in val_i]
        metas = [meta_by_name[n] for n in val_names]
        preds = apply_postproc(probs[val_i], metas, cfg)
        gts = np.stack([masks_by_name[n] for n in val_names]).astype(np.uint8)
        scores[val_i] = miou_batch(preds, gts)
    return scores, fold_cfgs


def summarize(names: list[str], scores: np.ndarray, meta_by_name: dict[str, dict[str, Any]], folds: list[list[str]]) -> dict[str, Any]:
    methods = np.array([meta_by_name[n]["method"] for n in names])
    kinds = np.array([meta_by_name[n]["kind"] for n in names])
    idx = {n: i for i, n in enumerate(names)}
    fold_means = [float(np.nanmean(scores[[idx[n] for n in f if n in idx]])) for f in folds]
    out: dict[str, Any] = {
        "mean": round(float(np.nanmean(scores)), 4),
        "fold_std": round(float(np.std(fold_means)), 4),
        "fold_means": [round(v, 4) for v in fold_means],
    }
    for m in METHODS:
        out[METHOD_SHORT[m]] = round(float(np.nanmean(scores[methods == m])), 4)
    for k in ("full", "patch"):
        out[k] = round(float(np.nanmean(scores[kinds == k])), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="Run directories containing val_probs.npz")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    args = ap.parse_args()

    meta_by_name, masks_by_name = load_train_ground_truth(args.data)
    folds = json.loads((ROOT_DIR / "splits" / "folds.json").read_text(encoding="utf-8"))["folds"]
    rows = []
    for run in args.runs:
        run_dir = Path(run)
        names, probs = load_probs(run_dir / "val_probs.npz")
        names = [str(n) for n in names]
        probs = probs.astype(np.float32)
        gts = np.stack([masks_by_name[n] for n in names]).astype(np.uint8)
        raw = float(miou_batch((probs >= 0.5).astype(np.uint8), gts).mean())
        if len(names) < len(meta_by_name):
            print(f"{run_dir.name}: val covers {len(names)}/{len(meta_by_name)} images; cross-fitting needs all510. raw@0.5={raw:.4f}")
            continue
        scores, fold_cfgs = cross_fitted_scores(names, probs, meta_by_name, masks_by_name, folds)
        summary = summarize(names, scores, meta_by_name, folds)
        summary["raw_at_0.5"] = round(raw, 4)
        result = {"run": run_dir.name, "honest": summary, "fold_post_cfgs": fold_cfgs}
        (run_dir / "honest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append((run_dir.name, summary))

    if rows:
        print(f"\n{'run':32s} {'raw@.5':>7s} {'honest':>7s} {'±std':>6s} {'PS':>6s} {'TM':>6s} {'SS':>6s} {'full':>6s} {'patch':>6s}")
        for name, s in sorted(rows, key=lambda r: -r[1]["mean"]):
            print(
                f"{name:32s} {s['raw_at_0.5']:7.4f} {s['mean']:7.4f} {s['fold_std']:6.4f} "
                f"{s['PS']:6.4f} {s['TM']:6.4f} {s['SS']:6.4f} {s['full']:6.4f} {s['patch']:6.4f}"
            )
        print("\nA difference smaller than ~1 fold std between two runs is noise.")


if __name__ == "__main__":
    main()
