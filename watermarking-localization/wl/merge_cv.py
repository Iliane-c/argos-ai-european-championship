"""Merge K-fold CV runs into a single 510-image OOF val_probs.npz and averaged test_probs.npz."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from wl import ROOT_DIR
from wl.evaluate import (
    evaluate_predictions,
    format_metrics_summary,
    load_probs,
    load_train_ground_truth,
    save_probs,
    write_metrics,
)
from wl.postproc import fit_postproc_oof


def merge_cv_runs(
    run_dirs: Sequence[str | Path],
    out_dir: str | Path,
    data_root: str | Path = ROOT_DIR / "data" / "release",
    tune_oof: bool = True,
) -> dict[str, Any]:
    """Concatenate disjoint fold val_probs, average aligned test_probs, and compute 510-image OOF metrics."""
    if not run_dirs:
        raise ValueError("Provide at least one fold run directory to merge.")

    meta_by_name, masks_by_name = load_train_ground_truth(data_root)
    val_names_all: list[str] = []
    val_probs_all: list[np.ndarray] = []
    seen_val: set[str] = set()
    fold_scores: list[float] = []

    test_ref_names: list[str] | None = None
    test_prob_sum: np.ndarray | None = None
    test_count = 0

    for raw_dir in run_dirs:
        rdir = Path(raw_dir)
        val_path = rdir / "val_probs.npz"
        if not val_path.is_file():
            raise FileNotFoundError(f"Missing fold val predictions: {val_path}")

        v_names, v_probs = load_probs(val_path)
        overlap = seen_val.intersection(v_names)
        if overlap:
            raise ValueError(f"Duplicate val image(s) across folds in {rdir}: {sorted(overlap)[:5]}")
        seen_val.update(v_names)
        val_names_all.extend(v_names)
        val_probs_all.append(v_probs)

        fold_metrics = evaluate_predictions(v_names, v_probs, meta_by_name, masks_by_name, threshold=0.5)
        fold_scores.append(float(fold_metrics["mean"]))

        test_path = rdir / "test_probs.npz"
        if test_path.is_file():
            t_names, t_probs = load_probs(test_path)
            if test_ref_names is None:
                test_ref_names = t_names
                test_prob_sum = t_probs.astype(np.float64)
            else:
                if set(t_names) != set(test_ref_names) or len(t_names) != len(test_ref_names):
                    raise ValueError(f"Test image set mismatch in {test_path}")
                pos = {n: i for i, n in enumerate(t_names)}
                order = [pos[n] for n in test_ref_names]
                test_prob_sum = test_prob_sum + t_probs[order].astype(np.float64)
            test_count += 1

    merged_val = np.concatenate(val_probs_all, axis=0)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_probs(out_path / "val_probs.npz", val_names_all, merged_val)

    metrics = evaluate_predictions(val_names_all, merged_val, meta_by_name, masks_by_name, threshold=0.5)
    metrics.update(
        {
            "id": out_path.name,
            "parent": "cv_merge",
            "data": f"{len(run_dirs)}fold",
            "val": f"oof{len(val_names_all)}",
            "folds": [str(Path(d)) for d in run_dirs],
            "fold_means": [round(s, 6) for s in fold_scores],
            "fold_std": float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0,
        }
    )

    if tune_oof:
        oof = fit_postproc_oof(val_names_all, merged_val, meta_by_name, masks_by_name)
        metrics["oof_post_cfg"] = oof["post_cfg"]
        metrics["oof_mean"] = oof["oof_score"]
        metrics["oof_metrics"] = oof["metrics"]

    write_metrics(out_path, metrics)

    if test_ref_names is not None and test_prob_sum is not None and test_count == len(run_dirs):
        save_probs(out_path / "test_probs.npz", test_ref_names, (test_prob_sum / float(test_count)).astype(np.float32))
    elif test_count > 0:
        print(f"Warning: only {test_count}/{len(run_dirs)} folds had test_probs.npz; skipped merged test_probs.npz.")

    print(
        f"[{out_path.name}] merged {len(run_dirs)} folds ({len(val_names_all)} imgs, "
        f"fold_std={metrics['fold_std']:.4f}): {format_metrics_summary(metrics)}"
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge K-fold CV runs into a single OOF run directory.")
    ap.add_argument("--runs", nargs="+", required=True, help="Fold output directories (e.g. output/E150_w7_f0 ... f4)")
    ap.add_argument("--out", required=True, help="Merged output directory (e.g. output/E150_w7_5fold)")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--no-tune-oof", action="store_true", help="Skip OOF post-processing fit")
    args = ap.parse_args()

    merge_cv_runs(
        run_dirs=args.runs,
        out_dir=args.out,
        data_root=args.data,
        tune_oof=not args.no_tune_oof,
    )


if __name__ == "__main__":
    main()
