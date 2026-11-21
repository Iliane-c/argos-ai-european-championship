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


def eiou_decide(prob: np.ndarray) -> np.ndarray:
    """Mask maximizing the expected per-image mIoU (both classes) for independent cell beliefs.

    Expected IoU is approximated by E[intersection] / E[union] (Nowozin, CVPR 2014). For this
    objective the best mask is a top-k set by probability, so all k are scanned in O(N log N).
    """
    p = np.clip(prob.reshape(-1).astype(np.float64), 0.0, 1.0)
    order = np.argsort(-p)
    n, total = p.size, p.sum()
    k = np.arange(n + 1)
    s_k = np.concatenate([[0.0], np.cumsum(p[order])])
    i1, u1 = s_k, k + (total - s_k)
    i0, u0 = (n - k) - (total - s_k), (n - k) + (k - s_k)
    iou1 = np.where(u1 > 1e-9, i1 / np.maximum(u1, 1e-9), 1.0)
    iou0 = np.where(u0 > 1e-9, i0 / np.maximum(u0, 1e-9), 1.0)
    best_k = int(np.argmax(0.5 * (iou1 + iou0)))
    mask = np.zeros(n, dtype=np.uint8)
    mask[order[:best_k]] = 1
    return mask.reshape(prob.shape)


def fit_platt(probs: np.ndarray, gts: np.ndarray, max_cells: int = 200_000, seed: int = 0) -> tuple[float, float]:
    """2-parameter calibration p' = sigmoid(a * logit(p) + b), fitted on (sub-sampled) training cells."""
    rng = np.random.default_rng(seed)
    x = np.log(np.clip(probs.reshape(-1), 1e-4, 1 - 1e-4) / (1 - np.clip(probs.reshape(-1), 1e-4, 1 - 1e-4)))
    y = gts.reshape(-1).astype(np.float64)
    if x.size > max_cells:
        idx = rng.choice(x.size, max_cells, replace=False)
        x, y = x[idx], y[idx]
    a, b = 1.0, 0.0
    for _ in range(50):  # Newton steps on the logistic loss
        q = 1 / (1 + np.exp(-np.clip(a * x + b, -30, 30)))
        g = np.array([((q - y) * x).sum(), (q - y).sum()])
        w = q * (1 - q)
        h = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def cross_fitted_eiou(
    names: list[str],
    probs: np.ndarray,
    meta_by_name: dict[str, dict[str, Any]],
    masks_by_name: dict[str, np.ndarray],
    folds: list[list[str]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Per-method Platt calibration fitted on 4 folds, then the expected-mIoU decision on the 5th."""
    idx = {n: i for i, n in enumerate(names)}
    methods = np.array([meta_by_name[n]["method"] for n in names])
    gts = np.stack([masks_by_name[n] for n in names]).astype(np.uint8)
    scores = np.full(len(names), np.nan)
    cfgs = []
    for k, fold in enumerate(folds):
        val_i = np.array([idx[n] for n in fold if n in idx], dtype=int)
        tr_mask = np.ones(len(names), dtype=bool)
        tr_mask[val_i] = False
        cal = {}
        for m in METHODS:
            sel = tr_mask & (methods == m)
            cal[m] = fit_platt(probs[sel], gts[sel], seed=k) if sel.any() else (1.0, 0.0)
        for i in val_i:
            a, b = cal[methods[i]]
            x = np.log(np.clip(probs[i], 1e-4, 1 - 1e-4) / (1 - np.clip(probs[i], 1e-4, 1 - 1e-4)))
            pred = eiou_decide(1 / (1 + np.exp(-np.clip(a * x + b, -30, 30))))
            scores[i] = miou_batch(pred[None], gts[i][None])[0]
        cfgs.append({"fold": k, "platt": {m: [round(v, 4) for v in cal[m]] for m in METHODS}})
    return scores, cfgs


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
    ap.add_argument("--decision", choices=["thresholds", "eiou"], default="thresholds",
                    help="thresholds: cross-fitted post-processing; eiou: cross-fitted 2-param calibration + expected-mIoU decision")
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
        if args.decision == "eiou":
            scores, fold_cfgs = cross_fitted_eiou(names, probs, meta_by_name, masks_by_name, folds)
        else:
            scores, fold_cfgs = cross_fitted_scores(names, probs, meta_by_name, masks_by_name, folds)
        summary = summarize(names, scores, meta_by_name, folds)
        summary["raw_at_0.5"] = round(raw, 4)
        summary["decision"] = args.decision
        result = {"run": run_dir.name, "honest": summary, "fold_post_cfgs": fold_cfgs}
        (run_dir / f"honest_{args.decision}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
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
