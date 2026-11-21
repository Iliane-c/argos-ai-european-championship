"""Learned full/patch kind classifier and hybrid assembler (kind -> per-kind probability source).

Why: the full/patch decision alone is worth ~0.455 mIoU (E00d), and the hand-tuned rule in
`wl.analytic.predict_analytic_t1` is not calibrated. This module:

1. Builds a few decoder-based features per image from the feature cache (`wl.features`):
   whole-image bit accuracy and margin, plus agreement / matched-filter statistics.
2. Fits a per-method L2 logistic regression (5 features + bias, so very little to overfit),
   with out-of-fold predictions on the fixed 5 folds for train and a model fit on all 510
   images for test.
3. Assembles a hybrid probability map: images predicted `full` take the `--full-src` maps,
   images predicted `patch` take the `--patch-src` maps. A source is either `const`
   (all ones for full, all zeros for patch) or an experiment ID under `output/` that has
   `val_probs.npz` / `test_probs.npz`.

Usage (feature cache must exist: `python -m wl.features --split train|test --device cuda`):

    python -m wl.kind_hybrid --id E110_kind_const
    python -m wl.kind_hybrid --id E111_hybrid --full-src E100_analytic_t1 --patch-src E106_rect_v2

Outputs in output/<id>/: kind_probs.npz, val_probs.npz, test_probs.npz, metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wl import GRID, ROOT_DIR
from wl.evaluate import (
    evaluate_predictions,
    format_metrics_summary,
    load_probs,
    load_train_ground_truth,
    save_probs,
    write_metrics,
)
from wl.features import FeatureCache

METHODS = ("pixelseal", "trustmark", "stegastamp")
FEATURE_NAMES = ("bit_acc_full", "margin_full", "agree_mean", "agree_frac", "matched_mean")


# ----------------------------------------------------------------------------- features


def image_features(feats: dict[str, Any]) -> np.ndarray:
    agree = np.asarray(feats["agreement128"], dtype=np.float32)
    matched = np.asarray(feats.get("matched128", np.full((GRID, GRID), 0.5)), dtype=np.float32)
    return np.array(
        [
            float(feats.get("bit_acc_full", 0.5)),
            float(feats.get("margin_full", 0.0)),
            float(agree.mean()),
            float((agree > 0.5).mean()),
            float(matched.mean()),
        ],
        dtype=np.float64,
    )


def load_split_features(cache: FeatureCache, split: str, names: list[str]) -> tuple[np.ndarray, list[str]]:
    rows, missing = [], []
    for n in names:
        f = cache.load(split, n)
        if f is None:
            missing.append(n)
            rows.append(np.full(len(FEATURE_NAMES), np.nan))
        else:
            rows.append(image_features(f))
    return np.stack(rows), missing


# ----------------------------------------------------------------------------- logistic regression


class Logistic:
    """Standardized L2 logistic regression trained with Newton steps (tiny, dependency-free)."""

    def __init__(self, l2: float = 1.0, iters: int = 50):
        self.l2, self.iters = l2, iters

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Logistic":
        self.mu = np.nanmean(x, axis=0)
        self.sd = np.nanstd(x, axis=0) + 1e-6
        z = self._design(x)
        w = np.zeros(z.shape[1])
        reg = self.l2 * np.eye(z.shape[1])
        reg[0, 0] = 0.0  # no penalty on the bias
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
            grad = z.T @ (p - y) + reg @ w
            hess = (z * (p * (1 - p))[:, None]).T @ z + reg + 1e-6 * np.eye(z.shape[1])
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def _design(self, x: np.ndarray) -> np.ndarray:
        z = (np.nan_to_num(x, nan=0.0) - self.mu) / self.sd
        z = np.where(np.isnan(x), 0.0, z)
        return np.hstack([np.ones((len(z), 1)), z])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self._design(x) @ self.w, -30, 30)))


def fit_kind_models(
    train_names: list[str],
    x_train: np.ndarray,
    meta_by_name: dict[str, dict[str, Any]],
    folds: list[list[str]],
    l2: float,
) -> tuple[np.ndarray, dict[str, Logistic], dict[str, Any]]:
    """Per-method OOF p(full) on train plus per-method models fit on all train images."""
    name_to_idx = {n: i for i, n in enumerate(train_names)}
    methods = np.array([meta_by_name[n]["method"] for n in train_names])
    y = np.array([1.0 if meta_by_name[n]["kind"] == "full" else 0.0 for n in train_names])
    oof = np.full(len(train_names), np.nan)

    for fold in folds:
        val_idx = np.array([name_to_idx[n] for n in fold if n in name_to_idx])
        val_mask = np.zeros(len(train_names), dtype=bool)
        val_mask[val_idx] = True
        for m in METHODS:
            tr = (~val_mask) & (methods == m)
            va = val_mask & (methods == m)
            if va.any() and tr.sum() > 5:
                oof[va] = Logistic(l2).fit(x_train[tr], y[tr]).predict_proba(x_train[va])

    full_models = {m: Logistic(l2).fit(x_train[methods == m], y[methods == m]) for m in METHODS}

    report: dict[str, Any] = {}
    for m in METHODS:
        sel = methods == m
        acc = float(((oof[sel] >= 0.5) == (y[sel] == 1)).mean())
        report[m] = {"oof_kind_acc": round(acc, 4), "n": int(sel.sum())}
    report["all"] = {"oof_kind_acc": round(float(((oof >= 0.5) == (y == 1)).mean()), 4)}
    report["weights"] = {m: [round(float(v), 3) for v in full_models[m].w] for m in METHODS}
    report["feature_order"] = ["bias", *FEATURE_NAMES]
    return oof, full_models, report


# ----------------------------------------------------------------------------- hybrid assembly


def load_source(src: str, split: str, out_root: Path) -> dict[str, np.ndarray] | None:
    if src == "const":
        return None
    fname = "val_probs.npz" if split == "train" else "test_probs.npz"
    path = out_root / src / fname
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (source '{src}', split '{split}')")
    names, probs = load_probs(path)
    return {str(n): p.astype(np.float32) for n, p in zip(names, probs)}


def assemble(
    names: list[str],
    p_full: np.ndarray,
    full_src: dict[str, np.ndarray] | None,
    patch_src: dict[str, np.ndarray] | None,
    soft: bool,
) -> tuple[list[str], np.ndarray]:
    """Pick the per-kind source map for each image; images missing from a needed source are skipped."""
    out_names, out_probs = [], []
    ones = np.ones((GRID, GRID), dtype=np.float32)
    zeros = np.zeros((GRID, GRID), dtype=np.float32)
    for n, pf in zip(names, p_full):
        f_map = ones if full_src is None else full_src.get(n)
        p_map = zeros if patch_src is None else patch_src.get(n)
        if f_map is None or p_map is None:
            continue
        if soft:
            prob = pf * f_map + (1.0 - pf) * p_map
        else:
            prob = f_map if pf >= 0.5 else p_map
        out_names.append(n)
        out_probs.append(np.asarray(prob, dtype=np.float32))
    return out_names, np.stack(out_probs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="Experiment ID, e.g. E110_kind_const")
    ap.add_argument("--full-src", default="const", help="'const' or experiment ID used for images predicted full")
    ap.add_argument("--patch-src", default="const", help="'const' or experiment ID used for images predicted patch")
    ap.add_argument("--soft", action="store_true", help="Blend sources with p(full) instead of a hard switch")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--cache", default=str(ROOT_DIR / "output" / "cache" / "features"))
    ap.add_argument("--out", default=str(ROOT_DIR / "output"))
    args = ap.parse_args()

    out_root = Path(args.out)
    out_dir = out_root / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(args.cache)

    meta_by_name, masks_by_name = load_train_ground_truth(args.data)
    train_names = sorted(meta_by_name)
    folds = json.loads((ROOT_DIR / "splits" / "folds.json").read_text(encoding="utf-8"))["folds"]
    test_meta = json.loads((Path(args.data) / "test" / "metadata.json").read_text(encoding="utf-8"))
    test_names = [r["image_name"] for r in test_meta]
    test_methods = np.array([r["method"] for r in test_meta])

    x_train, miss_tr = load_split_features(cache, "train", train_names)
    x_test, miss_te = load_split_features(cache, "test", test_names)
    if miss_tr or miss_te:
        frac = (len(miss_tr) + len(miss_te)) / (len(train_names) + len(test_names))
        msg = (f"missing cached features: train={len(miss_tr)}/{len(train_names)} "
               f"test={len(miss_te)}/{len(test_names)} in {args.cache}")
        if frac > 0.05:
            raise SystemExit(f"ERROR: {msg}. Run `python -m wl.features --split train|test --device cuda` first.")
        print(f"WARNING: {msg} (treated as NaN)")

    oof, models, report = fit_kind_models(train_names, x_train, meta_by_name, folds, args.l2)
    p_test = np.zeros(len(test_names))
    for m in METHODS:
        sel = test_methods == m
        if sel.any():
            p_test[sel] = models[m].predict_proba(x_test[sel])

    print("Kind classifier (out-of-fold accuracy):", json.dumps(report, indent=2))
    print(f"Test images predicted full: {float((p_test >= 0.5).mean()):.3f} (train share of full: 0.5)")
    np.savez_compressed(
        out_dir / "kind_probs.npz",
        train_names=np.array(train_names),
        train_p_full_oof=oof.astype(np.float32),
        test_names=np.array(test_names),
        test_p_full=p_test.astype(np.float32),
    )

    v_names, v_probs = assemble(
        train_names, oof, load_source(args.full_src, "train", out_root), load_source(args.patch_src, "train", out_root), args.soft
    )
    save_probs(out_dir / "val_probs.npz", v_names, v_probs)
    metrics = evaluate_predictions(v_names, v_probs, meta_by_name, masks_by_name, threshold=0.5)
    metrics.update(
        {
            "id": args.id,
            "full_src": args.full_src,
            "patch_src": args.patch_src,
            "soft": args.soft,
            "kind_report": report,
            "val": "all510-oof" if len(v_names) == len(train_names) else f"subset-{len(v_names)}",
        }
    )
    write_metrics(out_dir, metrics)
    print(f"[{args.id}] {format_metrics_summary(metrics)}")

    t_names, t_probs = assemble(
        test_names, p_test, load_source(args.full_src, "test", out_root), load_source(args.patch_src, "test", out_root), args.soft
    )
    if len(t_names) != len(test_names):
        print(f"WARNING: test assembled for {len(t_names)}/{len(test_names)} images (missing source maps)")
    save_probs(out_dir / "test_probs.npz", t_names, t_probs)
    print(f"Wrote {out_dir}/val_probs.npz, test_probs.npz, metrics.json, kind_probs.npz")


if __name__ == "__main__":
    main()
