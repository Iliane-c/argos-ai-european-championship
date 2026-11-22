"""Evaluation utilities, official mIoU metric, OOF threshold tuning, and prediction I/O."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from wl import GRID, METHOD_SHORT, METHODS, ROOT_DIR


def save_probs(path: str | Path, names: Sequence[str], probs: np.ndarray) -> Path:
    """Save [N, 128, 128] probabilities as compressed float16 .npz."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probs_arr = np.asarray(probs, dtype=np.float32)
    names_arr = np.asarray(list(names), dtype=str)
    if probs_arr.ndim != 3 or probs_arr.shape != (len(names_arr), GRID, GRID):
        raise ValueError(
            f"Expected probs shape ({len(names_arr)}, {GRID}, {GRID}), got {probs_arr.shape}"
        )
    probs_f16 = np.clip(probs_arr, 0.0, 1.0).astype(np.float16)
    np.savez_compressed(out_path, names=names_arr, probs=probs_f16)
    return out_path


def load_probs(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load standard .npz probability file as (names, float32 probs [N, 128, 128])."""
    with np.load(path, allow_pickle=False) as z:
        names = [str(n) for n in z["names"].tolist()]
        probs = z["probs"].astype(np.float32)
    return names, probs


def downsample_mask(mask: np.ndarray) -> np.ndarray:
    """Official 512->128 cell rule: 1 if >= 50% of pixels in each cell are watermarked."""
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] % GRID != 0:
        raise ValueError(f"Expected square mask divisible by {GRID}, got {m.shape}")
    cell = m.shape[0] // GRID
    return (m.reshape(GRID, cell, GRID, cell).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)


def miou_image(pred: np.ndarray, gt: np.ndarray) -> float:
    """Official per-image mIoU matching kit/wmloc/utils.py exactly."""
    p = np.asarray(pred, dtype=np.uint8)
    g = np.asarray(gt, dtype=np.uint8)
    total = 0.0
    for c in (0, 1):
        union = np.logical_or(p == c, g == c).sum()
        total += 1.0 if union == 0 else float(np.logical_and(p == c, g == c).sum()) / float(union)
    return total / 2.0


def miou_batch(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Vectorized per-image mIoU across [N, 128, 128] binary masks."""
    p = np.asarray(preds, dtype=np.uint8)
    g = np.asarray(gts, dtype=np.uint8)
    if p.shape != g.shape or p.ndim != 3:
        raise ValueError(f"Shape mismatch: preds {p.shape} vs gts {g.shape}")
    scores = np.zeros(len(p), dtype=np.float64)
    for c in (0, 1):
        pc = p == c
        gc = g == c
        inter = np.logical_and(pc, gc).sum(axis=(1, 2), dtype=np.float64)
        union = np.logical_or(pc, gc).sum(axis=(1, 2), dtype=np.float64)
        iou_c = np.where(union == 0.0, 1.0, inter / np.maximum(union, 1.0))
        scores += iou_c
    return scores * 0.5


def load_train_ground_truth(
    data_root: str | Path = ROOT_DIR / "data" / "release",
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Load train metadata and ground-truth 128x128 masks indexed by image_name."""
    train_dir = Path(data_root) / "train"
    raw_meta = json.loads((train_dir / "metadata.json").read_text(encoding="utf-8"))
    meta_by_name: dict[str, dict[str, Any]] = {}
    for row in raw_meta:
        entry = dict(row)
        entry["kind"] = "full" if entry.get("full_watermark") else "patch"
        meta_by_name[entry["image_name"]] = entry

    with np.load(train_dir / "masks_128.npz", allow_pickle=False) as z:
        masks_by_name = {
            str(name): mk.astype(np.uint8)
            for name, mk in zip(z["names"].tolist(), z["masks"])
        }
    return meta_by_name, masks_by_name


def _threshold_for_item(
    prob: np.ndarray,
    method: str,
    threshold: float | dict[str, Any],
) -> float:
    if isinstance(threshold, (int, float)):
        return float(threshold)
    if isinstance(threshold, dict):
        entry = threshold.get(method, threshold.get("global", threshold.get("default", 0.5)))
        if isinstance(entry, dict):
            inferred_kind = "full" if float(prob.mean()) >= 0.5 else "patch"
            return float(entry.get(inferred_kind, entry.get("default", 0.5)))
        return float(entry)
    return 0.5


def binarize_probs(
    probs: np.ndarray,
    metas: Sequence[dict[str, Any]],
    threshold: float | dict[str, Any] = 0.5,
) -> np.ndarray:
    """Convert [N, 128, 128] probabilities into uint8 binary masks using scalar or per-method thresholds."""
    if isinstance(threshold, (int, float)):
        return (probs >= float(threshold)).astype(np.uint8)
    preds = np.empty(probs.shape, dtype=np.uint8)
    for i, meta in enumerate(metas):
        thr = _threshold_for_item(probs[i], meta["method"], threshold)
        preds[i] = (probs[i] >= thr).astype(np.uint8)
    return preds


def evaluate_predictions(
    names: Sequence[str],
    probs: np.ndarray,
    meta_by_name: dict[str, dict[str, Any]],
    masks_by_name: dict[str, np.ndarray],
    threshold: float | dict[str, Any] = 0.5,
    binarize_fn: Callable[[np.ndarray, Sequence[dict[str, Any]]], np.ndarray] | None = None,
    return_per_image: bool = False,
) -> dict[str, Any]:
    """Compute official mIoU overall and broken down by method and kind."""
    metas = [meta_by_name[n] for n in names]
    gts = np.stack([masks_by_name[n] for n in names], axis=0).astype(np.uint8)
    if binarize_fn is not None:
        preds = binarize_fn(probs, metas).astype(np.uint8)
    else:
        preds = binarize_probs(probs, metas, threshold)

    scores = miou_batch(preds, gts)
    methods_arr = np.array([m["method"] for m in metas])
    kinds_arr = np.array([m["kind"] for m in metas])

    metrics: dict[str, Any] = {
        "n_images": int(len(names)),
        "mean": round(float(scores.mean()), 6),
    }
    for method in METHODS:
        mask_m = methods_arr == method
        short = METHOD_SHORT[method]
        metrics[short] = round(float(scores[mask_m].mean()), 6) if mask_m.any() else None

    for kind in ("full", "patch"):
        mask_k = kinds_arr == kind
        metrics[kind] = round(float(scores[mask_k].mean()), 6) if mask_k.any() else None

    by_mk: dict[str, float | None] = {}
    for method in METHODS:
        for kind in ("full", "patch"):
            mask_mk = (methods_arr == method) & (kinds_arr == kind)
            key = f"{method}_{kind}"
            by_mk[key] = round(float(scores[mask_mk].mean()), 6) if mask_mk.any() else None
    metrics["by_method_kind"] = by_mk
    metrics["threshold"] = threshold

    if return_per_image:
        metrics["per_image_scores"] = {str(n): float(s) for n, s in zip(names, scores)}
    return metrics


def tune_thresholds_oof(
    names: Sequence[str],
    probs: np.ndarray,
    meta_by_name: dict[str, dict[str, Any]],
    masks_by_name: dict[str, np.ndarray],
    per_method: bool = True,
    grid: Sequence[float] | None = None,
) -> tuple[dict[str, float], float]:
    """Grid-search thresholds on out-of-fold predictions to maximize official mIoU."""
    candidates = np.asarray(grid if grid is not None else np.linspace(0.15, 0.85, 71), dtype=np.float32)
    metas = [meta_by_name[n] for n in names]
    gts = np.stack([masks_by_name[n] for n in names], axis=0).astype(np.uint8)
    methods_arr = np.array([m["method"] for m in metas])

    def best_thr_on_slice(sub_probs: np.ndarray, sub_gts: np.ndarray) -> tuple[float, float]:
        best_t = 0.5
        best_s = -1.0
        for t in candidates:
            s = float(miou_batch((sub_probs >= t).astype(np.uint8), sub_gts).mean())
            if s > best_s:
                best_s = s
                best_t = float(round(float(t), 4))
        return best_t, best_s

    global_thr, global_score = best_thr_on_slice(probs, gts)
    if not per_method:
        return {"global": global_thr}, float(global_score)

    best_dict: dict[str, float] = {}
    for method in METHODS:
        idx = methods_arr == method
        if idx.any():
            best_t, _ = best_thr_on_slice(probs[idx], gts[idx])
            best_dict[method] = best_t
        else:
            best_dict[method] = global_thr

    preds = binarize_probs(probs, metas, best_dict)
    oof_score = float(miou_batch(preds, gts).mean())
    return best_dict, oof_score


def write_metrics(out_dir: str | Path, metrics: dict[str, Any]) -> Path:
    """Persist metrics dictionary as formatted metrics.json inside out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "metrics.json"
    clean_metrics = {k: v for k, v in metrics.items() if k != "per_image_scores"}
    target.write_text(json.dumps(clean_metrics, indent=2) + "\n", encoding="utf-8")
    return target


def format_metrics_summary(metrics: dict[str, Any]) -> str:
    """Format a compact one-line breakdown string for terminal logging."""
    def fmt(val: Any) -> str:
        return "—" if val is None else f"{val:.4f}"

    return (
        f"mean={fmt(metrics.get('mean'))} | "
        f"PS={fmt(metrics.get('PS'))} TM={fmt(metrics.get('TM'))} SS={fmt(metrics.get('SS'))} | "
        f"full={fmt(metrics.get('full'))} patch={fmt(metrics.get('patch'))} "
        f"(n={metrics.get('n_images', '?')})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate 128x128 probability predictions.")
    ap.add_argument("--probs", required=True, help="Path to val_probs.npz")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"), help="Release data root")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tune-oof", action="store_true", help="Also tune per-method OOF thresholds")
    ap.add_argument("--out", default=None, help="Output JSON path (defaults to <probs_dir>/metrics.json)")
    args = ap.parse_args()

    names, probs = load_probs(args.probs)
    meta_by_name, masks_by_name = load_train_ground_truth(args.data)
    metrics = evaluate_predictions(names, probs, meta_by_name, masks_by_name, threshold=args.threshold)
    print("Raw (0.5):", format_metrics_summary(metrics))

    if args.tune_oof:
        thrs, _ = tune_thresholds_oof(names, probs, meta_by_name, masks_by_name, per_method=True)
        oof_metrics = evaluate_predictions(names, probs, meta_by_name, masks_by_name, threshold=thrs)
        metrics["oof_threshold"] = thrs
        metrics["oof_mean"] = oof_metrics["mean"]
        metrics["oof_metrics"] = {k: oof_metrics[k] for k in ("PS", "TM", "SS", "full", "patch", "by_method_kind")}
        print("OOF tuned:", format_metrics_summary(oof_metrics), "thrs=", thrs)

    out_path = Path(args.out) if args.out else Path(args.probs).parent / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
