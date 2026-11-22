"""Post-processing for 128x128 watermark probabilities and OOF parameter tuning."""
from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import numpy as np

from wl import GRID, MAX_ASPECT, METHODS
from wl.evaluate import evaluate_predictions, miou_batch, tune_thresholds_oof

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:
    _scipy_ndimage = None


def connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected components of a boolean/uint8 2D mask."""
    mask = np.asarray(binary_mask, dtype=bool)
    if _scipy_ndimage is not None:
        labels, n = _scipy_ndimage.label(mask)
        return labels, int(n)

    # Pure-NumPy BFS fallback keeps post-processing functional even without scipy.
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if labels[sy, sx] != 0:
            continue
        cur += 1
        labels[sy, sx] = cur
        q = deque([(sy, sx)])
        while q:
            y, x = q.popleft()
            if y > 0 and mask[y - 1, x] and labels[y - 1, x] == 0:
                labels[y - 1, x] = cur
                q.append((y - 1, x))
            if y + 1 < h and mask[y + 1, x] and labels[y + 1, x] == 0:
                labels[y + 1, x] = cur
                q.append((y + 1, x))
            if x > 0 and mask[y, x - 1] and labels[y, x - 1] == 0:
                labels[y, x - 1] = cur
                q.append((y, x - 1))
            if x + 1 < w and mask[y, x + 1] and labels[y, x + 1] == 0:
                labels[y, x + 1] = cur
                q.append((y, x + 1))
    return labels, cur


def infer_kind_from_probs(prob128: np.ndarray, kind_prob: float | None = None) -> str:
    """Infer whether an image is `full` (clean holes) or `patch` (small watermarked patch)."""
    score = float(kind_prob) if kind_prob is not None else float(np.mean(prob128))
    return "full" if score >= 0.5 else "patch"


def _resolve_threshold(prob128: np.ndarray, method: str, cfg_post: dict[str, Any], kind_hint: str) -> float:
    thrs = cfg_post.get("thresholds")
    if isinstance(thrs, dict):
        entry = thrs.get(method, thrs.get("global", 0.5))
        if isinstance(entry, dict):
            return float(entry.get(kind_hint, entry.get("default", 0.5)))
        return float(entry)
    thr_raw = cfg_post.get("threshold", 0.5)
    if isinstance(thr_raw, (int, float)):
        return float(thr_raw)
    if isinstance(thr_raw, dict):
        entry = thr_raw.get(method, thr_raw.get("global", 0.5))
        if isinstance(entry, dict):
            return float(entry.get(kind_hint, entry.get("default", 0.5)))
        return float(entry)
    return 0.5


def postprocess_single(
    prob128: np.ndarray,
    method: str,
    cfg_post: dict[str, Any] | None = None,
    kind_prob: float | None = None,
) -> np.ndarray:
    """Apply thresholding, never-empty protection, speck removal, and optional rectangle snap."""
    cfg = cfg_post or {}
    p = np.asarray(prob128, dtype=np.float32)
    kind_hint = infer_kind_from_probs(p, kind_prob)
    thr = _resolve_threshold(p, method, cfg, kind_hint)

    mask = (p >= thr).astype(np.uint8)
    if kind_prob is not None:
        kind = "full" if float(kind_prob) >= 0.5 else "patch"
    else:
        kind = "full" if float(mask.mean()) >= 0.5 else "patch"

    # Minority region is the target that determines half of every image's official score.
    target = (mask == 0) if kind == "full" else (mask == 1)

    never_empty = bool(cfg.get("never_empty", True))
    min_comp = int(cfg.get("min_component_cells", 8))
    if min_comp > 1 and int(target.sum()) > 0:
        labels, n_comp = connected_components(target)
        if n_comp > 1:
            counts = np.bincount(labels.reshape(-1))[1:]
            keep = counts >= min_comp
            if not keep.any() and never_empty:
                keep[int(np.argmax(counts))] = True
            filtered = np.zeros_like(target)
            for comp_id, is_kept in enumerate(keep, start=1):
                if is_kept:
                    filtered |= labels == comp_id
            target = filtered

    min_cells = int(cfg.get("min_cells", 24))
    if never_empty and min_cells > 0 and int(target.sum()) < min_cells:
        affinity = (1.0 - p) if kind == "full" else p
        flat = affinity.reshape(-1)
        k = min(min_cells, flat.size)
        top_idx = np.argpartition(flat, -k)[-k:]
        target_flat = np.array(target, dtype=bool, copy=True).reshape(-1)
        target_flat[top_idx] = True
        target = target_flat.reshape(GRID, GRID)

    if bool(cfg.get("rect_snap", False)) and int(target.sum()) > 0:
        patch_only = bool(cfg.get("rect_snap_patch_only", True))
        if (not patch_only) or kind == "patch":
            labels, n_comp = connected_components(target)
            if n_comp == 1:
                ys, xs = np.nonzero(target)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                h, w = y1 - y0, x1 - x0
                fill = float(target.sum()) / float(h * w)
                aspect = float(max(h, w)) / float(max(1, min(h, w)))
                fill_thr = float(cfg.get("rect_snap_fill", 0.85))
                if fill >= fill_thr and aspect <= MAX_ASPECT.get(method, 16.0):
                    target = np.zeros_like(target)
                    target[y0:y1, x0:x1] = True

    return (1 - target.astype(np.uint8)) if kind == "full" else target.astype(np.uint8)


def apply_postproc(
    probs: np.ndarray,
    metas: Sequence[dict[str, Any]],
    cfg_post: dict[str, Any] | None = None,
    kind_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Apply postprocess_single to a batch of [N, 128, 128] probability maps."""
    out = np.empty(probs.shape, dtype=np.uint8)
    for i, meta in enumerate(metas):
        kp = None if kind_probs is None else float(kind_probs[i])
        out[i] = postprocess_single(probs[i], meta["method"], cfg_post, kind_prob=kp)
    return out


def fit_postproc_oof(
    names: Sequence[str],
    probs: np.ndarray,
    meta_by_name: dict[str, dict[str, Any]],
    masks_by_name: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Coordinate-search thresholds and cleanup hyperparameters on OOF predictions."""
    metas = [meta_by_name[n] for n in names]
    gts = np.stack([masks_by_name[n] for n in names], axis=0).astype(np.uint8)
    raw_score = float(miou_batch((probs >= 0.5).astype(np.uint8), gts).mean())

    best_thrs, _ = tune_thresholds_oof(names, probs, meta_by_name, masks_by_name, per_method=True)
    best_cfg: dict[str, Any] = {
        "thresholds": best_thrs,
        "never_empty": True,
        "min_cells": 24,
        "min_component_cells": 8,
        "rect_snap": False,
        "rect_snap_fill": 0.85,
        "rect_snap_patch_only": True,
    }
    best_score = float(miou_batch(apply_postproc(probs, metas, best_cfg), gts).mean())

    for mc in (0, 16, 24, 36):
        cand = dict(best_cfg, never_empty=mc > 0, min_cells=mc)
        s = float(miou_batch(apply_postproc(probs, metas, cand), gts).mean())
        if s > best_score:
            best_score, best_cfg = s, cand

    for min_comp in (0, 4, 8):
        cand = dict(best_cfg, min_component_cells=min_comp)
        s = float(miou_batch(apply_postproc(probs, metas, cand), gts).mean())
        if s > best_score:
            best_score, best_cfg = s, cand

    for snap, fill in ((False, 0.85), (True, 0.82), (True, 0.88), (True, 0.92)):
        cand = dict(best_cfg, rect_snap=snap, rect_snap_fill=fill)
        s = float(miou_batch(apply_postproc(probs, metas, cand), gts).mean())
        if s > best_score:
            best_score, best_cfg = s, cand

    oof_metrics = evaluate_predictions(
        names,
        probs,
        meta_by_name,
        masks_by_name,
        threshold=best_thrs,
        binarize_fn=lambda pr, ms: apply_postproc(pr, ms, best_cfg),
    )
    return {
        "post_cfg": best_cfg,
        "raw_score": round(raw_score, 6),
        "oof_score": round(best_score, 6),
        "metrics": oof_metrics,
    }
