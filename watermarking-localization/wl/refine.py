"""Post-hoc refinement of any run's probabilities with whole-image decoding and the kit decoders.

1. Kind gating. Decoding the whole image with the known message separates the two kinds almost
   perfectly (E01: bit accuracy ~0.97 on `full` images, ~0.53 on `patch`). A per-method threshold on
   the cached `bit_acc_full`, fit on the 510 train images, decides the kind. When a map disagrees with
   it, the map is re-ranked so its minority region gets the median size of that kind.
2. Patch box refinement (`--box`, needs the kit models and a GPU). The dataset embedded each patch at
   the scale of its own rectangle, so the exact rectangle decodes best (E01). For `patch` images, every
   edge of the main predicted component's bounding box is moved by coordinate ascent to maximize the
   decoder's signed logit agreement with the message on the crop (raw logits: the tanh margin
   saturates near the truth). Which rectangle-like components get snapped to the refined box, and how
   much the decoder score must rise, is chosen on the run's own validation images.

    python -m wl.refine --run output/E171_segb2_v2 --id E181_segb2_ref            # kind gating only, CPU
    python -m wl.refine --run output/E171_segb2_v2 --id E181_segb2_ref --box --device cuda

Writes output/<id>/{val_probs,test_probs}.npz, refine.json (choices + per-image boxes) and metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from wl import CELL, GRID, MAX_ASPECT, METHODS, NATIVE_RES, NBITS, ROOT_DIR
from wl.evaluate import (
    evaluate_predictions,
    format_metrics_summary,
    load_probs,
    load_train_ground_truth,
    miou_batch,
    save_probs,
    write_metrics,
)
from wl.features import FeatureCache, load_kit_models
from wl.postproc import connected_components, fit_postproc_oof

# Median minority-region size per kind on train (analysis/explore_dataset.py): holes 6 %, patches 4 %.
TYPICAL_FRAC = {"full": 0.06, "patch": 0.04}
FILL_GRID = (0.6, 0.7, 0.8, 0.9, 1.01)  # 1.01 = never snap
RHO_GRID = (0.5, 0.7, 0.85, 0.95, 1.0)  # accept a refined box if score0 <= rho * score


# ----------------------------------------------------------------------------- kind gating


def fit_kind_thresholds(cache: FeatureCache, meta_by_name: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-method threshold on whole-image bit accuracy that best separates full from patch on train."""
    out: dict[str, dict[str, float]] = {}
    for method in METHODS:
        accs, is_full = [], []
        for name, meta in meta_by_name.items():
            if meta["method"] != method:
                continue
            feats = cache.load("train", name)
            if feats is None:
                continue
            accs.append(float(feats["bit_acc_full"]))
            is_full.append(meta["kind"] == "full")
        if not accs:
            continue
        a = np.asarray(accs)
        y = np.asarray(is_full)
        cands = np.unique(a)
        mids = (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else cands
        best_t, best_acc = 0.75, -1.0
        for t in mids:
            acc = float(((a >= t) == y).mean())
            if acc > best_acc:
                best_t, best_acc = float(t), acc
        out[method] = {"threshold": round(best_t, 4), "train_acc": round(best_acc, 4), "n": int(len(a))}
    return out


def decide_kinds(
    names: Sequence[str],
    methods: Sequence[str],
    split: str,
    probs: np.ndarray,
    cache: FeatureCache,
    thresholds: dict[str, dict[str, float]],
) -> np.ndarray:
    """True = `full`. Uses the cached whole-image decode; falls back to the map for uncached images."""
    kinds = np.empty(len(names), dtype=bool)
    for i, (n, m) in enumerate(zip(names, methods)):
        feats = cache.load(split, n)
        if feats is not None and m in thresholds:
            kinds[i] = float(feats["bit_acc_full"]) >= thresholds[m]["threshold"]
        else:
            kinds[i] = float((probs[i] >= 0.5).mean()) >= 0.5
    return kinds


def gate_kind(prob: np.ndarray, is_full: bool, tau: float = 0.02) -> np.ndarray:
    """Return `prob` unchanged if its kind matches, else a re-ranked map with a typical minority size."""
    if (float((prob >= 0.5).mean()) >= 0.5) == is_full:
        return prob
    flat = prob.reshape(-1)
    k = max(1, int(TYPICAL_FRAC["full" if is_full else "patch"] * flat.size))
    # full: the k lowest cells become holes; patch: the k highest cells become the patch.
    q = np.partition(flat, k)[k] if is_full else np.partition(flat, flat.size - k)[flat.size - k]
    z = np.clip((prob - q) / tau, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


# ----------------------------------------------------------------------------- box refinement


class BoxScorer:
    """Decoder agreement with the known message on rectangle crops, resized to the method's resolution."""

    def __init__(self, wm_model: Any, method: str, img_uint8: np.ndarray, bits: np.ndarray, device: str, batch: int = 64):
        import torch

        self.torch = torch
        self.batch = int(batch)
        self.wm = wm_model
        self.res = int(NATIVE_RES[method])
        self.mode = "bicubic" if method == "stegastamp" else "bilinear"
        self.img = torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1)[None].float().div(255.0).to(device)
        nb = NBITS[method]
        self.signed = torch.from_numpy(2.0 * np.asarray(bits[:nb], dtype=np.float32) - 1.0)
        self.cache: dict[tuple[int, int, int, int], float] = {}

    def __call__(self, boxes: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
        torch = self.torch
        todo = [b for b in dict.fromkeys(boxes) if b not in self.cache]
        for i in range(0, len(todo), self.batch):
            chunk = todo[i : i + self.batch]
            crops = [
                torch.nn.functional.interpolate(
                    self.img[:, :, y0 * CELL : y1 * CELL, x0 * CELL : x1 * CELL],
                    size=(self.res, self.res), mode=self.mode, align_corners=False, antialias=True,
                ).clamp(0.0, 1.0)
                for (y0, y1, x0, x1) in chunk
            ]
            with torch.no_grad():
                logits = self.wm.logits(torch.cat(crops, 0)).detach().float().cpu()
            # Raw logits: the tanh margin saturates near 1 for every box close to the truth.
            margins = (logits * self.signed[None, :]).mean(dim=1).numpy()
            self.cache.update(zip(chunk, margins.tolist()))
        return np.asarray([self.cache[b] for b in boxes], dtype=np.float64)


def _valid_box(b: tuple[int, int, int, int], max_aspect: float) -> bool:
    y0, y1, x0, x1 = b
    if y0 < 0 or x0 < 0 or y1 > GRID or x1 > GRID:
        return False
    h, w = y1 - y0, x1 - x0
    if h < 2 or w < 2:
        return False
    return max(h, w) / min(h, w) <= max_aspect


def refine_box(
    scorer: BoxScorer,
    box0: tuple[int, int, int, int],
    max_aspect: float,
    rounds: int = 5,
    deltas: Sequence[int] = (-6, -3, -2, -1, 1, 2, 3, 6),
) -> tuple[tuple[int, int, int, int], float, float]:
    """Coordinate ascent over the 4 edges plus whole-box shifts. Returns (box, score, initial score)."""
    best = box0
    s0 = float(scorer([box0])[0])
    best_s = s0
    for _ in range(rounds):
        cands: list[tuple[int, int, int, int]] = []
        for e in range(4):
            for d in deltas:
                b = list(best)
                b[e] += d
                cands.append(tuple(b))
        for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, 0), (1, 0), (0, -1), (0, 1)):
            y0, y1, x0, x1 = best
            cands.append((y0 + dy, y1 + dy, x0 + dx, x1 + dx))
        cands = [c for c in cands if _valid_box(c, max_aspect)]
        if not cands:
            break
        s = scorer(cands)
        i = int(np.argmax(s))
        if s[i] <= best_s + 1e-4:
            break
        best, best_s = cands[i], float(s[i])
    return best, best_s, s0


def main_component(prob: np.ndarray, thr: float = 0.5) -> tuple[tuple[int, int, int, int], float] | None:
    """Bounding box (cells) and fill ratio of the largest predicted watermarked component."""
    labels, n = connected_components((prob >= thr).astype(np.uint8))
    if n == 0:
        return None
    counts = np.bincount(labels.reshape(-1))[1:]
    comp = labels == (int(np.argmax(counts)) + 1)
    ys, xs = np.nonzero(comp)
    box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    fill = float(comp.sum()) / float((box[1] - box[0]) * (box[3] - box[2]))
    return box, fill


def apply_box(prob: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.full_like(prob, 0.03)
    y0, y1, x0, x1 = box
    out[y0:y1, x0:x1] = 0.97
    return out


def compute_boxes(
    split: str,
    names: Sequence[str],
    probs: np.ndarray,
    is_full: np.ndarray,
    meta_rows: dict[str, dict[str, Any]],
    data_root: Path,
    wm_models: dict[str, Any],
    device: str,
    batch: int = 64,
) -> dict[str, dict[str, Any]]:
    """Refined box per `patch` image: {name: {box0, box, score0, score, fill}}."""
    out: dict[str, dict[str, Any]] = {}
    for i, n in enumerate(names):
        if is_full[i]:
            continue
        comp = main_component(probs[i])
        if comp is None:
            continue
        box0, fill = comp
        meta = meta_rows[n]
        method = meta["method"]
        img = np.asarray(Image.open(data_root / split / "images" / f"{n}.png").convert("RGB"), dtype=np.uint8)
        bits = np.array([int(c) for c in meta["message"]], dtype=np.uint8)
        scorer = BoxScorer(wm_models[method], method, img, bits, device, batch=batch)
        box, s, s0 = refine_box(scorer, box0, MAX_ASPECT[method])
        out[n] = {"box0": list(box0), "box": list(box), "score0": round(s0, 4), "score": round(s, 4), "fill": round(fill, 4)}
    return out


def apply_refinement(
    names: Sequence[str],
    probs: np.ndarray,
    boxes: dict[str, dict[str, Any]],
    fill_thr: float,
    rho: float,
) -> np.ndarray:
    """Snap rectangle-like components to their refined box when refinement raised the decoder score
    clearly (score0 <= rho * score). Near the true box the decoder tolerates a slightly smaller crop,
    so a box that already decodes well is kept as predicted."""
    out = probs.copy()
    for i, n in enumerate(names):
        info = boxes.get(n)
        if info is None or info["fill"] < fill_thr or info["score"] <= 0.0:
            continue
        if info["score0"] <= rho * info["score"]:
            out[i] = apply_box(probs[i], tuple(info["box"]))
    return out


# ----------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description="Kind gating + decoder box refinement for any run.")
    ap.add_argument("--run", required=True, help="Run dir with val_probs.npz (train images) and test_probs.npz")
    ap.add_argument("--id", required=True, help="Output run ID under output/")
    ap.add_argument("--box", action="store_true", help="Also refine patch rectangles with the kit decoders")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64, help="Crops per decoder call (lower it on CPU)")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--cache", default=str(ROOT_DIR / "output" / "cache" / "features"))
    ap.add_argument("--out", default=str(ROOT_DIR / "output"))
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out) / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data)
    cache = FeatureCache(args.cache)
    meta_by_name, masks_by_name = load_train_ground_truth(data_root)
    test_meta = {r["image_name"]: r for r in json.loads((data_root / "test" / "metadata.json").read_text(encoding="utf-8"))}

    thresholds = fit_kind_thresholds(cache, meta_by_name)
    print("kind thresholds on bit_acc_full (train):", json.dumps(thresholds))

    v_names, v_probs = load_probs(run_dir / "val_probs.npz")
    v_methods = [meta_by_name[n]["method"] for n in v_names]
    v_full = decide_kinds(v_names, v_methods, "train", v_probs, cache, thresholds)
    gt_full = np.array([meta_by_name[n]["kind"] == "full" for n in v_names])
    print(f"val kind accuracy (decode): {float((v_full == gt_full).mean()):.4f} | "
          f"map kind accuracy: {float((((v_probs >= 0.5).mean(axis=(1, 2)) >= 0.5) == gt_full).mean()):.4f}")
    gts = np.stack([masks_by_name[n] for n in v_names]).astype(np.uint8)

    def score(p: np.ndarray) -> float:
        return float(miou_batch((p >= 0.5).astype(np.uint8), gts).mean())

    v_gated = np.stack([gate_kind(p, bool(f)) for p, f in zip(v_probs, v_full)])
    report: dict[str, Any] = {"run": str(run_dir), "kind_thresholds": thresholds,
                              "val_raw": round(score(v_probs), 4), "val_gated": round(score(v_gated), 4)}
    print(f"val mIoU @0.5: raw {report['val_raw']} -> kind-gated {report['val_gated']}")

    t_names, t_probs = load_probs(run_dir / "test_probs.npz")
    t_methods = [test_meta[n]["method"] for n in t_names]
    t_full = decide_kinds(t_names, t_methods, "test", t_probs, cache, thresholds)
    t_gated = np.stack([gate_kind(p, bool(f)) for p, f in zip(t_probs, t_full)])
    v_final, t_final = v_gated, t_gated

    if args.box:
        methods_used = sorted(set(v_methods) | set(t_methods))
        wm_models = load_kit_models(methods_used, device=args.device)
        v_boxes = compute_boxes("train", v_names, v_gated, v_full, meta_by_name, data_root, wm_models, args.device, args.batch)
        best = (1.01, 0.0, score(v_gated))
        for fill_thr in FILL_GRID:
            for rho in RHO_GRID:
                s = score(apply_refinement(v_names, v_gated, v_boxes, fill_thr, rho))
                if s > best[2] + 1e-6:
                    best = (fill_thr, rho, s)
        fill_thr, rho, s_best = best
        print(f"box refinement on val: fill>={fill_thr} rho={rho} -> {s_best:.4f} "
              f"({len(v_boxes)} patch images searched)")
        report.update({"box_fill_thr": fill_thr, "box_rho": rho, "val_box": round(s_best, 4), "val_boxes": v_boxes})
        v_final = apply_refinement(v_names, v_gated, v_boxes, fill_thr, rho)
        if fill_thr <= 1.0:
            t_boxes = compute_boxes("test", t_names, t_gated, t_full, test_meta, data_root, wm_models, args.device, args.batch)
            t_final = apply_refinement(t_names, t_gated, t_boxes, fill_thr, rho)
            report["test_boxes"] = t_boxes
            print(f"test: {len(t_boxes)} patch images refined")

    (out_dir / "refine.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    save_probs(out_dir / "val_probs.npz", v_names, v_final)
    save_probs(out_dir / "test_probs.npz", t_names, t_final)

    metrics = evaluate_predictions(v_names, v_final, meta_by_name, masks_by_name, threshold=0.5)
    oof = fit_postproc_oof(v_names, v_final, meta_by_name, masks_by_name)
    metrics.update({"id": args.id, "parent": run_dir.name, "data": "refine",
                    "val": f"{len(v_names)} imgs", "oof_post_cfg": oof["post_cfg"],
                    "oof_mean": oof["oof_score"], "oof_metrics": oof["metrics"]})
    write_metrics(out_dir, metrics)
    print(f"[{args.id}] {format_metrics_summary(metrics)} | OOF post-proc {oof['oof_score']:.4f}")


if __name__ == "__main__":
    main()
