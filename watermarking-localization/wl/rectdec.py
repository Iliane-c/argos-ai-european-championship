"""Bayes decision over rectangles for patch images (no fitted parameter).

`wl.shapes`: patches are single components (80 %) with a median bounding-box fill of 0.97-0.98, i.e. near
rectangles (TrustMark crops a box with aspect <= 2). The expected-IoU decision of `wl.honest_eval` picks
cells independently; restricting the mask to an axis-aligned rectangle injects this prior. For every image
predicted `patch` (kind classifier), the rectangle maximising the expected per-image mIoU (Nowozin's ratio
approximation, both classes) is found by exhaustive search on a stride-2 grid with integral images, then
refined at stride 1. `full` images are left unchanged (holes are irregular: fill ~0.45).

    python -m wl.rectdec --run output/E198g_final_gate --kind output/E120_kind_const/kind_probs.npz --id E206_rect
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wl import GRID, MAX_ASPECT, METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import evaluate_predictions, load_probs, load_train_ground_truth, miou_batch, save_probs, write_metrics
from wl.honest_eval import eiou_decide

N = GRID * GRID


def _score(s, k, total):
    i1, u1 = s, k + (total - s)
    i0, u0 = (N - k) - (total - s), (N - k) + (k - s)
    return 0.5 * (i1 / np.maximum(u1, 1e-9) + i0 / np.maximum(u0, 1e-9))


def best_rectangle(p: np.ndarray, max_aspect: float, max_area: float = 0.5) -> tuple[tuple[int, int, int, int], float]:
    P = np.zeros((GRID + 1, GRID + 1)); P[1:, 1:] = p.cumsum(0).cumsum(1)
    total = float(p.sum())

    def rect_sum(y0, y1, x0, x1):
        return P[y1, x1] - P[y0, x1] - P[y1, x0] + P[y0, x0]

    coords = np.arange(0, GRID + 1, 2)
    best, best_box = -1.0, (0, 2, 0, 2)
    for a, y0 in enumerate(coords):
        for y1 in coords[a + 1:]:
            h = y1 - y0
            row = P[y1, coords] - P[y0, coords]                      # sums over x in [0, x)
            s = row[None, :] - row[:, None]                           # [x0, x1]
            w = coords[None, :] - coords[:, None]
            k = (h * w).astype(float)
            ok = (w > 0) & (k <= max_area * N) & (np.maximum(h, w) <= max_aspect * np.minimum(h, np.maximum(w, 1)))
            sc = np.where(ok, _score(s, k, total), -1.0)
            i = int(np.argmax(sc))
            if sc.flat[i] > best:
                best, best_box = float(sc.flat[i]), (int(y0), int(y1), int(coords[i // len(coords)]), int(coords[i % len(coords)]))
    y0, y1, x0, x1 = best_box
    for _ in range(2):                                               # stride-1 refinement
        for dy0 in range(-2, 3):
            for dy1 in range(-2, 3):
                for dx0 in range(-2, 3):
                    for dx1 in range(-2, 3):
                        b = (y0 + dy0, y1 + dy1, x0 + dx0, x1 + dx1)
                        if not (0 <= b[0] < b[1] <= GRID and 0 <= b[2] < b[3] <= GRID):
                            continue
                        h, w = b[1] - b[0], b[3] - b[2]
                        k = h * w
                        if k > max_area * N or max(h, w) > max_aspect * min(h, w):
                            continue
                        sc = _score(rect_sum(*b), k, total)
                        if sc > best:
                            best, best_box = float(sc), b
        y0, y1, x0, x1 = best_box
    return best_box, best


def apply(probs, names, methods, p_full):
    out = probs.copy()
    for i, (n, m) in enumerate(zip(names, methods)):
        if p_full.get(n, 1.0) >= 0.5:
            continue
        (y0, y1, x0, x1), _ = best_rectangle(probs[i].astype(np.float64), MAX_ASPECT[m])
        r = np.zeros((GRID, GRID), dtype=np.float32); r[y0:y1, x0:x1] = 1.0
        out[i] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    args = ap.parse_args()
    meta, masks = load_train_ground_truth(args.data)
    kz = np.load(args.kind)
    kind = {**{str(n): float(p) for n, p in zip(kz["train_names"], kz["train_p_full_oof"])},
            **{str(n): float(p) for n, p in zip(kz["test_names"], kz["test_p_full"])}}
    run = Path(args.run)
    v_names, v_probs = load_probs(run / "val_probs.npz"); v_names = [str(n) for n in v_names]
    v_meth = [meta[n]["method"] for n in v_names]
    gts = np.stack([masks[n] for n in v_names]).astype(np.uint8)
    base = np.array([miou_batch(eiou_decide(p)[None], g[None])[0] for p, g in zip(v_probs.astype(np.float32), gts)])
    v_new = apply(v_probs.astype(np.float32), v_names, v_meth, kind)
    new = np.array([miou_batch((p >= 0.5).astype(np.uint8)[None], g[None])[0] for p, g in zip(v_new, gts)])
    kinds = np.array([meta[n]["kind"] for n in v_names]); vm = np.array(v_meth)
    print(f"[{args.id}] eiou on {run.name}: {base.mean():.4f} -> rectangle prior on predicted patches: {new.mean():.4f}")
    for m in METHODS:
        for kd in ("full", "patch"):
            s = (vm == m) & (kinds == kd)
            print(f"   {METHOD_SHORT[m]} {kd:5s} {base[s].mean():.4f} -> {new[s].mean():.4f}")
    d = new - base
    print(f"   per-image: wins {int((d > 1e-6).sum())}, losses {int((d < -1e-6).sum())}")
    out = ROOT_DIR / "output" / args.id
    save_probs(out / "val_probs.npz", v_names, v_new)
    metrics = evaluate_predictions(v_names, v_new, meta, masks, threshold=0.5)
    metrics.update({"id": args.id, "parent": run.name, "val": "f0", "data": "rect"})
    write_metrics(out, metrics)
    t_names, t_probs = load_probs(run / "test_probs.npz"); t_names = [str(n) for n in t_names]
    tmeta = {r["image_name"]: r for r in json.loads((Path(args.data) / "test" / "metadata.json").read_text())}
    save_probs(out / "test_probs.npz", t_names, apply(t_probs.astype(np.float32), t_names, [tmeta[n]["method"] for n in t_names], kind))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
