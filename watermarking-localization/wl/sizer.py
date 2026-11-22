"""Size-calibrated Bayes decision: keep the model's cell ranking, predict how big the small region is.

`wl.oracle_k` on E151: same ranking with the true region size gives 0.687 vs 0.653 (eiou), so part of
the loss is the *size* of the predicted hole / patch, not where it is. Here the size of the small
region (holes for `full`, the patch for `patch`) is regressed, in log space, from per-image statistics
that do not use the ground truth:
  * log size of the expected-IoU decision and of the 0.5 threshold,
  * log of the total probability mass of the small class,
  * whole-image decoder bit accuracy and margin (more unmarked area -> lower global agreement).
One ridge regression per kind (method as one-hot), cross-fitted over 5 groups for the honest score;
the final mask is the top-k of the ranking with k from the prediction (geometric mean with the eiou k,
a shrinkage towards the model's own decision).

    python -m wl.sizer output/E151_verify_e143 --kind output/E120_kind_const/kind_probs.npz [--submit]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wl import GRID, METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch, save_probs
from wl.honest_eval import eiou_decide

N = GRID * GRID


def small_count(mask: np.ndarray, is_full: bool) -> int:
    return int(N - mask.sum()) if is_full else int(mask.sum())


def features(p: np.ndarray, is_full: bool, method: str, gfeat: tuple[float, float]) -> np.ndarray:
    e = eiou_decide(p)
    s_e = small_count(e, is_full)
    s_05 = small_count((p >= 0.5).astype(np.uint8), is_full)
    mass = float((1 - p).sum() if is_full else p.sum())
    onehot = [1.0 if method == m else 0.0 for m in METHODS]
    return np.array([np.log1p(s_e), np.log1p(s_05), np.log1p(mass), gfeat[0], gfeat[1], *onehot], dtype=np.float64)


def ridge_fit(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = np.c_[(x - mu) / sd, np.ones(len(x))]
    reg = l2 * np.eye(xs.shape[1]); reg[-1, -1] = 0
    w = np.linalg.solve(xs.T @ xs + reg, xs.T @ y)
    return w, mu, sd


def ridge_pred(model, x: np.ndarray) -> np.ndarray:
    w, mu, sd = model
    return np.c_[(x - mu) / sd, np.ones(len(x))] @ w


def topk_mask(p: np.ndarray, small: int, is_full: bool) -> np.ndarray:
    k = N - small if is_full else small
    k = int(np.clip(k, 1, N - 1))
    order = np.argsort(-p.reshape(-1))
    m = np.zeros(N, dtype=np.uint8); m[order[:k]] = 1
    return m.reshape(GRID, GRID)


def decide(p, is_full, pred_log_small):
    s_e = small_count(eiou_decide(p), is_full)
    s = int(round(np.sqrt(max(s_e, 1) * max(np.expm1(pred_log_small), 1))))  # shrink towards eiou
    return topk_mask(p, s, is_full)


def global_feats(names, split, cache):
    from wl.features import FeatureCache
    fc = FeatureCache(cache)
    out = []
    for n in names:
        f = fc.load(split, n)
        out.append((f["bit_acc_full"], f["margin_full"]) if f else (0.5, 0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--kind", required=True)
    ap.add_argument("--cache", default=str(ROOT_DIR / "output" / "cache" / "features"))
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    meta, masks = load_train_ground_truth(args.data)
    kz = np.load(args.kind)
    kind_tr = {str(n): float(p) for n, p in zip(kz["train_names"], kz["train_p_full_oof"])}
    kind_te = {str(n): float(p) for n, p in zip(kz["test_names"], kz["test_p_full"])}

    names, probs = load_probs(run / "val_probs.npz"); names = [str(n) for n in names]
    probs = probs.astype(np.float32)
    gf = global_feats(names, "train", args.cache)
    isf = np.array([kind_tr[n] >= 0.5 for n in names])
    meth = np.array([meta[n]["method"] for n in names])
    gts = np.stack([masks[n] for n in names]).astype(np.uint8)
    x = np.stack([features(p, f, m, g) for p, f, m, g in zip(probs, isf, meth, gf)])
    y = np.array([np.log1p(small_count(g, f)) for g, f in zip(gts, isf)])

    rng = np.random.default_rng(0)
    group = np.zeros(len(names), dtype=int)
    for m in METHODS:
        idx = np.flatnonzero(meth == m); group[rng.permutation(idx)] = np.arange(len(idx)) % 5
    base = np.array([miou_batch(eiou_decide(p)[None], g[None])[0] for p, g in zip(probs, gts)])
    honest = np.zeros(len(names)); val_masks = np.zeros_like(gts)
    for g in range(5):
        for kd in (True, False):
            tr = (group != g) & (isf == kd); va = (group == g) & (isf == kd)
            if va.sum() == 0:
                continue
            model = ridge_fit(x[tr], y[tr])
            for i, pl in zip(np.flatnonzero(va), ridge_pred(model, x[va])):
                val_masks[i] = decide(probs[i], isf[i], pl)
                honest[i] = miou_batch(val_masks[i][None], gts[i][None])[0]
    gm = [honest[group == g].mean() - base[group == g].mean() for g in range(5)]
    print(f"[{run.name}] eiou {base.mean():.4f} -> sizer (cross-fitted) {honest.mean():.4f}  "
          f"gain per group {' '.join(f'{v:+.3f}' for v in gm)}")
    kinds = np.array([meta[n]["kind"] for n in names])
    for m in METHODS:
        for kd in ("full", "patch"):
            s = (meth == m) & (kinds == kd)
            print(f"   {METHOD_SHORT[m]} {kd:5s} {base[s].mean():.4f} -> {honest[s].mean():.4f}")

    out_dir = ROOT_DIR / "output" / f"{run.name}_sizer"
    save_probs(out_dir / "val_probs.npz", names, val_masks.astype(np.float32))
    t_names, t_probs = load_probs(run / "test_probs.npz"); t_names = [str(n) for n in t_names]
    tgf = global_feats(t_names, "test", args.cache)
    test_meta = {r["image_name"]: r for r in json.loads((Path(args.data) / "test" / "metadata.json").read_text())}
    t_isf = np.array([kind_te[n] >= 0.5 for n in t_names])
    tx = np.stack([features(p.astype(np.float32), f, test_meta[n]["method"], g) for p, f, n, g in zip(t_probs, t_isf, t_names, tgf)])
    t_masks = np.zeros((len(t_names), GRID, GRID), dtype=np.uint8)
    for kd in (True, False):
        model = ridge_fit(x[isf == kd], y[isf == kd])
        sel = np.flatnonzero(t_isf == kd)
        for i, pl in zip(sel, ridge_pred(model, tx[sel])):
            t_masks[i] = decide(t_probs[i].astype(np.float32), t_isf[i], pl)
    save_probs(out_dir / "test_probs.npz", t_names, t_masks.astype(np.float32))
    (out_dir / "sizer.json").write_text(json.dumps({"eiou": round(float(base.mean()), 4), "honest": round(float(honest.mean()), 4),
                                                    "group_gain": [round(float(v), 4) for v in gm]}, indent=2) + "\n")
    print(f"Wrote {out_dir} (val masks are cross-fitted)")
    if args.submit:
        import torch
        from predict import submit, validate_submission
        order = [r for r in test_meta]
        pos = {n: i for i, n in enumerate(t_names)}
        out = out_dir / "submission.pt"
        torch.save({"image_names": order, "masks": torch.from_numpy(t_masks[[pos[n] for n in order]]).bool()}, out)
        validate_submission(out, order)
        submit(out)


if __name__ == "__main__":
    main()
