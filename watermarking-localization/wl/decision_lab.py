"""Decision theory lab on fixed cell beliefs (CPU). All rules are parameter-free; f0 scores are honest.

The expected-IoU decision (Nowozin's ratio approximation E[I]/E[U]) gave +0.007 public. Two principled refinements:
  * exact: the expected per-image mIoU (both classes) of every top-k mask, estimated by Monte Carlo from the
    independent-Bernoulli beliefs (S samples, common random numbers); the ratio approximation is poorest exactly
    for small regions, which decide the score here;
  * kind: the kind classifier (OOF accuracy 0.978) says whether the image is `full` (majority watermarked) or
    `patch`; the top-k is restricted to the consistent half (k >= N/2 for full, k <= N/2 for patch).

  * como / mix (spatially coherent truth): independent Bernoulli cells are the wrong uncertainty model here, since
    errors are coherent (a whole hole or patch is found or missed). The comonotone model draws ONE level u ~ U(0,1)
    per image and sets truth = {p > u}: no parameter, it is the maximal-correlation coupling with the same cell
    marginals (Frechet upper bound). Expected mIoU of top-k is then exact by quadrature over u. `mix` averages the
    independent and comonotone curves (the two extreme couplings, equal weight by symmetry, no fitting).

    python -m wl.decision_lab output/E198g_final_gate --kind output/E120_kind_const/kind_probs.npz [--write exact|kind|exact_kind]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wl import GRID, METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch

N = GRID * GRID


def topk_curves(p: np.ndarray, samples: int, rng: np.random.Generator):
    """Ratio-approximation and Monte-Carlo expected mIoU for every k (top-k by belief)."""
    q = np.clip(p.reshape(-1).astype(np.float64), 0, 1)
    order = np.argsort(-q)
    qs = q[order]
    k = np.arange(N + 1)
    tot = qs.sum()
    s_k = np.concatenate([[0.0], np.cumsum(qs)])
    i1, u1 = s_k, k + (tot - s_k)
    i0, u0 = (N - k) - (tot - s_k), (N - k) + (k - s_k)
    ratio = 0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1e-9), 1) + np.where(u0 > 0, i0 / np.maximum(u0, 1e-9), 1))
    y = (rng.random((samples, N)) < qs[None, :]).astype(np.float64)      # sampled truths, in belief order
    tp = np.concatenate([np.zeros((samples, 1)), np.cumsum(y, 1)], 1)    # [S, N+1]
    pos = y.sum(1, keepdims=True)
    fp, fn = k[None, :] - tp, pos - tp
    tn = N - k[None, :] - fn
    iou1 = np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1e-9), 1)
    iou0 = np.where(tn + fp + fn > 0, tn / np.maximum(tn + fp + fn, 1e-9), 1)
    exact = (0.5 * (iou1 + iou0)).mean(0)
    return order, ratio, exact


def como_curve(p: np.ndarray, levels: int = 512) -> np.ndarray:
    """Expected mIoU of every top-k mask when truth = {p > u}, u ~ U(0,1) (comonotone coupling)."""
    qs = np.sort(np.clip(p.reshape(-1).astype(np.float64), 0, 1))       # ascending
    u = (np.arange(levels) + 0.5) / levels
    m = (N - np.searchsorted(qs, u, side="right")).astype(np.float64)    # |{p > u}|, nested in belief order
    k = np.arange(N + 1, dtype=np.float64)[:, None]
    lo, hi = np.minimum(k, m[None]), np.maximum(k, m[None])
    iou1 = np.where(hi > 0, lo / np.maximum(hi, 1e-9), 1)
    iou0 = np.where(N - lo > 0, (N - hi) / np.maximum(N - lo, 1e-9), 1)
    return (0.5 * (iou1 + iou0)).mean(1)


def decide(p, p_full, rule, samples, rng):
    order, ratio, exact = topk_curves(p, samples, rng)
    if rule.startswith("como"):
        curve = como_curve(p)
    elif rule.startswith("mix"):
        curve = 0.5 * (exact + como_curve(p))
    else:
        curve = exact if "exact" in rule else ratio
    if "kind" in rule:
        curve = curve.copy()
        if p_full >= 0.5:
            curve[: N // 2] = -1
        else:
            curve[N // 2 + 1:] = -1
    k = int(np.argmax(curve))
    m = np.zeros(N, dtype=np.uint8); m[order[:k]] = 1
    return m.reshape(GRID, GRID)


RULES = ("ratio", "exact", "kind", "exact_kind", "como", "mix", "como_kind", "mix_kind")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--kind", required=True)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--write", default=None, choices=RULES)
    ap.add_argument("--ref", default=None, help="reference run (e.g. the submitted E198g): paired eiou (ratio) comparison")
    args = ap.parse_args()
    run = Path(args.run)
    meta, masks = load_train_ground_truth(ROOT_DIR / "data" / "release")
    kz = np.load(args.kind)
    kind = {**{str(n): float(p) for n, p in zip(kz["train_names"], kz["train_p_full_oof"])},
            **{str(n): float(p) for n, p in zip(kz["test_names"], kz["test_p_full"])}}
    names, probs = load_probs(run / "val_probs.npz"); names = [str(n) for n in names]
    gts = np.stack([masks[n] for n in names]).astype(np.uint8)
    rng = np.random.default_rng(0)
    sc = {r: np.array([miou_batch(decide(p, kind[n], r, args.samples, rng)[None], g[None])[0]
                       for n, p, g in zip(names, probs, gts)]) for r in RULES}
    meth = np.array([meta[n]["method"] for n in names]); kinds = np.array([meta[n]["kind"] for n in names])
    print(f"[{run.name}] f0 (n={len(names)}): " + "  ".join(f"{r}={sc[r].mean():.4f}" for r in RULES))
    for r in RULES[1:]:
        d = sc[r] - sc["ratio"]
        print(f"   {r:10s} vs ratio: {d.mean():+.4f}  wins {int((d > 1e-6).sum())} / losses {int((d < -1e-6).sum())}")
    for m in METHODS:
        for kd in ("full", "patch"):
            s = (meth == m) & (kinds == kd)
            print(f"   {METHOD_SHORT[m]} {kd:5s} " + "  ".join(f"{r}={sc[r][s].mean():.4f}" for r in RULES))
    if args.ref:
        r_names, r_probs = load_probs(Path(args.ref) / "val_probs.npz")
        rp = {str(n): p for n, p in zip(r_names, r_probs)}
        ref = np.array([miou_batch(decide(rp[n], kind[n], "ratio", args.samples, rng)[None], g[None])[0]
                        for n, g in zip(names, gts)])
        print(f"   paired vs {Path(args.ref).name} (ratio = eiou decision, ref {ref.mean():.4f}):")
        bs = np.random.default_rng(1)
        for r in RULES:
            d = sc[r] - ref
            boot = np.array([d[bs.integers(0, len(d), len(d))].mean() for _ in range(2000)])
            per = "  ".join(f"{METHOD_SHORT[m]} {d[meth == m].mean():+.4f}" for m in METHODS)
            print(f"   {r:10s} {sc[r].mean():.4f}  diff {d.mean():+.4f} CI95 [{np.quantile(boot, .025):+.4f}, {np.quantile(boot, .975):+.4f}]"
                  f"  wins {int((d > 1e-6).sum())} / losses {int((d < -1e-6).sum())}  | {per}")
    if args.write:
        test_meta = json.loads((ROOT_DIR / "data" / "release" / "test" / "metadata.json").read_text(encoding="utf-8"))
        order = [r["image_name"] for r in test_meta]
        t_names, t_probs = load_probs(run / "test_probs.npz")
        pos = {str(n): i for i, n in enumerate(t_names)}
        m = np.stack([decide(t_probs[pos[n]], kind[n], args.write, args.samples, rng) for n in order]).astype(bool)
        from predict import validate_submission
        out = run / f"submission_{args.write}.pt"
        torch.save({"image_names": order, "masks": torch.from_numpy(m)}, out)
        validate_submission(out, order)
        print(f"Wrote {out} (validated, not uploaded)")


if __name__ == "__main__":
    main()
