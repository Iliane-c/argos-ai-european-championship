"""Statistical decision layer on top of any model run, and a submission builder.

Input: a run directory with `val_probs.npz` (validation predictions, any subset of the 510 train
images that the model did NOT train on) and `test_probs.npz`. Optionally:
  --kind   a `kind_probs.npz` from `wl.kind_hybrid` (out-of-fold p(full) for train, p(full) for test)
  --bits   an npz with per-cell bit logits (`names`, `bit_logits` [N, nbits, H, W]), e.g. the
           `bit_logits` head of the W2 / E103 model. They are turned into a-contrario z-maps:
           agreement with the true message against K random messages, cell by cell.

Decision:
1. Per-method logistic calibration of each cell, p' = sigmoid(w . [logit p, 2 p_full - 1, z_bits] + b)
   (2 to 4 parameters per method, fitted on validation cells).
2. The mask that maximizes the expected per-image mIoU of the calibrated beliefs (top-k rule,
   `wl.honest_eval.eiou_decide`), instead of a tuned threshold.

Before writing anything it prints an honest estimate: the calibration is cross-fitted (5 groups of
validation images, stratified by method), so no image's label is used to set its own decision.

    python -m wl.finalize output/E103_w2_wam_multitask --kind output/E120_kind_const/kind_probs.npz
    python -m wl.finalize output/E103_w2_wam_multitask --kind ... --submit     # upload (needs .env)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wl import GRID, METHODS, METHOD_SHORT, ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch
from wl.honest_eval import eiou_decide


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def bits_to_z(bit_logits: np.ndarray, message_bits: np.ndarray, n_keys: int = 256, seed: int = 0) -> np.ndarray:
    """A-contrario z-map from per-cell bit logits [nbits, H, W]: true key vs random keys, resized to 128."""
    from PIL import Image

    nbits = bit_logits.shape[0]
    t = np.tanh(0.5 * bit_logits.astype(np.float32)).reshape(nbits, -1)  # [nbits, HW]
    rng = np.random.default_rng(seed)
    keys = np.vstack([2.0 * message_bits[:nbits] - 1.0, rng.choice([-1.0, 1.0], size=(n_keys, nbits))]).astype(np.float32)
    scores = keys @ t / nbits  # [K+1, HW]
    z = (scores[0] - scores[1:].mean(0)) / (scores[1:].std(0) + 1e-6)
    z = z.reshape(bit_logits.shape[1:])
    if z.shape != (GRID, GRID):
        z = np.asarray(Image.fromarray(z.astype(np.float32), mode="F").resize((GRID, GRID), Image.Resampling.BILINEAR))
    return z.astype(np.float32)


class CellCalibrator:
    """Per-method L2 logistic regression on per-cell features, trained with Newton steps."""

    def __init__(self, l2: float = 1e-3, max_cells: int = 300_000, seed: int = 0):
        self.l2, self.max_cells, self.seed, self.w = l2, max_cells, seed, {}

    def fit(self, feats: np.ndarray, gts: np.ndarray, methods: np.ndarray) -> "CellCalibrator":
        rng = np.random.default_rng(self.seed)
        for m in METHODS:
            sel = methods == m
            if not sel.any():
                continue
            x = feats[sel].reshape(-1, feats.shape[-1]).astype(np.float64)
            y = gts[sel].reshape(-1).astype(np.float64)
            if len(x) > self.max_cells:
                idx = rng.choice(len(x), self.max_cells, replace=False)
                x, y = x[idx], y[idx]
            x = np.hstack([x, np.ones((len(x), 1))])
            w = np.zeros(x.shape[1])
            w[0] = 1.0
            for _ in range(60):
                q = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
                grad = x.T @ (q - y) / len(y) + self.l2 * w
                hess = (x * (q * (1 - q))[:, None]).T @ x / len(y) + self.l2 * np.eye(len(w))
                step = np.linalg.solve(hess, grad)
                w -= step
                if np.abs(step).max() < 1e-8:
                    break
            self.w[m] = w
        return self

    def predict(self, feats_one: np.ndarray, method: str) -> np.ndarray:
        w = self.w[method]
        x = np.concatenate([feats_one, np.ones(feats_one.shape[:-1] + (1,))], axis=-1)
        return 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))


def load_ac_z(split: str, names: list[str]) -> dict[str, np.ndarray]:
    """Random-key calibrated decoder z-maps from wl.acontrario (native-scale evidence), per image."""
    from wl.acontrario import load_ac

    out = {}
    for n in names:
        ac = load_ac(split, n)
        if ac is None:
            raise SystemExit(f"ERROR: no a-contrario map for {split}/{n}; run `python -m wl.acontrario maps` first")
        out[n] = ac["z128"]
    return out


def load_resynth(split: str, names: list[str], sub: str = "resynth") -> dict[str, np.ndarray]:
    """Differential re-synthesis z-maps from wl.resynth (pixel-level, native scale for `full` images)."""
    path = ROOT_DIR / "output" / sub / f"{split}.npz"
    if not path.is_file():
        raise SystemExit(f"ERROR: {path} missing; run `python -m wl.resynth --split {split}` first")
    z = np.load(path)
    table = {str(n): np.nan_to_num(m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) for n, m in zip(z["names"], z["z128"])}
    missing = [n for n in names if n not in table]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} images missing from {path}")
    return {n: table[n] for n in names}


def build_features(probs: np.ndarray, names: list[str], kind: dict[str, float] | None, zbits: dict[str, np.ndarray] | None,
                   zdec: dict[str, np.ndarray] | None = None, zres: dict[str, np.ndarray] | None = None,
                   zbox: dict[str, np.ndarray] | None = None, zatt: dict[str, np.ndarray] | None = None) -> np.ndarray:
    parts = [_logit(probs)[..., None]]
    if kind is not None:
        pf = np.array([kind[n] for n in names], dtype=np.float32)
        parts.append(np.broadcast_to((2 * pf - 1)[:, None, None, None], probs.shape + (1,)))
    if zbits is not None:
        parts.append(np.clip(np.stack([zbits[n] for n in names]), -10, 10)[..., None] / 5.0)
    if zdec is not None:
        parts.append(np.clip(np.stack([zdec[n] for n in names]), -20, 20)[..., None] / 5.0)
    if zres is not None:
        zr = np.clip(np.stack([zres[n] for n in names]), -10, 10) / 5.0
        parts.append(zr[..., None])
        if kind is not None:  # the re-synthesis is at native scale only for `full` images: let it act there
            pf = np.array([kind[n] for n in names], dtype=np.float32)[:, None, None]
            parts.append((zr * pf)[..., None])
    if zbox is not None:  # box-wise re-synthesis, only filled for predicted patches
        parts.append((np.clip(np.stack([zbox[n] for n in names]), -10, 10) / 5.0)[..., None])
    if zatt is not None:  # decoder attribution z (PS, TM); interaction with p_full so it can act on holes
        za = np.clip(np.stack([zatt[n] for n in names]), -10, 10) / 5.0
        parts.append(za[..., None])
        if kind is not None:
            pf = np.array([kind[n] for n in names], dtype=np.float32)[:, None, None]
            parts.append((za * pf)[..., None])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def load_kind(path: str | None) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    if not path:
        return None, None
    z = np.load(path)
    train = {str(n): float(p) for n, p in zip(z["train_names"], z["train_p_full_oof"])}
    test = {str(n): float(p) for n, p in zip(z["test_names"], z["test_p_full"])}
    return train, test


def load_bits(path: str | None, messages: dict[str, np.ndarray], n_keys: int) -> dict[str, np.ndarray] | None:
    if not path:
        return None
    z = np.load(path)
    return {str(n): bits_to_z(b, messages[str(n)], n_keys, seed=i) for i, (n, b) in enumerate(zip(z["names"], z["bit_logits"]))}


MRF_BETAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def mrf_smooth(prob: np.ndarray, beta: float, iters: int = 6) -> np.ndarray:
    """Mean-field inference for an Ising/Potts prior on the cell grid (MAP-MRF segmentation).

    q_i <- sigmoid(logit p_i + beta * mean_{j in N8(i)} (2 q_j - 1)): calibrated cell evidence plus a
    coupling that rewards spatially coherent regions (holes and patches are compact).
    """
    if beta <= 0:
        return prob
    l = _logit(prob)
    q = prob.astype(np.float64)
    for _ in range(iters):
        s = 2 * q - 1
        sp = np.pad(s, 1, mode="edge")
        nb = sum(sp[1 + dy : 1 + dy + s.shape[0], 1 + dx : 1 + dx + s.shape[1]]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)) / 8.0
        q = 1.0 / (1.0 + np.exp(-(l + beta * nb)))
    return q.astype(np.float32)


def choose_betas(cal: "CellCalibrator", feats, gts, methods) -> dict[str, float]:
    """One coupling per method, chosen on the calibration images only (inner selection, 5-value grid)."""
    out = {}
    for m in METHODS:
        idx = np.flatnonzero(methods == m)
        beliefs = [cal.predict(feats[i], m) for i in idx]
        best = max(MRF_BETAS, key=lambda b: float(np.mean([
            miou_batch(eiou_decide(mrf_smooth(p, b))[None], gts[i][None])[0] for p, i in zip(beliefs, idx)])))
        out[m] = best
    return out


def honest_estimate(feats, gts, methods, names, n_groups: int = 5, seed: int = 0, mrf: bool = False) -> tuple[np.ndarray, list[float]]:
    """Cross-fitted per-image mIoU: calibrate on 4 groups, decide on the 5th (stratified by method)."""
    rng = np.random.default_rng(seed)
    group = np.zeros(len(names), dtype=int)
    for m in METHODS:
        idx = np.flatnonzero(methods == m)
        group[rng.permutation(idx)] = np.arange(len(idx)) % n_groups
    scores = np.zeros(len(names))
    for g in range(n_groups):
        tr, va = group != g, group == g
        cal = CellCalibrator(seed=g).fit(feats[tr], gts[tr], methods[tr])
        betas = choose_betas(cal, feats[tr], gts[tr], methods[tr]) if mrf else {m: 0.0 for m in METHODS}
        if mrf:
            print(f"   group {g}: MRF beta " + " ".join(f"{METHOD_SHORT[m]}={b}" for m, b in betas.items()))
        for i in np.flatnonzero(va):
            pred = eiou_decide(mrf_smooth(cal.predict(feats[i], methods[i]), betas[methods[i]]))
            scores[i] = miou_batch(pred[None], gts[i][None])[0]
    group_means = [float(scores[group == g].mean()) for g in range(n_groups)]
    return scores, group_means


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="Run directory with val_probs.npz and test_probs.npz")
    ap.add_argument("--kind", default=None, help="kind_probs.npz from wl.kind_hybrid")
    ap.add_argument("--bits-val", default=None, help="npz with names + bit_logits for the validation images")
    ap.add_argument("--bits-test", default=None, help="npz with names + bit_logits for the test images")
    ap.add_argument("--keys", type=int, default=256)
    ap.add_argument("--ac-z", action="store_true", help="Fuse the a-contrario decoder z-maps (wl.acontrario) as evidence")
    ap.add_argument("--mrf", action="store_true", help="Mean-field MRF smoothing of the calibrated beliefs (beta chosen per method, inner CV)")
    ap.add_argument("--attrib", action="store_true", help="Fuse decoder attribution z-maps (wl.attrib)")
    ap.add_argument("--resynth-box", action="store_true", help="Also fuse box-wise re-synthesis z (output/resynth_box)")
    ap.add_argument("--resynth", action="store_true", help="Fuse the re-synthesis matched-filter z-maps (wl.resynth)")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--out", default=None, help="Default: <run>/submission_final.pt")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    meta_by_name, masks_by_name = load_train_ground_truth(args.data)
    test_meta = json.loads((Path(args.data) / "test" / "metadata.json").read_text(encoding="utf-8"))
    messages = {m["image_name"]: np.array([int(c) for c in m["message"]], dtype=np.float32) for m in [*meta_by_name.values(), *test_meta]}
    kind_tr, kind_te = load_kind(args.kind)

    v_names, v_probs = load_probs(run / "val_probs.npz")
    v_names = [str(n) for n in v_names]
    v_methods = np.array([meta_by_name[n]["method"] for n in v_names])
    v_gts = np.stack([masks_by_name[n] for n in v_names]).astype(np.uint8)
    v_feats = build_features(v_probs.astype(np.float32), v_names, kind_tr, load_bits(args.bits_val, messages, args.keys),
                             load_ac_z("train", v_names) if args.ac_z else None,
                             load_resynth("train", v_names) if args.resynth else None,
                             load_resynth("train", v_names, "resynth_box") if args.resynth_box else None,
                             load_resynth("train", v_names, "attrib") if args.attrib else None)

    scores, group_means = honest_estimate(v_feats, v_gts, v_methods, v_names, mrf=args.mrf)
    kinds = np.array([meta_by_name[n]["kind"] for n in v_names])
    raw = float(miou_batch((v_probs >= 0.5).astype(np.uint8), v_gts).mean())
    print(f"[{run.name}] n_val={len(v_names)} raw@0.5={raw:.4f} | finalize (cross-fitted) = {scores.mean():.4f} ± {np.std(group_means):.4f}")
    print("   " + "  ".join(f"{METHOD_SHORT[m]}={scores[v_methods == m].mean():.4f}" for m in METHODS)
          + f"  full={scores[kinds == 'full'].mean():.4f}  patch={scores[kinds == 'patch'].mean():.4f}")

    # Final calibration on all validation images, applied to the test set.
    cal = CellCalibrator().fit(v_feats, v_gts, v_methods)
    betas = choose_betas(cal, v_feats, v_gts, v_methods) if args.mrf else {m: 0.0 for m in METHODS}
    t_names, t_probs = load_probs(run / "test_probs.npz")
    pos = {str(n): i for i, n in enumerate(t_names)}
    expected = [str(r["image_name"]) for r in test_meta]
    missing = [n for n in expected if n not in pos]
    if missing:
        raise SystemExit(f"ERROR: test_probs is missing {len(missing)} images")
    t_probs = t_probs[[pos[n] for n in expected]].astype(np.float32)
    t_feats = build_features(t_probs, expected, kind_te, load_bits(args.bits_test, messages, args.keys),
                             load_ac_z("test", expected) if args.ac_z else None,
                             load_resynth("test", expected) if args.resynth else None,
                             load_resynth("test", expected, "resynth_box") if args.resynth_box else None,
                             load_resynth("test", expected, "attrib") if args.attrib else None)
    masks = np.stack([eiou_decide(mrf_smooth(cal.predict(t_feats[i], r["method"]), betas[r["method"]]))
                      for i, r in enumerate(test_meta)]).astype(np.uint8)

    import torch

    from predict import submit, validate_submission

    out = Path(args.out) if args.out else Path("output") / f"final_{run.name}{'_acz' if args.ac_z else ''}{'_rs' if args.resynth else ''}{'_rsb' if args.resynth_box else ''}{'_att' if args.attrib else ''}{'_mrf' if args.mrf else ''}.pt"
    torch.save({"image_names": expected, "masks": torch.from_numpy(masks).bool()}, out)
    validate_submission(out, expected)
    frac = masks.mean(axis=(1, 2))
    print(f"Wrote {out} | test images predicted mostly watermarked: {float((frac >= 0.5).mean()):.3f}")
    (run / "finalize.json").write_text(json.dumps({
        "honest_mean": round(float(scores.mean()), 4), "group_std": round(float(np.std(group_means)), 4),
        "raw_at_0.5": round(raw, 4), "kind": args.kind, "bits": bool(args.bits_val), "ac_z": args.ac_z, "resynth": args.resynth, "mrf_betas": betas,
        "weights": {m: [round(float(v), 4) for v in w] for m, w in cal.w.items()},
    }, indent=2) + "\n", encoding="utf-8")
    if args.submit:
        submit(out)


if __name__ == "__main__":
    main()
