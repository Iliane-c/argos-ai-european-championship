"""Probability ensembling (linear, logit, rank), OOF simplex weight optimization, and 5-fold stacking CNN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from wl import GRID, METHOD_TO_ID, METHODS, ROOT_DIR
from wl.evaluate import (
    evaluate_predictions,
    format_metrics_summary,
    load_probs,
    load_train_ground_truth,
    miou_batch,
    save_probs,
    write_metrics,
)
from wl.postproc import fit_postproc_oof


def align_prob_files(paths: Sequence[str | Path]) -> tuple[list[str], np.ndarray]:
    """Load multiple .npz probability files, align to first file's image order, return [M, N, 128, 128]."""
    if not paths:
        raise ValueError("At least one probability file is required")
    loaded = [load_probs(p) for p in paths]
    ref_names = loaded[0][0]
    ref_set = set(ref_names)
    aligned = [loaded[0][1]]
    for idx, (names, probs) in enumerate(loaded[1:], start=1):
        if set(names) != ref_set or len(names) != len(ref_names):
            raise ValueError(f"Image names mismatch in {paths[idx]} vs {paths[0]}")
        pos = {n: i for i, n in enumerate(names)}
        order = [pos[n] for n in ref_names]
        aligned.append(probs[order])
    return ref_names, np.stack(aligned, axis=0).astype(np.float32)


def blend_probs(
    stacked_probs: np.ndarray,
    weights: Sequence[float] | None = None,
    mode: str = "linear",
) -> np.ndarray:
    """Blend [M, N, 128, 128] probabilities into [N, 128, 128]."""
    m = stacked_probs.shape[0]
    w = np.ones(m, dtype=np.float32) / float(m) if weights is None else np.asarray(weights, dtype=np.float32)
    w = np.maximum(w, 0.0)
    w = w / float(max(w.sum(), 1e-8))

    if mode == "linear":
        return np.clip(np.tensordot(w, stacked_probs, axes=(0, 0)), 0.0, 1.0).astype(np.float32)
    if mode == "logit":
        clipped = np.clip(stacked_probs, 1e-4, 1.0 - 1e-4)
        logits = np.log(clipped / (1.0 - clipped))
        blended_logits = np.tensordot(w, logits, axes=(0, 0))
        return (1.0 / (1.0 + np.exp(-np.clip(blended_logits, -20.0, 20.0)))).astype(np.float32)
    if mode == "rank":
        n = stacked_probs.shape[1]
        ranked = np.empty_like(stacked_probs, dtype=np.float32)
        for mi in range(m):
            for ni in range(n):
                flat = stacked_probs[mi, ni].reshape(-1)
                order = np.argsort(np.argsort(flat)).astype(np.float32) / float(max(1, flat.size - 1))
                ranked[mi, ni] = order.reshape(GRID, GRID)
        return np.clip(np.tensordot(w, ranked, axes=(0, 0)), 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported blend mode {mode!r}")


def _search_simplex_weights(
    sub_probs: np.ndarray,
    sub_gts: np.ndarray,
    mode: str = "linear",
    seed: int = 0,
    n_random: int = 250,
) -> tuple[np.ndarray, float]:
    """Find best simplex weights for [M, K, 128, 128] predictions via Dirichlet search + coordinate polish."""
    m = sub_probs.shape[0]
    if m == 1:
        s = float(miou_batch((sub_probs[0] >= 0.5).astype(np.uint8), sub_gts).mean())
        return np.array([1.0], dtype=np.float32), s

    rng = np.random.default_rng(seed)
    candidates: list[np.ndarray] = [np.full(m, 1.0 / m, dtype=np.float32)]
    for i in range(m):
        e = np.zeros(m, dtype=np.float32)
        e[i] = 1.0
        candidates.append(e)
    for _ in range(n_random):
        candidates.append(rng.dirichlet(np.ones(m)).astype(np.float32))

    best_w = candidates[0]
    best_s = -1.0
    for w in candidates:
        blended = blend_probs(sub_probs, w, mode=mode)
        s = float(miou_batch((blended >= 0.5).astype(np.uint8), sub_gts).mean())
        if s > best_s:
            best_s = s
            best_w = w

    # Local perturbation coordinate polish around the best random weight vector.
    for step in (0.15, 0.05):
        for _ in range(30):
            cand = np.maximum(0.0, best_w + rng.normal(0.0, step, size=m).astype(np.float32))
            if cand.sum() <= 1e-8:
                continue
            cand /= cand.sum()
            blended = blend_probs(sub_probs, cand, mode=mode)
            s = float(miou_batch((blended >= 0.5).astype(np.uint8), sub_gts).mean())
            if s > best_s:
                best_s = s
                best_w = cand

    return best_w, best_s


def optimize_ensemble_weights_oof(
    val_paths: Sequence[str | Path],
    meta_by_name: dict[str, dict[str, Any]],
    masks_by_name: dict[str, np.ndarray],
    mode: str = "linear",
    per_method: bool = True,
) -> dict[str, Any]:
    """Optimize ensemble weights on OOF predictions globally or per watermarking method."""
    names, stacked = align_prob_files(val_paths)
    metas = [meta_by_name[n] for n in names]
    gts = np.stack([masks_by_name[n] for n in names], axis=0).astype(np.uint8)
    methods_arr = np.array([m["method"] for m in metas])

    global_w, global_s = _search_simplex_weights(stacked, gts, mode=mode)
    if not per_method:
        blended = blend_probs(stacked, global_w, mode=mode)
        return {"names": names, "val_probs": blended, "weights": {"global": global_w.tolist()}, "oof_score": global_s}

    weights_by_method: dict[str, list[float]] = {}
    blended_out = np.empty((len(names), GRID, GRID), dtype=np.float32)
    for method in METHODS:
        idx = methods_arr == method
        if not idx.any():
            weights_by_method[method] = global_w.tolist()
            continue
        w_m, _ = _search_simplex_weights(stacked[:, idx], gts[idx], mode=mode)
        weights_by_method[method] = [round(float(x), 4) for x in w_m.tolist()]
        blended_out[idx] = blend_probs(stacked[:, idx], w_m, mode=mode)

    oof_score = float(miou_batch((blended_out >= 0.5).astype(np.uint8), gts).mean())
    return {
        "names": names,
        "val_probs": blended_out,
        "weights": weights_by_method,
        "oof_score": oof_score,
    }


def apply_weights_to_probs(
    stacked_probs: np.ndarray,
    metas: Sequence[dict[str, Any]],
    weights: dict[str, list[float]],
    mode: str = "linear",
) -> np.ndarray:
    """Apply scalar or per-method ensemble weights to stacked [M, N, 128, 128] test probabilities."""
    if "global" in weights:
        return blend_probs(stacked_probs, weights["global"], mode=mode)
    out = np.empty((stacked_probs.shape[1], GRID, GRID), dtype=np.float32)
    methods_arr = np.array([m["method"] for m in metas])
    for method in METHODS:
        idx = methods_arr == method
        if idx.any():
            w = weights.get(method)
            out[idx] = blend_probs(stacked_probs[:, idx], w, mode=mode)
    return out


def train_stacking_cnn_oof(
    val_paths: Sequence[str | Path],
    test_paths: Sequence[str | Path] | None,
    meta_by_name: dict[str, dict[str, Any]],
    masks_by_name: dict[str, np.ndarray],
    folds_path: str | Path = ROOT_DIR / "splits" / "folds.json",
    epochs: int = 15,
    lr: float = 3e-3,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a 5-fold out-of-fold 2D stacking CNN over [M model probs + method channels]."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from wl.losses import SoftDualIoULoss

    torch.manual_seed(seed)
    val_names, val_stack = align_prob_files(val_paths)  # [M, N, 128, 128]
    m_models, n_val, _, _ = val_stack.shape
    gts = np.stack([masks_by_name[n] for n in val_names], axis=0).astype(np.float32)
    fjson = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    fold_of = {str(n): k for k, arr in enumerate(fjson["folds"]) for n in arr}
    val_folds = np.array([fold_of[n] for n in val_names], dtype=np.int64)

    def _build_features(stack: np.ndarray, metas: Sequence[dict[str, Any]]) -> np.ndarray:
        _, n_items, h, w = stack.shape
        x = np.zeros((n_items, m_models + 3, h, w), dtype=np.float32)
        x[:, :m_models] = stack.transpose(1, 0, 2, 3)
        for i, meta in enumerate(metas):
            x[i, m_models + METHOD_TO_ID[meta["method"]]] = 1.0
        return x

    val_metas = [meta_by_name[n] for n in val_names]
    x_val_all = torch.from_numpy(_build_features(val_stack, val_metas))
    y_val_all = torch.from_numpy(gts)

    test_names: list[str] = []
    x_test_all = None
    if test_paths:
        test_names, test_stack = align_prob_files(test_paths)
        test_meta_list = json.loads((ROOT_DIR / "data" / "release" / "test" / "metadata.json").read_text(encoding="utf-8"))
        test_meta_map = {r["image_name"]: r for r in test_meta_list}
        x_test_all = torch.from_numpy(_build_features(test_stack, [test_meta_map[n] for n in test_names]))

    class _StackCNN(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mean_p = x[:, :m_models].mean(dim=1).clamp(1e-4, 1.0 - 1e-4)
            base_logits = torch.log(mean_p / (1.0 - mean_p))
            return base_logits + self.net(x).squeeze(1)

    oof_probs = np.zeros((n_val, GRID, GRID), dtype=np.float32)
    test_prob_sum = np.zeros((len(test_names), GRID, GRID), dtype=np.float32) if test_names else None
    loss_iou = SoftDualIoULoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    present_folds = sorted(set(val_folds.tolist()))
    # If val_probs only contains 1 fold (e.g., f0), use deterministic K=5 random subsets inside val for OOF safety.
    if len(present_folds) <= 1:
        rng = np.random.default_rng(seed)
        val_folds = rng.integers(0, 5, size=n_val)
        present_folds = list(range(5))

    for k in present_folds:
        tr_idx = torch.from_numpy(np.where(val_folds != k)[0])
        va_idx = torch.from_numpy(np.where(val_folds == k)[0])
        net = _StackCNN(m_models + 3).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        xb, yb = x_val_all[tr_idx].to(device), y_val_all[tr_idx].to(device)
        for _ in range(epochs):
            opt.zero_grad()
            logits = net(xb)
            loss = loss_iou(logits, yb) + 0.5 * F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            oof_probs[va_idx.numpy()] = torch.sigmoid(net(x_val_all[va_idx].to(device))).cpu().numpy()
            if x_test_all is not None and test_prob_sum is not None:
                test_prob_sum += torch.sigmoid(net(x_test_all.to(device))).cpu().numpy()

    test_probs = (test_prob_sum / float(len(present_folds))) if test_prob_sum is not None else None
    return {
        "names": val_names,
        "val_probs": oof_probs,
        "test_names": test_names,
        "test_probs": test_probs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble or stack multiple runs.")
    ap.add_argument("--runs", nargs="+", required=True, help="Run dirs (output/E101) or val_probs.npz paths")
    ap.add_argument("--out", required=True, help="Output ensemble dir (e.g. output/E290_ens)")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--mode", default="linear", choices=["linear", "logit", "rank", "stack"])
    ap.add_argument("--global-weights", action="store_true", help="Share weights across all 3 methods")
    ap.add_argument("--tune-oof", action="store_true", help="Also tune OOF post-processing")
    ap.add_argument("--test-runs", nargs="+", default=None,
                    help="Same length/order as --runs: take TEST probabilities from these runs instead (e.g. the "
                         "all-folds retrains), with the weights fitted on the validation of --runs")
    args = ap.parse_args()

    val_paths: list[Path] = []
    test_paths: list[Path] = []
    for r in args.runs:
        rp = Path(r)
        if rp.is_dir():
            val_paths.append(rp / "val_probs.npz")
            tp = rp / "test_probs.npz"
            if tp.is_file():
                test_paths.append(tp)
        else:
            val_paths.append(rp)
            tp = rp.parent / "test_probs.npz"
            if tp.is_file():
                test_paths.append(tp)

    if args.test_runs:
        if len(args.test_runs) != len(args.runs):
            raise SystemExit("--test-runs must list one run per --runs entry")
        test_paths = [Path(r) / "test_probs.npz" for r in args.test_runs]
        missing = [str(p) for p in test_paths if not p.is_file()]
        if missing:
            raise SystemExit(f"missing test probabilities: {missing}")

    meta_by_name, masks_by_name = load_train_ground_truth(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    have_all_tests = len(test_paths) == len(val_paths)

    if args.mode == "stack":
        res = train_stacking_cnn_oof(
            val_paths, test_paths if have_all_tests else None, meta_by_name, masks_by_name
        )
        v_names, v_probs = res["names"], res["val_probs"]
        t_names, t_probs = res["test_names"], res["test_probs"]
        extra_meta: dict[str, Any] = {"mode": "stack", "runs": [str(p) for p in val_paths]}
    else:
        opt = optimize_ensemble_weights_oof(
            val_paths, meta_by_name, masks_by_name, mode=args.mode, per_method=not args.global_weights
        )
        v_names, v_probs = opt["names"], opt["val_probs"]
        extra_meta = {"mode": args.mode, "weights": opt["weights"], "runs": [str(p) for p in val_paths],
                      "test_runs": [str(p) for p in test_paths]}
        t_names, t_probs = [], None
        if have_all_tests:
            test_meta_list = json.loads((Path(args.data) / "test" / "metadata.json").read_text(encoding="utf-8"))
            test_meta_map = {r["image_name"]: r for r in test_meta_list}
            t_names, t_stack = align_prob_files(test_paths)
            t_probs = apply_weights_to_probs(t_stack, [test_meta_map[n] for n in t_names], opt["weights"], mode=args.mode)

    save_probs(out_dir / "val_probs.npz", v_names, v_probs)
    metrics = evaluate_predictions(v_names, v_probs, meta_by_name, masks_by_name, threshold=0.5)
    metrics.update({"id": out_dir.name, "parent": "ensemble", "data": "oof", "val": "f0", **extra_meta})

    if args.tune_oof:
        oof = fit_postproc_oof(v_names, v_probs, meta_by_name, masks_by_name)
        metrics["oof_post_cfg"] = oof["post_cfg"]
        metrics["oof_mean"] = oof["oof_score"]
        metrics["oof_metrics"] = oof["metrics"]

    write_metrics(out_dir, metrics)
    if t_probs is not None and t_names:
        save_probs(out_dir / "test_probs.npz", t_names, t_probs)
    print(f"[{out_dir.name}] ensemble ({args.mode}): {format_metrics_summary(metrics)}")


if __name__ == "__main__":
    main()
