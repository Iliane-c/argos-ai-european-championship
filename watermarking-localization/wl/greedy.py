"""Greedy forward selection of ensemble members on f0 (shared weights), with optional all-folds test twins.

Starts from the best single candidate, then repeatedly adds the candidate that raises the f0 score of the
shared-weight logit ensemble the most, and stops when the best gain is below --min-gain (default 0.002).
Few free parameters (one weight per member, shared across methods) to limit the optimism of fitting on f0.

    python -m wl.greedy --cand E161_tta:E169_tta E143_tta:E167_tta E160_tta:E165_tta E142_w2_long \
        E190_w8ns_seed1_tta E191_w8ns_lovasz_tta E192_w8ns_35ep_tta E193_w8ns_seed4_tta --out E199_greedy_ens
Each candidate is `run` or `run:test_twin` (run dirs under output/). The chosen command is printed and run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from wl import ROOT_DIR
from wl.ensemble import optimize_ensemble_weights_oof
from wl.evaluate import evaluate_predictions, load_train_ground_truth

OUT = ROOT_DIR / "output"


def score(runs, meta, masks) -> float:
    opt = optimize_ensemble_weights_oof([OUT / r / "val_probs.npz" for r in runs], meta, masks, mode="logit", per_method=False)
    return float(evaluate_predictions(opt["names"], opt["val_probs"], meta, masks, threshold=0.5)["mean"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gain", type=float, default=0.002)
    args = ap.parse_args()
    pairs = [(c.split(":")[0], c.split(":")[1] if ":" in c else c.split(":")[0]) for c in args.cand]
    pairs = [p for p in pairs if (OUT / p[0] / "val_probs.npz").is_file() and (OUT / p[1] / "test_probs.npz").is_file()]
    meta, masks = load_train_ground_truth(ROOT_DIR / "data" / "release")
    singles = {p: score([p[0]], meta, masks) for p in pairs}
    for p, s in sorted(singles.items(), key=lambda kv: -kv[1]):
        print(f"  single {p[0]:28s} {s:.4f}")
    chosen = [max(singles, key=singles.get)]
    best = singles[chosen[0]]
    print(f"start: {chosen[0][0]} {best:.4f}")
    while True:
        trials = {p: score([c[0] for c in chosen] + [p[0]], meta, masks) for p in pairs if p not in chosen}
        if not trials:
            break
        p, s = max(trials.items(), key=lambda kv: kv[1])
        if s - best < args.min_gain:
            print(f"stop: best addition {p[0]} gives {s:.4f} (+{s - best:.4f} < {args.min_gain})")
            break
        chosen.append(p); best = s
        print(f"add {p[0]:28s} -> {s:.4f}")
    cmd = [sys.executable, "-m", "wl.ensemble", "--runs", *[f"output/{c[0]}" for c in chosen],
           "--test-runs", *[f"output/{c[1]}" for c in chosen], "--out", f"output/{args.out}", "--mode", "logit", "--global-weights"]
    print("final:", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


if __name__ == "__main__":
    main()
