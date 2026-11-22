"""Triage + targeted rescue (CPU, seconds).

Where does the score go? An image whose final mask has NO minority region (a patch image with an empty mask, or a
full image predicted marked everywhere) scores about 0.5 at best: the small class gets IoU 0. Under this metric a
wrong small guess there costs ~0.01 while a right one gains ~0.3-0.5, so any other model that DOES see a region is
worth trusting on exactly those images (break-even precision ~3%).

    python -m wl.triage <base_run> <alt_run> [<alt_run> ...] [--out NAME]

1. counts, on f0 and on test, the images where the base eiou mask has no minority region (< 0.5% of cells);
2. on those images only, takes the eiou mask of the first alternative run that has a minority region on the same
   side (same kind), and reports the paired f0 change vs the base; writes ~/wl_out/NAME/submission_eiou.pt.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from wl import ROOT_DIR, METHODS, METHOD_SHORT
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch
from wl.honest_eval import eiou_decide

EMPTY = 0.005


def runmap(r, f):
    return dict(zip(*load_probs(os.path.join(os.path.expanduser(r), f))))


def minority(m):
    a = float(m.mean())
    return min(a, 1 - a), a > 0.5


def rescue(base, alts, names):
    out, n_empty, n_fix = {}, 0, 0
    for n in names:
        m = eiou_decide(base[n].astype(np.float32))
        mi, full = minority(m)
        if mi < EMPTY:
            n_empty += 1
            for alt in alts:
                if n not in alt:
                    continue
                ma = eiou_decide(alt[n].astype(np.float32))
                mia, fulla = minority(ma)
                if mia >= EMPTY and fulla == full:
                    m = ma; n_fix += 1
                    break
        out[n] = m
    return out, n_empty, n_fix


def main() -> None:
    args = sys.argv[1:]
    name = "triage"
    if "--out" in args:
        i = args.index("--out"); name = args[i + 1]; del args[i:i + 2]
    base_dir, alt_dirs = args[0], args[1:]
    meta, masks = load_train_ground_truth(ROOT_DIR / "data" / "release")
    b = runmap(base_dir, "val_probs.npz"); alts = [runmap(a, "val_probs.npz") for a in alt_dirs]
    names = list(b)
    base_m = {n: eiou_decide(b[n].astype(np.float32)) for n in names}
    new_m, ne, nf = rescue(b, alts, names)
    s0 = np.array([miou_batch(base_m[n][None], masks[n][None])[0] for n in names])
    s1 = np.array([miou_batch(new_m[n][None], masks[n][None])[0] for n in names])
    d = s1 - s0
    meth = np.array([meta[n]["method"] for n in names])
    print(f"f0: {ne} of {len(names)} images have no minority region (they score {s0[[minority(base_m[n])[0] < EMPTY for n in names]].mean():.3f} on average)")
    print(f"f0: rescued from alternatives {nf};  {s1.mean():.4f} vs base {s0.mean():.4f}  diff {d.mean():+.4f}  "
          f"wins {(d > 1e-6).sum()} / losses {(d < -1e-6).sum()}  | " +
          "  ".join(f"{METHOD_SHORT[m]} {d[meth == m].mean():+.4f}" for m in METHODS))
    bt = runmap(base_dir, "test_probs.npz"); at = [runmap(a, "test_probs.npz") for a in alt_dirs]
    order = [x["image_name"] for x in json.loads((ROOT_DIR / "data/release/test/metadata.json").read_text())]
    if any(n not in bt for n in order):
        sys.exit("base run does not cover every test image: not written")
    tm, te, tf = rescue(bt, at, order)
    print(f"test: {te} of {len(order)} images have no minority region; {tf} would take an alternative's region")
    od = os.path.expanduser(f"~/wl_out/{name}"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, "submission_eiou.pt")
    torch.save({"image_names": order, "masks": torch.from_numpy(np.stack([tm[n] for n in order]).astype(bool))}, p)
    from predict import validate_submission
    validate_submission(p, order)
    print("Wrote", p)


if __name__ == "__main__":
    main()
