"""Equal-weight blend of validated runs' cell beliefs + expected-IoU decision; fast f0 check vs a reference run.

    python -m wl.blend <out_name> <run_dir> <run_dir> [...] [--ref ~/wl_out/E224avg]

Prints the paired f0 difference (eiou decision) against the reference, writes ~/wl_out/<out_name>/submission_eiou.pt
(validated, never uploaded) when every test image is covered.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from wl import ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch
from wl.honest_eval import eiou_decide


def blend(runs, f):
    ds = [dict(zip(*load_probs(os.path.join(os.path.expanduser(r), f)))) for r in runs]
    names = [n for n in ds[0] if all(n in d for d in ds)]
    return names, {n: sum(d[n] for d in ds) / len(ds) for n in names}


def main() -> None:
    args = sys.argv[1:]
    ref_dir = "~/wl_out/E224avg"
    if "--ref" in args:
        i = args.index("--ref"); ref_dir = args[i + 1]; del args[i:i + 2]
    out, runs = args[0], args[1:]
    _, masks = load_train_ground_truth(ROOT_DIR / "data" / "release")
    vn, v = blend(runs, "val_probs.npz")
    ref = dict(zip(*load_probs(os.path.join(os.path.expanduser(ref_dir), "val_probs.npz"))))
    sc = lambda p, n: miou_batch(eiou_decide(p.astype(np.float32))[None], masks[n][None])[0]
    new = np.array([sc(v[n], n) for n in vn]); old = np.array([sc(ref[n], n) for n in vn])
    d = new - old
    print(f"{out}: n={len(vn)}  {new.mean():.4f} vs ref {old.mean():.4f}  diff {d.mean():+.4f}  "
          f"wins {(d > 1e-6).sum()} / losses {(d < -1e-6).sum()}")
    tn, t = blend(runs, "test_probs.npz")
    order = [x["image_name"] for x in json.loads((ROOT_DIR / "data/release/test/metadata.json").read_text())]
    if any(n not in t for n in order):
        sys.exit(f"test images covered {len(tn)} / {len(order)}: not written")
    od = os.path.expanduser(f"~/wl_out/{out}"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, "submission_eiou.pt")
    m = np.stack([eiou_decide(t[n].astype(np.float32)) for n in order]).astype(bool)
    torch.save({"image_names": order, "masks": torch.from_numpy(m)}, p)
    from predict import validate_submission
    validate_submission(p, order)
    print("Wrote", p)


if __name__ == "__main__":
    main()
