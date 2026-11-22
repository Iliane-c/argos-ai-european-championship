"""Parameter-free final decision: expected-IoU top-k mask per image instead of the OOF-tuned post-processing.

`scripts/runs/submit.sh` fits a post-processing (threshold, minimum sizes) on the validation images; the
expected-IoU decision (Nowozin, CVPR 2014; `wl.honest_eval.eiou_decide`) has no parameter at all and is the
Bayes decision for the per-image mIoU under independent cell beliefs. On E198g: 0.6866 (eiou) vs 0.6849
(OOF post-processing, cross-fitted). Prints both on f0, writes and validates a submission, never uploads.

    python -m wl.eiou_submit output/E198g_final_gate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from wl import ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch
from wl.honest_eval import eiou_decide


def main() -> None:
    run = Path(sys.argv[1])
    meta, masks = load_train_ground_truth(ROOT_DIR / "data" / "release")
    v_names, v_probs = load_probs(run / "val_probs.npz")
    gts = np.stack([masks[str(n)] for n in v_names]).astype(np.uint8)
    thr = miou_batch((v_probs >= 0.5).astype(np.uint8), gts).mean()
    eiou = np.mean([miou_batch(eiou_decide(p.astype(np.float32))[None], g[None])[0] for p, g in zip(v_probs, gts)])
    print(f"[{run.name}] f0: threshold 0.5 = {thr:.4f} | expected-IoU decision = {eiou:.4f}")
    test_meta = json.loads((ROOT_DIR / "data" / "release" / "test" / "metadata.json").read_text(encoding="utf-8"))
    order = [r["image_name"] for r in test_meta]
    t_names, t_probs = load_probs(run / "test_probs.npz")
    pos = {str(n): i for i, n in enumerate(t_names)}
    m = np.stack([eiou_decide(t_probs[pos[n]].astype(np.float32)) for n in order]).astype(bool)
    from predict import validate_submission
    out = run / "submission_eiou.pt"
    torch.save({"image_names": order, "masks": torch.from_numpy(m)}, out)
    validate_submission(out, order)
    print(f"Wrote {out} (validated, not uploaded) | test frac watermarked {m.mean():.4f}")


if __name__ == "__main__":
    main()
