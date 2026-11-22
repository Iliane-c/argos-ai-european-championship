#!/usr/bin/env bash
# T3-012 on the 2x Quadro RTX 8000 server: sweep the sharpened (--target-scale) delta-shift recipe.
# Run from the T3-011 directory (task_template.py, model.pt, forget_images.npz, coco224.npy, gen0/1.npy).
# Turing has no bf16, so training uses --amp fp16. No exports: pick a run and gamma from the logs first.
set -euo pipefail
cd "$(dirname "$0")/.."  # remote task root
export PYTHONPATH=.

# Retain pool = generated + COCO, same order as ssh_t3011.sh step 2 (rebuilt only if it was deleted)
[ -e retain_mix.npy ] || python - <<'PY'
import numpy as np
a = np.concatenate([np.load("gen0.npy"), np.load("gen1.npy"), np.load("coco224.npy")])
np.save("retain_mix.npy", a[np.random.default_rng(0).permutation(len(a))])
print("retain_mix", a.shape)
PY

mkdir -p out
run() {  # run GPU TAG ARGS...
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python unlearning/shift_distill.py --out out/t12_$tag.pt --retain retain_mix.npy \
      --amp fp16 "$@" > out/t12_$tag.log 2>&1
}

# Round 1 (~12 min): smaller shift; heavier margin tail
run 0 d8 --target-scale 2.5 --margin-power -3.0 --delta 8 &
run 1 mp5 --target-scale 2.5 --margin-power -5.0 --delta 12 &
wait
# Round 2 (~18 min): longer training at scale 4 and 2.5
run 0 ts4e12 --target-scale 4 --margin-power -3.0 --delta 12 --epochs 12 &
run 1 e12 --target-scale 2.5 --margin-power -3.0 --delta 12 --epochs 12 &
wait

# Strength curves (scale 2.5 scored best at gamma 1.5 or above)
for m in d8 mp5 ts4e12 e12; do
  CUDA_VISIBLE_DEVICES=0 python unlearning/interpolate.py out/t12_$m.pt --gammas 1.4,1.6,1.8,2.0 \
      --retain retain_mix.npy | tee out/t12_$m.interp.log
done
