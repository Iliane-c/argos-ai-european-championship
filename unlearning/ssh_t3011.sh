#!/usr/bin/env bash
# T3-011 on the 2x Quadro RTX 8000 server: in-domain generated retention, then delta-shift training.
# Run from the remote task directory that holds task_template.py, model.pt, forget_images.npz,
# coco224.npy and the unlearning/*.py scripts (see the rsync commands in EXPERIMENTS.md).
# Turing has no bf16, so training uses --amp fp16.
set -euo pipefail
cd "$(dirname "$0")/.."  # remote task root
export PYTHONPATH=.

# 1. ~5,400 images of the 45 forget classes, one shard per GPU (SDXL-Turbo, ~7 GB download once)
CUDA_VISIBLE_DEVICES=0 python unlearning/gen_indomain.py --out gen0.npy --shard 0 --nshards 2 &
CUDA_VISIBLE_DEVICES=1 python unlearning/gen_indomain.py --out gen1.npy --shard 1 --nshards 2 &
wait

# 2. Retain pool = generated + COCO, shuffled so the 500-image drift audit is mixed
python - <<'EOF'
import numpy as np
a = np.concatenate([np.load("gen0.npy"), np.load("gen1.npy"), np.load("coco224.npy")])
np.save("retain_mix.npy", a[np.random.default_rng(0).permutation(len(a))])
print("retain_mix", a.shape)
EOF

# 3. Best local recipe (per-image delta, margin power -2.0; 0.0956) and a stronger tail (-3.0)
mkdir -p out
CUDA_VISIBLE_DEVICES=0 python unlearning/shift_distill.py --out out/gen_mp20.pt --retain retain_mix.npy \
    --amp fp16 --margin-power -2.0 > out/gen_mp20.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python unlearning/shift_distill.py --out out/gen_mp30.pt --retain retain_mix.npy \
    --amp fp16 --margin-power -3.0 > out/gen_mp30.log 2>&1 &
wait
tail -n 5 out/gen_mp20.log out/gen_mp30.log

# 4. Strength curves and exports (best COCO runs peaked at gamma 1.5)
for m in gen_mp20 gen_mp30; do
  CUDA_VISIBLE_DEVICES=0 python unlearning/interpolate.py out/$m.pt --gammas 1.3,1.5,1.7 \
      --retain retain_mix.npy --export out/export | tee out/$m.interp.log
done
