#!/usr/bin/env bash
# T3-013 on the 2x Quadro RTX 8000 server: img2img siblings of the forget set (strength 0.6 and 0.8) and
# retention anchors for the 55 classes outside the forget set. Run from the T3-011 directory, which holds
# task_template.py, model.pt, forget_images.npz, hf_cache (SDXL-Turbo) and .venv.
# GPU_A / GPU_B pick the two cards (default 0 and 1); if one is busy, set both to the free one.
set -euo pipefail
cd "$(dirname "$0")/.."  # remote task root
export PYTHONPATH=. HF_HOME=$PWD/hf_cache
mkdir -p out13

(
  CUDA_VISIBLE_DEVICES=${GPU_A:-0} python unlearning/gen_siblings.py --strength 0.6 --out sib06.npy > out13/sib06.log 2>&1
  CUDA_VISIBLE_DEVICES=${GPU_A:-0} python unlearning/gen_bank.py --shard 0 --nshards 2 --out bank0.npy > out13/bank0.log 2>&1
) & a=$!
(
  CUDA_VISIBLE_DEVICES=${GPU_B:-1} python unlearning/gen_siblings.py --strength 0.8 --out sib08.npy > out13/sib08.log 2>&1
  CUDA_VISIBLE_DEVICES=${GPU_B:-1} python unlearning/gen_bank.py --shard 1 --nshards 2 --out bank1.npy > out13/bank1.log 2>&1
) & b=$!
fail=0
wait $a || fail=1
wait $b || fail=1

tail -n 3 out13/sib06.log out13/sib08.log out13/bank0.log out13/bank1.log
exit $fail
