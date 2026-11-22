#!/bin/bash
# Run a command on a GPU that is idle inside one of our running jobs (no new queue wait).
#   scripts/run_on_job.sh <jobid> <gpu_index> "<command>"
# Example: scripts/run_on_job.sh 15904808 0 "python -m wl.verify --id E158_verify_e157g ..."
set -euo pipefail
jobid="$1"; gpu="$2"; shift 2
root="$(cd "$(dirname "$0")/.." && pwd)"
srun --jobid="$jobid" --overlap --ntasks=1 --cpus-per-task=8 \
  bash -lc "cd '$root' && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=$gpu && $*"
