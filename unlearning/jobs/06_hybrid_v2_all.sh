#!/bin/bash
#SBATCH --job-name=t3-hybrid-v2
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=output/t3-hybrid-v2-%j.out
#SBATCH --error=output/t3-hybrid-v2-%j.err
#
# One-command HPC runner for Task 3 Hybrid v2 (JURECA 4x A100 Slurm OR 2x RTX 8000 SSH server).
# automatically shards SDXL-Turbo across all visible GPUs and runs the 4 main branches in parallel.
#
# Usage on JURECA:
#   sbatch unlearning/jobs/06_hybrid_v2_all.sh
# Usage on 2x RTX 8000 SSH server / interactive GPU node:
#   TASK_DIR=/path/to/task3 bash unlearning/jobs/06_hybrid_v2_all.sh 2>&1 | tee hybrid_v2.log

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_common.sh" ]]; then
    REPO_UNLEARNING_DIR="$SLURM_SUBMIT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/unlearning/jobs/_common.sh" ]]; then
    REPO_UNLEARNING_DIR="$SLURM_SUBMIT_DIR/unlearning"
else
    REPO_UNLEARNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -f "$PWD/model.pt" && -f "$PWD/forget_images.npz" ]]; then
    export TASK_DIR="${TASK_DIR:-$PWD}"
else
    # shellcheck disable=SC1091
    source "$REPO_UNLEARNING_DIR/jobs/_common.sh"
    activate_unlearning_venv || true
    export TASK_DIR
fi

export OUT_DIR="${OUT_DIR:-$TASK_DIR/output/hybrid_v2}"
export MODE="${MODE:-full}"
export STAGES="${STAGES:-indomain,retain,oracle,wave1,wave2,soup}"
export SKIP_SHADOW="${SKIP_SHADOW:-1}"

bash "$REPO_UNLEARNING_DIR/run_pipeline.sh"
