#!/bin/bash
#SBATCH --job-name=t3-shadow-gold
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=output/t3-shadow-gold-%j.out
#SBATCH --error=output/t3-shadow-gold-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_common.sh" ]]; then
    REPO_UNLEARNING_DIR="$SLURM_SUBMIT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/unlearning/jobs/_common.sh" ]]; then
    REPO_UNLEARNING_DIR="$SLURM_SUBMIT_DIR/unlearning"
else
    REPO_UNLEARNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# shellcheck disable=SC1091
source "$REPO_UNLEARNING_DIR/jobs/_common.sh"
activate_unlearning_venv
mkdir -p "$OUT_DIR"

RETAIN_FIT="${RETAIN_FIT:-$OUT_DIR/retain/retain_fit.npy}"
if [[ ! -f "$RETAIN_FIT" ]]; then
    echo "ERROR: Missing retain proxy at $RETAIN_FIT. Run jobs/01_build_retain_and_oracle.sh first." >&2
    exit 1
fi

run_ul train_shadow_gold.py \
    --task-dir "$TASK_DIR" \
    --retain "$RETAIN_FIT" \
    --out "$OUT_DIR/shadow_gold.pt" \
    --num-models "${NUM_SHADOW_MODELS:-3}" \
    --epochs "${SHADOW_EPOCHS:-12}" \
    --bs "${SHADOW_BS:-32}" \
    --lr "${SHADOW_LR:-1e-4}" \
    --device cuda
