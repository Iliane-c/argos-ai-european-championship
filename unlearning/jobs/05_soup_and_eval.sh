#!/bin/bash
#SBATCH --job-name=t3-soup-eval
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=output/t3-soup-eval-%j.out
#SBATCH --error=output/t3-soup-eval-%j.err

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
mkdir -p "$OUT_DIR/final"

CANDIDATES=()
for ckpt in \
    "$OUT_DIR/wave1_kfold/merged_best.pt" \
    "$OUT_DIR/wave1_salun/salun_mlp.pt" \
    "$OUT_DIR/wave2_branches/branch_full_hybrid.pt" \
    "$OUT_DIR/wave2_branches/branch_lora_ogp.pt" \
    "$OUT_DIR/wave2_branches/branch_prune_heal.pt"; do
    [[ -f "$ckpt" ]] && CANDIDATES+=("$ckpt")
done

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
    echo "ERROR: No candidate checkpoints found under $OUT_DIR" >&2
    exit 1
fi

EXTRA=()
[[ -f "$OUT_DIR/retain/retain_audit.npy" ]] && EXTRA+=(--retain-audit "$OUT_DIR/retain/retain_audit.npy")
[[ -f "$OUT_DIR/shadow_gold.pt" ]] && EXTRA+=(--shadow-gold "$OUT_DIR/shadow_gold.pt")

run_ul soup_and_eval.py \
    --task-dir "$TASK_DIR" \
    --models "${CANDIDATES[@]}" \
    --search-weights \
    --search-gammas \
    --out "$OUT_DIR/final/soup_submission.pt" \
    --metrics-out "$OUT_DIR/final/soup_metrics.json" \
    "${EXTRA[@]}" \
    --device cuda
