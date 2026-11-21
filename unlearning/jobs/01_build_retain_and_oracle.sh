#!/bin/bash
#SBATCH --job-name=t3-retain-oracle
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=output/t3-retain-oracle-%j.out
#SBATCH --error=output/t3-retain-oracle-%j.err

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
mkdir -p "$OUT_DIR/retain" "$OUT_DIR/oracle"

POOL_PATH="${POOL_PATH:-$TASK_DIR/coco224.npy}"
if [[ ! -e "$POOL_PATH" ]]; then
    echo "ERROR: Set POOL_PATH to a domain-matched image pool (.npy/.npz/dir); missing: $POOL_PATH" >&2
    exit 1
fi

run_ul build_retain_proxy.py \
    --task-dir "$TASK_DIR" \
    --pool "$POOL_PATH" \
    --out-dir "$OUT_DIR/retain" \
    --per-class "${PER_CLASS:-400}" \
    --min-per-class "${MIN_PER_CLASS:-20}" \
    --audit-count "${AUDIT_COUNT:-500}" \
    --dedup-cos "${DEDUP_COS:-0.85}" \
    --mixup-fallback \
    --device cuda

ORACLE_EXTRA=()
if [[ -f "$OUT_DIR/shadow_gold.pt" ]]; then
    ORACLE_EXTRA+=(--shadow-gold "$OUT_DIR/shadow_gold.pt")
fi
if [[ -n "${EXT_EMBEDDINGS:-}" && -f "${EXT_EMBEDDINGS}" ]]; then
    ORACLE_EXTRA+=(--ext-embeddings "$EXT_EMBEDDINGS")
fi

run_ul build_oracle.py \
    --task-dir "$TASK_DIR" \
    --out "$OUT_DIR/oracle/oracle_targets.pt" \
    "${ORACLE_EXTRA[@]}" \
    --device cuda
