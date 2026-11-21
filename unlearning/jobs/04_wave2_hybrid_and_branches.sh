#!/bin/bash
#SBATCH --job-name=t3-wave2-branches
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=output/t3-wave2-branches-%j.out
#SBATCH --error=output/t3-wave2-branches-%j.err

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
mkdir -p "$OUT_DIR/wave2_branches"

RETAIN_FIT="${RETAIN_FIT:-$OUT_DIR/retain/retain_fit.npy}"
if [[ ! -f "$RETAIN_FIT" ]]; then
    echo "ERROR: Missing $RETAIN_FIT. Run jobs/01_build_retain_and_oracle.sh first." >&2
    exit 1
fi

SHARED=()
[[ -f "$OUT_DIR/retain/retain_audit.npy" ]] && SHARED+=(--retain-audit "$OUT_DIR/retain/retain_audit.npy")
[[ -f "$OUT_DIR/oracle/oracle_targets.pt" ]] && SHARED+=(--oracle "$OUT_DIR/oracle/oracle_targets.pt")
[[ -f "$OUT_DIR/shadow_gold.pt" ]] && SHARED+=(--shadow-gold "$OUT_DIR/shadow_gold.pt")

# Branch A: Full hybrid (Oracle + Wasserstein moment + NPO + Smoothness + SCAR + RUM curriculum)
run_ul hybrid_unlearn.py \
    --task-dir "$TASK_DIR" \
    --retain "$RETAIN_FIT" \
    --out "$OUT_DIR/wave2_branches/branch_full_hybrid.pt" \
    --scope late_mlp_attn_proj \
    --train-blocks 8 \
    --ssd-warmstart \
    --salun-ratio 0.6 \
    --salun-mode relative_ssd \
    --epochs 10 \
    --lr 3e-5 \
    --w-kd-t2 1.0 \
    --w-kd-t1 0.35 \
    --w-moment 0.35 \
    --moment-mode wasserstein \
    --w-npo 0.20 \
    --w-smooth 0.30 \
    --w-scar 0.15 \
    --w-retain 1.5 \
    --w-feat-retain 0.5 \
    --w-head-gaussian 0.3 \
    --w-ewc 1.0 \
    --rum-curriculum \
    --polish-epochs 2 \
    "${SHARED[@]}" \
    --device cuda

# Branch B: Mergeable LoRA + Orthogonal Gradient Projection
run_ul hybrid_unlearn.py \
    --task-dir "$TASK_DIR" \
    --retain "$RETAIN_FIT" \
    --out "$OUT_DIR/wave2_branches/branch_lora_ogp.pt" \
    --train-blocks 8 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --ogp-rank 64 \
    --ogp-var 0.95 \
    --epochs 8 \
    --lr 1e-4 \
    --w-moment 0.25 \
    --w-npo 0.15 \
    --w-smooth 0.25 \
    --w-retain 1.5 \
    --polish-epochs 1 \
    "${SHARED[@]}" \
    --device cuda

# Branch C: Blind L1-sparse prune warmstart + retain-heal (independent diversity for soup)
run_ul hybrid_unlearn.py \
    --task-dir "$TASK_DIR" \
    --retain "$RETAIN_FIT" \
    --out "$OUT_DIR/wave2_branches/branch_prune_heal.pt" \
    --scope mlp_only \
    --train-blocks 12 \
    --prune-warmstart \
    --prune-sparsity 0.15 \
    --epochs 6 \
    --lr 2e-5 \
    --w-kd-t2 0.5 \
    --w-moment 0.20 \
    --w-npo 0.0 \
    --w-retain 2.0 \
    --w-feat-retain 0.75 \
    --w-ewc 1.2 \
    --polish-epochs 2 \
    "${SHARED[@]}" \
    --device cuda
