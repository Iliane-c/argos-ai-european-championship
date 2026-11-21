#!/bin/bash
#SBATCH --job-name=t3-wave1
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=output/t3-wave1-%j.out
#SBATCH --error=output/t3-wave1-%j.err

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
mkdir -p "$OUT_DIR/wave1_kfold" "$OUT_DIR/wave1_salun"

OPT_ARGS=()
[[ -f "$OUT_DIR/retain/retain_fit.npy" ]] && OPT_ARGS+=(--retain "$OUT_DIR/retain/retain_fit.npy")
[[ -f "$OUT_DIR/retain/retain_audit.npy" ]] && OPT_ARGS+=(--retain-audit "$OUT_DIR/retain/retain_audit.npy")
[[ -f "$OUT_DIR/oracle/oracle_targets.pt" ]] && OPT_ARGS+=(--oracle "$OUT_DIR/oracle/oracle_targets.pt")
[[ -f "$OUT_DIR/shadow_gold.pt" ]] && OPT_ARGS+=(--shadow-gold "$OUT_DIR/shadow_gold.pt")

run_ul kfold_unlearn.py \
    --task-dir "$TASK_DIR" \
    --out-dir "$OUT_DIR/wave1_kfold" \
    --folds 5 \
    --merge-method ties \
    --alphas "0.8,1.0,1.2,1.5" \
    --scope late_mlp \
    --train-blocks 6 \
    --epochs 6 \
    --lr 3e-5 \
    --w-retain 1.5 \
    --w-ewc 1.0 \
    --w-moment 0.25 \
    --w-npo 0.15 \
    --merged-polish-epochs 1 \
    "${OPT_ARGS[@]}" \
    --device cuda

if [[ -f "$OUT_DIR/retain/retain_fit.npy" ]]; then
    SALUN_EXTRA=()
    [[ -f "$OUT_DIR/retain/retain_audit.npy" ]] && SALUN_EXTRA+=(--retain-audit "$OUT_DIR/retain/retain_audit.npy")
    [[ -f "$OUT_DIR/oracle/oracle_targets.pt" ]] && SALUN_EXTRA+=(--oracle "$OUT_DIR/oracle/oracle_targets.pt")
    [[ -f "$OUT_DIR/shadow_gold.pt" ]] && SALUN_EXTRA+=(--shadow-gold "$OUT_DIR/shadow_gold.pt")

    run_ul hybrid_unlearn.py \
        --task-dir "$TASK_DIR" \
        --retain "$OUT_DIR/retain/retain_fit.npy" \
        --out "$OUT_DIR/wave1_salun/salun_mlp.pt" \
        --scope late_mlp \
        --train-blocks 6 \
        --salun-ratio 0.5 \
        --salun-mode relative_ssd \
        --epochs 8 \
        --lr 3e-5 \
        --w-retain 1.5 \
        --w-feat-retain 0.5 \
        --w-ewc 1.0 \
        --polish-epochs 1 \
        "${SALUN_EXTRA[@]}" \
        --device cuda
fi
