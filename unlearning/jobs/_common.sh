#!/bin/bash
# Shared JURECA paths, venv activation, and Slurm helpers for Task 3 unlearning jobs.

[[ $- == *i* ]] || set -euo pipefail

if [[ -n "${REPO_UNLEARNING_DIR:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_common.sh" ]]; then
    REPO_UNLEARNING_DIR="$SLURM_SUBMIT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/unlearning/jobs/_common.sh" ]]; then
    REPO_UNLEARNING_DIR="$SLURM_SUBMIT_DIR/unlearning"
else
    REPO_UNLEARNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

TASK_DIR="${TASK_DIR:-/p/scratch/training2666/chander2/argos/unlearning}"
OUT_DIR="${OUT_DIR:-$TASK_DIR/output/hybrid}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-training2666}"
SLURM_PARTITION="${SLURM_PARTITION:-dc-gpu}"
SLURM_RESERVATION="${SLURM_RESERVATION:-cispahack}"
SLURM_CPUS="${SLURM_CPUS:-8}"

activate_unlearning_venv() {
    if [[ -n "${UNLEARNING_PYTHON:-}" ]]; then
        return 0
    fi
    if [[ -f "$TASK_DIR/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$TASK_DIR/.venv/bin/activate"
    elif [[ -f "$REPO_UNLEARNING_DIR/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$REPO_UNLEARNING_DIR/.venv/bin/activate"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        return 0
    else
        echo "ERROR: No .venv found in $TASK_DIR/.venv or $REPO_UNLEARNING_DIR/.venv" >&2
        exit 1
    fi
}

run_ul() {
    local script_name="$1"
    shift
    local py="${UNLEARNING_PYTHON:-python}"
    PYTHONPATH="$REPO_UNLEARNING_DIR:$TASK_DIR:${PYTHONPATH:-}" "$py" -u "$REPO_UNLEARNING_DIR/$script_name" "$@"
}

sbatch_unlearn() {
    local job_name="$1"
    local walltime="$2"
    local script_path="$3"
    shift 3
    mkdir -p "$LOG_DIR"
    sbatch \
        --job-name="$job_name" \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --reservation="$SLURM_RESERVATION" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres=gpu:1 \
        --cpus-per-task="$SLURM_CPUS" \
        --time="$walltime" \
        --output="$LOG_DIR/${job_name}-%j.out" \
        --error="$LOG_DIR/${job_name}-%j.err" \
        "$script_path" "$@"
}
