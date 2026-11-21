#!/bin/bash
# Shared paths, venv activation, and Slurm defaults for all scripts/.

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$COMMON_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Overridable Slurm defaults (match jobs/run.sh and jobs/run_node.sh)
SLURM_ACCOUNT="${SLURM_ACCOUNT:-training2666}"
SLURM_PARTITION="${SLURM_PARTITION:-dc-gpu}"
SLURM_RESERVATION="${SLURM_RESERVATION:-cispahack}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_NODE_CPUS="${SLURM_NODE_CPUS:-32}"
SLURM_NODE_GPUS="${SLURM_NODE_GPUS:-4}"

activate_venv() {
    if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$ROOT_DIR/.venv/bin/activate"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        return 0
    else
        echo "ERROR: No .venv found at $ROOT_DIR/.venv. Run scripts/setup/01_jureca_shared_env.sh or 03_link_workspace.sh first." >&2
        exit 1
    fi
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command not found in PATH: $cmd" >&2
        exit 1
    fi
}

sbatch_gpu() {
    local job_name="$1"
    local walltime="$2"
    local wrap_cmd="$3"
    require_cmd sbatch
    mkdir -p "$ROOT_DIR/output"
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
        --output="$ROOT_DIR/output/${job_name}-%j.out" \
        --error="$ROOT_DIR/output/${job_name}-%j.err" \
        --wrap="cd '$ROOT_DIR' && source .venv/bin/activate && $wrap_cmd"
}

sbatch_node() {
    local job_name="$1"
    local walltime="$2"
    shift 2
    require_cmd sbatch
    mkdir -p "$ROOT_DIR/output/logs"
    sbatch \
        --job-name="$job_name" \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --reservation="$SLURM_RESERVATION" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres="gpu:${SLURM_NODE_GPUS}" \
        --cpus-per-task="$SLURM_NODE_CPUS" \
        --time="$walltime" \
        --output="$ROOT_DIR/output/${job_name}-%j.out" \
        --error="$ROOT_DIR/output/${job_name}-%j.err" \
        "$ROOT_DIR/jobs/run_node.sh" "$@"
}
