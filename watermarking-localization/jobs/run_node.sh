#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm-node-%j.out
#SBATCH --error=output/slurm-node-%j.err

# Single-node 4-GPU dynamic task pool runner.
# Packs up to 4 independent experiments or commands onto ONE 4-GPU node so no GPUs sit idle
# when the cluster limits each team/user to 1 node at a time.
#
# Usage:
#   sbatch jobs/run_node.sh <manifest.txt>
#   sbatch jobs/run_node.sh <cfg1.yaml> <cfg2.yaml> [cfg3.yaml ...]
#
# Manifest format (one task per non-empty, non-# line):
#   configs/E101_unet_r34.yaml
#   configs/E105_w7_dual_stream.yaml id=E150_f0 data.val=f0 data.train_folds=[1,2,3,4]
#   CMD: python -m wl.features --split train --device cuda
#
# Optional env vars:
#   POST_CMD="..."   Shell command to run once after all tasks succeed (e.g. merge_cv).
#   OMP_NUM_THREADS  Defaults to 4 to avoid CPU oversubscription across 4 GPU workers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p output/logs

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch jobs/run_node.sh <manifest.txt | cfg1.yaml [cfg2.yaml ...]>" >&2
    exit 1
fi

TASKS=()
if [[ $# -eq 1 && "$1" == *.txt ]]; then
    MANIFEST="$1"
    if [[ ! -f "$MANIFEST" ]]; then
        echo "ERROR: Manifest file not found: $MANIFEST" >&2
        exit 1
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "$line" || "$line" == \#* ]] && continue
        TASKS+=("$line")
    done < "$MANIFEST"
else
    for arg in "$@"; do
        TASKS+=("$arg")
    done
fi

NUM_TASKS="${#TASKS[@]}"
if [[ "$NUM_TASKS" -eq 0 ]]; then
    echo "ERROR: No tasks to run." >&2
    exit 1
fi

# Resolve GPU list from CUDA_VISIBLE_DEVICES (set by Slurm) or default to 0,1,2,3
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_IDS=(0 1 2 3)
fi
NUM_WORKERS="${#GPU_IDS[@]}"
if [[ "$NUM_TASKS" -lt "$NUM_WORKERS" ]]; then
    NUM_WORKERS="$NUM_TASKS"
fi

echo "[$(date -Is)] Node pool starting: ${NUM_TASKS} task(s) across ${NUM_WORKERS} GPU(s) (${GPU_IDS[*]})"

QUEUE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wl_node_${SLURM_JOB_ID:-$$}.XXXXXX")"
IDX_FILE="$QUEUE_DIR/next_idx"
LOCK_FILE="$QUEUE_DIR/lock"
FAIL_FILE="$QUEUE_DIR/failed"
echo 0 > "$IDX_FILE"
: > "$FAIL_FILE"
trap 'rm -rf "$QUEUE_DIR"' EXIT

next_task_index() {
    local idx
    (
        flock -x 9
        idx="$(cat "$IDX_FILE")"
        if [[ "$idx" -ge "$NUM_TASKS" ]]; then
            echo -1
        else
            echo $((idx + 1)) > "$IDX_FILE"
            echo "$idx"
        fi
    ) 9>"$LOCK_FILE"
}

run_one_task() {
    local gpu_id="$1"
    local task_idx="$2"
    local task_line="${TASKS[$task_idx]}"
    local job_tag="${SLURM_JOB_ID:-local}"

    if [[ "$task_line" == CMD:* ]]; then
        local cmd="${task_line#CMD:}"
        local log_file="output/logs/task${task_idx}_gpu${gpu_id}_${job_tag}.log"
        echo "[$(date -Is)] [GPU ${gpu_id}] START task #${task_idx} (CMD) -> ${log_file}"
        if CUDA_VISIBLE_DEVICES="$gpu_id" bash -lc "cd '$ROOT_DIR' && [[ -f .venv/bin/activate ]] && source .venv/bin/activate; $cmd" >"$log_file" 2>&1; then
            echo "[$(date -Is)] [GPU ${gpu_id}] DONE  task #${task_idx} (CMD)"
        else
            local rc=$?
            echo "[$(date -Is)] [GPU ${gpu_id}] FAIL  task #${task_idx} (rc=${rc}) -> see ${log_file}" >&2
            printf "task#%d (rc=%d): %s\n" "$task_idx" "$rc" "$task_line" >> "$FAIL_FILE"
        fi
        return 0
    fi

    # Split task line into config + overrides (no spaces inside override values)
    read -r -a parts <<< "$task_line"
    local cfg="${parts[0]}"
    local overrides=("${parts[@]:1}")

    local model_name exp_id
    model_name="$(python -c "import sys; from wl import load_config; ov=[a for a in sys.argv[2:] if '=' in a and not a.startswith('-')]; print(load_config(sys.argv[1], overrides=ov).get('model', {}).get('name', ''))" "$cfg" "${overrides[@]}")"
    exp_id="$(python -c "import sys; from wl import load_config; ov=[a for a in sys.argv[2:] if '=' in a and not a.startswith('-')]; print(load_config(sys.argv[1], overrides=ov).get('id', 'run'))" "$cfg" "${overrides[@]}")"

    local log_file="output/logs/${exp_id}_${job_tag}.log"
    echo "[$(date -Is)] [GPU ${gpu_id}] START ${exp_id} (${cfg}) -> ${log_file}"

    local rc=0
    if [[ "$model_name" == "analytic" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu_id" python -m wl.analytic --config "$cfg" --device cuda "${overrides[@]}" >"$log_file" 2>&1 || rc=$?
    else
        CUDA_VISIBLE_DEVICES="$gpu_id" python -m wl.train --config "$cfg" "${overrides[@]}" >"$log_file" 2>&1 || rc=$?
    fi

    if [[ "$rc" -eq 0 ]]; then
        if [[ -f "output/$exp_id/val_probs.npz" ]]; then
            CUDA_VISIBLE_DEVICES="$gpu_id" python -m wl.gallery --probs "output/$exp_id/val_probs.npz" >>"$log_file" 2>&1 || true
        fi
        echo "[$(date -Is)] [GPU ${gpu_id}] DONE  ${exp_id}"
    else
        echo "[$(date -Is)] [GPU ${gpu_id}] FAIL  ${exp_id} (rc=${rc}) -> see ${log_file}" >&2
        printf "%s (rc=%d): %s\n" "$exp_id" "$rc" "$task_line" >> "$FAIL_FILE"
    fi
}

worker_loop() {
    local gpu_id="$1"
    while true; do
        local idx
        idx="$(next_task_index)"
        if [[ "$idx" -lt 0 ]]; then
            break
        fi
        run_one_task "$gpu_id" "$idx"
    done
}

PIDS=()
for ((w = 0; w < NUM_WORKERS; w++)); do
    worker_loop "${GPU_IDS[$w]}" &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

if [[ -s "$FAIL_FILE" ]]; then
    echo "[$(date -Is)] Node pool finished WITH FAILURES:" >&2
    cat "$FAIL_FILE" >&2
    python -m wl.report || true
    exit 1
fi

if [[ -n "${POST_CMD:-}" ]]; then
    echo "[$(date -Is)] Running POST_CMD: $POST_CMD"
    bash -lc "cd '$ROOT_DIR' && [[ -f .venv/bin/activate ]] && source .venv/bin/activate; $POST_CMD"
fi

python -m wl.report || true
echo "[$(date -Is)] Node pool finished cleanly (${NUM_TASKS} task(s))."
