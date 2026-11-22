#!/bin/bash
# Submit ONE 4-GPU node job to generate synthetic training shards in parallel across all 4 GPUs.
# Excludes fold 0 by default so validation fold 0 remains strictly leak-free.
#
# Usage:
#   ./scripts/prep/generate_synth.sh [version] [num_per_method] [shard_size] [walltime] [clean_dir]
# Examples:
#   ./scripts/prep/generate_synth.sh v1 5000 200 01:30:00
#   ./scripts/prep/generate_synth.sh v2 6000 200 01:30:00 data/clean/val2017
# With a clean_dir, 85 % of the clean bases come from it and `full` holes are filled by
# restore / paste / inpaint at 0.6 / 0.3 / 0.1 (override with SYNTH_EXTRA="--fill-probs a b c ...").
# RUN_LOCAL=1 runs the workers on the current node (inside a job) instead of submitting one.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

VERSION="${1:-v1}"
NUM_PER_METHOD="${2:-5000}"
SHARD_SIZE="${3:-200}"
WALLTIME="${4:-01:30:00}"
CLEAN_DIR="${5:-}"
NUM_WORKERS="${SLURM_NODE_GPUS:-4}"
PER_WORKER=$(( (NUM_PER_METHOD + NUM_WORKERS - 1) / NUM_WORKERS ))

EXTRA=""
if [[ -n "$CLEAN_DIR" ]]; then
    if [[ ! -d "$CLEAN_DIR" ]]; then
        echo "ERROR: clean_dir not found: $CLEAN_DIR" >&2
        exit 1
    fi
    EXTRA="--clean-dir ${CLEAN_DIR} --external-prob 0.85 --fill-probs 0.6 0.3 0.1"
fi
EXTRA="${EXTRA} ${SYNTH_EXTRA:-}"

mkdir -p "$ROOT_DIR/output/manifests"
MANIFEST="$ROOT_DIR/output/manifests/prep_synth_${VERSION}.txt"
: > "$MANIFEST"

for ((w = 0; w < NUM_WORKERS; w++)); do
    echo "CMD: python -m wl.synth --out data/synth/${VERSION} --num-per-method ${PER_WORKER} --shard-size ${SHARD_SIZE} --train-folds 1 2 3 4 --seed ${w} --shard-prefix shard_w${w} --device cuda ${EXTRA}" >> "$MANIFEST"
done

if [[ "${RUN_LOCAL:-0}" == "1" ]]; then
    # Already inside a GPU node job: run the 4 workers here instead of submitting.
    echo "Running synth-${VERSION} on this node (${NUM_WORKERS} x ${PER_WORKER}/method)..."
    "$ROOT_DIR/jobs/run_node.sh" "$MANIFEST"
else
    echo "Submitting 4-GPU parallel synth-${VERSION} job (${NUM_WORKERS} x ${PER_WORKER}/method)..."
    sbatch_node "prep_synth_${VERSION}" "$WALLTIME" "$MANIFEST"
fi
