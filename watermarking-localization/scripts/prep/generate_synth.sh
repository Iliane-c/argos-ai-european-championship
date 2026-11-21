#!/bin/bash
# Submit ONE 4-GPU node job to generate synthetic training shards in parallel across all 4 GPUs.
# Excludes fold 0 by default so validation fold 0 remains strictly leak-free.
#
# Usage:
#   ./scripts/prep/generate_synth.sh [version] [num_per_method] [shard_size] [walltime]
# Examples:
#   ./scripts/prep/generate_synth.sh v1 5000 200 01:30:00
#   ./scripts/prep/generate_synth.sh v1_big 20000 500 03:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

VERSION="${1:-v1}"
NUM_PER_METHOD="${2:-5000}"
SHARD_SIZE="${3:-200}"
WALLTIME="${4:-01:30:00}"
NUM_WORKERS="${SLURM_NODE_GPUS:-4}"
PER_WORKER=$(( (NUM_PER_METHOD + NUM_WORKERS - 1) / NUM_WORKERS ))

mkdir -p "$ROOT_DIR/output/manifests"
MANIFEST="$ROOT_DIR/output/manifests/prep_synth_${VERSION}.txt"
: > "$MANIFEST"

for ((w = 0; w < NUM_WORKERS; w++)); do
    echo "CMD: python -m wl.synth --out data/synth/${VERSION} --num-per-method ${PER_WORKER} --shard-size ${SHARD_SIZE} --train-folds 1 2 3 4 --seed ${w} --shard-prefix shard_w${w} --device cuda" >> "$MANIFEST"
done

echo "Submitting 4-GPU parallel synth-${VERSION} job (${NUM_WORKERS} x ${PER_WORKER}/method)..."
sbatch_node "prep_synth_${VERSION}" "$WALLTIME" "$MANIFEST"
