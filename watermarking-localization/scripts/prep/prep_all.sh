#!/bin/bash
# Submit ONE 4-GPU node job that runs feature caching (GPU 0) and synth-v1 generation (GPU 1)
# concurrently on the same node, avoiding idle GPUs under a 1-node-at-a-time limit.
#
# Usage:
#   ./scripts/prep/prep_all.sh [synth_version] [num_per_method] [shard_size] [walltime]
# Example:
#   ./scripts/prep/prep_all.sh v1 5000 200 02:30:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

VERSION="${1:-v1}"
NUM_PER_METHOD="${2:-5000}"
SHARD_SIZE="${3:-200}"
WALLTIME="${4:-02:00:00}"
PER_WORKER=$(( (NUM_PER_METHOD + 2) / 3 ))

mkdir -p "$ROOT_DIR/output/manifests"
MANIFEST="$ROOT_DIR/output/manifests/prep_all_${VERSION}.txt"
cat > "$MANIFEST" <<EOF
CMD: python -m wl.analytic --mode e01 --device cuda && python -m wl.features --split train --device cuda && python -m wl.features --split test --device cuda
CMD: python -m wl.synth --out data/synth/${VERSION} --num-per-method ${PER_WORKER} --shard-size ${SHARD_SIZE} --train-folds 1 2 3 4 --seed 0 --shard-prefix shard_w0 --device cuda
CMD: python -m wl.synth --out data/synth/${VERSION} --num-per-method ${PER_WORKER} --shard-size ${SHARD_SIZE} --train-folds 1 2 3 4 --seed 1 --shard-prefix shard_w1 --device cuda
CMD: python -m wl.synth --out data/synth/${VERSION} --num-per-method ${PER_WORKER} --shard-size ${SHARD_SIZE} --train-folds 1 2 3 4 --seed 2 --shard-prefix shard_w2 --device cuda
EOF

echo "Submitting 1-node parallel prep job (GPU 0: features, GPUs 1-3: synth-${VERSION})..."
sbatch_node "prep_all_${VERSION}" "$WALLTIME" "$MANIFEST"
