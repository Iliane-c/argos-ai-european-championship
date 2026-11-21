#!/bin/bash
# Submit 1-GPU Slurm job to generate synthetic training shards on JURECA scratch.
# Excludes fold 0 by default so validation fold 0 remains strictly leak-free.
#
# Usage:
#   ./scripts/prep/generate_synth.sh [version] [num_per_method] [shard_size] [walltime]
# Examples:
#   ./scripts/prep/generate_synth.sh v1 5000 200 02:00:00
#   ./scripts/prep/generate_synth.sh v1_big 20000 500 04:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

VERSION="${1:-v1}"
NUM_PER_METHOD="${2:-5000}"
SHARD_SIZE="${3:-200}"
WALLTIME="${4:-02:00:00}"

sbatch_gpu "prep_synth_${VERSION}" "$WALLTIME" "\
python -m wl.synth \
  --out data/synth/${VERSION} \
  --num-per-method ${NUM_PER_METHOD} \
  --shard-size ${SHARD_SIZE} \
  --train-folds 1 2 3 4 \
  --device cuda"
