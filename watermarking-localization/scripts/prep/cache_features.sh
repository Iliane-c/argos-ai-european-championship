#!/bin/bash
# Submit 1-GPU Slurm job to run E01 box-decode verification and precompute
# agreement128 + matched128 feature caches for train and test splits.
#
# Usage:
#   ./scripts/prep/cache_features.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

sbatch_gpu "prep_features" "01:00:00" "\
python -m wl.analytic --mode e01 --device cuda && \
python -m wl.features --split train --device cuda && \
python -m wl.features --split test --device cuda"
