#!/bin/bash
# Round 4: build linear, logit, and 5-fold stacking CNN ensembles from >=2 runs with OOF tuning.
#
# Usage:
#   ./scripts/runs/round4_ensemble.sh <ens_base_id> <run_dir_1> <run_dir_2> [run_dir_3 ...]
# Example:
#   ./scripts/runs/round4_ensemble.sh E190 output/E100_analytic_t1 output/E150_w7_5fold output/E160_segb5_5fold

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_venv

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <ens_base_id> <run_dir_1> <run_dir_2> [run_dir_3 ...]" >&2
    exit 1
fi

ENS_ID="$1"
shift
RUNS=("$@")

for R in "${RUNS[@]}"; do
    if [[ ! -f "$R/val_probs.npz" ]]; then
        echo "ERROR: Missing $R/val_probs.npz" >&2
        exit 1
    fi
done

for MODE in linear logit stack; do
    OUT="output/${ENS_ID}_ens_${MODE}"
    echo "=== Building ${MODE} ensemble -> ${OUT} ==="
    python -m wl.ensemble --runs "${RUNS[@]}" --out "$OUT" --mode "$MODE" --tune-oof
    python -m wl.gallery --probs "$OUT/val_probs.npz" || true
done

echo "=== Updated leaderboard table ==="
python -m wl.report --markdown
