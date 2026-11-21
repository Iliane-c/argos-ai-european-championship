#!/bin/bash
# Round 3: submit full 5-fold CV (f0..f4) for any finalist config.
#
# Usage:
#   ./scripts/runs/round3_5fold.sh <config.yaml> <base_id> [epochs] [walltime] [extra key=val overrides...]
# Example:
#   ./scripts/runs/round3_5fold.sh configs/E105_w7_dual_stream.yaml E150_w7 40 04:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
require_cmd sbatch

CFG="${1:?Usage: $0 <config.yaml> <base_id> [epochs] [walltime] [extra overrides...]}"
BASE_ID="${2:?Usage: $0 <config.yaml> <base_id> [epochs] [walltime] [extra overrides...]}"
EPOCHS="${3:-40}"
WALLTIME="${4:-04:00:00}"
shift $(( $# >= 4 ? 4 : $# ))

if [[ ! -f "$CFG" ]]; then
    echo "ERROR: Config file not found: $CFG" >&2
    exit 1
fi

for F in 0 1 2 3 4; do
    TRAIN_FOLDS="[$(for k in 0 1 2 3 4; do [[ $k -ne $F ]] && printf "%s," "$k"; done | sed 's/,$//')]"
    FOLD_ID="${BASE_ID}_f${F}"
    echo "Submitting ${FOLD_ID} (val=f${F}, train_folds=${TRAIN_FOLDS}, epochs=${EPOCHS})..."
    sbatch --job-name="$FOLD_ID" --time="$WALLTIME" jobs/run.sh "$CFG" \
        "id=${FOLD_ID}" \
        "data.val=f${F}" \
        "data.train_folds=${TRAIN_FOLDS}" \
        "train.epochs=${EPOCHS}" \
        "$@"
done

echo "All 5 folds submitted. When finished, merge with:"
echo "  ./scripts/runs/merge_5fold.sh ${BASE_ID}"
