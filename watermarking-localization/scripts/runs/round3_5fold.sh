#!/bin/bash
# Round 3: pack all 5 folds (f0..f4) onto ONE 4-GPU node job.
# GPUs 0-3 run folds 0-3 concurrently; the first GPU to finish picks up fold 4.
# Once all 5 folds finish, automatically merges OOF + test predictions via wl.merge_cv.
#
# Usage:
#   ./scripts/runs/round3_5fold.sh <config.yaml> <base_id> [epochs] [walltime] [extra key=val overrides...]
# Example:
#   ./scripts/runs/round3_5fold.sh configs/E105_w7_dual_stream.yaml E150_w7 40 06:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

CFG="${1:?Usage: $0 <config.yaml> <base_id> [epochs] [walltime] [extra overrides...]}"
BASE_ID="${2:?Usage: $0 <config.yaml> <base_id> [epochs] [walltime] [extra overrides...]}"
EPOCHS="${3:-40}"
WALLTIME="${4:-06:00:00}"
shift $(( $# >= 4 ? 4 : $# ))

if [[ ! -f "$CFG" ]]; then
    echo "ERROR: Config file not found: $CFG" >&2
    exit 1
fi

mkdir -p "$ROOT_DIR/output/manifests"
MANIFEST="$ROOT_DIR/output/manifests/${BASE_ID}_5fold.txt"
: > "$MANIFEST"

EXTRA_STR=""
if [[ $# -gt 0 ]]; then
    EXTRA_STR=" $*"
fi

for F in 0 1 2 3 4; do
    TRAIN_FOLDS="[$(for k in 0 1 2 3 4; do [[ $k -ne $F ]] && printf "%s," "$k"; done | sed 's/,$//')]"
    FOLD_ID="${BASE_ID}_f${F}"
    echo "${CFG} id=${FOLD_ID} data.val=f${F} data.train_folds=${TRAIN_FOLDS} train.epochs=${EPOCHS}${EXTRA_STR}" >> "$MANIFEST"
done

echo "Generated 5-fold manifest: $MANIFEST"
echo "Submitting 1-node (4-GPU) job for ${BASE_ID} (auto-merges after f0..f4 complete)..."
POST_CMD="python -m wl.merge_cv --base-id '${BASE_ID}' --out-id '${BASE_ID}_5fold'" \
    sbatch_node "${BASE_ID}_5fold" "$WALLTIME" "$MANIFEST"
