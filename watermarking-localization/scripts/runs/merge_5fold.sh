#!/bin/bash
# Merge completed 5-fold CV runs (output/<base_id>_f0..f4) into output/<base_id>_5fold,
# compute full 510-image OOF metrics + postproc, and build a worst-case gallery.
#
# Usage:
#   ./scripts/runs/merge_5fold.sh <base_id>
# Example:
#   ./scripts/runs/merge_5fold.sh E150_w7

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_venv

BASE_ID="${1:?Usage: $0 <base_id>}"
OUT_DIR="output/${BASE_ID}_5fold"

RUN_DIRS=()
for F in 0 1 2 3 4; do
    D="output/${BASE_ID}_f${F}"
    if [[ ! -f "$D/val_probs.npz" ]]; then
        echo "ERROR: Missing fold output: $D/val_probs.npz" >&2
        exit 1
    fi
    RUN_DIRS+=("$D")
done

python -m wl.merge_cv --runs "${RUN_DIRS[@]}" --out "$OUT_DIR"
python -m wl.gallery --probs "$OUT_DIR/val_probs.npz" || true
python -m wl.report
