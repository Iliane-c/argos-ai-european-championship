#!/bin/bash
# Build, validate, and optionally upload submission.pt from any completed run directory.
#
# Usage:
#   ./scripts/runs/submit.sh <run_dir> [--submit] [config.yaml]
# Examples:
#   ./scripts/runs/submit.sh output/E100_analytic_t1                # validate locally only
#   ./scripts/runs/submit.sh output/E190_ens_stack --submit         # validate + upload to leaderboard

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_venv

RUN_DIR="${1:?Usage: $0 <run_dir> [--submit] [config.yaml]}"
DO_SUBMIT="${2:-}"
CFG="${3:-configs/base.yaml}"

TEST_PROBS="${RUN_DIR%/}/test_probs.npz"
VAL_PROBS="${RUN_DIR%/}/val_probs.npz"
OUT_PT="${RUN_DIR%/}/submission.pt"

if [[ ! -f "$TEST_PROBS" ]]; then
    echo "ERROR: Missing test predictions: $TEST_PROBS" >&2
    exit 1
fi

ARGS=(--probs "$TEST_PROBS" --config "$CFG" --out "$OUT_PT")
if [[ -f "$VAL_PROBS" ]]; then
    ARGS+=(--val-probs "$VAL_PROBS")
fi

if [[ "$DO_SUBMIT" == "--submit" ]]; then
    if [[ -z "${WMLOC_BASE_URL:-}" || -z "${WMLOC_API_KEY:-}" ]]; then
        echo "ERROR: Export WMLOC_BASE_URL and WMLOC_API_KEY before using --submit." >&2
        exit 1
    fi
    ARGS+=(--submit)
fi

python predict.py "${ARGS[@]}"
