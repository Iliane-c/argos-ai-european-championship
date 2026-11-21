#!/bin/bash
# Quick overview of Slurm jobs and ranked experiment results.
#
# Usage:
#   ./scripts/runs/status.sh [--markdown]

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

if command -v squeue >/dev/null 2>&1; then
    echo "=== Active Slurm jobs ($USER) ==="
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.10l %.12R" || true
    echo ""
fi

activate_venv
echo "=== Experiment Rankings ==="
python -m wl.report "$@"
