#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm-%A_%a.out
#SBATCH --error=output/slurm-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p output

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

TARGET="${1:?Usage: sbatch jobs/run.sh <config.yaml | sweep_manifest.txt> [overrides...]}"
shift || true

if [[ "$TARGET" == *.txt ]]; then
    IDX="${SLURM_ARRAY_TASK_ID:-0}"
    CFG="$(sed -n "$((IDX + 1))p" "$TARGET")"
    if [[ -z "$CFG" ]]; then
        echo "No config found at line $((IDX + 1)) in $TARGET" >&2
        exit 1
    fi
else
    CFG="$TARGET"
fi

echo "[$(date -Is)] Running config: $CFG (task=${SLURM_ARRAY_TASK_ID:-single})"

MODEL_NAME="$(python -c "import sys; from wl import load_config; ov=[a for a in sys.argv[2:] if '=' in a and not a.startswith('-')]; print(load_config(sys.argv[1], overrides=ov).get('model', {}).get('name', ''))" "$CFG" "$@")"
EXP_ID="$(python -c "import sys; from wl import load_config; ov=[a for a in sys.argv[2:] if '=' in a and not a.startswith('-')]; print(load_config(sys.argv[1], overrides=ov).get('id', 'run'))" "$CFG" "$@")"

if [[ "$MODEL_NAME" == "analytic" ]]; then
    python -m wl.analytic --config "$CFG" --device cuda "$@"
else
    python -m wl.train --config "$CFG" "$@"
fi

if [[ -f "output/$EXP_ID/val_probs.npz" ]]; then
    python -m wl.gallery --probs "output/$EXP_ID/val_probs.npz" || true
fi
python -m wl.report || true
