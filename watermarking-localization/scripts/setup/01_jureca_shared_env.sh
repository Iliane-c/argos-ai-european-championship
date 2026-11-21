#!/bin/bash
# Run ONCE per team on a JURECA LOGIN node after hackathon_setup.sh completes.
# Unzips starter kit + dataset and creates the shared .venv with all required packages.
#
# Usage:
#   ./scripts/setup/01_jureca_shared_env.sh /p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task

set -euo pipefail

TEAM_TASK="${1:-${TEAM_TASK:-}}"
if [[ -z "$TEAM_TASK" ]]; then
    echo "Usage: $0 <path-to-shared-Hackathon_Watermark_Localisation_Task>" >&2
    echo "Or set TEAM_TASK env var first." >&2
    exit 1
fi
if [[ ! -d "$TEAM_TASK" ]]; then
    echo "ERROR: Shared task directory does not exist: $TEAM_TASK" >&2
    exit 1
fi

cd "$TEAM_TASK"
mkdir -p data/release data/synth output

if [[ ! -d "kit/wmloc" ]]; then
    echo "[1/4] Unzipping starter_kit.zip -> kit/"
    unzip -q -o starter_kit.zip -d kit
else
    echo "[1/4] kit/wmloc already present, skipping."
fi

if [[ ! -f "data/release/train/metadata.json" ]]; then
    echo "[2/4] Unzipping train.zip -> data/release/"
    unzip -q -o train.zip -d data/release
else
    echo "[2/4] data/release/train already present, skipping."
fi

if [[ ! -f "data/release/test/metadata.json" ]]; then
    echo "[3/4] Unzipping test.zip -> data/release/"
    unzip -q -o test.zip -d data/release
else
    echo "[3/4] data/release/test already present, skipping."
fi

echo "[4/4] Creating/updating Python 3.11 .venv with kit + extra packages"
if [[ ! -d ".venv" ]]; then
    uv venv -p 3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r kit/requirements.txt scipy matplotlib segmentation-models-pytorch pyyaml pilgram

echo "Done! Next run: ./scripts/setup/02_download_weights.sh '$TEAM_TASK'"
