#!/bin/bash
# Run ONCE per teammate inside their own repo clone on JURECA.
# Symlinks .venv, kit, data, and output to the shared $TEAM_TASK scratch folder.
#
# Usage:
#   ./scripts/setup/03_link_workspace.sh /p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TEAM_TASK="${1:-${TEAM_TASK:-}}"
if [[ -z "$TEAM_TASK" ]]; then
    echo "Usage: $0 <path-to-shared-Hackathon_Watermark_Localisation_Task>" >&2
    exit 1
fi
if [[ ! -d "$TEAM_TASK" ]]; then
    echo "ERROR: Shared task directory does not exist: $TEAM_TASK" >&2
    exit 1
fi

link_dir() {
    local name="$1"
    local target="$TEAM_TASK/$name"
    local dest="$ROOT_DIR/$name"

    if [[ ! -e "$target" ]]; then
        echo "ERROR: Expected shared target does not exist: $target" >&2
        exit 1
    fi
    if [[ -e "$dest" && ! -L "$dest" ]]; then
        echo "ERROR: $dest exists and is NOT a symlink. Move or remove it manually before linking." >&2
        exit 1
    fi
    ln -sfn "$target" "$dest"
    echo "  linked $name -> $target"
}

echo "Linking workspace $ROOT_DIR to $TEAM_TASK..."
link_dir ".venv"
link_dir "kit"
link_dir "data"
link_dir "output"

echo "Verifying imports..."
# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"
python -c "import torch, wl; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available(), 'ROOT_DIR:', wl.ROOT_DIR)"
echo "Workspace linked and ready."
