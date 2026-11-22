#!/bin/bash
# Round 2: pack watermark-specific custom architectures (W2, W3, W7) onto ONE 4-GPU node
# (GPUs 0, 1, 2 run E103, E104, E105 simultaneously).
# Prerequisite: synth-v1 shards (data/synth/v1) and feature cache (output/cache/features).
#
# Usage:
#   ./scripts/runs/round2_custom.sh [walltime]

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

WALLTIME="${1:-03:00:00}"

if [[ ! -d "$ROOT_DIR/data/synth/v1" ]] || ! ls "$ROOT_DIR/data/synth/v1"/*.npz >/dev/null 2>&1; then
    echo "ERROR: data/synth/v1 has no .npz shards. Run scripts/prep/prep_all.sh or generate_synth.sh first." >&2
    exit 1
fi

sbatch_node "round2_custom" "$WALLTIME" \
    "configs/E103_w2_wam_multitask.yaml" \
    "configs/E104_w3_msg_cond.yaml" \
    "configs/E105_w7_dual_stream.yaml"
