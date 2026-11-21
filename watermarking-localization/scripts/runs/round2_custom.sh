#!/bin/bash
# Round 2: launch watermark-specific custom architectures (W2, W3, W7; 3 GPU jobs).
# Prerequisite: synth-v1 shards (data/synth/v1) and feature cache (output/cache/features).
#
# Usage:
#   ./scripts/runs/round2_custom.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
require_cmd sbatch

if [[ ! -d "$ROOT_DIR/data/synth/v1" ]] || ! ls "$ROOT_DIR/data/synth/v1"/*.npz >/dev/null 2>&1; then
    echo "ERROR: data/synth/v1 has no .npz shards. Run scripts/prep/generate_synth.sh first." >&2
    exit 1
fi

sbatch --job-name=E103_w2_wam    jobs/run.sh configs/E103_w2_wam_multitask.yaml
sbatch --job-name=E104_w3_msg    jobs/run.sh configs/E104_w3_msg_cond.yaml
sbatch --job-name=E105_w7_dual   jobs/run.sh configs/E105_w7_dual_stream.yaml
