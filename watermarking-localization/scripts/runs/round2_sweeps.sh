#!/bin/bash
# Round 2: submit ablation grids via jobs/sweep.py (capped at 4 concurrent GPUs).
#
# Usage:
#   ./scripts/runs/round2_sweeps.sh <preset> [start_id] [--dry-run]
# Presets:
#   data_channels  : A/B test real vs real+synth-v1 x [rgb] vs [rgb,srm,agreement] on SegFormer-B2
#   encoders_loss  : Compare mit_b2 vs mit_b5 x lovasz 0.0 vs 0.3
#   augment_post   : Compare copy_paste_prob 0.0 vs 0.3 x rect_snap false vs true on W7

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_venv

PRESET="${1:-}"
START_ID="${2:-110}"
EXTRA_ARGS=(--submit)
if [[ "${3:-}" == "--dry-run" ]]; then
    EXTRA_ARGS=()
fi

case "$PRESET" in
    data_channels)
        python jobs/sweep.py configs/E102_segformer_b2.yaml \
            --start-id "$START_ID" --max-concurrent 4 "${EXTRA_ARGS[@]}" \
            "data.train=[real],[real,synth-v1]" \
            "input.channels=[rgb],[rgb,srm,agreement]"
        ;;
    encoders_loss)
        python jobs/sweep.py configs/E102_segformer_b2.yaml \
            --start-id "$START_ID" --max-concurrent 4 "${EXTRA_ARGS[@]}" \
            "model.encoder=mit_b2,mit_b5" \
            "loss.lovasz=0.0,0.3"
        ;;
    augment_post)
        python jobs/sweep.py configs/E105_w7_dual_stream.yaml \
            --start-id "$START_ID" --max-concurrent 4 "${EXTRA_ARGS[@]}" \
            "augment.copy_paste_prob=0.0,0.3" \
            "post.rect_snap=false,true"
        ;;
    *)
        echo "Usage: $0 <data_channels|encoders_loss|augment_post> [start_id] [--dry-run]" >&2
        exit 1
        ;;
esac
