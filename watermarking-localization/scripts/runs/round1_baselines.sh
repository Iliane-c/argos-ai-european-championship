#!/bin/bash
# Round 1: pack training-free T1 baseline + fast real-data baselines onto ONE 4-GPU node
# (GPUs 0, 1, 2 run E100, E101, E102 simultaneously).
#
# Usage:
#   ./scripts/runs/round1_baselines.sh [walltime]

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

WALLTIME="${1:-02:00:00}"

sbatch_node "round1_baselines" "$WALLTIME" \
    "configs/E100_analytic_t1.yaml" \
    "configs/E101_unet_r34.yaml" \
    "configs/E102_segformer_b2.yaml"
