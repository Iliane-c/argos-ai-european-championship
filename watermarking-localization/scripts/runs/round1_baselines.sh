#!/bin/bash
# Round 1: launch training-free T1 baseline + fast real-data baselines (3 GPU jobs).
# Note: E100 benefits from scripts/prep/cache_features.sh completing first, but can also compute on the fly.
#
# Usage:
#   ./scripts/runs/round1_baselines.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
require_cmd sbatch

sbatch --job-name=E100_analytic_t1 jobs/run.sh configs/E100_analytic_t1.yaml
sbatch --job-name=E101_unet_r34    jobs/run.sh configs/E101_unet_r34.yaml
sbatch --job-name=E102_segformer   jobs/run.sh configs/E102_segformer_b2.yaml
