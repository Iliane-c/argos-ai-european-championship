#!/bin/bash
# Train one ensemble member on an idle GPU of a running job, in a detached tmux session (survives a closed
# terminal), then run its shift TTA. Log: output/logs/<id>_bg.log
#   scripts/bg_member.sh <jobid> <gpu> <config.yaml>
set -euo pipefail
jobid="$1"; gpu="$2"; cfg="$3"
id="$(basename "$cfg" .yaml)"
root="$(cd "$(dirname "$0")/.." && pwd)"
tmux new-session -d -s "$id" "cd '$root' && scripts/run_on_job.sh $jobid $gpu 'python -m wl.train --config $cfg && python -m wl.infer --run output/$id --id ${id}_tta' > output/logs/${id}_bg.log 2>&1"
echo "started tmux session $id on job $jobid GPU $gpu -> output/logs/${id}_bg.log"
