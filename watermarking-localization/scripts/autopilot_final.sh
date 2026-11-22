#!/bin/bash
# Final-assembly autopilot (deterministic, no AI, NEVER submits). Run it in tmux on the login node:
#   tmux new -d -s autopilot "scripts/autopilot_final.sh 15904808 08:10 > output/logs/autopilot.log 2>&1"
# 1. waits (until the cut-off time) for the members' TTA predictions; runs the TTA of E164 itself;
# 2. greedy forward selection on f0 (shared weights, min gain 0.002, all-folds test twins);
# 3. decoder verification with the size gate on the least-loaded GPU of <jobid>;
# 4. paired comparison with the current best (E198g) and a summary in output/AUTOPILOT_SUMMARY.txt.
# Guardrails: read-only on existing runs, writes only new run dirs, prints the submit command, never runs it.
set -uo pipefail
jobid="$1"; cutoff="${2:-08:10}"
cd "$(dirname "$0")/.."
source .venv/bin/activate
log() { echo "[$(date +%H:%M:%S)] $*"; }
deadline=$(date -d "today $cutoff" +%s)
free_gpu() {  # least-used GPU of the job, never GPU 1 while E164 trains
  srun --jobid="$jobid" --overlap --ntasks=1 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | sort -t, -k2 -n | head -1 | cut -d, -f1
}
wanted=(E190_w8ns_seed1_tta E191_w8ns_lovasz_tta E192_w8ns_35ep_tta E193_w8ns_seed4_tta E194_tta E195_tta
        E200_w2_r50_seed1_tta E201_w8ns_r34_tta E202_w8ns_synth60_tta E164_tta)
e164_started=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ $e164_started -eq 0 ] && [ -f output/E164_w8_tmrm/test_probs.npz ]; then
    g=$(free_gpu); log "E164 finished -> TTA on GPU $g"
    scripts/run_on_job.sh "$jobid" "$g" "python -m wl.infer --run output/E164_w8_tmrm --id E164_tta" && e164_started=1
  fi
  missing=(); for r in "${wanted[@]}"; do [ -f "output/$r/test_probs.npz" ] || missing+=("$r"); done
  [ ${#missing[@]} -eq 0 ] && { log "all members ready"; break; }
  log "waiting for: ${missing[*]}"; sleep 120
done
cands="E161_tta:E169_tta E143_tta:E167_tta E160_tta:E165_tta E166_tta:E168_tta E142_tta"
for r in "${wanted[@]}"; do [ -f "output/$r/test_probs.npz" ] && cands="$cands $r"; done
log "greedy over: $cands"
python -m wl.greedy --cand $cands --out E210_auto_ens || { log "greedy failed"; exit 1; }
g=$(free_gpu); log "verify on GPU $g"
scripts/run_on_job.sh "$jobid" "$g" "python -m wl.verify --id E211_auto_final --sources output/E210_auto_ens output/E121_analytic_rect_v2 --kind output/E120_kind_const/kind_probs.npz --methods pixelseal trustmark --max-area-q 0.9" \
  || python -m wl.verify --id E211_auto_final --sources output/E210_auto_ens output/E121_analytic_rect_v2 --kind output/E120_kind_const/kind_probs.npz --methods pixelseal trustmark --max-area-q 0.9 --device cpu
{
  echo "AUTOPILOT $(date)"; echo "candidates: $cands"
  python -m wl.compare output/E198g_final_gate output/E211_auto_final
  python -m wl.report | grep -E "E198g|E210|E211"
  echo "If E211 beats E198g by >= 0.005 on f0 with a mostly positive CI, submit (>= 20 min after the last submission):"
  echo "  source .env && ./scripts/runs/submit.sh output/E211_auto_final --submit"
} | tee output/AUTOPILOT_SUMMARY.txt
log "done"
