#!/bin/bash
# Round v2 on JURECA in one command, from a login node. Queues a chain of 1-node jobs:
#   1. synth-v2 (skipped when data/synth/v2 is already complete)      ~20 min
#   2. E170-E173 on real + synth-v2, one model per GPU                    ~2.5 h (4 h limit)
#   3. post: refine each model (kind gating + boxes), ensemble the finished models, refine the
#      ensemble, diagnose, and write output/round_v2_summary.txt with the best run by OOF.
# The chain waits for your own queued/running jobs, plus WAIT_FOR (e.g. your partner's job id, colon-separated)
# because the reservation allows one node per team. Submitting stays manual:
#   ./scripts/runs/submit.sh output/<best> --submit
#
# Usage:
#   [WAIT_FOR=<jobid>] ./scripts/runs/round_v2.sh   # login node: download COCO val2017 if missing, queue the chain
#   ./scripts/runs/round_v2.sh --coco               # login node: only download COCO val2017
#   ./scripts/runs/round_v2.sh --all                # inside ONE 4-GPU job: synth, training and post in a row
#   ./scripts/runs/round_v2.sh --post               # what job 3 runs on the GPU node

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

RUNS=(E170_segb5_v2 E171_segb2_v2 E172_w2r34_v2 E173_hrnet_v2)
CONFIGS=(configs/E170_segb5_v2.yaml configs/E171_segb2_v2.yaml configs/E172_w2r34_v2.yaml configs/E173_hrnet_v2.yaml)
SYNTH_PER_METHOD=6000
SHARD_SIZE=200
CLEAN_DIR=data/clean/val2017

post() {
    activate_venv
    local done_runs=() refined=() pids=() gpu=0 r
    for r in "${RUNS[@]}"; do
        [[ -f "output/$r/test_probs.npz" ]] && done_runs+=("output/$r") || echo "missing: $r (see output/logs/${r}_*.log)"
    done
    if [[ ${#done_runs[@]} -eq 0 ]]; then
        echo "ERROR: no round v2 model finished" >&2
        exit 1
    fi

    # One refine per GPU; their logs go to output/logs/.
    for r in "${done_runs[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python -m wl.refine --run "$r" --id "$(basename "$r")_ref" --box --device cuda \
            > "output/logs/$(basename "$r")_ref.log" 2>&1 &
        pids+=($!)
        refined+=("output/$(basename "$r")_ref")
        gpu=$((gpu + 1))
    done
    for p in "${pids[@]}"; do wait "$p" || echo "a refine failed: see output/logs/*_ref.log"; done

    if [[ ${#done_runs[@]} -ge 2 ]]; then
        ./scripts/runs/round4_ensemble.sh E190 "${done_runs[@]}" > output/logs/E190_ens.log 2>&1 || echo "ensemble failed: see output/logs/E190_ens.log"
        if [[ -f output/E190_ens_logit/test_probs.npz ]]; then
            CUDA_VISIBLE_DEVICES=0 python -m wl.refine --run output/E190_ens_logit --id E191_ens_ref --box --device cuda \
                > output/logs/E191_ens_ref.log 2>&1 || echo "ensemble refine failed: see output/logs/E191_ens_ref.log"
        fi
    fi

    local candidates=() d
    for d in "${done_runs[@]}" "${refined[@]}" output/E190_ens_linear output/E190_ens_logit output/E190_ens_stack output/E191_ens_ref; do
        [[ -f "$d/val_probs.npz" && -f "$d/test_probs.npz" ]] && candidates+=("$d")
    done
    {
        python - "${candidates[@]}" <<'PY'
import json, sys
from pathlib import Path
from wl.evaluate import format_metrics_summary
rows = []
for d in map(Path, sys.argv[1:]):
    if (d / "metrics.json").is_file() and (d / "test_probs.npz").is_file():
        m = json.loads((d / "metrics.json").read_text())
        rows.append((m.get("oof_mean") or m.get("mean") or 0.0, d, m))
rows.sort(key=lambda t: -t[0])
for s, d, m in rows:
    print(f"{d.name:24s} OOF {s:.4f} | {format_metrics_summary(m)}")
if rows:
    print(f"\nbest: {rows[0][1]}  (E104 on the leaderboard: 0.6258 OOF)")
    print(f"submit: ./scripts/runs/submit.sh {rows[0][1]} --submit")
PY
        echo
        python -m wl.diagnose "${candidates[@]}" || true
    } > output/round_v2_summary.txt 2>&1
    cat output/round_v2_summary.txt
}

synth_complete() {
    local per_worker=$(( (SYNTH_PER_METHOD + SLURM_NODE_GPUS - 1) / SLURM_NODE_GPUS ))
    SYNTH_EXPECTED=$(( SLURM_NODE_GPUS * ((per_worker * 3 + SHARD_SIZE - 1) / SHARD_SIZE) ))
    SYNTH_HAVE="$(find data/synth/v2 -name '*.npz' 2>/dev/null | wc -l || true)"
    [[ "$SYNTH_HAVE" -ge "$SYNTH_EXPECTED" ]]
}

coco_complete() {
    [[ "$(find "$CLEAN_DIR" -name '*.jpg' 2>/dev/null | wc -l || true)" -ge 4000 ]]
}

fetch_coco() {
    if ! coco_complete; then
        echo "[1/4] Downloading COCO val2017 (780 MB) -> $CLEAN_DIR"
        rm -rf "$CLEAN_DIR"; mkdir -p data/clean
        curl -sSfL --retry 5 -o data/clean/val2017.zip http://images.cocodataset.org/zips/val2017.zip
        unzip -q data/clean/val2017.zip -d data/clean && rm data/clean/val2017.zip
    else
        echo "[1/4] $CLEAN_DIR present ($(ls "$CLEAN_DIR" | wc -l) images)"
    fi
}

# Everything inside the current 4-GPU job (compute nodes have no internet: fetch COCO first on a login node).
run_all() {
    activate_venv
    export SLURM_SUBMIT_DIR="$ROOT_DIR"   # jobs/run_node.sh resolves the repo from it
    mkdir -p output/logs output/manifests
    coco_complete || { echo "ERROR: $CLEAN_DIR missing. On a login node run: ./scripts/runs/round_v2.sh --coco" >&2; exit 1; }
    if synth_complete; then
        echo "[$(date +%H:%M)] synth-v2 complete ($SYNTH_HAVE shards), skipping"
    else
        rm -rf data/synth/v2
        echo "[$(date +%H:%M)] synth-v2"
        RUN_LOCAL=1 ./scripts/prep/generate_synth.sh v2 "$SYNTH_PER_METHOD" "$SHARD_SIZE" - "$CLEAN_DIR"
        synth_complete || { echo "ERROR: synth-v2 incomplete ($SYNTH_HAVE of $SYNTH_EXPECTED shards), see output/logs/task*" >&2; exit 1; }
    fi
    echo "[$(date +%H:%M)] training E170-E173 (one per GPU; logs: output/logs/<id>_${SLURM_JOB_ID:-local}.log)"
    ./jobs/run_node.sh "${CONFIGS[@]}" || echo "some models failed; post uses the ones that finished"
    echo "[$(date +%H:%M)] refine + ensemble"
    post
}

submit_chain() {
    require_cmd sbatch
    mkdir -p output/logs output/manifests
    fetch_coco

    # Wait for your own jobs and WAIT_FOR (one node per team).
    local team_jobs dep=""
    team_jobs="$( { squeue -h -u "$USER" -o %i; tr ':' '\n' <<< "${WAIT_FOR:-}"; } | grep -E '^[0-9]+$' | sort -u | paste -sd: || true)"
    if [[ -n "$team_jobs" ]]; then
        dep="afterany:$team_jobs"
        echo "      chain waits for jobs: $team_jobs"
    fi
    echo "      partner's jobs are not seen; pass them as WAIT_FOR=<id>. Team account view:"
    squeue -A "$SLURM_ACCOUNT" --reservation="$SLURM_RESERVATION" -o '      %.10i %.10u %.20j %.8T %.10M' || true

    local synth_id=""
    if synth_complete; then
        echo "[2/4] synth-v2 complete ($SYNTH_HAVE shards), skipping"
    else
        [[ "$SYNTH_HAVE" -gt 0 ]] && echo "      removing incomplete synth-v2 ($SYNTH_HAVE of $SYNTH_EXPECTED shards)" && rm -rf data/synth/v2
        set_dep "$dep"
        synth_id="$(./scripts/prep/generate_synth.sh v2 "$SYNTH_PER_METHOD" "$SHARD_SIZE" 01:00:00 "$CLEAN_DIR" \
            | awk '/Submitted batch job/ {print $4}')"
        [[ -n "$synth_id" ]] || { echo "ERROR: synth job was not submitted" >&2; exit 1; }
        echo "[2/4] synth-v2 job $synth_id"
        dep="afterok:$synth_id"
    fi

    local train_id
    set_dep "$dep"
    train_id="$(sbatch_node round_v2 04:00:00 "${CONFIGS[@]}" | awk '/Submitted batch job/ {print $4}')"
    [[ -n "$train_id" ]] || { echo "ERROR: training job was not submitted" >&2; exit 1; }
    echo "[3/4] E170-E173 job $train_id"

    # afterany: the post step uses whichever models finished.
    local post_id
    set_dep "afterany:$train_id"
    post_id="$(sbatch_gpu_node round_v2_post 01:00:00 "./scripts/runs/round_v2.sh --post" \
        | awk '/Submitted batch job/ {print $4}')"
    [[ -n "$post_id" ]] || { echo "ERROR: post job was not submitted" >&2; exit 1; }
    echo "[4/4] refine + ensemble job $post_id"
    echo
    echo "Progress:  squeue -u \$USER   |   tail -f output/round_v2-${train_id}.out"
    echo "Result:    cat output/round_v2_summary.txt   (after job $post_id)"
}

# sbatch reads SBATCH_DEPENDENCY; an empty value is not "no dependency", so unset it instead.
set_dep() {
    if [[ -n "$1" ]]; then export SBATCH_DEPENDENCY="$1"; else unset SBATCH_DEPENDENCY; fi
}

# The post step needs all 4 GPUs of one node, like sbatch_node, but runs a single command.
sbatch_gpu_node() {
    local job_name="$1" walltime="$2" cmd="$3"
    sbatch \
        --job-name="$job_name" \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --reservation="$SLURM_RESERVATION" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres="gpu:${SLURM_NODE_GPUS}" \
        --cpus-per-task="$SLURM_NODE_CPUS" \
        --time="$walltime" \
        --output="$ROOT_DIR/output/${job_name}-%j.out" \
        --error="$ROOT_DIR/output/${job_name}-%j.err" \
        --wrap="cd '$ROOT_DIR' && $cmd"
}

case "${1:-}" in
    --post) post ;;
    --all) run_all ;;
    --coco) fetch_coco ;;
    "") submit_chain ;;
    *) echo "Unknown option: $1 (use --all, --coco, --post or nothing)" >&2; exit 1 ;;
esac
