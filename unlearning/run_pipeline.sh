#!/usr/bin/env bash
# One-command runner for the full Task 3 hybrid unlearning pipeline.
# Works on Google Colab, local GPU machines, and JURECA interactive GPU nodes.
#
# Quick start:
#   TASK_DIR=/path/to/task3 bash unlearning/run_pipeline.sh
#
# Key environment variables (all optional except TASK_DIR when files aren't in $PWD):
#   TASK_DIR            Directory containing model.pt and forget_images.npz (and optional task_template.py).
#   OUT_DIR             Output directory (default: $TASK_DIR/output/hybrid).
#   POOL_PATH           Retain proxy image pool (.npy, .npz, or directory of images).
#                       Defaults to $TASK_DIR/coco224.npy if present, else auto-downloads COCO val2017.
#   AUTO_DOWNLOAD_COCO  1 (default) to download COCO val2017 (~778MB) if POOL_PATH is missing; 0 to fail instead.
#   MODE                full (default, competition quality) | fast (Colab T4 friendly) | smoke (tiny dry-run).
#   STAGES              all (default) or comma-separated subset of:
#                       diagnose,retain,shadow,oracle,wave1,wave2,soup
#   SKIP_SHADOW         0 (default) | 1 (skip Shadow-Gold training to save GPU time).
#   FORCE               0 (default, skip stages whose outputs exist so reruns resume) | 1 (overwrite all).
#   DEVICE              auto (default) | cuda | cpu.
#   AMP                 auto (default: bf16 on Ampere+, fp16 on T4/Volta, fp32 on CPU) | bf16 | fp16 | fp32.
#   PYTHON_BIN          Explicit Python interpreter (auto-detects venv / python3 otherwise).
#   AUTO_INSTALL_DEPS   0 (default) | 1 (pip install missing timm/numpy/pillow automatically).

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

REPO_UNLEARNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK_DIR="${TASK_DIR:-$PWD}"
OUT_DIR="${OUT_DIR:-$TASK_DIR/output/hybrid}"
MODE="${MODE:-full}"
STAGES="${STAGES:-all}"
SKIP_SHADOW="${SKIP_SHADOW:-0}"
FORCE="${FORCE:-0}"
AUTO_DOWNLOAD_COCO="${AUTO_DOWNLOAD_COCO:-1}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"
DEVICE="${DEVICE:-auto}"
AMP="${AMP:-auto}"

# ---------------------------------------------------------------------------
# 1. Resolve Python interpreter & verify dependencies
# ---------------------------------------------------------------------------
resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "$PYTHON_BIN"
    elif [[ -n "${UNLEARNING_PYTHON:-}" ]]; then
        echo "$UNLEARNING_PYTHON"
    elif [[ -x "$TASK_DIR/.venv/bin/python" ]]; then
        echo "$TASK_DIR/.venv/bin/python"
    elif [[ -x "$REPO_UNLEARNING_DIR/.venv/bin/python" ]]; then
        echo "$REPO_UNLEARNING_DIR/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        echo "ERROR: No Python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
        exit 1
    fi
}

PY="$(resolve_python)"

check_deps() {
    local need_timm="1"
    if [[ -f "$TASK_DIR/task_template.py" ]] && ! grep -q "timm" "$TASK_DIR/task_template.py"; then
        need_timm="0"
    fi
    local missing
    missing="$("$PY" - "$need_timm" <<'PY'
import importlib, sys
reqs = [("torch", "torch"), ("numpy", "numpy"), ("PIL", "pillow")]
if sys.argv[1] == "1":
    reqs.append(("timm", "timm"))
bad = []
for mod, pkg in reqs:
    try:
        importlib.import_module(mod)
    except Exception:
        bad.append(pkg)
print(" ".join(bad))
PY
)"
    if [[ -n "$missing" ]]; then
        if [[ "$AUTO_INSTALL_DEPS" == "1" ]]; then
            echo "[setup] Installing missing Python packages: $missing"
            "$PY" -m pip install -q $missing
        else
            echo "ERROR: Missing Python packages: $missing" >&2
            echo "Install them with: $PY -m pip install $missing (or re-run with AUTO_INSTALL_DEPS=1)" >&2
            exit 1
        fi
    fi
}

check_deps

# ---------------------------------------------------------------------------
# 2. Validate task inputs & detect hardware / AMP precision
# ---------------------------------------------------------------------------
if [[ ! -f "$TASK_DIR/model.pt" || ! -f "$TASK_DIR/forget_images.npz" ]]; then
    echo "ERROR: TASK_DIR ($TASK_DIR) must contain model.pt and forget_images.npz." >&2
    echo "Example: TASK_DIR=/content/task3 bash unlearning/run_pipeline.sh" >&2
    exit 1
fi

if [[ "$DEVICE" == "auto" ]]; then
    DEVICE="$("$PY" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")"
fi

if [[ "$AMP" == "auto" ]]; then
    if [[ "$DEVICE" == cuda* ]]; then
        AMP="$("$PY" -c "import torch; print('bf16' if torch.cuda.is_bf16_supported() else 'fp16')")"
    else
        AMP="fp32"
    fi
fi

mkdir -p "$OUT_DIR"

echo "=================================================================="
echo " Task 3 Hybrid Unlearning Pipeline"
echo "------------------------------------------------------------------"
echo " REPO_DIR   : $REPO_UNLEARNING_DIR"
echo " TASK_DIR   : $TASK_DIR"
echo " OUT_DIR    : $OUT_DIR"
echo " MODE       : $MODE (STAGES=$STAGES, SKIP_SHADOW=$SKIP_SHADOW, FORCE=$FORCE)"
echo " DEVICE/AMP : $DEVICE / $AMP"
echo " PYTHON     : $PY"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 3. Mode hyperparameters (full | fast | smoke)
# ---------------------------------------------------------------------------
case "$MODE" in
    full)
        DIAG_GRIDS="${SAVE_GRIDS:-20}"
        RETAIN_PER_CLASS="${PER_CLASS:-400}"
        RETAIN_MIN_PER_CLASS="${MIN_PER_CLASS:-20}"
        RETAIN_AUDIT="${AUDIT_COUNT:-500}"
        SHADOW_MODELS="${NUM_SHADOW_MODELS:-3}"
        SHADOW_EPOCHS="${SHADOW_EPOCHS:-12}"
        SHADOW_BS="${SHADOW_BS:-32}"
        KFOLD_FOLDS=5
        KFOLD_EPOCHS=6
        SALUN_EPOCHS=8
        HYBRID_EPOCHS=10
        LORA_EPOCHS=8
        PRUNE_EPOCHS=6
        POLISH_EPOCHS=2
        TRAIN_BLOCKS_W1=6
        TRAIN_BLOCKS_W2=8
        PRUNE_BLOCKS=12
        ORACLE_WEIGHT_SAMPLES=4
        ;;
    fast)
        DIAG_GRIDS="${SAVE_GRIDS:-8}"
        RETAIN_PER_CLASS="${PER_CLASS:-200}"
        RETAIN_MIN_PER_CLASS="${MIN_PER_CLASS:-10}"
        RETAIN_AUDIT="${AUDIT_COUNT:-400}"
        SHADOW_MODELS="${NUM_SHADOW_MODELS:-2}"
        SHADOW_EPOCHS="${SHADOW_EPOCHS:-6}"
        SHADOW_BS="${SHADOW_BS:-32}"
        KFOLD_FOLDS=3
        KFOLD_EPOCHS=4
        SALUN_EPOCHS=5
        HYBRID_EPOCHS=6
        LORA_EPOCHS=5
        PRUNE_EPOCHS=4
        POLISH_EPOCHS=1
        TRAIN_BLOCKS_W1=6
        TRAIN_BLOCKS_W2=8
        PRUNE_BLOCKS=10
        ORACLE_WEIGHT_SAMPLES=2
        ;;
    smoke)
        DIAG_GRIDS=0
        RETAIN_PER_CLASS=4
        RETAIN_MIN_PER_CLASS=1
        RETAIN_AUDIT=4
        SHADOW_MODELS=2
        SHADOW_EPOCHS=1
        SHADOW_BS=8
        KFOLD_FOLDS=2
        KFOLD_EPOCHS=1
        SALUN_EPOCHS=1
        HYBRID_EPOCHS=1
        LORA_EPOCHS=1
        PRUNE_EPOCHS=1
        POLISH_EPOCHS=1
        TRAIN_BLOCKS_W1=1
        TRAIN_BLOCKS_W2=2
        PRUNE_BLOCKS=2
        ORACLE_WEIGHT_SAMPLES=1
        ;;
    *)
        echo "ERROR: Unknown MODE='$MODE' (expected full, fast, or smoke)" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
stage_enabled() {
    local name="$1"
    [[ "$STAGES" == "all" ]] && return 0
    [[ ",$STAGES," == *",$name,"* ]]
}

needs_run() {
    local artifact="$1"
    [[ "$FORCE" == "1" || ! -f "$artifact" ]]
}

run_ul() {
    local script_name="$1"
    shift
    PYTHONPATH="$REPO_UNLEARNING_DIR:$TASK_DIR:${PYTHONPATH:-}" \
        "$PY" -u "$REPO_UNLEARNING_DIR/$script_name" "$@"
}

ensure_pool_path() {
    if [[ -n "${POOL_PATH:-}" ]]; then
        if [[ ! -e "$POOL_PATH" ]]; then
            echo "ERROR: Specified POOL_PATH does not exist: $POOL_PATH" >&2
            exit 1
        fi
        return 0
    fi

    if [[ -f "$TASK_DIR/coco224.npy" ]]; then
        POOL_PATH="$TASK_DIR/coco224.npy"
        return 0
    fi

    local coco_dir="$OUT_DIR/pools/val2017"
    if [[ -d "$coco_dir" ]] && find "$coco_dir" -maxdepth 1 -name '*.jpg' -print -quit | grep -q .; then
        POOL_PATH="$coco_dir"
        return 0
    fi

    if [[ "$AUTO_DOWNLOAD_COCO" != "1" ]]; then
        echo "ERROR: No POOL_PATH set and $TASK_DIR/coco224.npy not found (AUTO_DOWNLOAD_COCO=0)." >&2
        exit 1
    fi

    echo "[retain] Downloading COCO val2017 (~778 MB) into $OUT_DIR/pools ..."
    mkdir -p "$OUT_DIR/pools"
    local zip_path="$OUT_DIR/pools/val2017.zip"
    if [[ ! -f "$zip_path" ]]; then
        if command -v curl >/dev/null 2>&1; then
            curl -fL --progress-bar -o "$zip_path.tmp" "http://images.cocodataset.org/zips/val2017.zip"
            mv "$zip_path.tmp" "$zip_path"
        elif command -v wget >/dev/null 2>&1; then
            wget -q --show-progress -O "$zip_path.tmp" "http://images.cocodataset.org/zips/val2017.zip"
            mv "$zip_path.tmp" "$zip_path"
        else
            "$PY" - "$zip_path" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", sys.argv[1])
PY
        fi
    fi

    echo "[retain] Extracting COCO val2017 ..."
    "$PY" - "$zip_path" "$OUT_DIR/pools" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "r") as zf:
    zf.extractall(sys.argv[2])
PY
    rm -f "$zip_path"
    POOL_PATH="$coco_dir"
}

# ---------------------------------------------------------------------------
# Stage 0: Forensic Diagnostics
# ---------------------------------------------------------------------------
if stage_enabled diagnose; then
    mkdir -p "$OUT_DIR/diagnose"
    if needs_run "$OUT_DIR/diagnose/diagnostics.json"; then
        echo ""
        echo ">>> [Stage 0/6] Forensic Diagnostics (diagnose.py)"
        run_ul diagnose.py \
            --task-dir "$TASK_DIR" \
            --out-dir "$OUT_DIR/diagnose" \
            --save-grids "$DIAG_GRIDS" \
            --device "$DEVICE" \
            --amp "$AMP"
    else
        echo ">>> [Stage 0/6] diagnose already exists (set FORCE=1 to rerun)"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 1: Deduplicated Class-Balanced Retain Proxy
# ---------------------------------------------------------------------------
RETAIN_FIT="$OUT_DIR/retain/retain_fit.npy"
RETAIN_AUDIT_FILE="$OUT_DIR/retain/retain_audit.npy"

if stage_enabled retain; then
    mkdir -p "$OUT_DIR/retain"
    if needs_run "$RETAIN_FIT"; then
        echo ""
        echo ">>> [Stage 1/6] Build Retain Proxy (build_retain_proxy.py)"
        ensure_pool_path
        echo "[retain] Using POOL_PATH=$POOL_PATH"
        run_ul build_retain_proxy.py \
            --task-dir "$TASK_DIR" \
            --pool "$POOL_PATH" \
            --out-dir "$OUT_DIR/retain" \
            --per-class "$RETAIN_PER_CLASS" \
            --min-per-class "$RETAIN_MIN_PER_CLASS" \
            --audit-count "$RETAIN_AUDIT" \
            --dedup-cos "${DEDUP_COS:-0.85}" \
            --mixup-fallback \
            --device "$DEVICE" \
            --amp "$AMP"
    else
        echo ">>> [Stage 1/6] retain proxy already exists (set FORCE=1 to rerun)"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2: Shadow-Gold Reference Ensemble (trained strictly without forget)
# ---------------------------------------------------------------------------
SHADOW_GOLD="$OUT_DIR/shadow_gold.pt"

if stage_enabled shadow && [[ "$SKIP_SHADOW" != "1" ]]; then
    if [[ ! -f "$RETAIN_FIT" ]]; then
        echo "ERROR: Missing $RETAIN_FIT required for Shadow-Gold training." >&2
        exit 1
    fi
    if needs_run "$SHADOW_GOLD"; then
        echo ""
        echo ">>> [Stage 2/6] Train Shadow-Gold Ensemble (train_shadow_gold.py)"
        run_ul train_shadow_gold.py \
            --task-dir "$TASK_DIR" \
            --retain "$RETAIN_FIT" \
            --out "$SHADOW_GOLD" \
            --num-models "$SHADOW_MODELS" \
            --epochs "$SHADOW_EPOCHS" \
            --bs "$SHADOW_BS" \
            --lr "${SHADOW_LR:-1e-4}" \
            --device "$DEVICE" \
            --amp "$AMP"
    else
        echo ">>> [Stage 2/6] shadow_gold.pt already exists (set FORCE=1 to rerun)"
    fi
elif [[ "$SKIP_SHADOW" == "1" ]]; then
    echo ">>> [Stage 2/6] Skipping Shadow-Gold training (SKIP_SHADOW=1)"
fi

# ---------------------------------------------------------------------------
# Stage 3: Memorization Scoring & Hybrid Gold-Target Oracle
# ---------------------------------------------------------------------------
ORACLE_FILE="$OUT_DIR/oracle/oracle_targets.pt"

if stage_enabled oracle; then
    mkdir -p "$OUT_DIR/oracle"
    if needs_run "$ORACLE_FILE"; then
        echo ""
        echo ">>> [Stage 3/6] Build Memorization & OOF Oracle (build_oracle.py)"
        ORACLE_EXTRA=()
        [[ -f "$SHADOW_GOLD" ]] && ORACLE_EXTRA+=(--shadow-gold "$SHADOW_GOLD")
        [[ -n "${EXT_EMBEDDINGS:-}" && -f "${EXT_EMBEDDINGS}" ]] && ORACLE_EXTRA+=(--ext-embeddings "$EXT_EMBEDDINGS")
        run_ul build_oracle.py \
            --task-dir "$TASK_DIR" \
            --out "$ORACLE_FILE" \
            --num-weight-samples "$ORACLE_WEIGHT_SAMPLES" \
            "${ORACLE_EXTRA[@]}" \
            --device "$DEVICE" \
            --amp "$AMP"
    else
        echo ">>> [Stage 3/6] oracle_targets.pt already exists (set FORCE=1 to rerun)"
    fi
fi

# Shared optional arguments for training waves
SHARED_OPT=()
[[ -f "$RETAIN_AUDIT_FILE" ]] && SHARED_OPT+=(--retain-audit "$RETAIN_AUDIT_FILE")
[[ -f "$ORACLE_FILE" ]] && SHARED_OPT+=(--oracle "$ORACLE_FILE")
[[ -f "$SHADOW_GOLD" ]] && SHARED_OPT+=(--shadow-gold "$SHADOW_GOLD")

# ---------------------------------------------------------------------------
# Stage 4: Wave 1 — K-Fold Cross-Fitted TIES + SalUn Late-MLP
# ---------------------------------------------------------------------------
if stage_enabled wave1; then
    mkdir -p "$OUT_DIR/wave1_kfold" "$OUT_DIR/wave1_salun"

    if needs_run "$OUT_DIR/wave1_kfold/merged_best.pt"; then
        echo ""
        echo ">>> [Stage 4a/6] Wave 1a: ${KFOLD_FOLDS}-Fold Cross-Fitted TIES Unlearning (kfold_unlearn.py)"
        KFOLD_EXTRA=("${SHARED_OPT[@]}")
        [[ -f "$RETAIN_FIT" ]] && KFOLD_EXTRA+=(--retain "$RETAIN_FIT")
        run_ul kfold_unlearn.py \
            --task-dir "$TASK_DIR" \
            --out-dir "$OUT_DIR/wave1_kfold" \
            --folds "$KFOLD_FOLDS" \
            --merge-method ties \
            --alphas "0.8,1.0,1.2,1.5" \
            --scope late_mlp \
            --train-blocks "$TRAIN_BLOCKS_W1" \
            --epochs "$KFOLD_EPOCHS" \
            --lr 3e-5 \
            --w-retain 1.5 \
            --w-ewc 1.0 \
            --w-moment 0.25 \
            --w-npo 0.15 \
            --merged-polish-epochs 1 \
            "${KFOLD_EXTRA[@]}" \
            --device "$DEVICE" \
            --amp "$AMP"
    else
        echo ">>> [Stage 4a/6] wave1_kfold/merged_best.pt already exists (set FORCE=1 to rerun)"
    fi

    if [[ -f "$RETAIN_FIT" ]]; then
        if needs_run "$OUT_DIR/wave1_salun/salun_mlp.pt"; then
            echo ""
            echo ">>> [Stage 4b/6] Wave 1b: SalUn Late-MLP Hybrid (hybrid_unlearn.py)"
            run_ul hybrid_unlearn.py \
                --task-dir "$TASK_DIR" \
                --retain "$RETAIN_FIT" \
                --out "$OUT_DIR/wave1_salun/salun_mlp.pt" \
                --scope late_mlp \
                --train-blocks "$TRAIN_BLOCKS_W1" \
                --salun-ratio 0.5 \
                --salun-mode relative_ssd \
                --epochs "$SALUN_EPOCHS" \
                --lr 3e-5 \
                --w-retain 1.5 \
                --w-feat-retain 0.5 \
                --w-ewc 1.0 \
                --polish-epochs 1 \
                "${SHARED_OPT[@]}" \
                --device "$DEVICE" \
                --amp "$AMP"
        else
            echo ">>> [Stage 4b/6] wave1_salun/salun_mlp.pt already exists (set FORCE=1 to rerun)"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Stage 5: Wave 2 — Diverse Multi-Paradigm Unlearning Branches
# ---------------------------------------------------------------------------
if stage_enabled wave2; then
    mkdir -p "$OUT_DIR/wave2_branches"
    if [[ ! -f "$RETAIN_FIT" ]]; then
        echo "ERROR: Missing $RETAIN_FIT required for Wave 2 branches." >&2
        exit 1
    fi

    if needs_run "$OUT_DIR/wave2_branches/branch_full_hybrid.pt"; then
        echo ""
        echo ">>> [Stage 5a/6] Wave 2a: Full Hybrid (SSD + SalUn + Wasserstein + NPO + Smooth + SCAR + RUM)"
        run_ul hybrid_unlearn.py \
            --task-dir "$TASK_DIR" \
            --retain "$RETAIN_FIT" \
            --out "$OUT_DIR/wave2_branches/branch_full_hybrid.pt" \
            --scope late_mlp_attn_proj \
            --train-blocks "$TRAIN_BLOCKS_W2" \
            --ssd-warmstart \
            --salun-ratio 0.6 \
            --salun-mode relative_ssd \
            --epochs "$HYBRID_EPOCHS" \
            --lr 3e-5 \
            --w-kd-t2 1.0 \
            --w-kd-t1 0.35 \
            --w-moment 0.35 \
            --moment-mode wasserstein \
            --w-npo 0.20 \
            --w-smooth 0.30 \
            --w-scar 0.15 \
            --w-retain 1.5 \
            --w-feat-retain 0.5 \
            --w-head-gaussian 0.3 \
            --w-ewc 1.0 \
            --rum-curriculum \
            --polish-epochs "$POLISH_EPOCHS" \
            "${SHARED_OPT[@]}" \
            --device "$DEVICE" \
            --amp "$AMP"
    else
        echo ">>> [Stage 5a/6] branch_full_hybrid.pt already exists (set FORCE=1 to rerun)"
    fi

    if needs_run "$OUT_DIR/wave2_branches/branch_lora_ogp.pt"; then
        echo ""
        echo ">>> [Stage 5b/6] Wave 2b: Mergeable LoRA + Orthogonal Gradient Projection"
        run_ul hybrid_unlearn.py \
            --task-dir "$TASK_DIR" \
            --retain "$RETAIN_FIT" \
            --out "$OUT_DIR/wave2_branches/branch_lora_ogp.pt" \
            --train-blocks "$TRAIN_BLOCKS_W2" \
            --lora-rank 16 \
            --lora-alpha 32 \
            --ogp-rank 64 \
            --ogp-var 0.95 \
            --epochs "$LORA_EPOCHS" \
            --lr 1e-4 \
            --w-moment 0.25 \
            --w-npo 0.15 \
            --w-smooth 0.25 \
            --w-retain 1.5 \
            --polish-epochs 1 \
            "${SHARED_OPT[@]}" \
            --device "$DEVICE" \
            --amp "$AMP"
    else
        echo ">>> [Stage 5b/6] branch_lora_ogp.pt already exists (set FORCE=1 to rerun)"
    fi

    if needs_run "$OUT_DIR/wave2_branches/branch_prune_heal.pt"; then
        echo ""
        echo ">>> [Stage 5c/6] Wave 2c: L1-Sparse Prune Warmstart + Retain-Heal"
        run_ul hybrid_unlearn.py \
            --task-dir "$TASK_DIR" \
            --retain "$RETAIN_FIT" \
            --out "$OUT_DIR/wave2_branches/branch_prune_heal.pt" \
            --scope mlp_only \
            --train-blocks "$PRUNE_BLOCKS" \
            --prune-warmstart \
            --prune-sparsity 0.15 \
            --epochs "$PRUNE_EPOCHS" \
            --lr 2e-5 \
            --w-kd-t2 0.5 \
            --w-moment 0.20 \
            --w-npo 0.0 \
            --w-retain 2.0 \
            --w-feat-retain 0.75 \
            --w-ewc 1.2 \
            --polish-epochs "$POLISH_EPOCHS" \
            "${SHARED_OPT[@]}" \
            --device "$DEVICE" \
            --amp "$AMP"
    else
        echo ">>> [Stage 5c/6] branch_prune_heal.pt already exists (set FORCE=1 to rerun)"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 6: Weight-Space Soup + Coordinate Layerwise Gamma Search
# ---------------------------------------------------------------------------
if stage_enabled soup; then
    mkdir -p "$OUT_DIR/final"
    CANDIDATES=()
    for ckpt in \
        "$OUT_DIR/wave1_kfold/merged_best.pt" \
        "$OUT_DIR/wave1_salun/salun_mlp.pt" \
        "$OUT_DIR/wave2_branches/branch_full_hybrid.pt" \
        "$OUT_DIR/wave2_branches/branch_lora_ogp.pt" \
        "$OUT_DIR/wave2_branches/branch_prune_heal.pt"; do
        [[ -f "$ckpt" ]] && CANDIDATES+=("$ckpt")
    done

    if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
        echo "ERROR: No candidate checkpoints found under $OUT_DIR for Stage 6." >&2
        exit 1
    fi

    echo ""
    echo ">>> [Stage 6/6] Multi-Branch Weight Soup & Submission Export (soup_and_eval.py)"
    SOUP_EXTRA=()
    [[ -f "$RETAIN_AUDIT_FILE" ]] && SOUP_EXTRA+=(--retain-audit "$RETAIN_AUDIT_FILE")
    [[ -f "$SHADOW_GOLD" ]] && SOUP_EXTRA+=(--shadow-gold "$SHADOW_GOLD")

    run_ul soup_and_eval.py \
        --task-dir "$TASK_DIR" \
        --models "${CANDIDATES[@]}" \
        --search-weights \
        --search-gammas \
        --out "$OUT_DIR/final/soup_submission.pt" \
        --metrics-out "$OUT_DIR/final/soup_metrics.json" \
        "${SOUP_EXTRA[@]}" \
        --device "$DEVICE" \
        --amp "$AMP"
fi

echo ""
echo "=================================================================="
echo " Pipeline complete!"
if [[ -f "$OUT_DIR/final/soup_submission.pt" ]]; then
    echo " Final submission : $OUT_DIR/final/soup_submission.pt"
    echo " Metrics JSON     : $OUT_DIR/final/soup_metrics.json"
fi
echo "=================================================================="
