#!/bin/bash
# Run ONCE per team on a JURECA LOGIN node (compute nodes have no internet).
# Downloads kit watermark models + TrustMark weights + ImageNet/timm encoder weights.
#
# Usage:
#   ./scripts/setup/02_download_weights.sh /p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task

set -euo pipefail

TEAM_TASK="${1:-${TEAM_TASK:-}}"
if [[ -z "$TEAM_TASK" ]]; then
    echo "Usage: $0 <path-to-shared-Hackathon_Watermark_Localisation_Task>" >&2
    exit 1
fi
if [[ ! -f "$TEAM_TASK/.venv/bin/activate" ]]; then
    echo "ERROR: Missing $TEAM_TASK/.venv. Run 01_jureca_shared_env.sh first." >&2
    exit 1
fi

cd "$TEAM_TASK"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[1/3] Downloading starter kit watermark models (PixelSeal, StegaStamp)..."
python kit/download_models.py

echo "[2/3] Triggering TrustMark weight download..."
python -c "import sys; sys.path.insert(0, 'kit'); from wmloc.watermarks import load_watermark; load_watermark('trustmark', device='cpu')"

echo "[3/3] Pre-caching ImageNet/timm encoders for offline compute nodes..."
python - <<'PY'
import segmentation_models_pytorch as smp
import timm

smp_encoders = [
    ("Unet", "resnet34"),
    ("Segformer", "mit_b2"),
    ("Segformer", "mit_b5"),
    ("Unet", "tu-convnextv2_base"),
]
for arch, enc in smp_encoders:
    try:
        getattr(smp, arch)(encoder_name=enc, encoder_weights="imagenet")
        print(f"  cached smp {arch}/{enc}")
    except Exception as exc:
        print(f"  warning: could not cache smp {arch}/{enc}: {exc}")

timm_encoders = [
    "hrnet_w48",
    "convnextv2_base.fcmae_ft_in22k_in1k_384",
]
for name in timm_encoders:
    try:
        timm.create_model(name, pretrained=True, features_only=True)
        print(f"  cached timm {name}")
    except Exception as exc:
        print(f"  warning: could not cache timm {name}: {exc}")
PY

echo "Done! Next each teammate runs: ./scripts/setup/03_link_workspace.sh '$TEAM_TASK'"
