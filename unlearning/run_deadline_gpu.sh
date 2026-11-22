#!/usr/bin/env bash
# Run from the task directory with model.pt, task_template.py and forget_images.npz.
# GPU 0: CUDA_VISIBLE_DEVICES=0 bash /path/to/run_deadline_gpu.sh class
# GPU 1: CUDA_VISIBLE_DEVICES=1 bash /path/to/run_deadline_gpu.sh crop /path/to/scored-parent.pt
set -euo pipefail
scripts=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mode=${1:-}
retain=${RETAIN:-retain_mix.npy}
python=${PYTHON:-python}
if [[ "$mode" != class && "$mode" != crop ]]; then echo 'Expected class or crop' >&2; exit 2; fi
parent=${2:-output/t3-012/dep6rep_g1.20_head1.21.pt}
for input in task_template.py model.pt forget_images.npz "$retain"; do
    [[ -r "$input" ]] || { echo "Missing $input" >&2; exit 2; }
done
if [[ "$mode" == crop && ! -r "$parent" ]]; then echo "Missing $parent" >&2; exit 2; fi
mkdir -p output
out=$(mktemp -d "output/deadline-$mode-XXXXXX")
echo "Writing logs and checkpoints to $out"
exec > >(tee "$out/run.log") 2>&1
train=("$python" -u "$scripts/shift_distill.py" --retain "$retain" --amp "${AMP:-fp16}")
interp=("$python" -u "$scripts/interpolate.py" --retain "$retain" --export "$out")
repair=(--delta 0 --train-blocks 1 --epochs 2 --lr 1e-5 --retain-weight 3)
if [[ "$mode" == class ]]; then
    "${train[@]}" --class-depth --depth-weight 1 --depth-frac 0.6 --depth-lo 6 --depth-hi 16 \
        --delta 12 --margin-power -3 --target-scale 2.5 --out "$out/class.pt"
    "${interp[@]}" "$out/class.pt" --gammas 1.15,1.20,1.225,1.245,1.265,1.30,1.35 --match-ce 2.40
    "${train[@]}" "${repair[@]}" --repair "$out/class_matched.pt" --out "$out/class_repaired.pt"
    "${interp[@]}" "$out/class_repaired.pt" --gammas 1.25,1.30,1.35,1.40,1.45 --match-ce 3.54
else
    "${train[@]}" "${repair[@]}" --repair "$parent" --crop-weight 0.5 --out "$out/crop_repaired.pt"
fi
