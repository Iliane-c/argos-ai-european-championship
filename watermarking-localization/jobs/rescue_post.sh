#!/bin/bash
# After the rescue TTA: greedy selection on f0 (shared weights, all-folds test twins), gated verify,
# paired comparison with the current best. Never submits. Summary: output/RESCUE_SUMMARY.txt
cd "$(dirname "$0")/.." && source .venv/bin/activate
python -m wl.greedy --cand E161_tta:E169_tta E143_tta:E167_tta E160_tta:E165_tta E166_tta:E168_tta E142_tta E195_tta E194_tta \
  E192_w8ns_35ep_tta E191_w8ns_lovasz_tta E190_w8ns_seed1_tta E193_w8ns_seed4_tta E164_tta \
  E200_w2_r50_seed1_tta E201_w8ns_r34_tta E202_w8ns_synth60_tta --out E220_rescue_ens
python -m wl.verify --id E221_rescue_final --sources output/E220_rescue_ens output/E121_analytic_rect_v2 \
  --kind output/E120_kind_const/kind_probs.npz --methods pixelseal trustmark --max-area-q 0.9
{ date; python -m wl.compare output/E198g_final_gate output/E221_rescue_final; python -m wl.report | grep -E "E198g|E220|E221|_tta"; } > output/RESCUE_SUMMARY.txt 2>&1
