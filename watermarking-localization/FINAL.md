# Final submission procedure (deadline 11:00, target 10:00)

The best public submission counts, the server rate-limits submissions (HTTP 429): submit only validated
jumps, at least 20 min apart. All commands run on JURECA from `watermarking-localization/`.

## Principle

Weights and every choice are fitted on f0 with models that never saw f0. The all-folds retrains (E165,
E167, E168, E169) saw f0, so they are never scored or weighted on f0: each one *replaces* its f0
counterpart at test time, with the counterpart's weight (`wl.ensemble --test-runs`).

| f0-validated member (weights) | test-time replacement (all folds) |
|---|---|
| E161_tta (W8 no interleave) | E169_tta (E161 warm start, all folds) |
| E160_tta (W8) | E165_tta (W8, all folds) |
| E143_tta (W2 ResNet-50) | E167_tta (W2 ResNet-50, all folds) |
| E166_tta (W8 seed 1) | E168_tta (W8 seed 1, all folds) |
| E142_w2_long, wave 2 *_tta | themselves |

## Steps

1. TTA of every finished member (idle GPU of a running job, no queue):
   `scripts/run_on_job.sh <jobid> <gpu> "python -m wl.infer --run output/E165_w8_allfolds --id E165_tta"`
   (same for E166, E167, E168, E169 when they are finished).
2. Validated ensemble (shared weights) and its all-folds test version:
   ```
   python -m wl.ensemble --runs output/E161_tta output/E143_tta output/E160_tta output/E166_tta output/E142_w2_long [wave-2 *_tta] \
       --test-runs output/E169_tta output/E167_tta output/E165_tta output/E168_tta output/E142_w2_long [same wave-2] \
       --out output/E199_final_ens --mode logit --global-weights
   ```
   Keep a member only if the f0 score of the ensemble does not drop when it is added.
3. Verify (PS, TM) with the size gate (E182: oversized refined boxes are containing boxes, +0.007 on f0):
   `scripts/run_on_job.sh <jobid> <gpu> "python -m wl.verify --id E199_final --sources output/E199_final_ens output/E121_analytic_rect_v2 --kind output/E120_kind_const/kind_probs.npz --methods pixelseal trustmark --max-area-q 0.9"`
   If verify already ran without the gate, apply it on CPU (seconds, no GPU) and check the paired difference:
   `python -m wl.verify --id E199_final_g --regate output/E199_final --max-area-q 0.9 && python -m wl.compare output/E199_final output/E199_final_g`
4. Submit if f0 beats the last submitted candidate: `source .env && ./scripts/runs/submit.sh output/E199_final --submit`
5. Copy the submitted file to `/p/scratch/training2666/chander2/argos/submissions/` and log it in EXPERIMENTS.md.
