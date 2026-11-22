# Experiments: Task 1, Watermark Localization

This is the shared experiment log for Task 1. **Newest entries first.** Add one row per experiment as soon
as it is launched (status `running`), then fill in the result, including negative results. How we work
(base library, evaluation, rounds) is in [PLAN.md](PLAN.md).

## Team

There are no fixed roles: either of us runs any experiment. The "builds" column only splits the base
library (round 0).

| Person | Name | Experiment IDs | Builds in round 0 |
|---|---|---|---|
| 1 | | E100–E199 | B1 data, B5 synth + augment, B6 features, B7 analytic |
| 2 | | E200–E299 | B2 evaluate, B3 models, B4 train + losses, B8 postproc + ensemble + predict, B9 jobs + report + gallery |

## Task summary

**Goal:** maximize mIoU (classes 0 and 1, computed per image, then averaged) on 1,020 test images. Masks
are evaluated on a 128×128 grid.

| Item | Details |
|---|---|
| Training data | 510 images (170 per method) with ground-truth mask, method, embedded message and `full_watermark` flag |
| Test data | 1,020 images with method and embedded message (no flag, no mask) |
| Two kinds of images | `full` (50 %): whole image watermarked except clean holes (median 6 %). `patch` (50 %): small watermarked patch (median 4 %); half rectangles, half objects. See `analysis/explore_dataset.py`. |
| Edit types | Outpainting, object removal, object addition, object transplantation |
| Geometry | Watermark applied to a rectangular region; aspect ratio ≤ 16:1 (PixelSeal, StegaStamp), ≤ 2:1 (TrustMark) |
| Leaderboard | Public: 30 % of the test set. Final ranking: the other 70 %. Do not tune thresholds on the public score. |

### Method notes

- **PixelSeal:**
  - Watermark computed at 256×256, upscaled and modulated by a JND map.
  - Luma (Y) channel only.
  - ConvNeXt-Tiny extractor with mean pooling and a linear head, so per-position logits are exact.
  - 256-bit payload. The signal is concentrated on object edges.
- **TrustMark (variant Q, 100 bits, no ECC):**
  - 256×256 residual rescaled to the image size.
  - Center crop when the aspect ratio is > 2; the dataset only uses ≤ 2.
  - 1 % feathered paste.
  - Removal network (TrustMark-RM) available.
- **StegaStamp:**
  - Embedding at 400×400; the decoder resizes to 400×400 (bicubic). Both confirmed in the kit wrapper.
  - Dense decoder head, so localization is weak by design.

### Working hypothesis

- **Main track:** segmentation networks trained on real data plus synthetic data made with the kit
  embedders, validated on fixed folds.
- **Most promising:** watermark-specific architectures (W1–W8 in PLAN.md §5).
- **Training-free track:** methods that use the known message (box decoding, agreement maps, matched
  filter). They give baselines, extra input channels and ensemble members.
- **E01 test:** decoding the exact watermarked rectangle should give the highest bit accuracy, because each
  watermark is generated at a fixed resolution and rescaled to its rectangle. E01 checks this first.

## Logging conventions

1. **ID:** from your own block (Person 1: E100–E199, Person 2: E200–E299; shared baselines: E00x).
   Reuse it in `configs/<ID>_<name>.yaml`, `output/<ID>_<name>/` and the commit message.
2. **Claim before launching:** add the row with status `running`, then `git pull --rebase` and push.
3. **Hypothesis:** one sentence: what is tested and the expected outcome.
4. **Parent:** the experiment this one changes. Make exactly one change.
5. **Data:**
   - `real`: real folds 1–4.
   - `synth-vN`: synthetic version N.
   - `real+synth-vN`: both.
   - `none`: training-free.
6. **Val:**
   - `f0`: real fold 0, 102 images.
   - `cv5`: 5-fold mean ± std.
   - `all510`: all real train images. Only allowed when no real image was used for training or tuning.
7. **Numbers:** from `metrics.json` (official metric, threshold 0.5 unless stated). Report PS / TM / SS,
   Full / Patch, and the mean. Run `python -m wl.report` to rank all runs.
8. **Verdict:** **Keep**, **Drop** or **Investigate**, with a one-line justification. A gain below 1 CV
   std counts as noise.

## Experiment log

| ID | Time | Owner | Parent | Change | Hypothesis | Data | Val | PS | TM | SS | Full | Patch | Mean | Public | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E180 | 11-22 | 1 | E104 | `wl.refine`: kind gating from cached whole-image bit accuracy | Wrong-kind maps cost ~0.5 each; decoding fixes them for free | none | f0 | | | | | | | | planned |
| E173 | 11-22 | 1 | E170 | HRNet-W48 + FPN | High-res stream helps faint patterns; ensemble diversity | real+synth-v2 | f0 | | | | | | | | planned |
| E172 | 11-22 | 1 | E103 | synth-v2, 35 epochs | Same W2 recipe gains from data diversity | real+synth-v2 | f0 | | | | | | | | planned |
| E171 | 11-22 | 1 | E170 | SegFormer-B2 | Direct comparison with E104: gain of synth-v2 + fixes | real+synth-v2 | f0 | | | | | | | | planned |
| E170 | 11-22 | 1 | E104 | synth-v2 (COCO bases, restore/paste/inpaint holes), no wrong-message negatives, no cached maps, SegFormer-B5 | Data diversity + capacity lift PixelSeal and `full` holes | real+synth-v2 | f0 | | | | | | | | planned |
| E140_w3_noneg, E141_w3_rgbsrm, E142_w3_b5, E143_w2_40ep | 11-22 | 1 | E104/E103 | Ablations on synth-v1: no negatives; + drop agreement; + SegFormer-B5; W2 40 epochs. Numbers clash with E141_w2_msgmaps / E142_w2_long / E143_w2_r50 below; output dirs differ | The agreement channel is constant on synth and misaligned on real; negatives add label noise | real+synth-v1 | f0 | | | | | | | | running |
| E151 | 11-22 | 2 | E143 | `wl.verify` on E143 with E121 boxes, `--methods pixelseal trustmark` | Verification gains (E150) stack on the best model | none | f0 | | | | | | | | planned |
| E145_w2_ps_only | 11-22 | 2 | E103 | W2 trained on PixelSeal only (`data.methods: [pixelseal]`) | A specialist lifts the weakest method (PS ~0.50) | real+synth-v1 | f0 | | | | | | | | planned |
| E143_w2_r50 + finalize --ac-z | 11-22 | 2 | E143 | `wl.finalize` cell calibrator: model logit + kind + decoder z (a-contrario), expected-IoU decision, cross-fitted | Product-of-experts fusion of model and decoder evidence | none | f0 | | | | | | | | planned |
| E150_verify_e103 | 11-22 | 2 | E103 | `wl.verify`: re-decode candidate boxes at native scale, z vs 256 random keys, coordinate-ascent refinement; keep model map when its own box verifies (z ≥ 4) | Native-scale decoding confirms/repairs patch boxes | none | f0 ∩ all510 | 0.502 | 0.624 | 0.751 | 0.602 | 0.650 | 0.626 | | Helps PS (+0.03) and TM, hurts SS: apply to PS/TM only |
| E143_w2_r50 | 11-22 | 2 | E103 | W2 with ResNet-50 encoder | More capacity for faint PixelSeal / holes | real+synth-v1 | f0 | 0.496 | 0.616 | 0.817 | 0.627 | 0.659 | 0.643 | | **Best single model** (OOF 0.6456) |
| E142_w2_long | 11-22 | 2 | E103 | W2, 50 epochs | E103 under-trained | real+synth-v1 | f0 | | | | | | | | running (job 15904353) |
| E141_w2_msgmaps | 11-22 | 2 | E103 | W2 + cached agreement/matched maps as input | Decoder evidence as extra channels | real+synth-v1 | f0 | | | | | | 0.616 | | No gain (channel constant on synth, see Decisions) |
| E103_w2_wam_multitask | 11-22 | 1 | — | WAM-style multitask: detection + per-cell bits vs known message | Message-aware model beats plain segmentation | real+synth-v1 | f0 | 0.468 | 0.621 | 0.775 | 0.602 | 0.640 | 0.621 | 0.621 | Baseline for W2; finalize (kind) cross-fitted 0.6214±0.029, no gain |
| E102_segformer_b2 | 11-21 | 2 | — | SegFormer-B2 segmentation | Plain segmentation baseline | real | f0 | | | | | | 0.407 | 0.4239 | Weak (OOF 0.4315) |
| E101_unet_r34 | 11-21 | 2 | — | U-Net ResNet-34 segmentation | Plain segmentation baseline | real | f0 | | | | | | 0.421 | | Weak |
| E132_ac_hybrid | 11-22 | 2 | E121 | A-contrario per-cell z (true key vs 256 random keys) + kind hybrid | Calibrated p-values beat raw agreement | none | all510 | 0.488 | 0.535 | 0.457 | 0.455 | 0.532 | 0.493 | | Raw (not cross-fitted); ≈ E121 |
| E127_scan_hybrid | 11-22 | 2 | E120 | `wl.scan`: Gaussian scan statistic over rectangles on decoder evidence | Scan statistic localizes patches | none | all510 | | | | | | 0.447 ± 0.006 | | Honest (cross-fitted); no gain over constant |
| E121_analytic_rect_v2 | 11-22 | 2 | E100 | Analytic decoder, multi-scale rectangle search (sizes 6–128, top-5 boxes) | Native-scale boxes recover patches | none | all510 | 0.490 | 0.534 | 0.452 | 0.451 | 0.533 | 0.492 ± 0.019 | | Best training-free (honest) |
| E120_kind_const | 11-22 | 2 | — | `wl.kind_hybrid`: per-method logistic kind classifier (OOF acc 0.978) + constant maps | Kind alone ≈ oracle E00d | none | all510 | 0.449 | 0.450 | 0.456 | 0.450 | 0.454 | 0.452 ± 0.005 | 0.4451 | Matches E00d; training-free floor |
| E100_analytic_t1 | 11-21 | 1 | — | Analytic decoder map, threshold | Training-free baseline | none | all510 | | | | | | 0.450 | | Honest |
| E01 | | | — | Decode the ground-truth bounding box vs. the full image vs. the box shifted by 32 px (report bit accuracy, not mIoU) | The exact box yields markedly higher bit accuracy | none | all510 | | | | | | | | |
| E00e | 11-21 | | — | Oracle: ground-truth bounding box of the part to find | Shows how much a box-only prediction can reach | none | all510 | 0.718 | 0.692 | 0.714 | 0.549 | 0.867 | 0.708 | — | Reference (upper bound for boxes) |
| E00d | 11-21 | | — | Oracle: the right constant per image (all-ones for `full`, all-zeros for `patch`) | What a perfect kind classifier alone gives | none | all510 | 0.452 | 0.452 | 0.460 | 0.453 | 0.456 | 0.455 | — | Reference: the rest of the score comes from the small part |
| E00c | 11-21 | | — | Mean train mask per method, threshold 0.5 (in-sample) | Location prior alone | none | all510 | 0.300 | 0.308 | 0.301 | 0.277 | 0.329 | 0.303 | — | Floor |
| E00b | 11-21 | | — | All ones | Score floor | none | all510 | 0.247 | 0.252 | 0.246 | 0.453 | 0.044 | 0.248 | — | Floor |
| E00a | 11-21 | | — | All zeros | Score floor | none | all510 | 0.253 | 0.248 | 0.254 | 0.047 | 0.456 | 0.252 | — | Floor |

## Round 0: base library (17:00–20:00)

| # | Module | Builder | Status |
|---|---|---|---|
| B1 | `wl/data.py`: real dataset with folds, shard dataset, augmentation hook, extra channels | 1 | done |
| B2 | `wl/evaluate.py`: official metric, per method × kind, out-of-fold threshold, `metrics.json` | 2 | done |
| B3 | `wl/models.py`: registry (smp, HF, timm) + wrappers (channels, method/message conditioning, extra heads) | 2 | done |
| B4 | `wl/train.py`, `wl/losses.py`: YAML-driven training, writes `val_probs` / `test_probs` | 2 | done |
| B5 | `wl/synth.py`, `wl/augment.py`: generator, mask library, watermark-safe augmentation, copy-paste | 1 | done |
| B6 | `wl/features.py`: decoder agreement, matched filter, TrustMark-RM residual, SRM; cached | 1 | done |
| B7 | `wl/analytic.py`: training-free localizers writing the standard probability files | 1 | done |
| B8 | `wl/postproc.py`, `wl/ensemble.py`, `predict.py`: thresholds, clean-up, ensembles, submission | 2 | done |
| B9 | `jobs/run.sh`, `jobs/sweep.py`, `wl/report.py`, `wl/gallery.py` | 2 | done |
| ✔ | End-to-end CPU smoke test at 18:30: real data → U-Net → probs → `metrics.json` → `submission.pt` | both | done |

## Backlog (anyone claims a row by putting their name in Owner)

**Data (synthetic and augmentation)**

| # | Idea | Owner | Status |
|---|---|---|---|
| D1 | synth-v1: `full` = embed the whole image + clean holes; `patch` = embed a box + optional object mask | | |
| D2 | Real only vs real + synth-v1 vs synth-v1 only (scored on all510) | | |
| D3 | Copy-paste between real images (integer shift, no resize) | | |
| D4 | Instagram filters (`pilgram`) / color / JPEG **before** embedding vs **after** embedding (A/B) | | |
| D5 | synth-v2: holes inpainted (LaMa / SD-inpaint), outpainting and transplantation done with real editing models | | |
| D6 | Synthetic-to-real ratio and volume (5k / 20k / 50k per method) | | |

**Models (PLAN.md §5)**

| # | Idea | Owner | Status |
|---|---|---|---|
| M1 | T1 baselines: U-Net ResNet-34, SegFormer-B2 (real data, f0) | | |
| M2 | T2: SegFormer-B5, UPerNet + ConvNeXt-V2 / Swin-V2, HRNet-W48, Mask2Former | | |
| M3 | T4 forensic: TruFor fine-tune; IML-ViT / Mesorch / SparseViT via IMDL-BenCo | | |
| M4 | Inputs: RGB vs + SRM vs + agreement / matched-filter maps vs + `native_view` | | |
| M5 | One model per method vs a shared model with a method input | | |
| M6 | Loss: BCE + Dice vs per-image Lovász vs + boundary loss; kind head | | |

**Watermark-specific architectures (PLAN.md §5, T5)**

| # | Idea | Owner | Status |
|---|---|---|---|
| W1 | Unpooled decoder: own extractor without pooling + message agreement + FPN head | | |
| W2 | WAM-style multi-task: per-pixel detection + bits (optionally from WAM weights) | | |
| W3 | Message-conditioned segmenter (FiLM / cross-attention on bits, trained with wrong-message negatives) | | |
| W4 | Reference two-stream + correlation volume (`x` high-pass vs `embed(x, m) − x`) | | |
| W5 | Auxiliary residual regression `x − c` on synthetic data | | |
| W6 | Two-stage: find the box, segment at native scale | | |
| W7 | Dual-stream noise (SRM / Bayar, full resolution) + semantic (DINOv3 / SAM 2) | | |
| W8 | SAM 2 refiner prompted with the coarse map | | |

**Beyond training (PLAN.md §6)**

| # | Idea | Owner | Status |
|---|---|---|---|
| T1 | Training-free maps: decoder agreement (PS, TM), blurred-patch occlusion (SS), matched filter, TrustMark-RM; whole-image decode for the kind | | |
| T2 | Rectangle search (coarse 64 px grid, then 4 px refinement, then matched-filter refinement) maximizing the log-likelihood of the known bits | | |
| T3 | Test-time optimization through the differentiable decoders (box / mask refinement) | | |
| T4 | Pseudo-labels on test (Noisy Student), **only if the organizers allow it** | | |
| T5 | Watermark-safe TTA: integer pixel shifts | | |
| T6 | Post-processing on out-of-fold predictions: thresholds per method × kind, never-empty small part, rectangle snap, morphology, SAM 2 snap | | |
| T7 | Ensembles and stacking (small CNN on probs + T1 maps, trained out-of-fold) | | |
| T8 | Region-wise binomial p-values on known-bit agreement (calibrated maps); multi-scale sliding-window decoding + linear probe (SS) | | |

## Leaderboard submissions

| Time | Owner | Experiment | CV (f0 / cv5) | Public score | File / commit |
|---|---|---|---|---|---|
| 11-22 | 1 | E103_w2_wam_multitask | 0.621 / — | 0.621 | backup in `/p/scratch/training2666/chander2/argos/submissions` |
| 11-22 | 2 | E120_kind_const | 0.452 (honest all510) | 0.4451 | below best, leaderboard unchanged |
| 11-22 | 2 | E102_segformer_b2 | 0.407 / OOF 0.432 | 0.4239 | below best |
| 11-22 | 2 | E143_w2_r50 | 0.643 / OOF 0.646 | pending | next submission |

## Decisions

- 2026-11-21: No fixed roles. We split the base library build only; afterwards anyone runs any experiment.
  ID blocks: E100–E199 and E200–E299.
- 2026-11-21: Main track is segmentation networks on real + synthetic data, plus the training-free track.
  Everything writes the same probability files, so anything can be evaluated and ensembled.
- 2026-11-21: Fixed 5-fold split in `splits/folds.json` (seed 0, 17 images per method × kind per fold).
  Never regenerate it.
- 2026-11-22: Do not use the cached `agreement` / `matched` maps as input channels. Synth shards have no
  cached maps, so the channel is a constant 0.5 on every synth sample (80 % of batches), and on real images
  the shift augmentation moves the image but not the cached map (up to 32 px off). E104 / E105 /
  E141_w2_msgmaps therefore learn to ignore it. Also keep `wrong_msg_prob` at 0: the negatives empty the
  target while the image still carries the watermark.
- 2026-11-22: Whole-image re-embedding correlation (NCC of the image with `embed(x, m) - x`, 32 px windows)
  measured on train with the real embedders: AUC 0.72–0.77 for `full` holes (TrustMark, StegaStamp), ~0.5
  for patches (wrong scale). A side cue only.
- 2026-11-22: Training-free ceiling is ~0.45–0.49 (honest, cross-fitted); learned W2 models reach 0.62–0.64.
  The statistical track is used as a verifier/fusion stage on top of W2, not as a standalone predictor.
- 2026-11-22: `wl.verify` is applied to PixelSeal and TrustMark only (E150: helps PS/TM, hurts StegaStamp).
- 2026-11-22: All run outputs live on scratch (`output` symlinked to `/p/scratch/training2666/...`); jobs
  outside the reservation via `sbatch_node_public` are fine. Final choice by CV (cross-fitted), not by public score.
