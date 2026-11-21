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
| | | | | | |

## Decisions

- 2026-11-21: No fixed roles. We split the base library build only; afterwards anyone runs any experiment.
  ID blocks: E100–E199 and E200–E299.
- 2026-11-21: Main track is segmentation networks on real + synthetic data, plus the training-free track.
  Everything writes the same probability files, so anything can be evaluated and ensembled.
- 2026-11-21: Fixed 5-fold split in `splits/folds.json` (seed 0, 17 images per method × kind per fold).
  Never regenerate it.
