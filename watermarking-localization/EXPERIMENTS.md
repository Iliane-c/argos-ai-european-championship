# Experiments: Task 1, Watermark Localization

This is the shared experiment log for Task 1. **Newest entries first.** Add one row per experiment as soon
as a result is available, including negative results. How we work (roles, evaluation, syncs) is in
[PLAN.md](PLAN.md).

## Team

| Role | Name | Owns |
|---|---|---|
| **A**: Data & Evaluation | | `eval.py`, `splits/`, `synth/` (synthetic data, augmentation, extra input channels), thresholds and post-processing |
| **B**: Models & Training | | `train.py`, `configs/`, `jobs/`, architectures, losses, ensembling, `predict.py` |

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

- **Main track:** a segmentation network trained on the real images plus synthetic images made with the kit
  embedders, validated on fixed folds (PLAN.md §3–5).
- **Side track:** the decoder-based ideas (box decoding, per-position agreement, matched filter) give
  message-aware maps. These maps become extra input channels for the network.
- **E01 test:** because each watermark is generated at a fixed resolution and rescaled to its rectangle,
  decoding the exact watermarked rectangle should give the highest bit accuracy. E01 checks this first.

## Logging conventions

1. Use sequential IDs (E01, E02, …). Reuse the ID in the output folder `output/E01_<short-name>/`, the
   config `configs/E01_<short-name>.yaml` and the commit message.
2. Write a one-sentence hypothesis: what is tested and the expected outcome.
3. **Parent**: the experiment this one changes. Make exactly one change per experiment.
4. **Data**:
   - `real`: real folds 1–4.
   - `synth-vN`: synthetic version N.
   - `real+synth-vN`: both.
5. **Val**:
   - `f0`: real fold 0, 102 images.
   - `cv5`: 5-fold mean ± std.
   - `all510`: all real train images. Only allowed when training used no real image.
6. Use the official metric function (`eval.py`) at threshold 0.5, unless the row says otherwise. Report
   PS / TM / SS, Full / Patch, and the mean.
7. Verdict: **Keep**, **Drop** or **Investigate**, with a one-line justification. A gain below 1 CV std
   counts as noise.
8. Before pushing: `git pull --rebase`, then commit and push, to avoid merge conflicts.

## Experiment log

| ID | Time | Owner | Parent | Change | Hypothesis | Data | Val | PS | TM | SS | Full | Patch | Mean | Public | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E01 | | A | — | Decode the ground-truth bounding box vs. the full image vs. the box shifted by 32 px (report bit accuracy, not mIoU) | The exact box yields markedly higher bit accuracy | — | all510 | | | | | | | | |
| E00e | 11-21 | A | — | Oracle: ground-truth bounding box of the part to find | Shows how much a box-only prediction can reach | — | all510 | 0.718 | 0.692 | 0.714 | 0.549 | 0.867 | 0.708 | — | Reference (upper bound for boxes) |
| E00d | 11-21 | A | — | Oracle: the right constant per image (all-ones for `full`, all-zeros for `patch`) | What a perfect kind classifier alone gives | — | all510 | 0.452 | 0.452 | 0.460 | 0.453 | 0.456 | 0.455 | — | Reference: the rest of the score comes from the small part |
| E00c | 11-21 | A | — | Mean train mask per method, threshold 0.5 (in-sample) | Location prior alone | — | all510 | 0.300 | 0.308 | 0.301 | 0.277 | 0.329 | 0.303 | — | Floor |
| E00b | 11-21 | A | — | All ones | Score floor | — | all510 | 0.247 | 0.252 | 0.246 | 0.453 | 0.044 | 0.248 | — | Floor |
| E00a | 11-21 | A | — | All zeros | Score floor | — | all510 | 0.253 | 0.248 | 0.254 | 0.047 | 0.456 | 0.252 | — | Floor |

## Backlog (by priority within each role)

**A: Data & Evaluation**

| Priority | Idea | Target WM | Owner | Status |
|---|---|---|---|---|
| A1 | `eval.py` (official metric, per method × kind, writes `metrics.json`) + loader for real data with `splits/folds.json` | All | A | |
| A2 | Synthetic generator v1: `full` = embed the whole image, put clean pixels back in holes; `patch` = embed a box at its own size, optionally keep only an object mask | All | A | |
| A3 | Mask library: real masks (flip / rotate / shift / scale), random blobs and strokes, rectangles, COCO or SAM object masks | All | A | |
| A4 | Copy-paste augmentation between real images (integer shift, no resize) | All | A | |
| A5 | Augmentation A/B: Instagram filters (`pilgram`), color jitter and JPEG **before** embedding (safe) vs **after** embedding (risky) | All | A | |
| A6 | Message-aware input channels: per-position bit agreement in extractor maps (PS, TM); matched filter `embed(x, m) − x` vs `x − denoise(x)`; residual `x − TrustMark-RM(x)` | All | A | |
| A7 | Kind feature: whole-image decode score (`full` vs `patch`) as input or as a check | All | A | |
| A8 | Synthetic generator v2: holes filled by an inpainting model (LaMa) for realism | All | A | |
| A9 | Thresholds chosen on out-of-fold predictions; "never empty small part" fallback; rectangle snap; morphology | All | A | |

**B: Models & Training**

| Priority | Idea | Target WM | Owner | Status |
|---|---|---|---|---|
| B1 | `train.py` (YAML config, smp / HF models, writes `val_probs.npz`, `test_probs.npz`) + `jobs/train.sh` | All | B | |
| B2 | Baseline: U-Net ResNet-34, 512 input, 128 output, BCE + Dice, real folds 1–4 → f0 | All | B | |
| B3 | SegFormer MiT-B2 (then B5): output stride 4 gives 128×128 directly | All | B | |
| B4 | smp sweep: UNet++ / DeepLabV3+ / FPN × EfficientNet-B4 / ConvNeXt-T | All | B | |
| B5 | TruFor init (forgery-localization pretrained, sees noise traces) | All | B | |
| B6 | Watermark Anything extractor fine-tuned to our 3 watermarks | All | B | |
| B7 | The method's own decoder backbone (PixelSeal ConvNeXt, TrustMark ResNet-50) + pixel head | PS, TM | B | |
| B8 | Input: RGB vs + SRM high-pass (KB, KV, SPAM 5×5) vs + A6 channels | All | B | |
| B9 | One model per method vs a shared model with a method input | All | B | |
| B10 | Loss: BCE + Dice vs per-image 2-class Lovász | All | B | |
| B11 | Ensemble of finalists; test-time augmentation only if it keeps the watermark (no resize) | All | B | |

**Side track (if time):** rectangle search (coarse 64 px grid, then 4 px refinement) maximizing the
log-likelihood of the known bits; patch occlusion / input-gradient saliency; multi-scale sliding-window
decoding + linear probe (SS); region-wise binomial p-values on known-bit agreement.

## Leaderboard submissions

| Time | Owner | Experiment | CV (f0 / cv5) | Public score | File / commit |
|---|---|---|---|---|---|
| | | | | | |

## Decisions

- 2026-11-21: Main track is a segmentation network trained on real + synthetic data. Decoder-based maps
  become input channels.
- 2026-11-21: Fixed 5-fold split in `splits/folds.json` (seed 0, 17 images per method × kind per fold).
  Never regenerate it.
