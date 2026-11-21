# Experiments — Task 1: Watermark Localization

Shared experiment log for Task 1. **Newest entries first.** Add one row per experiment as soon as a result is available, including negative results.

## Task summary

**Goal:** maximize mIoU (classes 0 and 1, computed per image, then averaged) on 1,020 test images, with masks evaluated on a 128×128 grid.

| Item | Details |
|---|---|
| Training data | 510 images (170 per method) with ground-truth mask, method and embedded message |
| Test data | 1,020 images with method and embedded message provided |
| Edit types | Outpainting, object removal, object addition, object transplantation |
| Geometry | Watermark applied to a rectangular region; aspect ratio ≤ 16:1 (PixelSeal, StegaStamp), ≤ 2:1 (TrustMark) |
| Leaderboard | Public: 30% of test set. Final ranking: remaining 70%. Do not tune thresholds on the public score. |

### Method notes

- **PixelSeal:** watermark computed at 256×256, upscaled and modulated by a JND map; ConvNeXt-v2 Tiny extractor; 256-bit payload; signal concentrated on object edges.
- **TrustMark:** 256×256 residual rescaled to image size; center crop when aspect ratio > 2; removal network (TrustMark-RM) available.
- **StegaStamp:** fixed input resolution (400×400, to be confirmed); dense decoder head, so weak intrinsic localization.

### Working hypothesis

Because each watermark is generated at a fixed resolution and rescaled to its rectangle, decoding the exact watermarked rectangle should yield the highest bit accuracy. Combined with the known message, this turns a global decoder into a spatial hypothesis test. To be validated first (E01).

## Logging conventions

1. Sequential IDs (E01, E02, …), reused in the output folder `output/E01_<short-name>/` and in the commit message.
2. One-sentence hypothesis: what is tested and the expected outcome.
3. Local mIoU on the training split using the official metric function, per method (PS / TM / SS) and averaged.
4. Verdict: **Keep**, **Drop** or **Investigate**, with a one-line justification.
5. Before pushing: `git pull --rebase`, then commit and push, to avoid merge conflicts.

## Experiment log

| ID | Date | Owner | Method / change | WM | Hypothesis | PS | TM | SS | Mean | Public | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E01 | 2026-11-21 | | Decode the ground-truth bounding box vs. full image vs. box shifted by 32 px | All | The exact box yields markedly higher bit accuracy | | | | | | |
| E00 | 2026-11-21 | | Baselines: all-ones, all-zeros, mean mask per method | All | Establishes the score floor | | | | | | |

## Backlog (by priority)

| Priority | Idea | Target WM | Owner | Status |
|---|---|---|---|---|
| 1 | Rectangle search (coarse 64 px grid, then 4 px refinement) maximizing the log-likelihood of the known bits | All | | |
| 2 | Per-pixel bit agreement in the extractor's intermediate feature maps | PS | | |
| 3 | Matched filter: local normalized correlation between `embed(crop, m) − crop` and the residual `x − denoise(x)` | PS, TM | | |
| 4 | Residual map `x − TrustMark-RM(x)` | TM | | |
| 5 | Patch occlusion / input-gradient saliency on the message likelihood | All | | |
| 6 | Multi-scale sliding-window decoding + linear probe on extractor features | SS | | |
| 7 | SRM high-pass filters (KB, KV, SPAM 5×5) as residual input channels | All | | |
| 8 | Per-method segmentation network (small U-Net) trained on synthetic data covering the four edit types | All | | |
| 9 | Post-processing: rectangle fit + hole detection, morphology, per-image threshold maximizing expected mIoU | All | | |
| 10 | Region-wise binomial p-values on known-bit agreement (calibrated maps) | All | | |
| 11 | Test-time augmentation for the segmentation network only (inverse-transform masks before averaging) | All | | |

## Leaderboard submissions

| Time | Owner | Experiment | Public score | File / commit |
|---|---|---|---|---|
| | | | | |

## Decisions

- 
