# Watermark localization: method summary (draft for the jury)

Task 1 of the CISPA x Fraunhofer AISEC Paris 2026 hackathon: for each 512 x 512 image, predict which
cells of a 128 x 128 grid carry a known watermark (PixelSeal, TrustMark-Q or StegaStamp, known message).
Score: per-image mIoU over both classes. Half of the images are `full` (watermarked except holes), half
are `patch` (one small watermarked region). The score is decided by the small region of each image.

## 1. Measure before building

`wl.diagnose` and `wl.oracle_k` split the loss of our best model by method, image kind and region size:

* kind errors (full vs patch) are almost zero, so a kind classifier would not help;
* the biggest loss is the holes of PixelSeal / TrustMark `full` images (small-region IoU 0.16-0.18,
  versus 0.68 for StegaStamp);
* with the true region size on the same cell ranking the score would rise from 0.653 to 0.687, but even
  the best size caps PS/TM holes at 0.56-0.59: the ranking (evidence), not the size, is the ceiling.

## 2. Learned model with the decoder in the loop (W8)

WAM-style multitask network (per-cell detection + per-cell message bits compared with the known message),
ResNet-50 U-Net, trained on real + synthetic images made with the real embedders.
W8 runs the frozen official extractors (PixelSeal, TrustMark) inside the forward pass on the augmented
image, compares their per-position bit logits with the known message (sub-stride interleave) and fuses
this evidence with the learned features. This fixes why feeding cached decoder maps had failed (they
were constant on synthetic data and misaligned with the shift augmentation).

## 3. Statistical layer: every decision backed by a random-key null

* Decoder verification (`wl.verify`): candidate boxes are re-decoded at their native scale; the true
  message is compared against 256 random messages (the a-contrario test of Three Bricks, Fernandez et
  al. 2023). A box only overrides the model if z >= 4. Applied to PixelSeal and TrustMark only.
* Differential re-synthesis (`wl.resynth`): the embedders are deterministic, so re-embedding the observed
  image with the known message reproduces the added residual. Re-embedding with random messages gives
  templates that share the content-shaped part but not the message part; the per-cell z of the true
  template against this null cancels the content bias and gives pixel-level evidence (holes), and the
  box-wise version re-embeds a candidate crop at the patch's own scale.
* Fusion and decision (`wl.finalize`): per-method logistic calibration of all evidence (product of
  experts), optional mean-field MRF prior, then the expected-IoU-optimal top-k mask (Nowozin, CVPR 2014).

## 4. Honest evaluation

Every number is out of sample: a fixed fold (f0, never seen in training) or cross-fitting (calibrate on 4
groups, score the 5th). A change is kept only if it beats the fold noise (~0.01-0.02), moves the slice it
targets, and is confirmed on the public leaderboard. Negative results are kept in the log.

## 5. Ablation (validation f0, 102 images)

| Step | Mean | PS | TM | SS | Public |
|---|---|---|---|---|---|
| Training-free decoder + rectangle search (E121) | 0.492 (cross-fitted) | 0.490 | 0.534 | 0.452 | |
| Plain segmentation (SegFormer-B2, E102) | 0.407 | 0.312 | 0.220 | 0.689 | 0.424 |
| WAM-style multitask (E103) | 0.614 | 0.470 | 0.607 | 0.764 | 0.621 |
| + ResNet-50 (E143) | 0.643 | 0.496 | 0.616 | 0.817 | 0.654 |
| + decoder verification (E151) | 0.653 | 0.527 | 0.615 | 0.817 | 0.663 |
| + ensemble with E142 (E152, before verify) | 0.651 | 0.504 | 0.630 | 0.820 | |
| W8 decoder in the loop (E160) | running | | | | |
| + re-synthesis fusion | running | | | | |

Rejected out of sample: decoder z fused in the calibrator (no gain), MRF on the CNN maps (inner CV picks
beta = 0), size calibration (-0.022), PixelSeal-only specialist (-0.04 on PS).
