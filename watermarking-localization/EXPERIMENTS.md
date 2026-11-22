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
| E187_multiprop | 11-22 | 1 | E182 | `wl.verify --extra-proposals 5`: when no main component verifies, the next connected components of the first source's map (thresholds 0.5 and 0.2) are refined too; accepted at z ≥ 5 (`--extra-z-min`), size gate kept | On f0 the map had found a TrustMark patch (p = 1 on it) but verify only tried its largest component, a wrong blob | none | f0 | | | | | | 0.6858 | | **Keep** (+0.0043 vs E182, 1 win / 0 losses: TM `f10e01fec41b` 0.45 → 0.95; 20 unverified f0 images searched, no false accept). Test, CPU run on E182: 7 boxes accepted out of 122 searched (5 PS, 2 TM, z 6.1–9.8). Opt-in flag; costs about 10 decodes per unverified image |
| E188_cpu_pixel_stats | 11-22 | 1 | — | CPU statistics, round 2: (a) period-2 trace of the residual (PS and TM compute it at 256 and upsample 2× bilinear, which makes the third difference exactly 0 at even columns), plus a phase test at periods 2–16; (b) steganalysis rich-model features (11 residual filters × 6 colour channels, energies and cross-channel correlations, 3 scales) with logistic / MLP trained on folds 1–4; (c) TrustMark remover residual \|RM(y) − y\|; (d) TrustMark re-synthesis matched filter (re-embed the true message, local NCC and regression slope) | Pixel-level evidence for PS/TM holes without a GPU | none | f0 / all510 | | | | | | | | **Drop**: (a) hole AUC 0.50 and no periodicity at any period (no JPEG grid either): the images were resampled after embedding; (b) f0 hole AUC PS 0.56, TM 0.64, SS 0.88 against 0.71 / 0.81 / 0.975 for the model, and fused with the model 0.6672 vs 0.668; (c) AUC 0.57–0.60; (d) AUC 0.47–0.55. With E205 (gradient attribution), every pixel-level probe is at chance |
| E189_decodability | 11-22 | 1 | — | Decoder z at the true box (oracle, train patches; TM and PS decoders on CPU), its decay under box perturbations, z-plateau box expansion, whole-image z as a kind check | Where can a better box search still win? | none | train / f0 | | | | | | | | **Findings**: TM rectangles decode at their true box (z ≈ 10), object-shaped TM patches do not (z 0–2). PS compact rectangles reach z 9–17, but thin bars (3–4 cells wide) and tiny boxes stay at z 0–4 even at the exact box: 8 of the 10 PS rectangles that E182 misses on f0 are of that kind, so no search can recover them. z is flat when the box shrinks inside the patch and collapses when it grows 10 % outside it. Expanding verified boxes up to that plateau helps PS (+0.12 on 2 images) but hurts TM, whose z saturates at ~10 (−0.10 on 6): net −0.006, **dropped**. Whole-image z separates kinds (PS full ≥ 11.2, TM full ≥ 3.7, train); on test, 6 images predicted `full` fall below every training full image (1 PS, 5 TM), but only one has a decodable box, so the kinds are left unchanged |
| E182_gate_e158 | 11-22 | 1 | E158 | `wl.verify --regate --max-area-q 0.9` (CPU, no decoding): a refined box larger than the 90 % quantile of the method's training patch box areas (folds 1–4 when scoring f0: PS 0.36, TM 0.39, SS 0.30 of the image) reverts to the first source | A significant z only proves that the message is somewhere in the box; oversized boxes are *containing* boxes and cost clean-class IoU | none | f0 | 0.545 | 0.654 | 0.846 | 0.631 | 0.732 | 0.682 | | **Keep** (+0.007 vs E158; +0.006 / +0.007 on E153 / E156; any quantile 0.75–0.95 gives +0.005–0.007 and never a loss). Only 2 f0 images flip (2 wins / 0 losses), but the test statistics support the mechanism: accepted test boxes with z < 8 have a median area of 0.16–0.33, those with z ≥ 8 0.02–0.08. Reverts 21 of the 89 refined test boxes |
| E183_decision_rules | 11-22 | 1 | E161_tta | Cross-fitted Platt calibration, then Bayes-optimal decisions: RankSEG-RMA (= current `eiou_decide`), dependence-aware variant (per-image latent logit shift, s = 0.5–3, after Wang & Dai 2026), rectangle model selection by expected mIoU, rectangle snap | Decision theory recovers part of the per-image threshold oracle (+0.03) | none | f0 | 0.514 | 0.642 | 0.839 | | | 0.6646 | | **Drop**: same masks (Δ = 0.0000; rectangle snap −0.002 to −0.005). The maps are near-binary (4–10 % of cells in (0.05, 0.95)), so every decision rule returns the same mask |
| E184_stack_context | 11-22 | 1 | E161_tta | (a) Logarithmic opinion pool: per-method cross-fitted logistic stacking of the E161_tta, E143_tta, E160_tta and E142 logits (extremizing allowed); (b) auto-context: per-cell logistic regression on multi-scale smoothed logits, local max, image statistics and border distance | Learned pooling or spatial context re-ranks the cells | none | f0 | | | | | | 0.652 / 0.645 | | **Drop**: −0.013 (stacking), −0.020 (auto-context, sign test p = 0.002). The cell log-loss does not track per-image mIoU |
| E185_edge_aware | 11-22 | 1 | E161_tta | Edge-aware refinement of the calibrated beliefs before the eIoU decision: guided filter (r 2–4), SLIC / Felzenszwalb superpixel pooling, bilateral dense-CRF mean field (9×9 window); setting chosen by nested cross-fitting | Snap masks to object boundaries | none | f0 | | | | | | 0.6645 | | **Drop**: −0.0001 with the nested choice, ±0.000 for the best fixed setting: nothing to sharpen on near-binary maps |
| E186_forensic_stats | 11-22 | 1 | — | Blind forensic statistics on the 128 grid: Splicebuster (3rd-order residual co-occurrences, per-image 2-component Gaussian mixture by EM), local high-pass energy, local noise level (MAD) | Edit traces (inpainting, transplantation) show regions the watermark does not | none | all510 | | | | | | | | **Drop**: within-image cell AUC 0.50 (Splicebuster), 0.59 (energy), 0.58 (noise); on f0 PixelSeal 0.48–0.51, where the model has 0.67–0.69 |
| E180 | 11-22 | 1 | E104 | `wl.refine`: kind gating from cached whole-image bit accuracy | Wrong-kind maps cost ~0.5 each; decoding fixes them for free | none | f0 | | | | | | | | planned |
| E173 | 11-22 | 1 | E170 | HRNet-W48 + FPN | High-res stream helps faint patterns; ensemble diversity | real+synth-v2 | f0 | | | | | | | | planned |
| E172 | 11-22 | 1 | E103 | synth-v2, 35 epochs | Same W2 recipe gains from data diversity | real+synth-v2 | f0 | | | | | | | | planned |
| E171 | 11-22 | 1 | E170 | SegFormer-B2 | Direct comparison with E104: gain of synth-v2 + fixes | real+synth-v2 | f0 | | | | | | | | planned |
| E170 | 11-22 | 1 | E104 | synth-v2 (COCO bases, restore/paste/inpaint holes), no wrong-message negatives, no cached maps, SegFormer-B5 | Data diversity + capacity lift PixelSeal and `full` holes | real+synth-v2 | f0 | | | | | | | | planned |
| E140_w3_noneg, E141_w3_rgbsrm, E142_w3_b5, E143_w2_40ep | 11-22 | 1 | E104/E103 | Ablations on synth-v1: no negatives; + drop agreement; + SegFormer-B5; W2 40 epochs. Numbers clash with E141_w2_msgmaps / E142_w2_long / E143_w2_r50 below; output dirs differ | The agreement channel is constant on synth and misaligned on real; negatives add label noise | real+synth-v1 | f0 | | | | | | | | running |
| E190–E193 (wave 2) | 11-22 | 2 | E161 | Best recipe again: seed 1 (E190), + Lovász-hinge loss 0.5 (E191, Berman et al. CVPR 2018: surrogate of the Jaccard index the task is scored on), 35 epochs (E192), seed 4 (E193); TTA chained | More diverse members of the best recipe for the final ensemble | real+synth-v1 | f0 | | | | | | | | queued (job 15905158) |
| E207 / E208 / E209 (encoders) | 11-22 | 2 | E161 / E161 / E143 | Larger or different encoders: W8 no-interleave with ResNet-101 (E207) and ConvNeXt-Small via timm (E208), W2 with ResNet-101 (E209); weights cached on the login node, `HF_HUB_OFFLINE=1` | ResNet-34 -> 50 was the largest single-model gain (+0.03) | real+synth-v1 | f0 | | | | | | | | queued (job 15905449), ends ~09:05; enters a last greedy pass if in time |
| E200 / E201 / E202 (diverse members) | 11-22 | 2 | E143 / E161 / E161 | W2 seed 1; W8 with ResNet-34; W8 with synth ratio 0.6. Trained on idle GPUs of job 15904808 via `scripts/bg_member.sh` (tmux), TTA chained | Ensemble diversity | real+synth-v1 | f0 | | | | | | | | running, ends ~08:15; picked up by `scripts/autopilot_final.sh` |
| E194 / E195 (FiLM test) | 11-22 | 2 | E161 | E194: per-method FiLM on the fused features (zero init), warm start from E161, 8 epochs on folds 1–4; E195: same warm start and schedule without FiLM (control) | The shared head ignores which method it sees | real+synth-v1 | f0 | 0.514 / 0.520 | 0.641 / 0.645 | 0.825 / 0.836 | 0.625 / 0.630 | 0.695 / 0.704 | 0.660 / **0.667** (TTA) | | **FiLM rejected**: −0.007 vs its control, CI [−0.019, +0.005], 34 wins / 45 losses. E195_tta (warm-start control) is the best single model (0.667) and a candidate member |
| E214 = E198g + finalize (calibration + eiou) | 11-22 | 2 | E198g | Per-method logistic calibration of the cell beliefs (cross-fitted) before the expected-IoU decision | eiou assumes calibrated beliefs | none | f0 | 0.545 | 0.654 | 0.845 | 0.631 | 0.731 | 0.6810 ± 0.039 | | Rejected (−0.006 vs plain eiou 0.6866): the verified maps are partly binary, recalibration adds noise |
| E198g_eiou | 11-22 | 2 | E198g | Expected-IoU top-k decision per image (Nowozin, CVPR 2014), no parameter, instead of the OOF-tuned post-processing | Bayes decision for the per-image mIoU | none | f0 | | | | | | 0.6866 | **0.6936** | **Current best** (+0.0072 public) |
| E215_decision_lab | 11-22 | 2 | E198g | `wl.decision_lab`: 8 parameter-free decision rules on the same beliefs: Nowozin ratio (= E198g_eiou), exact Monte-Carlo expected mIoU (RankSEG-like), kind-consistent top-k, comonotone coupling (truth = {p > u}, one u per image: spatially coherent errors), mix of the two extreme couplings, and kind variants | Independent-cell eiou may be the wrong uncertainty model (holes/patches found or missed as a whole) | none | f0 | | | | | | ratio 0.6866, exact 0.6866, exact_kind 0.6867, como 0.6837, mix 0.6846, mix_kind 0.6848 | | **None kept**. Ratio approximation = exact Bayes (decision layer saturated under the cell model). Comonotone: more image wins (53/35) but −0.003 mean, driven by PS full −0.016 (holes behave like independent cells) and tiny patch gains (PS +0.004, TM +0.001); a kind-specific coupling would be chosen on f0, not tried |
| E213_dupes | 11-22 | 2 | — | `wl.dupes`: near-duplicate base photos across train/test and within test (32×32 blurred thumbnails, correlation) | If a photo is reused, the image difference localises the marks exactly | none | train+test | | | | | | | | **Closed: no duplicates** (0 pairs > 0.95, max 0.926, mostly one low-texture image) |
| E206_rect | 11-22 | 2 | E198g | `wl.rectdec`: expected-IoU optimal axis-aligned rectangle for images predicted `patch` (`wl.shapes`: patches are single components, bbox fill 0.97–0.98; holes are irregular, fill ~0.45) | Shape prior, no fitted parameter | none | f0 | | | | | | 0.6749 (eiou base 0.6866) | | **Rejected** (−0.012; 51 wins / 43 losses but SS patch −0.04, TM patch −0.010): the models' maps already follow the near-rectangular shapes, and a forced rectangle loses the corners/feathering the remaining 50 % of patches have |
| E205_attrib | 11-22 | 2 | — | `wl.attrib`: input-x-gradient attribution of the frozen PS/TM extractors for the true message, z vs 8 random messages per cell | Pixel-level evidence for PS/TM holes | none | train (60 img) | | | | | | | | **Rejected at the 60-image check: no signal** (cell AUC PS full 0.48, PS patch 0.49, TM 0.49–0.51). With the re-synthesis result, two independent pixel-level probes find nothing: the readable evidence is the decoders' coarse output, exploited by W8 and verify |
| E198g_final_gate | 11-22 | 2 | E198 | Size gate on verified boxes (Rahul, E182): reject refined boxes larger than the 0.9 quantile of training patch areas, CPU re-gate | z only proves the message is inside the box; oversized boxes are containing boxes | none | f0 | 0.545 | 0.654 | 0.846 | 0.631 | 0.732 | 0.6816 | **0.6863** | Best. f0: 2 images changed (2/0, CI [0, +0.021]); public +0.005 is the independent confirmation; 20 test boxes reverted |
| E203_ens_142tta | 11-22 | 2 | E197 | E142 replaced by E142_tta (0.633 vs 0.623 alone) | Stronger member | none | f0 | | | | | | 0.6682 | | No gain (+0.0002) |
| E198_final_verify | 11-22 | 2 | E197 | verify (PS, TM) on E197 = E157g weights/validation with all-folds twins at test time (E169_tta, E167_tta, E165_tta, E142) via `ensemble --test-runs` | +25 % data at test without biasing the validation | none | f0 | 0.543 | 0.634 | 0.846 | 0.631 | 0.717 | 0.674 | **0.6813** | **Current best.** Test masks agree 98.5 % with E158 (sanity check of the swap) |
| E158_verify_e157g | 11-22 | 2 | E157g | `wl.verify` (PS, TM) on the TTA ensemble | Verify gain stacks | none | f0 | 0.543 | 0.634 | 0.846 | 0.631 | 0.717 | 0.674 | 0.6786 | +0.006 f0, +0.013 public vs E155g |
| E196g_ens_plus166 | 11-22 | 2 | E157g | E157g + E166_tta (W8 seed 1) | More members | none | f0 | | | | | | 0.6678 | | Rejected (no gain, too close to E160) |
| E157g_ens_tta | 11-22 | 2 | E161_tta, E143_tta, E160_tta, E142 | Shared-weight logit ensemble of the TTA versions | TTA and ensembling add up | none | f0 | 0.516 | 0.641 | 0.846 | 0.631 | 0.705 | 0.668 | | Best on f0 (+0.008 over E155g) |
| E161_tta / E143_tta / E160_tta | 11-22 | 2 | E161 / E143 / E160 | `wl.infer`: 9 integer cell shifts (fixed set, no tuning), shift undone on the output, mean | Variance reduction for free | none | f0 | 0.514 / 0.515 / 0.505 | 0.642 / 0.627 / 0.615 | 0.839 / 0.823 / 0.798 | | | 0.665 / 0.655 / 0.639 | | +0.011 / +0.012 / −0.001: keep TTA for every member |
| E155g_ens_w8_global | 11-22 | 2 | E161, E160, E143, E142 | Logit ensemble of 4 models, weights shared across methods (4 parameters) | W8 models add diversity | none | f0 | 0.511 | 0.646 | 0.824 | 0.627 | 0.694 | 0.660 | 0.6656 | **Current best public.** Per-method weights (E155) give 0.6645 on f0: the 0.004 difference is the optimism of 12 weights fitted on 102 images; the shared-weight version was submitted |
| E161_w8_noshift (final) | 11-22 | 2 | E160 | W8 without the 2×2 interleave (one extractor pass) | Is the interleave needed? | real+synth-v1 | f0 | 0.504 | 0.640 | 0.819 | 0.622 | 0.687 | 0.654 | | **Best single model** (+0.011 vs E143, TM and patches), 4× cheaper than E160 |
| E160_w8_decoder_loop (final) | 11-22 | 2 | E143 | W8 with 2×2 interleave | Decoder in the loop | real+synth-v1 | f0 | ≈0.50 | ≈0.61 | ≈0.80 | ≈0.61 | ≈0.66 | 0.641 | | Ties E143: the epoch-10 lead (+0.025) was faster convergence, not a better optimum |
| E153_verify_e152 | 11-22 | 2 | E152 | `wl.verify` (PS, TM) on the E143+E142 ensemble | Verify gain stacks on the ensemble | none | f0 | 0.541 | 0.628 | 0.820 | 0.634 | 0.692 | 0.663 | 0.661 | +0.010 on f0 but −0.002 public vs E151: not confirmed, treated as a tie (~0.66) |
| E152_ens_143_142 | 11-22 | 2 | E143, E142 | Logit ensemble of E143 (ResNet-50) and E142 (ResNet-34, 50 epochs) | Different errors average out | none | f0 | 0.504 | 0.630 | 0.820 | 0.634 | 0.669 | 0.651 | | Keep (+0.008 over E143, mostly TM). E153 = verify on it, queued |
| E142_w2_long | 11-22 | 2 | E103 | W2, 50 epochs | E103 under-trained | real+synth-v1 | f0 | 0.488 | 0.604 | 0.778 | 0.611 | 0.636 | 0.623 | | +0.010 over E103, below E143: capacity matters more than length |
| E151 + finalize --mrf | 11-22 | 2 | E151 | Mean-field Ising prior on calibrated beliefs, beta per method chosen by inner CV | Holes/patches are compact | none | f0 | 0.527 | 0.615 | 0.817 | 0.627 | 0.679 | 0.6529 ± 0.014 | | No gain: inner CV picks beta=0 almost always (CNN maps already smooth). Kept for noisy evidence (resynth) |
| E151 + wl.sizer | 11-22 | 2 | E151 | Size-calibrated top-k: ridge on log region size from image statistics, shrunk to the eiou size | `wl.oracle_k`: true size on the same ranking gives 0.687 vs 0.653 | none | f0 | 0.552 | — | — | — | — | 0.631 | | Rejected (−0.022, negative in 3/5 groups). Ranking, not size, is the ceiling: oracle best-k only 0.56–0.59 on PS/TM holes |
| E154_resynth | 11-22 | 2 | E151 | `wl.resynth`: re-embed with the true message and 8 random messages, local correlation of the high-pass image with the templates, z vs the random-message null; whole image + box-wise for patches; fused via `finalize --resynth [--resynth-box] [--mrf]` | Pixel-level evidence for PS/TM holes | none | all510 / f0 | 0.527 | 0.615 | 0.817 | 0.627 | 0.679 | 0.6529 ± 0.014 | | **Rejected: no signal.** Cell AUC PS full 0.498, PS patch 0.492, TM 0.51–0.53, SS 0.55–0.60; fusion unchanged (calibrator gives it ~0 weight). Re-embedding does not reproduce the residual well enough after the data pipeline, while the (robust) decoders still read the mark: localisation must come from decoder evidence, not template matching. First run had NaN maps (fixed in 91f833b) |
| E160–E164 (W8) | 11-22 | 2 | E143 | `w8_decoder_loop`: frozen PixelSeal/TrustMark extractors in the forward pass, per-position agreement with the known message (2×2 interleave) fused with learned features. E161 no interleave, E162 PS+TM only, E164 + TrustMark-RM residual | Fixes why E141 failed (evidence constant on synth, misaligned with shifts) | real+synth-v1 | f0 | | | | | | | | E160 epoch 23: 0.640 (caught up by E143, the early lead was faster convergence); E161 epoch 23: 0.651 (+0.008 vs E143, TM and patches), 4x cheaper. Epoch 10: E160 0.635 / E161 0.631 vs E143 0.610 at the same epoch (+0.025, gain on PS, TM and SS). E164 ~10 min/epoch (RM at 512 px), ends ~08:00 |
| E151_verify_e143 | 11-22 | 2 | E143 | `wl.verify` on E143 with E121 boxes, `--methods pixelseal trustmark` | Verification gains (E150) stack on the best model | none | f0 | 0.527 | 0.615 | 0.817 | 0.627 | 0.679 | 0.653 | 0.663 | **Current best** (+0.010 val, +0.009 public). 34 PS/TM patches: 5 confirmed, 8 repaired, 21 untouched |
| E145_w2_ps_only | 11-22 | 2 | E103 | W2 trained on PixelSeal only (`data.methods: [pixelseal]`) | A specialist lifts the weakest method (PS ~0.50) | real+synth-v1 | f0 | 0.455 | — | — | 0.484 | 0.427 | 0.455 | | Worse than E143 on PS (0.496): less data hurts more than specialisation helps |
| E143_w2_r50 + finalize (±--ac-z) | 11-22 | 2 | E143 | `wl.finalize` cell calibrator: model logit + kind (+ decoder z), expected-IoU decision, cross-fitted | Product-of-experts fusion of model and decoder evidence | none | f0 | 0.497 | 0.616 | 0.817 | 0.627 | 0.659 | 0.6433 ± 0.040 (0.6431 without z) | | No gain: the model already absorbs the decoder evidence |
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

## Lessons learned (night of 11-21/22)

1. Never judge a model mid-training: W8 led by +0.025 at epoch 10 and tied E143 at the end (faster convergence, same optimum). Compare final checkpoints only.
2. Check a new evidence map on a few images (no NaN, AUC) before a full run and a fusion: the first re-synthesis run had NaN maps.
3. What wins is decoder evidence and model diversity: verify +0.010 (three times), ensembles +0.008 / +0.013. Post-processing smooth CNN maps (MRF, sizer, decoder z in the calibrator) and template matching (re-synthesis) gave nothing.
4. f0 has 102 images: prefer the version with the fewest free parameters (shared ensemble weights), and require a public confirmation (E153: +0.010 f0, −0.002 public).

## Statistical methods round (E182–E189)

Question: which state-of-the-art statistical (non-deep-learning) methods still add accuracy on top of the
W2 / W8 maps? Every test runs on f0 (102 images), cross-fitted or with a rule fixed in advance, and is
compared image by image with `python -m wl.compare` (paired bootstrap CI and sign test).

What limits the post-hoc layer:

- The maps are near-binary: only 4–10 % of the cells lie in (0.05, 0.95). Even an oracle that picks the
  best threshold for each image only reaches 0.6955 (E161_tta: 0.6647). Decision rules can win at most
  +0.03, and none wins any of it.
- The remaining error is missing evidence. On f0, E161_tta predicts nothing for 8 of 17 PixelSeal patches
  (none with IoU > 0.5), for 5 of 17 PixelSeal holes and for 6 of 17 TrustMark holes.
- Decoder evidence is bimodal: on f0, the verified patch boxes have z ≥ 9.8 or z ≤ 2.3. The accepted test
  boxes with z < 8 are far larger than real patches (median area 0.16–0.33 against 0.04).

| Family | State of the art | Run | Result |
|---|---|---|---|
| Bayes-optimal IoU decision | RankSEG / RankSEG-RMA (Dai & Li, JMLR 2023; arXiv 2510.15362); dependence-aware RankSEG (Wang & Dai, arXiv 2609.38930) | E183 | 0: `eiou_decide` already is the RMA rule |
| Calibration and forecast aggregation | Platt scaling; logarithmic opinion pool with extremizing (Satopää et al., IJF 2014) | E184a | −0.013 |
| Contextual stacking | Auto-context (Tu & Bai, TPAMI 2010) | E184b | −0.020 |
| Edge-aware MRF / CRF | Dense CRF (Krähenbühl & Koltun, NeurIPS 2011), fast bilateral solver (Barron & Poole, ECCV 2016), guided filter, superpixel pooling | E185 | 0 |
| Blind forensic statistics | Splicebuster (Cozzolino, Poggi & Verdoliva, WIFS 2015), local noise inconsistency | E186 | AUC 0.50–0.59 |
| A-contrario detection and multiple testing | Number of false alarms (Desolneux, Moisan & Morel), random-key z (Fernandez et al., WIFS 2023), e-BH block tests for watermark localization (arXiv 2611.02030) | E182 | **+0.007** (size gate; +0.005 public on E198) |
| Learned proposals, statistical verification | DL map components as proposals, a-contrario z with a stricter threshold for the extra candidates | E187 | **+0.004** |
| Resampling and steganalysis statistics | Resampling / cyclostationarity detection (Popescu & Farid, IEEE TSP 2005); spatial rich models (Fridrich & Kodovský, IEEE TIFS 2012) | E188 | AUC 0.50–0.64 |
| Embedder-based tests | Watermark remover residual, re-synthesis matched filter | E188 | AUC 0.47–0.60 |

Other checks, all negative: blind border priors for holes (frames, strips) give at most +0.004; no image is
a near-duplicate of another (32×32 thumbnail correlation < 0.95 everywhere), so StegaStamp masks cannot be
transferred to PixelSeal or TrustMark twins.

Alternative to the size gate: z ≥ 6 for searched boxes (Bonferroni / NFA over the rectangle search) gives
the same f0 gain. On test, however, it keeps 9 boxes larger than the 90 % quantile (z 6–8) and drops small
boxes with z 4–6. That is the wrong way round under the asymmetric loss: a wrong small box costs about
0.01, a right one gains about 0.3. The size gate is preferred.

## Ideas queue (what to try next, and why)

Ranked by expected gain; each one is tested out of sample (f0 or cross-fitted) and kept only if it beats the fold noise (~0.01–0.02) and moves the slice it targets.

| Idea | Targets | Evidence that motivates it | Status |
|---|---|---|---|
| Differential re-synthesis matched filter (`wl.resynth`) | PS/TM `full` holes | `wl.diagnose`, `wl.oracle_k` | **rejected**: cell AUC ~0.5 on PS, 0.53 on TM |
| Box-wise re-synthesis for patches (re-embed the candidate crop at native scale, `wl.resynth --boxes-from`, `finalize --resynth-box`) | PS patch (IoU 0.23), TM patch | verify (native-scale decoding) already +0.01; same principle at pixel level | coded (jobs/resynth_box.txt); launched only if whole-image resynth shows signal |
| Ensemble of W8 + E143 + E142 with OOF weights, then verify | all | E152 (E143+E142) +0.008; verify +0.010 twice | after W8 finishes |
| Test-time augmentation with 9 integer cell shifts (`wl.infer`), fixed set, no tuning | all (variance) | standard, watermark-safe (shift undone on the output) | coded, jobs/tta_ft.txt |
| Warm start (`train.init_from`): fine-tune the best recipe from its checkpoint on all folds, 8 epochs (E169) | all, +25 % data at ~1/3 of the cost | E161 is the best W8 recipe (0.651 f0) | coded, jobs/tta_ft.txt |
| Retrain the finalist on all 5 folds (fixed epochs from CV) | all | +25 % data | E165/E167/E168 running (job finals) |
| MRF prior on resynth evidence | holes | MRF useless on smooth CNN maps, expected to help on noisy pixel evidence | runs in the resynth job |
| Rejected: size calibration (`wl.sizer`), MRF on E151, `--ac-z` fusion, PixelSeal-only specialist (E145) | — | all negative out of sample, see log | — |

## Leaderboard submissions

| Time | Owner | Experiment | CV (f0 / cv5) | Public score | File / commit |
|---|---|---|---|---|---|
| 11-22 | 1 | E103_w2_wam_multitask | 0.621 / — | 0.621 | backup in `/p/scratch/training2666/chander2/argos/submissions` |
| 11-22 | 2 | E120_kind_const | 0.452 (honest all510) | 0.4451 | below best, leaderboard unchanged |
| 11-22 | 2 | E102_segformer_b2 | 0.407 / OOF 0.432 | 0.4239 | below best |
| 11-22 | 2 | E143_w2_r50 | 0.643 / OOF 0.646 | 0.654 | new best at the time |
| 11-22 | 2 | E151_verify_e143 | 0.653 / OOF-postproc 0.657 | 0.663 | best until 05:07 |
| 11-22 | 2 | E153_verify_e152 | 0.663 / OOF-postproc 0.668 | 0.661 | not improved (within public noise) |
| 11-22 | 2 | E155g_ens_w8_global | 0.660 | 0.6656 | best until 05:41 |
| 11-22 | 2 | E157g_ens_tta | 0.668 | — | refused (HTTP 429 rate limit) |
| 11-22 | 2 | E158_verify_e157g | 0.674 | 0.6786 | best until 06:07 |
| 11-22 | 2 | E198_final_verify | 0.674 (all-folds test members) | 0.6813 | best until 06:32 |
| 11-22 | 2 | E198g_final_gate | 0.6816 (E198 + Rahul's size gate `--max-area-q 0.9`, `--regate`) | 0.6863 | best until 07:23 (+0.005 public: confirms the gate) |
| 11-22 | 2 | E198g + expected-IoU decision (`wl.eiou_submit`) | 0.6866 (vs 0.6849 for the OOF-tuned post-processing of submit.sh) | **0.6936** | **current best**: parameter-free Bayes decision for the per-image mIoU replaces the post-processing fitted on f0 |

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
- 2026-11-22: Verify runs use `--max-area-q 0.9`; on a run that already exists, `wl.verify --regate` applies
  the same gate on CPU in seconds (E182). Post-hoc decision rules, smoothing, stacking and blind forensic
  statistics are exhausted on these maps (E183–E186): further gains need new evidence for PixelSeal patches
  and PixelSeal / TrustMark holes.
- 2026-11-22: Training-free ceiling is ~0.45–0.49 (honest, cross-fitted); learned W2 models reach 0.62–0.64.
  The statistical track is used as a verifier/fusion stage on top of W2, not as a standalone predictor.
- 2026-11-22: `wl.verify` is applied to PixelSeal and TrustMark only (E150: helps PS/TM, hurts StegaStamp).
- 2026-11-22: All run outputs live on scratch (`output` symlinked to `/p/scratch/training2666/...`); jobs
  outside the reservation via `sbatch_node_public` are fine. Final choice by CV (cross-fitted), not by public score.
