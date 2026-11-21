# Watermark Localization: Plan

The task: for each 512 x 512 test image, return a 128 x 128 binary mask (1 = watermarked) for PixelSeal,
TrustMark or StegaStamp. The score is the per-image mIoU over {clean, watermarked}, averaged over the 1,020
test images. The hackathon ends **Sun 22 Nov, 12:00**. Freeze the final submission by about 11:00.

At test time we know the **method** and the **embedded message** for every image. We also have the **exact
embedders** used to build the dataset. Most methods below rely on these three facts.

---

## 1. What the train data says

Numbers come from `python analysis/mask_stats.py --data data/release`, run on the train split (510 images).

| method | `full_watermark` | n | watermarked area, median [p10–p90] | rectangular masks (fill > 0.98) | bbox aspect, median (max) | all-1 | all-0 | GT bbox as prediction |
|---|---|---|---|---|---|---|---|---|
| PixelSeal  | True  | 85 | 0.94 [0.74–0.99] | 20 % | 1.0 (1.2)   | 0.449 | 0.051 | 0.452 |
| PixelSeal  | False | 85 | 0.05 [0.01–0.22] | 49 % | 2.2 (16)    | 0.045 | 0.455 | 0.879 |
| TrustMark  | True  | 85 | 0.94 [0.80–0.99] | 25 % | 1.0 (1.3)   | 0.454 | 0.046 | 0.460 |
| TrustMark  | False | 85 | 0.04 [0.01–0.30] | 51 % | 1.5 (16)    | 0.049 | 0.451 | 0.864 |
| StegaStamp | True  | 85 | 0.94 [0.81–0.99] | 21 % | 1.0 (1.4)   | 0.456 | 0.044 | 0.463 |
| StegaStamp | False | 85 | 0.04 [0.01–0.16] | 51 % | 2.1 (16.3)  | 0.036 | 0.464 | 0.858 |

What this means:

1. **There are two regimes, split exactly 50/50 within each method.**
   - **Regime A (`full_watermark=True`):** the whole image was watermarked. Then one or more **clean holes**
     were restored from the original (hole area: median 6 %, maximum 49 %). Every image has a hole.
   - **Regime B (`full_watermark=False`):** a **small watermarked patch** sits on a clean image (median
     4 % of the image, maximum 44 %). About half of these patches are exact rectangles. The rest are
     object-shaped (for example a horse, a boat or a plant), or a rectangle with a cut-out.
   - The test metadata has no regime flag, so we must predict the regime. Section 3, M0 covers this.
2. **The small region decides the score.** If you predict the wrong constant for the regime, you score
   ≈ 0.05. The right constant scores ≈ 0.45, because the small class has IoU = 0. The full gain from 0.45
   to 1.0 comes from finding the small region. **Rule: always predict a non-empty small region.** In
   regime A, at least guess a hole. In regime B, at least guess a patch. A wrong guess costs only a few
   percent of the large class's IoU. A partly right guess gains up to +0.5 on that image.
3. **The regime-B rectangle alone gives a lot.** Predicting the exact GT bounding box in regime B scores
   0.86–0.88. So a good rectangle search is worth more than precise carving.
4. **Shapes:**
   - Regime-A holes are random blobs (amoeba or brush-stroke style, as in inpainting masks), rectangles or
     object silhouettes (people, phone booth).
   - Regime-B regions are thin strips (up to 16:1, also along the image border), small squares or single
     objects.
5. **There are no visible editing traces.** The `full_watermark` flag says "a part restored to the clean
   original", and the montage and high-pass views show no seams. So this is a pure **watermark-signal**
   problem. Generic forensic detectors (TruFor, CAT-Net, MVSS-Net) should not help. Do not spend time on
   them.
6. **Boundaries matter.** A median regime-B patch covers about 25 x 25 cells. A one-cell boundary error
   costs about 15 % of that region's IoU.

### Inferred generation process (replicate it for synthetic data)

```
Regime A: x = E_full(c, m), then x[H] = c[H] for the hole mask H.            GT = 1 - H
Regime B: x = c, then x[R] = E(c[R], m) for a rectangle R (aspect ≤ 16, ≤ 2 for TrustMark),
          optionally kept only on an object mask O inside R (the rest restored).   GT = R or O
```

`E` is the embedder from the starter kit (`wm.embed(img, bits)`). The kit README says these are the exact
embedders used to build the dataset.

---

## 2. Relevant internals of the three watermarks (from the starter kit and the upstream code)

| | PixelSeal | TrustMark (Q) | StegaStamp |
|---|---|---|---|
| bits | 256 | 100 (no ECC) | 100 |
| embed processing resolution | 256 | 256 | 400 |
| residual | **luma (Y) channel only**, U-Net, upsampled, **JND-attenuated**, `scaling_w=0.2`. Most invisible. | RGB, per-channel mean removed, bilinear upsample, 1 % feathered paste | RGB, channel mean removed, bilinear upsample. **Most visible.** |
| decoder input | resize to 256 | resize to 245 | resize to 400 (bicubic) |
| extractor | ConvNeXt-tiny, then **mean-pool, then linear** | ResNet-50, then **GAP, then fc** | STN + conv + dense (no GAP) |
| per-position map | exact: `features()["per_position_logits"]`, 8 x 8 | exact: `features()["per_position_logits"]`, 8 x 8 (layer4) | none. Use occlusion, gradients (onnx2torch) or sliding windows |
| extra | channel 0 = detection logit (also available per position) | **watermark remover** `tm.remove_watermark()` (TrustMark-RM) | ONNX graph. Any internal tensor can be exposed as an output |

Consequences:

- The residual of a patch embedded on rectangle `R` is computed at the processing resolution, then resized
  to the size of `R`. So the residual pattern is **tied to the geometry of R**. Decoding or correlating
  with the wrong R decorrelates the pattern. This makes the true R a sharp optimum (see M3).
- For PixelSeal and TrustMark, the global logits are the **average of the per-position logits** (the head
  is linear after pooling). So the per-position agreement with the known message is an exact localization
  signal, not a heuristic CAM.

---

## 3. Methods, ranked by value per hour

### M0: Regime classifier and baseline (do first, about 1 h)
- Decode the full image and compute the matched-filter score against the known message:
  `s = mean_k (2 m_k - 1) * logit_k` (or bit accuracy).
- Regime A (94 % watermarked) should decode almost perfectly. Regime B (4 %) should be near chance.
  Pick the threshold per method on train.
- Baseline submission: all-ones if A, all-zeros if B, which gives **≈ 0.455** if the classifier is
  perfect. Then add the "always guess a small region" rule as soon as any evidence map exists.

### M1: Differential matched filter on the known residual (regime A first, about 2–3 h)
In regime A the embedding geometry is known (the whole image), so we can compute the expected residual:
- `r_m = E(x, m) - x` ≈ the residual actually present, because x ≈ c (PSNR > 40 dB for PixelSeal and
  TrustMark).
- `r_j = E(x, m_j) - x` for J = 4–8 random messages. Their mean `r̄` estimates the content-driven part of
  the encoder output, which appears in both hypotheses.
- `d = HP(x)`, where `HP = x - GaussianBlur(x, σ≈1–2)` or `x - denoiser(x)`. Use luma only for PixelSeal.
- Local normalized correlation over a window σ_w ≈ 4–8 px:
  `score(p) = G * [d · (r_m - r̄)] / sqrt(G * d² · G * (r_m - r̄)²)`.
  Watermarked pixels give a positive score. Holes give about 0.
- Variant: compute it at the embed processing resolution (256 or 400). The residual is band-limited
  there, so the content noise is lower.
- Pool to the 128 grid, tune the threshold per method on train, then apply M7 post-processing.
- Expected strength: StegaStamp > TrustMark > PixelSeal.
- The same filter works in regime B once R is known (M3): use `E(x[R], m) - x[R]`.

### M2: Decoder evidence maps (about 2 h, works in both regimes)
- **PixelSeal and TrustMark:** `a(h, w) = Σ_k (2 m_k - 1) · ℓ_k(h, w)` from `per_position_logits`.
  Also test the PixelSeal detection channel 0 per position. To densify the 8 x 8 map, use
  **shift-and-stitch**: feed copies of the processing-resolution image shifted by stride/4 px, which gives
  a 32 x 32 map from 16 passes.
- **StegaStamp:** occlusion map. Replace a patch with `x - r_m` (the de-watermarked estimate) or with a
  blurred patch, then measure the drop in matched-filter score. Alternatively, convert the ONNX graph with
  `onnx2torch` and use gradient × (x − x̂_clean).
- Use these maps as seeds for M3 and as input channels for M5.

### M3: Rectangle search for regime B (about 3–4 h)
- Objective 1, for the coarse stage: `S(R)` = matched-filter score of
  `decode(resize(x[R] → decoder resolution))` against the known message, as a z-score against random
  messages.
- Coarse stage:
  - Sides {32, 48, 64, 96, 128, 192, 256, 384} px × aspect {1, 2, 4, 8, 16} × both orientations, stride
    = side/4.
  - Respect the aspect limits: TrustMark ≤ 2, the others ≤ 16.
  - Batch all crops on the GPU (thousands of crops per image are fine).
  - Seed extra candidates from the peaks of the M2 maps.
- Refine stage: coordinate ascent on (x0, y0, x1, y1) with steps of 16, 8, 4, 2, 1 px.
  - First maximize S(R).
  - Then switch to **objective 2**, the residual correlation `C(R) = corr(HP(x[R]), E(x[R], m) - x[R])`.
    It peaks sharply at the exact geometry, because a scale or offset mismatch decorrelates the pattern.
- Output R̂, then carve the object inside it with M4. Expected gain: up to about 0.86 on regime B if R̂ is
  exact (see the table).
- Risk: the decoders are trained to be robust to crops and resizes, so S(R) has a wide plateau. That is
  why C(R) does the final refinement.

### M4: Carving and shape snapping (about 2–3 h)
- In R̂ (regime B) or in the whole image (regime A), threshold the M1 map computed with the correct
  geometry.
- **Segment-then-verify:**
  1. Generate proposals: SAM automatic masks, SAM masks prompted with the bounding box of each map
     component, rectangles, and the thresholded blob itself.
  2. Score each proposal by its watermark evidence (mean map inside vs. outside, or Δ matched-filter score
     when the proposal is removed or kept).
  3. Keep the best proposal.
  - SAM fits object-shaped regions (horse, boat, people). Random blobs do not follow edges, so accept a
    SAM mask only when it raises agreement with the evidence map.
- Rectangle snap: if a component fills more than 90 % of its bounding box, output the box.

### M5: Learned localizer on synthetic data, WAM-style (main GPU track, about 6–10 h)
- **Generator** that replicates Section 1, using the kit embedders:
  - Clean images: COCO val2017 or DIV2K, as random 512 crops. Add the regime-B train images (clean
    outside the known GT) and optionally the test regime-B images (about 96 % clean, so the label noise is
    small).
  - Regime A: embed the full image with a random or known message, then paste clean holes back. Draw holes
    from a shape library: the 510 train GT shapes, random blobs and strokes, rectangles, and COCO or SAM
    object masks.
  - Regime B: random rectangle within the aspect limits. Embed the crop, then optionally keep only an
    object mask inside it.
  - Labels: `downsample_mask` from `wmloc.utils`.
  - Validate the generator by comparing M1 and M2 statistics on synthetic vs. train images.
- **Model:** UNet (`segmentation_models_pytorch`, ResNet-34 or EfficientNet encoder) at the **full 512
  resolution**. Downscaling destroys the residual. Output: 128 x 128 logits.
  - Inputs: RGB, HP(x), the regime-A residual-correlation map from M1, and the upsampled M2 maps.
  - Train one model per method, or one shared model with a method embedding.
  - Loss: BCE plus a per-image **Lovász / soft-IoU** term, which matches the per-image mIoU.
- Alternative: fine-tune the method's own extractor with a pixelwise head (as in Watermark Anything).
  The backbone already "sees" the watermark.
- **The whole train split stays a clean validation set**, because training uses synthetic data only.
  Optionally fine-tune on train with 5-fold cross-validation at the end.
- Rough cost: generation takes minutes on an A100. Training takes 1–3 h per model.

### M6: TrustMark remover (cheap extra, about 1 h)
- `x - tm.remove_watermark(x)` estimates the residual without knowing the message. Use its local energy,
  or its correlation with `r_m`, as one more map.
- In regime A it applies to the full image. In regime B, run it on M3 candidate rectangles.

### M7: Post-processing tuned to the metric
- Threshold per (method, regime), chosen to maximize **train mIoU** (not pixel accuracy).
- **Never output an empty minority class.** If thresholding gives nothing, output the top-k % cells of the
  evidence map. Take k from the train median: about 6 % of cells for holes, 4 % for patches.
- Morphology: drop components smaller than N cells, fill pinholes, keep the top-scoring components.
- Ensemble: average the calibrated maps of M1, M2, M5 and M6, then binarize.

### Not worth it here
- Generic splicing or inpainting forensics (there are no editing traces; see Section 1, point 5).
- Test-time augmentation with flips or rotations (the decoders are not invariant to them).
- Brute-force analysis-by-synthesis over masks (too slow for 20 h).

---

## 4. Validation and submission

- Score everything with `wmloc.utils.score` on the train split. Always report **per method × regime**.
  Keep a short results log (method, settings, train mIoU, public LB score).
- The public leaderboard uses only 30 % of test (about 306 images). Do not tune to it. Trust train
  validation.
- Submission: `{"image_names": [...], "masks": bool tensor [1020, 128, 128]}`, about 17 MB, limit 25 MB.
  Use `submit(path)` from `kit/submission_template.py`. Read the API key from an environment variable;
  **never commit it**.
- Task id: `34-watermark-localisation-image`. Leaderboard: http://34.147.229.55/leaderboard_page

## 5. Order of work (now until Sun 12:00)

| when | track | goal | expected train mIoU |
|---|---|---|---|
| first 1 h | setup + M0 | models load, regime classifier, first submission | ≈ 0.455 |
| +1–4 h | M1 + M2 + M7 on regime A | find holes, always guess a hole | regime A up from 0.45 |
| +1–6 h (parallel) | M3 on regime B | rectangle search | regime B toward 0.8 |
| +3–12 h (GPU, parallel) | M5 | synthetic generator, UNet per method | best single model |
| +10–17 h | M4, M6, ensemble | carving, SAM snap, fused maps | — |
| +17–19.5 h | freeze | final submission by about 11:00, 6-min pitch slides | — |

Team split for three people:
- **P1:** regime A (M1, M2, M6).
- **P2:** regime B (M3, M4).
- **P3:** generator and learned model (M5).
- Shared: the eval harness and M7.

### Quick checks before building (de-risk, about 30 min with the models loaded)
1. Plot the histograms of full-image matched-filter scores for regime A vs. B, per method. Are they
   separable?
2. Is `E(x, m) - x` really uncorrelated with HP(x) inside train holes, and correlated outside them?
3. On regime-B rectangle cases, decode the GT crop: is the bit accuracy high? How fast does it drop when
   the crop is shifted or scaled by 2, 4, 8 or 16 px? This shapes the M3 objective.
4. Is the embedder deterministic? Embed the same input twice (PixelSeal on GPU may not be bit-exact).

---

## 6. Setup

### Local (CPU only, for analysis and debugging)
```bash
cd watermarking-localization
uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download as s; \
  s('SprintML/Hackathon_Watermark_Localisation_Task', repo_type='dataset', local_dir='data')"
unzip -q data/starter_kit.zip -d kit
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
sed 's/^onnxruntime-gpu.*/onnxruntime/' kit/requirements.txt > /tmp/req-cpu.txt && uv pip install -r /tmp/req-cpu.txt scipy
python kit/download_models.py          # default target kit/models, which load_watermark() finds by itself
python -c "import sys; sys.path.insert(0, 'kit'); from wmloc.data import unzip; \
  unzip('data/train.zip', 'data/release'); unzip('data/test.zip', 'data/release')"
python analysis/mask_stats.py --data data/release
```

### JURECA (GPU)
- `hackathon_setup.sh` downloads the dataset to `$TEAM_DIR/Hackathon_Watermark_Localisation_Task`. That
  repo has **no top-level `requirements.txt`** (it is inside `starter_kit.zip`), so the script **skips
  creating the venv**. Create it by hand.
- Run `download_models.py` on a **login node**, because compute nodes usually have no internet.
- TrustMark downloads its weights on first use, so call `load_watermark("trustmark")` once on the login
  node too.

```bash
cd /p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task
unzip -q starter_kit.zip -d kit
uv venv -p 3.11 .venv && source .venv/bin/activate && uv pip install -r kit/requirements.txt scipy
python kit/download_models.py
python -c "import sys; sys.path.insert(0, 'kit'); from wmloc.watermarks import load_watermark; load_watermark('trustmark', device='cpu')"
python -c "import sys; sys.path.insert(0, 'kit'); from wmloc.data import unzip; unzip('train.zip', 'release'); unzip('test.zip', 'release')"
mkdir -p output && sbatch main.sh   # main.sh as in the hackathon README, section 5
```

### Proposed layout of this folder
```
watermarking-localization/
  PLAN.md                 this file
  analysis/mask_stats.py  train statistics and oracle baselines (section 1)
  wl/                     our code: evidence maps (M1, M2, M6), rect search (M3), carving (M4), postproc (M7)
  train_localizer.py      M5
  predict.py              writes submission.pt (uses kit/submission_template.submit)
  data/ kit/ (kit/models/) git-ignored
```

## References
- PixelSeal: https://arxiv.org/abs/2512.16874 (code: github.com/facebookresearch/videoseal, card `pixelseal`)
- TrustMark: https://arxiv.org/abs/2311.18297 (pip `trustmark`, includes the TrustMark-RM remover)
- StegaStamp: Tancik et al., CVPR 2020 (ONNX from github.com/umd-huang-lab/WAVES)
- Watermark Anything (WAM): Sander et al., ICLR 2025. The pixelwise extractor recipe for M5.
- Block-level watermark detection maps: https://arxiv.org/abs/2512.14994
- Local-watermark robustness benchmark (TrustMark adapted to boxes): https://arxiv.org/pdf/2609.16832
