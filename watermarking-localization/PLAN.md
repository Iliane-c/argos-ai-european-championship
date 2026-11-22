# Watermark Localization: Shared Plan

Results, the backlog and decisions live in [EXPERIMENTS.md](EXPERIMENTS.md). This file explains **how we
work**: build a base library once, then run many experiments in parallel, both of us, in rounds.

The hackathon ends **Sun 22 Nov, 12:00**. The final submission is due by **11:00**.

## 0. How we work

1. **Build the base library together (about 3 h).** We split it by module only to go faster (Section 2).
   Nobody owns an experiment area.
2. **Run experiments in rounds.** Either of us can run anything from the backlog:
   1. Claim a row in EXPERIMENTS.md.
   2. Write a config file.
   3. Launch it.
   4. At the next sync, compare and decide what to do next.
3. **Everything outputs the same file**: 128 x 128 probabilities per image. Trained models, training-free
   methods and ensembles all go through the same evaluation and can be mixed freely.

## 1. The problem in one screen

```
image 512x512 + method + message  -->  [model or training-free method]  -->  128x128 probabilities  -->  threshold  -->  mask
```

- **Train target:** the official `mask128` (a cell is 1 if at least 8 of its 4 x 4 pixels are
  watermarked).
- **Score:** per image, the mean of IoU(clean) and IoU(watermarked); then the mean over images
  (`kit/wmloc/utils.py`).
- **Data details:** `uv run analysis/explore_dataset.py` writes them to `output/explore/`.

| kind (train flag) | share | watermarked area | what must be found |
|---|---|---|---|
| `full` | 50 % | 51–100 %, median 94 % | **clean holes** (median 6 %): object silhouettes or blobs, often several, often at the border |
| `patch` | 50 % | 0.3–44 %, median 4 % | **a watermarked patch**: half rectangles (strips up to 16:1), half objects |

- The three methods have the same statistics.
- Test: 1,020 images, 340 per method, with no kind flag.
- The small part is half of every image's score (see E00 in EXPERIMENTS.md).
- The watermark is a faint **pixel-level** pattern. It was computed at the method's processing resolution
  (256 for PixelSeal and TrustMark, 400 for StegaStamp), then resized to the watermarked box.

## 2. Base library (build first, then freeze the interfaces)

| # | module | gives | built by | ready by |
|---|---|---|---|---|
| B1 | `wl/data.py` | real dataset with folds, synthetic shard dataset, augmentation hook, extra input channels | builder 1 | 18:30 |
| B2 | `wl/evaluate.py` | official metric; per method × kind; threshold from out-of-fold predictions; `metrics.json` | builder 2 | 18:30 |
| B3 | `wl/models.py` | model registry (Section 5): smp (any architecture × encoder), HF / timm backbones; wrappers for extra input channels, method or message conditioning, extra heads | builder 2 | 18:30 (smp), 20:00 (rest) |
| B4 | `wl/train.py`, `wl/losses.py` | YAML-driven training: AMP, AdamW + cosine, EMA, BCE / Dice / Lovász / auxiliary losses; writes `val_probs.npz` and `test_probs.npz` | builder 2 | 18:30 |
| B5 | `wl/synth.py`, `wl/augment.py` | synthetic generator (Section 4), mask library, watermark-safe augmentation, copy-paste | builder 1 | 20:00 |
| B6 | `wl/features.py` | message-aware maps: decoder agreement, matched filter, TrustMark-RM residual, SRM filters; cached per image | builder 1 | 20:00 |
| B7 | `wl/analytic.py` | training-free localizers (Section 6) that write probabilities in the standard format | builder 1 | 21:00 |
| B8 | `wl/postproc.py`, `wl/ensemble.py`, `predict.py` | thresholds, clean-up, ensembles, `submission.pt` + `submit()` | builder 2 | 20:00 |
| B9 | `jobs/run.sh`, `jobs/sweep.py`, `wl/report.py`, `wl/gallery.py` | Slurm launcher, grid to job array, ranked table of all runs, worst-case image galleries | builder 2 | 20:00 |

**Checkpoint at 18:30:** one tiny run goes end to end on CPU: real data, then U-Net, then probabilities,
then `metrics.json`, then `submission.pt`. After that, interfaces only grow; nothing gets renamed.

**Standard files:**

| what | format | where |
|---|---|---|
| folds | `splits/folds.json` (committed, never regenerated) | repo |
| training shard | `.npz`: `image` uint8 [N,512,512,3], `mask128` uint8 [N,128,128], `method`, `message`, `kind`, `source`, optional cached feature maps | `data/synth/<version>/` |
| predictions | `.npz`: `names` [N], `probs` float16 [N,128,128] | `output/<ID>/val_probs.npz`, `test_probs.npz` |
| metrics | `metrics.json`: mean, PS/TM/SS, full/patch, threshold used | `output/<ID>/` |

**An experiment is one config file**, which inherits `configs/base.yaml`:
```yaml
id: E104_segformer_b2_synth
parent: E101_unet_r34
data:  {train: [real, synth-v1], train_folds: [1, 2, 3, 4], val: f0, synth_ratio: 0.8}
input: {channels: [rgb, srm, agreement]}          # rgb | srm | agreement | matched | native_view
model: {name: smp, arch: Segformer, encoder: mit_b2, cond: none, heads: [mask]}   # cond: none | method | message
loss:  {bce: 1.0, dice: 1.0, lovasz: 0.0, aux_residual: 0.0}
train: {epochs: 20, batch: 16, lr: 3.0e-4, ema: true, seed: 0}
post:  {threshold: oof, never_empty: true, rect_snap: false}
```

## 3. Evaluation (against overfitting)

1. **Fixed split.** `splits/folds.json` has 5 folds × 102 images (17 per method × kind).
2. **Quick runs:** train on folds 1–4 (+ synthetic), report on **fold 0**.
3. **Finalists:** full 5-fold CV, mean ± std. A gain below 1 std is noise, so keep the simpler model.
4. **Models trained on synthetic data only:** report on **all 510 real images**. This is the honest
   synthetic-to-real number.
5. **Thresholds and post-processing:** choose them on out-of-fold predictions only.
6. **Public leaderboard (30 % of test):** submit at syncs only, to check the format and that LB ≈ CV. Never
   pick a model because of the LB.
7. **After every round, look at the 50 worst images** per method × kind (`wl/gallery.py`). They tell us
   what to try next.

## 4. Data: real, synthetic, augmentation

**Synthetic data** recreates the dataset process with the kit embedders, so labels come for free:

- **Clean images:**
  - the 255 real `patch` images (clean outside their known patch);
  - COCO / DIV2K / Open Images, resized to 512 **before** embedding;
  - AI-generated images, because the dataset looks generated.
- **`full` recipe:** embed the whole image with a random message, then put clean pixels back in the holes.
  v2: fill the holes with an inpainting model (LaMa or SD-inpainting), because real holes are real edits.
- **`patch` recipe:** embed a random box (aspect ≤ 16, ≤ 2 for TrustMark) at its own size. In half of the
  cases, keep only an object mask inside the box. v2: outpainting and object transplantation done with
  real editing models.
- **Mask library:**
  - the 510 real masks, flipped, rotated, shifted and scaled;
  - random blobs and strokes;
  - rectangles;
  - COCO or SAM object masks.
- **Volume and location:** about 20k images per method per version, generated on **JURECA scratch**. The
  local disk has about 2 GB free.

**Augmentation: which transforms keep the watermark intact**

| augmentation | applied to | verdict |
|---|---|---|
| Instagram filters (`pilgram`), color jitter, contrast, gamma, blur, JPEG, crop, flip, resize | clean image, **before** embedding | ✅ safe: the watermark is added afterwards |
| flip, rotate, shift, scale, elastic warp | the mask, before compositing | ✅ safe: masks are only shapes |
| copy-paste between real images (integer shift, no resize) | real images | ✅ safe: this is exactly object addition or transplantation |
| random crops at native resolution, integer shifts | any image | ✅ safe |
| resize the watermarked box to the method's processing resolution | any image | ✅ useful extra view (`native_view`): this is what the decoders do |
| filters, color jitter, JPEG | a watermarked image | ⚠️ A/B test only: weakens a pattern that test images still have intact |
| flips, 90° rotations | a watermarked image | ⚠️ A/B test only: the pattern never appears flipped in test |
| random rescale or arbitrary rotation | a watermarked image | ❌ avoid: the pattern is tied to its pixel grid |

## 5. Model zoo (ambitious)

All models take a 512 input (never downscaled) and output 128 x 128 logits. "Cost" is for one A100.

**T1: fast baselines (first round)**
- U-Net + ResNet-34 (smp): the reference.
- SegFormer MiT-B2: its output stride 4 gives exactly 128 x 128 at 512 input.

**T2: strong general segmenters**
- SegFormer-B5; UPerNet + ConvNeXt-V2 or Swin-V2; DeepLabV3+ / UNet++ + EfficientNet-B5.
- HRNet-W48: keeps a 1/4-resolution stream (128 x 128) through the whole network, which suits
  pixel-level signals.
- Mask2Former (Swin): handles holes made of several pieces. Heavy.

**T3: foundation backbones, used through a dual-stream design (see W7)**
- DINOv3 / DINOv2 ViT-B or ViT-L + DPT head, and the SAM 2.1 Hiera encoder with LoRA or adapters.
- They give excellent object boundaries, which matters because targets are objects.
- Their 16-px patches miss the pixel pattern, so pair them with a noise stream.

**T4: forensic state of the art (made to find faint low-level traces)**
- TruFor (Noiseprint++ + SegFormer, pretrained weights available).
- IML-ViT, Mesorch, SparseViT, CAT-Net (HRNet).
- Most of them are in one codebase: IMDL-BenCo.
- Fine-tune them from their forgery-localization weights.

**T5: watermark-specific architectures (our own; the most promising)**

| # | architecture | idea |
|---|---|---|
| W1 | **Unpooled decoder** | Take the method's own extractor (PixelSeal ConvNeXt, TrustMark ResNet-50, StegaStamp via `onnx2torch`) and remove the global pooling. Its per-position bit logits are compared with the known message, giving an agreement map. A light FPN head upsamples to 128 x 128, and everything is fine-tuned on synthetic data. The backbone already "sees" the exact watermark. |
| W2 | **WAM-style multi-task** | Per-pixel output with 1 + nbits channels: a "watermarked?" channel plus the message bits, as in Watermark Anything. Final map = detection × agreement between the predicted bits and the known message. Can also start from the WAM extractor weights. |
| W3 | **Message-conditioned segmenter** | The message bits go through an MLP and condition the decoder (FiLM or cross-attention), as in referring segmentation. Train with the true message and with wrong messages, whose target is empty. The model learns to localize *this* message. |
| W4 | **Reference two-stream + correlation volume** | Stream 1: the image's high-pass residual. Stream 2: the re-embedded reference `embed(x, m) − x`. A local correlation volume over small shifts and scales (as in optical-flow networks) feeds the decoder. This is a learned matched filter. |
| W5 | **Residual-regression auxiliary head** | Also predict the watermark residual `x − c`, which is known exactly in synthetic data. This dense extra supervision sharpens the encoder. |
| W6 | **Two-stage: box, then native scale** | Stage 1 finds the watermarked box (learned box head or analytic search). Stage 2 crops it, resizes it to the method's processing resolution (where the pattern has its native scale), segments it, and maps the result back. Mirrors how the data was made. |
| W7 | **Dual-stream noise + semantic** | A full-resolution noise stream (SRM / Bayar constrained conv, no early stride) fused with a DINOv3 or SAM 2 semantic stream, as in TruFor. The noise stream finds the watermark; the semantic stream snaps to object boundaries. |
| W8 | **SAM 2 refiner** | SAM 2's mask decoder (LoRA-tuned), prompted with the coarse map of the best model. Gives object-accurate masks. |

**Knobs to test on the best model** (one change per experiment):
- input channels (`rgb`, `srm`, `agreement`, `matched`, `native_view`);
- one model per method vs. a shared model with a method input;
- loss: BCE + Dice, per-image Lovász (closest to the metric), boundary loss;
- an image-level kind head (`full` vs. `patch`);
- the synthetic-to-real ratio;
- longer schedules and EMA.

**Budget per run:**
- Quick runs: ≤ 1 h on 1 GPU.
- Finalists: ≤ 4 h.
- ViT-L at 512 is slow, so start with the B / S variants.

## 6. Beyond training

Yes, there is more than training. All of this writes the same probability files, so it can be evaluated
and ensembled like a model.

| # | idea | needs |
|---|---|---|
| T1 | **Training-free localizers** using the known message: decoder agreement maps (PixelSeal, TrustMark); occlusion with blurred patches (StegaStamp); matched filter; TrustMark-RM residual; whole-image decode to tell `full` from `patch`. Most of these run on CPU, so they can start **while the compute is down**. | kit models |
| T2 | **Rectangle search:** find the box whose crop decodes the message best, then refine it to the pixel with the matched filter. Box-only gives up to 0.867 on `patch` images (E00e). | GPU for the full test set |
| T3 | **Test-time optimization** through the differentiable decoders: refine the box or mask of the best model so the decoder evidence is maximal. Example: find the largest region that can be blurred without hurting the message likelihood (= the clean region). | GPU, moonshot |
| T4 | **Semi-supervised on test:** pseudo-label the confident test predictions and retrain (Noisy Student). The test images are given without labels. **Ask the organizers first.** | after round 2 |
| T5 | **Watermark-safe TTA:** average over integer pixel shifts (no flips, no resizes). | trained model |
| T6 | **Post-processing on out-of-fold predictions:** threshold per method × kind; never leave the small part empty (mark the top-k % cells); snap near-rectangles to boxes; remove specks; SAM 2 snap. | any probabilities |
| T7 | **Ensembles and stacking:** average probabilities; or a small CNN trained out-of-fold on [model probs + T1 maps]. | ≥ 2 runs |

## 7. Running experiments in rounds

**Experiment IDs:**
- Person 1 uses **E100–E199**, Person 2 uses **E200–E299**. No ID collisions, even when we both log at
  once.
- E00x is for shared baselines.

**Per experiment:**
1. **Claim:** add the row with status `running` before launching, then `git pull --rebase` and push.
2. **Config:** `configs/<ID>_<name>.yaml`. For grids, run `python jobs/sweep.py configs/<base>.yaml
   model.encoder=resnet34,mit_b2 loss.lovasz=0,1`. It writes the configs and **one Slurm job array**.
3. **Run:** `sbatch jobs/run.sh configs/<ID>_<name>.yaml`. One job is one GPU. Keep **≤ 4 GPU jobs per
   person** (check the limit with the organizers). Long CPU jobs go on a CPU partition or on a laptop,
   **never on login nodes**.
4. **Report:** run `python -m wl.report`. It reads every `output/*/metrics.json` in the shared team folder
   and prints a ranked table. Copy the numbers into your row and give a verdict.

**Each sync (15 min):**
1. `wl.report` + galleries.
2. Verdicts.
3. Update "Decisions".
4. Each person claims the next ≤ 4 runs.
5. Submit to the LB only if CV improved.

| time | round | focus |
|---|---|---|
| 17:00–20:00 | **0: base** | build B1–B9; end-to-end CPU smoke test at 18:30; first LB submission from T1 (kind decision + agreement maps) if ready |
| 20:00–22:30 | **1: breadth** | T1 models on real data; T1 training-free maps on all images; generate synth-v1; first W1 / W3 prototypes |
| 22:30 sync | | keep the best 2–3 directions |
| 22:30–01:30 | **2: scale** | real vs. synthetic; T2 / T4 backbones; message-aware inputs; W1–W5 |
| 01:30 sync | | choose the overnight runs |
| 01:30–08:00 | **3: overnight** | 5-fold finalists, long schedules, per-method models, W6 / W7, pseudo-labels (if allowed); sleep in shifts |
| 08:00 sync | | read the overnight results |
| 08:00–10:30 | **4: polish** | ensembles, stacking, TTA, post-processing, test-time refinement |
| 10:30 | | choose the final by CV; **submit by 11:00**; slides (6-min pitch) |

## 8. Setup

Already done in this checkout: `data/` (zips + `data/release/{train,test}`) and `kit/` (starter kit).
Both are git-ignored.

Local (CPU):
```bash
cd watermarking-localization
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
sed 's/^onnxruntime-gpu.*/onnxruntime/' kit/requirements.txt > /tmp/req-cpu.txt
uv pip install -r /tmp/req-cpu.txt scipy matplotlib segmentation-models-pytorch pyyaml pilgram
python kit/download_models.py      # into kit/models; load_watermark() finds it there
```

Fresh checkout:
```bash
uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download as s; \
  s('SprintML/Hackathon_Watermark_Localisation_Task', repo_type='dataset', local_dir='data')"
unzip -q data/train.zip -d data/release && unzip -q data/test.zip -d data/release
unzip -q data/starter_kit.zip -d kit
```

JURECA (both of us need access to the shared team folder):
- `teammate.sh` must use project `training2666`. The downloaded copy says `training2625`.
- `hackathon_setup.sh` creates **no venv** for this task, because its `requirements.txt` is inside
  `starter_kit.zip`.
- Download models and weights on a **login node**, since compute nodes usually have no internet.

```bash
cd /p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task
unzip -q starter_kit.zip -d kit
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -r kit/requirements.txt scipy matplotlib segmentation-models-pytorch pyyaml pilgram
python kit/download_models.py
python -c "import sys; sys.path.insert(0, 'kit'); from wmloc.watermarks import load_watermark; load_watermark('trustmark', device='cpu')"
unzip -q train.zip -d release && unzip -q test.zip -d release
```

## 9. Folder layout
```
watermarking-localization/
  PLAN.md  EXPERIMENTS.md
  analysis/explore_dataset.py   dataset statistics and figures
  splits/                       make_splits.py + folds.json (fixed)
  wl/                           base library: data, evaluate, models, train, losses, synth, augment,
                                features, analytic, postproc, ensemble, report, gallery
  configs/                      base.yaml + one YAML per experiment
  jobs/                         run.sh (1 GPU), sweep.py (grid -> job array)
  predict.py                    probabilities -> submission.pt (+ submit())
  data/  kit/  output/          git-ignored
```

## References
- Watermarks:
  - PixelSeal: https://arxiv.org/abs/2512.16874
  - TrustMark: https://arxiv.org/abs/2311.18297
  - StegaStamp: Tancik et al., CVPR 2020
  - Watermark Anything: https://arxiv.org/abs/2411.07231
- Segmentation:
  - SegFormer: Xie et al., NeurIPS 2021
  - HRNet: Wang et al., TPAMI 2020
  - Mask2Former: Cheng et al., CVPR 2022
  - smp: github.com/qubvel-org/segmentation_models.pytorch
- Foundation backbones: DINOv2 / DINOv3 (Meta), SAM 2 (Meta)
- Forensics:
  - TruFor: Guillaro et al., CVPR 2023
  - IML-ViT: https://arxiv.org/abs/2307.14863
  - IMDL-BenCo codebase: github.com/scu-zjz/IMDLBenCo
- Tools:
  - `pilgram` (Instagram filters): github.com/akiomik/pilgram
  - `onnx2torch`: github.com/ENOT-AutoDL/onnx2torch
