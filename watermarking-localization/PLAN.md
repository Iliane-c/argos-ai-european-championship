# Watermark Localization: Shared Plan (2 people)

Results and the backlog live in [EXPERIMENTS.md](EXPERIMENTS.md). This file says **who does what, how we
evaluate, and how we run experiments in parallel**.

The hackathon ends **Sun 22 Nov, 12:00**. The final submission is due by **11:00**.

## 0. In short

- **Main approach:** a segmentation network. It takes the 512 x 512 image and outputs a 128 x 128 mask
  (1 = watermarked). We train it on the 510 real train images plus **synthetic images we create with the
  starter-kit embedders**, which gives unlimited labelled data.
- **Evaluation:** a fixed 5-fold split of the real train images (`splits/folds.json`). Fold 0 is for quick
  checks; all 5 folds are for final choices. The public leaderboard is only a sanity check.
- **Two roles:**
  - **A = Data & Evaluation:** evaluation script, synthetic data, augmentation, extra input channels.
  - **B = Models & Training:** training script, architectures, losses, post-processing, ensembles.
- **Rhythm:** each person keeps up to 4 GPU jobs running. We meet at fixed sync times (Section 7), read
  `EXPERIMENTS.md` together, keep or drop, and launch the next batch.

## 1. The task as a segmentation problem

```
image 512x512 (+ method, + message)  -->  segmentation model  -->  128x128 probabilities  -->  threshold  -->  mask
```

- **Train target:** the official `mask128` from the train set (a cell is 1 if at least 8 of its 4 x 4
  pixels are watermarked). Then no conversion step is needed.
- **Score:** per image, the mean of IoU(clean) and IoU(watermarked), then the mean over images. The
  official function is `kit/wmloc/utils.py: miou_image`.

What the data looks like (details: `uv run analysis/explore_dataset.py`, which writes `output/explore/`):

| kind (train flag `full_watermark`) | share | watermarked area | what the model must find |
|---|---|---|---|
| `full` | 50 % | 51–100 % (median 94 %) | **clean holes** where objects were removed or added (median 6 %). Silhouettes and blobs, often in several pieces, often at the border. |
| `patch` | 50 % | 0.3–44 % (median 4 %) | **a watermarked patch**. Half are exact rectangles (thin strips up to 16:1), half are object shapes. |

- **The three methods have the same statistics.** Test: 340 images per method, with no kind flag and no
  mask.
- **The small part is half the score.** If the model marks nothing there, that half is 0. This explains
  the baseline numbers in EXPERIMENTS.md (E00).
- **The watermark is a faint pixel-level pattern.** So **never downscale the input below 512**, and treat
  any resize of a watermarked image as harmful (Section 4).

## 2. Who does what

| | **A: Data & Evaluation** | **B: Models & Training** |
|---|---|---|
| first hour | `eval.py`, loader for real data, E00 and E01 | `train.py` (config-driven), smoke test of a U-Net on fold 0 |
| main work | synthetic generator, augmentation, extra input channels, kind feature | architecture sweep, losses, per-method vs shared models |
| end game | threshold choice by CV, post-processing | ensembling, test inference, submission file |
| experiments | data side: "same model, different data" | model side: "same data, different model" |

**Interface: agree on this in the first hour, then never break it.**

| what | format | path |
|---|---|---|
| fold split | `{"n_folds": 5, "folds": [[names...], ...]}` (already committed) | `splits/folds.json` |
| training data shard | `.npz` with `image` uint8 [N,512,512,3], `mask128` uint8 [N,128,128], `method`, `message`, `kind`, `source` (`real` / `synth-v1` / ...) | `data/synth/<version>/shard_XXX.npz` (git-ignored) |
| predictions | `.npz` with `names` [N] and `probs` float16 [N,128,128] (probability of watermarked) | `output/<ID>_<name>/val_probs.npz`, `test_probs.npz` |
| metrics | `eval.py output/<ID>_<name>` writes `metrics.json`: mean, per method, per kind, at threshold 0.5 | same folder |

Keep **probabilities**, not binary masks. That makes ensembling and threshold tuning possible later
without retraining.

## 3. Evaluation protocol (against overfitting)

1. **Fixed split.** `splits/folds.json` has 5 folds of 102 images (17 per method × kind). It was made by
   `splits/make_splits.py` and must never be regenerated.
2. **Quick experiments:** train on folds 1–4 (plus synthetic data), validate on **fold 0**. Report the
   mean, PS / TM / SS, and Full / Patch.
3. **Finalists:** full 5-fold CV. Report the mean ± std. A gain smaller than 1 std is noise, so keep the
   simpler model.
4. **Synthetic-only models** are evaluated on **all 510 real images**. This measures the synthetic-to-real
   gap and is our most honest generalization number.
5. **Threshold and post-processing** are chosen on out-of-fold predictions, never on the fold being
   reported.
6. **Public leaderboard (30 % of test, about 306 images):** submit only at syncs, to check the format and
   that LB ≈ CV. Never pick a model because of the LB.
7. Record every experiment, including failures, with its parent experiment, so we always know what one
   change did.

## 4. Training data: real, synthetic, augmentation

### Synthetic data (A, the biggest lever)
The kit embedders are the ones used to build the dataset, so we can recreate its process with labels for
free:
- **Clean images.** Use the 255 real `patch` images (clean outside their known patch) and a public image
  set (COCO / DIV2K / Open Images, resized to 512 **before** embedding). Generated-looking images also fit,
  because the dataset images look AI-generated.
- **`full` recipe:**
  1. Embed the whole image with a random message (`wm.embed`).
  2. Choose a hole mask.
  3. Put the clean pixels back inside the hole.
  - v2 option: fill the hole with an inpainting model such as LaMa, because real holes are real edits (in
    one example, cones were removed from a ski slope).
- **`patch` recipe:**
  1. Pick a random box. Sizes follow `output/explore/fig5`; aspect ratio ≤ 16 (≤ 2 for TrustMark).
  2. Embed the crop at its own size and paste it back.
  3. In 50 % of cases, keep only an object mask inside the box.
- **Mask library:**
  - The 510 real masks, freely flipped, rotated, shifted and scaled (masks are only shapes).
  - Random blobs and strokes, rectangles, and COCO or SAM object masks.
- **Labels:** `downsample_mask` from `kit/wmloc/utils.py`. Use 50 / 50 kinds and equal methods.
- **Volume:** about 20k images per method per version. Generate them **on JURECA scratch**: the local disk
  has about 2 GB free.

### Augmentation: what is safe for a watermark

| augmentation | applied to | verdict | why |
|---|---|---|---|
| Instagram-style filters (`pilgram` lib), color jitter, contrast, gamma, blur, JPEG, crop, flip, resize | the **clean** image, **before** embedding | ✅ safe | the watermark is added afterwards; this only diversifies content |
| flip, rotate, shift, scale, elastic warp | the **mask** before compositing | ✅ safe | masks are just shapes |
| copy-paste between real images (integer shift, **no resize**) | real train images | ✅ safe | it mimics object addition and transplantation exactly; labels stay correct |
| random crops at native resolution (e.g. 384 px, no resize) | any image | ✅ safe | the pattern is untouched |
| Instagram filters, color jitter, JPEG | a **watermarked** image (real, or synthetic after embedding) | ⚠️ A/B test only | it weakens the pattern, while test watermarks are intact |
| flips, 90° rotations | a watermarked image | ⚠️ A/B test only | the pattern orientation never appears flipped in test |
| resize, rescale, rotation by arbitrary angles | a watermarked image | ❌ avoid | the pattern is tied to its pixel grid; resampling destroys it |

## 5. Models to try (B)

All models: input 512 x 512, no downscaling, output 128 x 128 logits.

| # | model | why | source | cost |
|---|---|---|---|---|
| 1 | **U-Net, ResNet-34 encoder** | simple, strong baseline | `segmentation_models_pytorch` (smp) | low |
| 2 | **SegFormer MiT-B2** (B5 later) | strong general segmenter; output stride 4 gives exactly 128 x 128 at 512 input | HF `transformers` / smp | medium |
| 3 | UNet++, DeepLabV3+, FPN with EfficientNet-B4 or ConvNeXt-T | a sweep with one config flag | smp | low |
| 4 | **TruFor** (Noiseprint++ + SegFormer) | top forgery-localization model; pretrained to see low-level noise traces, which is what a watermark is | github `grip-unina/TruFor` | medium |
| 5 | IML-ViT | high-resolution ViT for manipulation localization | github `SunnyHaze/IML-ViT` | medium–high |
| 6 | **Watermark Anything (WAM) extractor**, fine-tuned | the only public model trained to output "watermarked or not" per pixel | github `facebookresearch/watermark-anything` | medium |
| 7 | each method's own decoder backbone + pixel head | already tuned to that exact watermark (PixelSeal ConvNeXt, TrustMark ResNet-50) | starter kit | medium |
| + | SAM 2 as mask refiner | sharp object boundaries for object-shaped targets | post-processing | low |

Knobs to test, one experiment each, against the best model so far:
- **Input channels:** RGB, vs. RGB + high-pass/SRM filters, vs. RGB + **message-aware channels** from A.
  The message-aware channels are the local correlation with `embed(x, m) − x` and the decoder's
  per-position agreement with the known message.
- **One model per method** vs. a shared model with a method input.
- **Loss:** BCE + Dice vs. per-image 2-class Lovász (the closest to our metric).
- **Kind head:** an image-level output "mostly watermarked or patch", or the whole-image decode score as
  an input.
- **Post-processing:**
  - If the small part is empty, mark the top-k % cells.
  - Snap near-rectangles to boxes.
  - Remove specks.

## 6. How to run an experiment

1. **Claim** a backlog row in EXPERIMENTS.md: put your name and the next free ID, then push.
2. **Config:** `configs/E07_segformer_b2.yaml`. It holds only the differences from its parent experiment.
3. **Launch:** `sbatch jobs/train.sh configs/E07_segformer_b2.yaml`. Each job takes 1 GPU, and one JURECA
   node has 4 A100s, so keep **at most 4 jobs per person**.
4. **Outputs:** the job writes `output/E07_segformer_b2/` with `config.yaml`, `log.txt`, `val_probs.npz`,
   `metrics.json`, `test_probs.npz` and `model.pt`.
5. **Log:** run `python eval.py output/E07_segformer_b2`, fill in the row, and give a verdict (Keep / Drop
   / Investigate).
6. **Commit** the config and the row as `E07: <change>`. Run `git pull --rebase` first.

Rules:
- One change per experiment.
- Fixed seeds.
- Fold 0 first; 5 folds only for finalists.
- Before sleeping, launch the long runs.

## 7. Timeline and syncs (each sync is 15 min, at a fixed time)

| time | A: Data & Eval | B: Models | sync |
|---|---|---|---|
| 16:45–18:30 | `eval.py`, real-data loader, E00 ✅, E01 | `train.py` + U-Net-R34 smoke test on fold 0 (CPU is fine while compute is down) | **18:30**: interface frozen, first numbers |
| 18:30–21:30 | synthetic generator v1 + copy-paste aug; generate synth-v1 on JURECA | architecture sweep on real data (U-Net, SegFormer-B2, UNet++-EffB4, TruFor init) | **21:30**: pick the 2 best archs |
| 21:30–00:30 | message-aware channels; augmentation A/B tests | the 2 best archs on synth-v1 + real; loss sweep | **00:30**: first serious LB submission |
| 00:30–08:00 | synth-v2 (fixes seen at 00:30); sleep in shifts | overnight: 5-fold finalists, per-method models | **08:00**: read the overnight results |
| 08:00–10:30 | threshold by CV, post-processing | ensemble of finalists, test inference | **10:30**: choose the final by CV |
| 10:30–12:00 | final submission **by 11:00**, slides | slides (6-min pitch) | end 12:00 |

At each sync:
1. Read the new rows.
2. Give each a verdict.
3. Update "Decisions".
4. Each person picks the next ≤ 4 experiments.
5. Submit to the LB only if CV improved.

## 8. Setup

Already done in this checkout: `data/` (zips + `data/release/{train,test}`) and `kit/` (starter kit).
Both are git-ignored.

Local (CPU):
```bash
cd watermarking-localization
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
sed 's/^onnxruntime-gpu.*/onnxruntime/' kit/requirements.txt > /tmp/req-cpu.txt
uv pip install -r /tmp/req-cpu.txt scipy matplotlib segmentation-models-pytorch
python kit/download_models.py      # into kit/models; load_watermark() finds it there
```

Fresh checkout:
```bash
uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download as s; \
  s('SprintML/Hackathon_Watermark_Localisation_Task', repo_type='dataset', local_dir='data')"
unzip -q data/train.zip -d data/release && unzip -q data/test.zip -d data/release
unzip -q data/starter_kit.zip -d kit
```

JURECA:
- `hackathon_setup.sh` downloads the data but creates **no venv** for this task, because the
  `requirements.txt` is inside `starter_kit.zip`.
- Download models and weights on a **login node**, since compute nodes usually have no internet.

```bash
cd /p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task
unzip -q starter_kit.zip -d kit
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -r kit/requirements.txt scipy matplotlib segmentation-models-pytorch
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
  eval.py                       A: official metric, per method x kind
  synth/                        A: generator + augmentation
  train.py  configs/  jobs/     B: training, one YAML per experiment, Slurm scripts
  predict.py                    B: test probs -> submission.pt (submit() from kit/submission_template.py)
  data/  kit/  output/          git-ignored
```

## References
- PixelSeal: https://arxiv.org/abs/2512.16874 · TrustMark: https://arxiv.org/abs/2311.18297 · StegaStamp:
  Tancik et al., CVPR 2020
- Watermark Anything: https://arxiv.org/abs/2411.07231
- TruFor: Guillaro et al., CVPR 2023 · IML-ViT: https://arxiv.org/abs/2307.14863
- SegFormer: Xie et al., NeurIPS 2021 · smp: github.com/qubvel-org/segmentation_models.pytorch
- Instagram filters for PIL: github.com/akiomik/pilgram
