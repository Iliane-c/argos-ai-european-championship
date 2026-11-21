# Watermark Localization: Plan

The hackathon ends **Sun 22 Nov, 12:00**. Freeze the final submission by about 11:00.

## 1. The task in one picture

```
test image 512x512  +  method (pixelseal / trustmark / stegastamp)  +  secret message (bits)
        |
        v
[1] SCORE MAP: "how strongly is THIS message's watermark present here?"  (per pixel or per small block)
        |
        v
[2] SEGMENTATION: threshold + clean-up  ->  binary pixel mask (1 = watermarked)
        |
        v
[3] CELL RULE: 512x512 -> 128x128 cells; a cell is 1 if at least 8 of its 16 pixels (4x4) are 1
        |
        v
submission.pt = {"image_names": [...], "masks": bool tensor [1020, 128, 128]}
```

- Step 3 is mechanical: `downsample_mask` in `kit/wmloc/utils.py`. Alternatively, average the score per 4x4
  cell and threshold that.
- Step 1 is the real problem.
- Step 2 matters because the metric is IoU, which punishes wrong shapes.

Score per image = (IoU of the clean class + IoU of the watermarked class) / 2. The final score is the mean
over images, and every image counts the same.

## 2. What the data looks like

Run `uv run analysis/explore_dataset.py`. It writes the tables, a per-image CSV and 7 figures to
`output/explore/`. All numbers below come from it.

| split | pixelseal | trustmark | stegastamp | given |
|---|---|---|---|---|
| train | 170 | 170 | 170 | image, method, message, mask (512 and 128), `full_watermark` flag |
| test | 340 | 340 | 340 | image, method, message |

- Every image is 512 x 512 RGB PNG.
- Message length: 256 bits for PixelSeal, 100 bits for TrustMark and StegaStamp.

**Each image is either almost all watermarked or only a little** (fig1). No image falls in between.

| kind (train flag) | what happened | watermarked share | what we must find |
|---|---|---|---|
| `full` (255 images, 85 per method) | The whole image was watermarked. Then objects were removed or added, and those spots lost the watermark. | 51–100 %, median 94 % | The **clean holes**: median 6 % of the image. In 55 % of images the holes come in several pieces, and 79 % touch the border. |
| `patch` (255 images, 85 per method) | Only a box, or one object, carries the watermark (watermarked box, outpainting, transplanted object). | 0.3–44 %, median 4 % | The **watermarked patch**. Half are exact rectangles, including thin strips up to 16:1. The other half are object shapes (horse, boat, person, airplane). |

- **Shapes** (fig3): holes are object silhouettes (people, cones, bushes) or blob shapes. Patches are boxes
  or single objects.
- **Location** (fig4): patches sit mostly near the centre. Holes can be anywhere, often at the bottom or
  the border.
- **Size** (fig5): the part to find covers 50 to 8,000 cells, out of 16,384 cells in the image.
- **Cells** (fig6): along a region's border, the 50 % rule decides the cell. Border cells equal about 15 %
  of the target size, so boundary precision matters for small targets.
- **The three methods have the same distributions**, by design.
- **The test set gives no kind flag and no mask** (fig7).
- **The edits are real.** In fig2, for example, cones were removed from a ski slope and filled with snow.
  So removed or added regions may also carry editing traces. These are secondary to the watermark signal.

## 3. What the metric rewards

| prediction (train) | mIoU |
|---|---|
| all zeros | 0.252 |
| all ones | 0.248 |
| the right constant per image (all ones for `full`, all zeros for `patch`) | 0.455 |
| true bounding box of the part to find | 0.708 (`patch` 0.867, `full` 0.549) |
| perfect | 1.000 |

- The large part of each image is easy and is worth about 0.45.
- **The other about 0.5 comes from the small part** (the holes or the patch). If we mark nothing there,
  that class gets IoU 0. **Rule: always mark some small region, even when unsure.** A wrong guess costs
  little. A partly right guess gains a lot.
- **Knowing the kind helps, and it is cheap to get.** Decode the whole image once with the known message.
  A `full` image (94 % watermarked) decodes well. A `patch` image (4 %) decodes at about chance. This is not
  a separate task. It is a prior for the segmentation: it tells us whether to look for holes or for a
  patch.

## 4. Step 1: building the score map

For every test image we know the method and the message. The starter kit has the **exact embedders and
decoders** used to build the dataset. There are three ways to get a score map, from quickest to strongest.
We will probably combine them.

### A. Ask the decoder where it reads the message (no training)
- **PixelSeal and TrustMark:** the decoder averages a grid of 8 x 8 local "votes" before its final answer.
  `wm.features(x)["per_position_logits"]` returns these votes. Score per block = how well the votes agree
  with the known message.
  - The grid is coarse (64 px blocks). Shifting the input by a few pixels and repeating gives a finer map.
- **StegaStamp:** the decoder has no such grid. Instead, cover patches one at a time (occlusion) and
  measure how much the decoding gets worse.
- **Box search for `patch` images:** decode many crops of the image. The crop that matches the
  watermarked box decodes the message best. Search coarse to fine over position, size and aspect (≤ 16:1,
  and ≤ 2:1 for TrustMark).

### B. Compare with the expected watermark pattern (no training, pixel-precise)
- Re-embed the known message into the image: `r = wm.embed(x, m) - x`. This is roughly the pattern that
  should be present.
- Correlate `r` locally with the image's fine detail (`x` minus a blurred `x`). Watermarked pixels score
  high. Holes and clean areas score about 0.
- Do the same with a few random messages and subtract. This cancels the part of the correlation that comes
  from image content alone.
- `full` images: the pattern was embedded on the whole image, so this works directly.
- `patch` images: the pattern depends on the box it was embedded in. Find the box first (box search
  above), then correlate inside it.
- Expected difficulty: StegaStamp is easiest (strongest pattern). PixelSeal is hardest (brightness channel
  only, very faint).
- TrustMark extra: its built-in remover gives a pattern estimate directly:
  `x - tm.remove_watermark(x)`.

### C. Train a segmentation network (strongest; needs a GPU)
- **Unlimited labelled data:**
  1. Take clean images (COCO, or the clean areas of our train images).
  2. Embed a watermark with the kit embedders.
  3. Cut holes, or keep only a box or an object, the same way the dataset was built.
  4. Labels come for free.
- **Model:** a U-Net at the full 512 resolution (shrinking the image destroys the faint pattern). It
  outputs 128 x 128 cells.
  - Extra inputs: the maps from A and B.
  - Loss: BCE plus an IoU loss, which matches the metric.
  - Train one model per method.
- Training uses synthetic data only, so **all 510 train images stay free for honest validation**.
- Alternative: fine-tune each method's own decoder to output a mask instead of bits, like Watermark
  Anything does.

## 5. Step 2: from score map to mask

1. Threshold, per method and per kind, chosen for the best train mIoU.
2. Remove specks and fill pinholes.
3. Snap shapes:
   - If a region is almost a box, make it a box (half of the patches are exact boxes).
   - If it is object-shaped, refine it with SAM, but only when the SAM mask agrees with the score map.
4. If the small part comes out empty, mark the top 4–6 % of cells by score (the rule from Section 3).
5. Apply the official cell rule and write `submission.pt`.

## 6. Order of work

| when | what | expected train mIoU |
|---|---|---|
| first 1 h | setup, load models, whole-image decode for the kind, first submission | ≈ 0.455 |
| +1–4 h | A + B on `full` images (find the holes) | `full` part up from 0.45 |
| +1–6 h | box search + B on `patch` images | `patch` part toward 0.8 |
| +3–12 h (GPU) | C: synthetic data generator + U-Net per method | best single model |
| +10–17 h | combine A, B and C; post-processing (Section 5) | — |
| +17–19.5 h | freeze, final submission, 6-min pitch slides | — |

Split for three people:
- **P1:** `full` images (A and B, holes).
- **P2:** `patch` images (box search, B, SAM).
- **P3:** C (generator and U-Net).
- Shared: evaluation and submission script.

**Quick checks first** (30 min, once the models load):
1. Whole-image decode score for `full` vs `patch` images. Are the two clearly separated?
2. In B on `full` images: is the correlation high outside the holes and about 0 inside them?
3. On `patch` boxes: does decoding the true box give high bit accuracy? How fast does it drop when the box
   shifts?

**Validation:** score everything on train with `wmloc.utils.score`, broken down by method × kind. The
public leaderboard uses only 30 % of test, so do not tune to it.

## 7. Setup

Already done in this checkout:
- `data/` holds the zips plus `data/release/{train,test}`.
- `kit/` holds the unzipped starter kit.
- Both are git-ignored.

**Disk:** `/home` has only about 2 GB free. CPU torch plus the models need about 1.5 GB.

Remaining local steps (CPU only):
```bash
cd watermarking-localization
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
sed 's/^onnxruntime-gpu.*/onnxruntime/' kit/requirements.txt > /tmp/req-cpu.txt
uv pip install -r /tmp/req-cpu.txt scipy matplotlib
python kit/download_models.py      # into kit/models; load_watermark() finds it there
```

Fresh checkout (download the data again):
```bash
uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download as s; \
  s('SprintML/Hackathon_Watermark_Localisation_Task', repo_type='dataset', local_dir='data')"
unzip -q data/train.zip -d data/release && unzip -q data/test.zip -d data/release
unzip -q data/starter_kit.zip -d kit
```

JURECA:
- `hackathon_setup.sh` downloads the dataset into `$TEAM_DIR/Hackathon_Watermark_Localisation_Task`, but
  **creates no venv**: the `requirements.txt` is inside `starter_kit.zip`, not at the top level.
- Run the download steps on a **login node**, because compute nodes usually have no internet.
  `load_watermark("trustmark")` fetches its weights on first use, so call it there once.

```bash
cd /p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task
unzip -q starter_kit.zip -d kit
uv venv -p 3.11 .venv && source .venv/bin/activate && uv pip install -r kit/requirements.txt scipy matplotlib
python kit/download_models.py
python -c "import sys; sys.path.insert(0, 'kit'); from wmloc.watermarks import load_watermark; load_watermark('trustmark', device='cpu')"
unzip -q train.zip -d release && unzip -q test.zip -d release
mkdir -p output && sbatch main.sh   # main.sh as in the hackathon README, section 5
```

## 8. Folder layout
```
watermarking-localization/
  PLAN.md                       this file
  analysis/explore_dataset.py   dataset statistics and figures (Section 2)
  wl/                           our code: score maps (A, B), box search, post-processing
  train_localizer.py            C
  predict.py                    writes submission.pt (submit() from kit/submission_template.py)
  data/  kit/  output/          git-ignored
```

## References
- PixelSeal: https://arxiv.org/abs/2512.16874 (code: github.com/facebookresearch/videoseal)
- TrustMark: https://arxiv.org/abs/2311.18297 (pip `trustmark`, includes the watermark remover)
- StegaStamp: Tancik et al., CVPR 2020 (ONNX from github.com/umd-huang-lab/WAVES)
- Watermark Anything: Sander et al., ICLR 2025 (a watermark decoder trained to output masks; idea for C)
- Block-level watermark detection maps: https://arxiv.org/abs/2512.14994
