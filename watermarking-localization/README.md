# Watermark localization: a weekend of experiments

**Task 1 of the ARGOS team, European AI Championship, Paris round (21–22 November 2026).**
Final public score **0.6957** (mean per-image IoU), reached by a parameter-free pipeline that
was validated out of sample at every step.

This README tells the story in the order it happened: what we saw, what we decided, and what it
changed. The full log of more than 100 experiments, the failed ones included, is in
[`EXPERIMENTS.md`](EXPERIMENTS.md); the method is written up in [`METHOD.md`](METHOD.md).

---

## The problem

Each image is 512×512. Part of it carries an invisible watermark from one of three methods,
**PixelSeal**, **TrustMark** or **StegaStamp**. We know the method and even the hidden message. We
must output a 128×128 mask saying where the watermark is.

Two cases, half the images each:

- **full**: marked everywhere except a few small holes (median 6 % of the image);
- **patch**: one small marked region (median 4 %), almost always a rectangle.

The score is the IoU of both classes, averaged per image. This detail drives everything that
follows: **miss every hole of a full image and a near-perfect answer drops to 0.47.** The small
region decides the score.

We had 510 training images. We kept 102 of them (fold `f0`) out of training for the whole
weekend, as an honest validation set.

---

## Chapter 1: networks that see the watermark (Saturday)

We started with plain segmentation (U-Net, SegFormer) and quickly moved to two families of our own.

- **W2, WAM-style multitask.** Inspired by *Watermark Anything*: the network predicts, cell by
  cell, whether the watermark is present *and* the bits of the message. Since we know the true
  message, agreement between predicted and true bits becomes extra evidence. With a ResNet-50
  encoder (E143): **0.643** on f0.
- **W8, decoder in the loop.** We put the official PixelSeal and TrustMark decoders, frozen,
  inside the network, and fed their internal maps to the segmentation head. **E161: 0.654**, our
  best single model.

**What we noticed:** a shift by a whole number of pixels leaves a watermark intact, while a
rotation or a blur damages it.
**What we did:** predict on 9 shifted copies of each image and average (test-time augmentation).
**Result:** **0.654 → 0.665**.

Then an ensemble of the best members, with one shared weight per model rather than per method (fewer
free parameters, less overfitting): **0.668**.

---

## Chapter 2: the decoder as a judge (Saturday night)

**What we noticed:** when we decoded the true patch boxes of f0 with the official decoder, the
evidence was strikingly bimodal: z ≥ 9.8 or z ≤ 2.3, never in between.

**What we did:** turn the decoder into a statistical judge (`wl.verify`). For each candidate
region proposed by the network, we decode it, count the bits that match the true message, and
compare with 256 random messages (the null distribution, measured rather than assumed). A region
is accepted when z ≥ 4. This is the *a contrario* approach of Desolneux, Moisan and Morel,
with random-key calibration as in Fernandez et al. (2023).

**Result:** our biggest jump of the weekend on the leaderboard: **0.6656 → 0.6786** public, then
0.6813 once applied to the final ensemble (0.668 → 0.674 on f0).

Then the first trap.

**What we noticed:** the boxes accepted on the test set with a moderate z (below 8) were huge:
16 to 33 % of the image, against 4 % for a real patch. A high z only proves that the message is
*somewhere inside* the box.
**What we did:** reject boxes larger than the 90 % quantile of training patch areas. The
threshold comes from training data, not from validation.
**Result:** **+0.005 public (0.6813 → 0.6863)**.

For 8 hours during the night, nothing else moved the score.

---

## Chapter 3: losing the GPUs, finding the statistics (Sunday morning)

Early on Sunday morning we lost GPU access on one account: the cluster allows one node per team, and we had been
running several jobs. From then on, most of the work had to run on CPU, on the predictions we
already had. That forced us to ask a different question: **not how to get better maps, but how to
make better decisions from the maps we have.**

**What we noticed:** thresholding probabilities at 0.5 maximizes the number of correct cells, not
the IoU, and IoU punishes a missed small region far more than a slightly too large one.
**What we did:** for each image, choose the mask that maximizes the *expected* IoU. For IoU, the
optimal mask is always the top-k most probable cells (Nowozin, CVPR 2014), so we only need to sort
and try every k. No parameter at all.
**Result:** **+0.007 public (0.6863 → 0.6936)**.

We then tried to push decision theory further (`wl.decision_lab`): exact expected IoU by Monte
Carlo, a kind-consistency constraint, and a spatially coherent error model (one threshold u per
image, truth = {p > u}).
**Result:** the approximation and the exact expectation gave **exactly the same score (0.6866)**.
Our decision layer was already Bayes-optimal for these maps. That negative result told us where
not to look any more.

**What we noticed next:** on one TrustMark image, the network had found the true patch with
probability 1, but the verification only tested the largest region, which was a wrong blob in a
field.
**What we did:** when the main region fails, test the next regions too, with a stricter threshold
(z ≥ 5) since we run more tests (Rahul's E187).
**Result:** that image went from **0.44 to 0.88**, **+0.004 on f0**.

---

## Chapter 4: two ensembles that disagree (Sunday, last hour)

**What we noticed:** two validated ensembles, at about the same level, did not win in the same
places. The rescue ensemble (E221) gained on PixelSeal (+0.010), the other (E222) on TrustMark
patches (+0.013).
**What we did:** average their probability maps with equal weights, then apply the same Bayes
decision. Nothing tuned.
**Result:** **0.6957 public, our best score (+0.0021)**. Adding a third model with a different
encoder (ResNet-101) reached **+0.0061 on f0**; the server's rate limit stopped us from
submitting it before the deadline.

---

## Chapter 5: the wall

Every gain above is small, and we wanted to know why.

**What we counted** (`wl.triage`): on **12 % of the test images (119 of 1,020)**, our final mask
has no small region at all. Those images score about 0.5 at best, and that is where the remaining
points are.
**What we tried:** on exactly those images, borrow the region seen by another model.
**Result:** 0 images gained, 7 lost. When our ensemble sees nothing, the other models' regions
are wrong. **No model has the information on those images.**

The other probes agreed:

| What we tried | Result | What it taught us |
|---|---|---|
| Pixel-level traces (resampling, steganalysis, Splicebuster, re-synthesis) | AUC 0.47–0.64 | The images were resampled after embedding: the fine trace is gone |
| Edge-aware refinement (dense CRF, superpixels, guided filter) | ±0.000 | The maps are already almost binary |
| Rectangle prior for patches | −0.012 | The networks already know the shape |
| Calibration and stacking | −0.006 to −0.020 | Per-cell loss does not track per-image IoU |
| Larger encoders (ResNet-101, ConvNeXt) | 0.59–0.66 vs 0.667 | Capacity is not the bottleneck |
| Thin PixelSeal bars, decoded at the exact box | z 0–4 | Undetectable by construction |

Even an oracle choosing the best threshold for each image would only gain +0.03. **We were
grinding against a ceiling set by the data, not by the method.**

---

## How we kept ourselves honest

With 102 validation images, noise is of the same order as most of our gains (about ±0.005), and
the final ranking is computed on 70 % of the test set that nobody sees. So we followed four rules
all weekend:

1. **Score only out of sample.** Every choice was measured on f0, never seen in training. Models
   retrained on all folds were never scored on f0; they only replaced their validated twin at
   test time.
2. **Paired comparisons.** Same images, image-by-image differences, with a paired bootstrap
   interval and a sign test (`wl.compare`).
3. **Decision rules written before the result.** For example: "submit if +0.003 on f0, more images
   won than lost, no method down". This kept us from choosing whatever happened to look good.
4. **The public leaderboard only confirms.** It never chooses. Submitting many variants means
   selecting the one that got lucky on 30 % of the test set. Example: E222 gained +0.0043 on f0
   and scored −0.0006 on the public split: noise, not failure.

Wherever possible we preferred **parameter-free steps** (expected-IoU decision, equal-weight
averages) and **thresholds fixed by theory or by training data** (z ≥ 4, the 90 % area quantile).

---

## The public score, step by step

| Submission | What changed | Public |
|---|---|---|
| E151 | Early ensemble | 0.6630 |
| E155g | Early ensemble, revised | 0.6656 |
| E158 | First decoder verification | 0.6786 |
| E198 | Verification on the final ensemble | 0.6813 |
| E198g | Size gate | 0.6863 |
| E198g + eIoU | Bayes decision (expected IoU) | 0.6936 |
| E222 | + next-region proposals | 0.6930 |
| E221 + eIoU | Rescue ensemble | 0.6922 |
| **E224** | **Equal-weight average of E222 and E221** | **0.6957** |

---

## What we would do next

The triage showed that the remaining loss is missing information, so the next gains must bring
new information, not better tuning.

1. **Synthetic data that matches the real pipeline.** Our analysis showed that the real images
   carry no embedding trace (period-2 ratio 0.993, where 1.0 means no trace), so they were resampled
   after watermarking. Our synthetic generator skipped that step, so the networks partly learned
   cues that never exist at test time. `wl/synth.py` now has `--resample-prob`, calibrated on
   training images only (`trace_report`), and `jobs/synthv3.txt` retrains the best recipe on it
   (E230), as a one-change ablation against E161.
2. **Object proposals checked by the decoder.** Test whether mask contours follow object edges
   (`wl.edge_align`); if they do, a generic segmenter can propose candidates that the decoder then
   verifies.
3. **Unfreeze the decoder** and train it to localize, conditioned on the message.
4. **An acceptance threshold set by the metric's asymmetry.** On an empty patch image, a wrong
   small box costs about 0.01 and a right one gains about 0.3, so the Bayes threshold is far below
   z ≥ 4.

---

## Reproducing

All commands run from this folder on JURECA. See [`FINAL.md`](FINAL.md) for the exact final
pipeline. In short:

```bash
python -m wl.infer --run output/<run> --id <run>_tta          # shift TTA
python -m wl.ensemble --runs ... --out output/<ens> --mode logit --global-weights
python -m wl.verify --id <ver> --sources output/<ens> output/E121_analytic_rect_v2 \
    --kind output/E120_kind_const/kind_probs.npz --methods pixelseal trustmark \
    --max-area-q 0.9 --extra-proposals 5
python -m wl.eiou_submit output/<ver>                          # Bayes decision
python -m wl.blend <name> <run_a> <run_b> [...]                # equal-weight blend + eIoU
```

Code map: `wl/models.py` (W2, W8), `wl/infer.py` (TTA), `wl/ensemble.py`, `wl/verify.py`
(decoder judge, size gate, extra proposals), `wl/honest_eval.py` (expected-IoU decision),
`wl/decision_lab.py`, `wl/triage.py`, `wl/blend.py`, `wl/compare.py` (paired bootstrap),
`wl/synth.py` (synthetic data).

## Who did what

Two people worked on this task, often in parallel through the night.

- **Rahul Chander** laid the foundations: the shared plan and the fixed folds, the `wl` base library
  (models, analytic tools), the single-node 4-GPU job runner (`jobs/run_node.sh`), the synthetic data
  generator (synth-v2, kind gating, decoder box refinement) and the one-command round v2 job chain.
  On Sunday he found the **size gate** (E182), wrote the paired run comparison, ran the statistical
  methods round (E183–E189, including the pixel-level probes that ruled out fine traces), added the
  **next-region proposals** (E187), and kept the GPU work going on his account after ours was cut:
  the rescue ensemble (E221) and the larger encoders (E207–E209).
- **Iliane Chikhaoui** built the decoder-as-judge line: the a contrario localization with a
  random-key null, the decoder verification (`wl.verify`), and the decision layer (expected-IoU
  decision, `wl.decision_lab`), then the final blends (`wl.blend`), the triage (`wl.triage`) and
  the synthetic-data fix (synth-v3).

The other two tasks of the team are in [`../model-equality-testing/`](../model-equality-testing/)
and [`../unlearning/`](../unlearning/).
