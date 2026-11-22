# ARGOS — European AI Championship, Paris round

**21–22 November 2026, organized by CISPA, SprintML and PoC Innovation.**

Three tasks, one weekend, one cluster, and a single rule we kept coming back to: **the leaderboard
you see is not the one that counts.** Each task was scored on a public split during the event,
and ranked at the end on a hidden one. So, on every task, we tried to win on data we had not used,
not on the public numbers.

This README tells how each task went, in the order things happened: what we saw, what we decided,
what it changed, including the long nights where nothing moved. Every task folder has its own
code, Slurm scripts and full experiment log.

| Task | Folder | What it asks | Best public score |
|---|---|---|---|
| 1. Watermark localization | [`watermarking-localization/`](watermarking-localization/) | Find where an invisible watermark sits in an image | **0.6957** mean IoU |
| 2. Model equality testing | [`model-equality-testing/`](model-equality-testing/) | Tell which of 33 black-box API endpoints serve the same model | **0.10** (families exact) |
| 3. Unlearning | [`unlearning/`](unlearning/) | Make a ViT-B/16 forget 3,500 images without the retain set | **0.3572** |

---

## Task 1 — Watermark localization: the decoder as a judge

*Full story, figures and code map: [`watermarking-localization/README.md`](watermarking-localization/README.md).*

Each 512×512 image carries an invisible watermark (PixelSeal, TrustMark or StegaStamp) on part of
it. We know the method and the hidden message, and must output a 128×128 mask. The metric averages
the IoU of both classes per image, so **missing the small region costs everything**: miss all the
holes of an almost fully marked image and a perfect answer elsewhere drops to 0.47.

1. **Networks first.** Segmentation networks with the official decoders frozen inside them reached
   **0.654** on our held-out fold. Shifting images by whole pixels keeps the watermark intact, so we
   averaged 9 shifted predictions (**0.665**), then an ensemble with one shared weight per model
   (**0.668**).
2. **The decoder as a judge.** Decoding the true boxes gave bimodal evidence: z ≥ 9.8 or z ≤ 2.3,
   nothing in between. We turned the decoder into a statistical test against 256 random messages
   (z ≥ 4). Our biggest jump on the leaderboard: **0.6656 → 0.6786**.
3. **The trap.** Accepted boxes with a moderate z were huge (16–33 % of the image, against 4 % for a
   real patch): a high z only proves the message is *inside* the box. A size gate taken from the
   training data gave **+0.005** (0.6863). Then, for 8 hours during the night, nothing moved.
4. **Decisions, not thresholds.** Without GPUs on Sunday morning, we worked on the decision itself:
   the mask that maximizes the *expected* IoU (the top-k most probable cells, no parameter).
   **0.6936.** An exact Monte Carlo version gave the same score: the decision layer was saturated.
5. **Two ensembles that disagree.** Two validated ensembles won in different places (PixelSeal vs
   TrustMark). Their equal-weight average: **0.6957**, our best.
6. **The wall.** On 12 % of the test images, no model sees any small region, and borrowing other
   models' regions is wrong every time. The real images were resampled after watermarking, which
   erased the fine traces; our synthetic data had skipped that step. The fix is in the generator,
   ready for the next run.

---

## Task 2 — Model equality testing: knowing what does not work

*Full write-up: [`model-equality-testing/FINDINGS.md`](https://github.com/Iliane-c/argos-ai-european-championship/blob/model-equality/model-equality-testing/FINDINGS.md) (branch `model-equality`).*

33 anonymous chat endpoints, a budget of 100 answers per endpoint for the whole event, at most 4
requests at a time. Score: 0.1 × MCC on model families + 0.9 × MCC on exact models inside families.
Nine practice endpoints with known answers served a small model in plain form, in four disguises
(system prompt, persona, watermark, temperature 1.0) and as four real variants (fp8, three LoRA
strengths).

1. **Families in the first hours.** Clustering the answers recovered the families exactly: the
   first submission scored **0.10**, i.e. family part 1.0, model part 0.
2. **A night of probes, all checked on the practice endpoints first.** Number and animal
   histograms, style features, 0/1 sequences, multiple choice, completions, proxy-model
   log-probabilities. Each one was tested out of sample on the practice set before spending budget.
3. **What we learned instead of a score.** Distribution tests on short answers detect the
   *disguises* as differences, and more samples make it worse, not better. fp8 was never separable.
   The only signal that replicated out of sample, overrunning an "exactly N sentences" instruction,
   caught strong LoRA on the small model; the larger endpoints followed length limits and showed
   nothing.
4. **Why splits never helped.** Every apparent outlier had a near-identical partner in its family,
   so any split also broke a real same-model pair. Seven different partitions all scored 0.10.

The result is a careful negative: we know which probes do not separate these models, and why.

---

## Task 3 — Unlearning: from 0.0 to 0.3572 by reading the metric

*Full log with every submission: [`unlearning/EXPERIMENTS.md`](https://github.com/Iliane-c/argos-ai-european-championship/blob/task3/baseline/unlearning/EXPERIMENTS.md) (branch `task3/baseline`).*

A ViT-B/16 classifier must forget 3,500 training images, without access to the data it should keep.
The score compares the model with a memorizing original and with "gold" models that never saw the
forget set.

1. **Hours at 0.0.** A head-only update, last-block fine-tuning, negative gradient: all scored 0.0.
2. **What we noticed:** the metric normalizes by the gap between the memorizing original and the gold
   models, so our edits, which moved the forget-set loss from 0.0003 to 0.0006, were worth about
   1e-5, rounded to 0.0. **We had to move much further.**
3. **What we did:** distill the model toward its own logits minus δ on the top class, on every
   augmented view, with a strength we could dial up afterwards. First non-zero score, then
   **0.0692**.
4. **A heavier tail.** At the same mean loss, making the loss heavier on low-margin images was the
   only change that kept improving the score (**0.0956**): the gold models seem confidently wrong on
   a subset of atypical images. Smoothing targets or protecting altered views lowered it.
5. **Retention without the retain set.** In-domain generated images to anchor what must be kept:
   **0.1097**.
6. **Sharper targets.** The largest single gain: **0.1930 → 0.2088**, and the best strength moved up.
7. **Depth on the hardest images, then repair.** A direct loss on the hardest forget images, followed
   by a short retention fine-tune: **0.2431**, then a wider depth band (60 %): **0.3044**.
8. **Sharpen and average.** Scaling the head logits, then a three-seed model soup: **0.3572**.

| Step | Public |
|---|---|
| Head-only, last block, negative gradient | 0.0 |
| Delta-shift distillation | 0.0692 |
| Heavier loss tail on low-margin images | 0.0956 |
| Generated in-domain retention | 0.1097 |
| Sharpened targets | 0.2088 |
| Depth loss on the hardest images + repair | 0.2431 |
| Wider depth band (60 %) | 0.3044 |
| Head scaling + three-seed soup | **0.3572** |

---

## What the three tasks taught us

- **Read the metric before the model.** Task 1's per-image IoU made the small region decisive;
  Task 3's normalization explained hours of 0.0; Task 2's MCC made every wrong split costly.
- **Validate out of sample, always.** A held-out fold on Task 1, the practice endpoints on Task 2.
  Decision rules were written before the results, and the public score only confirmed.
- **Negative results are results.** On every task, the log of what failed told us where not to
  look, and on Task 1 it explained exactly where the remaining points are.
- **Averaging models that disagree is the most robust gain.** It gave our best score on Task 1 and
  on Task 3.

---

## Compute

Jobs run on the JURECA cluster (Jülich), project `training2666`, using the shared team folder created
by `hackathon_setup.sh` from the [SprintML/hackathon](https://huggingface.co/datasets/SprintML/hackathon)
repository. The cluster allows one node (4 GPUs) per team at a time; `watermarking-localization/scripts/_common.sh`
refuses a second job.

## Conventions

- Datasets, model weights, caches and outputs are never committed (see `.gitignore`).
- Every experiment is logged in the task's `EXPERIMENTS.md` with an ID referenced in commit messages.
- Run `git pull --rebase` before pushing.
