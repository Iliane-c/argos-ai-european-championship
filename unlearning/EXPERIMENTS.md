# Task 3 experiments

## T3-001 — Unchanged checkpoint and head-only update

Run from the repository root with the organizer's starter files, `model.pt`, and `forget_images.npz` in `task3/`.

```bash
cd task3
uv sync --locked
OMP_NUM_THREADS=8 uv run --no-sync python task_template.py
OMP_NUM_THREADS=8 PYTHONPATH=. uv run --no-sync python ../unlearning/baseline.py
```

`submission.pt` is the unchanged control. The script caches its logits for all 3,500 forget images and fits a ridge-regularized update to the classifier head weights. It targets a 25% reduction in each original top-two margin, capped at 0.5 logit, by reducing the original top class logit. `output/head_only.pt` is the candidate for the first public evaluation. Both sets of logits are saved under `output/`.

The public evaluator is needed to measure the forget score, retain score, and the effect of the 70% validation accuracy gate. The local comparison uses the original model's top classes since the forget labels and gold predictions are unavailable.

### Local results

| Check | Result |
| --- | --- |
| Control checkpoint | All 176 tensors match the supplied model |
| Mean gap between the original top two classes | 12.2414 → 11.9793 |
| Forget images with the same top class | 3,500 of 3,500 |
| Candidate checkpoint | Only `head.weight` changed. Strict reload passed, all weights are finite float32, and the file is 343,629,067 bytes. |

### Public evaluation

The organizer accepted `head_only.pt` as submission **1123** and returned a public-split score of **0.0**. The response included only the aggregate score, so the forget component, retain component, and validation gate outcome are unknown. The starter client timed out during the first attempt. Streaming the same file to the documented evaluator endpoint succeeded.

## T3-002 — Last-block updates with synthetic retention probes

Start with `model.pt`. The script builds an image-specific soft target from the original model's predictions on the clean image, a horizontal flip, and two shifted crops. Each round updates only the last ViT block, first fitting those targets on the original forget images and then matching the original predictions on synthetic inputs. Both the rest of the backbone and the classifier head stay frozen.

The original model generates synthetic inputs from random starts by optimizing each one toward a requested class. The full run targets all 100 classes with four inputs per class for training and two independent inputs per class for validation. The log records how many requested classes the teacher actually predicted in each split.

On a GPU node, run from the repository root with the starter files in `task3/`.

```bash
cd task3
uv sync --locked
PYTHONPATH=. uv run --no-sync python -u ../unlearning/unlearn.py
```

The default run uses one GPU and writes `output/t3-002/round_01.pt` through `round_03.pt`. Add `--smoke` for a four-image pipeline check, or use `--output output/t3-002-repeat` to run again in a fresh directory. Smoke checkpoints are for pipeline verification.

`metrics.jsonl` includes the unchanged model at round 0 and the measurements after each forget and retention pass. Compare KL divergence to the soft targets on clean and blurred forget images, and compare KL divergence and top-class agreement with the original model on the held-out synthetic inputs. Blur is used only for evaluation. These are diagnostics for choosing a checkpoint. The evaluator supplies the score against the retrained reference and applies the held-out accuracy gate.

### GPU run results

The full run processed 3,500 forget images. The teacher predicted the requested class on every synthetic input in both splits.

| Round | Clean target KL | Blurred target KL | Mean probability of original top class | Held-out probe KL |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.004790 | 0.115301 | 98.546% | 0.000000006 |
| 1 | 0.003820 | 0.115186 | 98.667% | 0.000000492 |
| 2 | 0.003301 | 0.115401 | 98.698% | 0.000000983 |
| 3 | 0.002876 | 0.115501 | 98.702% | 0.000001441 |

All rounds kept 100% top-class agreement on held-out synthetic probes. The forget targets assigned the original top classes 98.641% mean probability, up from 98.546% in the original model. The checkpoints moved closer to the targets on clean inputs, while the mean probability of the original top class rose in every round.

`round_01.pt` is the first candidate for public evaluation. It has the lowest blurred target KL and the least synthetic probe drift among the three updated checkpoints. Its blurred target KL improved by only 0.10% relative to round 0. The public score and held-out accuracy remain pending.

## T3-003 — Negative-gradient unlearning with synthetic retention

Start again from `model.pt`. The original model supplies a pseudo-label only when its top class agrees across the clean image, flip, and both crops. Each round samples up to 128 eligible images and takes up to eight steps that increase cross-entropy on their pseudo-labels at temperature 2. Only the last ViT block is updated at a learning rate of `1e-5`. Each ascent pass is followed by the same synthetic retention pass as T3-002, with separate optimizer state for the two objectives.

In the Jülich `unlearning/` workspace, copy the updated `unlearn.py` alongside `task_template.py` and run on a GPU node.

```bash
uv run --no-sync python -u unlearn.py --neggrad
```

Locally, use `PYTHONPATH=. uv run --no-sync python -u ../unlearning/unlearn.py --neggrad` from `task3/`. Add `--smoke` to check the pipeline. The full run writes three checkpoints and `metrics.jsonl` under `output/t3-003/`.

The metrics include eligible image count, mean probability of the original top class on clean and blurred images, and agreement and KL drift on held-out synthetic probes. The view-averaged target KL is retained for comparison with T3-002. These diagnostics use pseudo-labels and synthetic inputs. The official evaluator measures gold-model agreement and held-out accuracy.

## T3-010 — Delta-shift distillation with strength interpolation

All previous submissions moved the forget-set CE from 0.0003 to at most 0.0006. The metric normalizes by the gap between the memorizing original and the never-seen gold models, so those edits scored about 1e-5. The server rounds responses to four decimals and displayed them as 0.0.

`shift_distill.py` fine-tunes the full model. For every augmented view of a forget image, the target logits are the original model's logits minus δ on the clean top class. This removes a memorization bonus and sends the freed mass to the original runner-ups, which share a block of related class IDs 62% of the time. Retention is teacher KD at T=2 on COCO val2017, center-cropped to 224 px, with the last 500 images held out for a drift audit. `interpolate.py` turns one run into a strength family θ(γ) = θ₀ + γ(θᵤ − θ₀). An 80/20 split of the forget set showed that forgetting reaches held-out forget images almost as strongly as trained ones, so drift on same-class control images is the main cost.

From `task3/`, with `coco224.npy` built from COCO val2017:

```bash
PYTHONPATH=. .venv/bin/python ../unlearning/shift_distill.py --out output/t3-010/d12.pt --retain coco224.npy --delta 12 --epochs 8
PYTHONPATH=. .venv/bin/python ../unlearning/interpolate.py output/t3-010/d12.pt --gammas 1.5 --retain coco224.npy --export output/t3-010
```

The submitted runs used the scratchpad prototype of these scripts, with retention on 1,024 CIFAR-100 images (δ=10) or all 5,000 COCO images (δ=12). Clean forget statistics cover all 3,500 images.

| Submission | Model | Forget clean CE mean/std, acc | Public score |
| --- | --- | ---: | ---: |
| 1612 | δ=10, CIFAR KD, 5 epochs, γ=1.0 | 0.27/0.49, 91% | 0.0236 |
| 1640 | same, γ=1.25 | 0.61/0.72, 80% | 0.0380 |
| 1676 | same, γ=1.5 | 1.07/0.89, 66% | 0.0440 |
| 1708 | δ=12, COCO KD, 8 epochs, γ=1.25 | 1.06/0.99, 62% | 0.0622 |
| **1731** | **same, γ=1.5** | **1.61/1.12, 44%** | **0.0692** |
| 1757 | same, γ=1.75 | 2.10/1.16, 32% | 0.0642 |

At matched clean CE, δ=12 with COCO KD beat δ=10 with CIFAR KD by about 40%, with a heavier loss tail and less KL from the original. The γ sweep peaks near 1.5.

### Follow-ups with the repository script

Each run changes one factor from δ=12, 8 epochs, COCO KD weight 1. Every run is submitted at the γ that matches the previous best clean strength (CE about 1.6). Scores within about ±0.005 of each other are treated as ties.

| Submission | Change | Clean CE mean/std at submitted γ | Public score |
| --- | --- | ---: | ---: |
| 1841 | View-ratio δ for altered views, γ=1.7 | 1.69/1.17 | 0.0562 |
| 1901 | Retention weight 3, γ=1.65 | 1.67/1.14 | 0.0657 |
| 1937 | Weight 3, 10% per-class confusion prior, γ=1.6 | 1.62/1.02 | 0.0504 |
| 1998 | Last 6 blocks only, γ=1.5 | 1.55/1.16 | 0.0708 |
| 1970 | Per-image δ, margin power −0.5, γ=1.5 | 1.58/1.31 | 0.0795 |
| 2027 | Margin power −1.0, γ=1.5 | 1.60/1.42 | 0.0869 |
| 2055 | Margin power −1.0, γ=1.7 | — | 0.0845 |
| 2090 | Margin power −1.5, γ=1.5 | 1.60/1.49 | 0.0920 |
| **2120** | **Margin power −2.0, γ=1.5** | **1.60/1.55** | **0.0956** |

At a matched mean loss, a heavier loss tail is the only change that kept improving the score. Smoothing the targets, protecting altered views and lowering COCO drift all lowered it. The gold models therefore seem confidently wrong on a subset of forget images, with the atypical low-margin images most affected. Gains from the margin power shrank from 0.010 to 0.004 per step. The best checkpoint is `task3/output/t3-010/mp20_g1.5.pt`.

## T3-011 — In-domain generated retention

The 80/20 forget split showed that forgetting spreads to every image of the 45 forget classes, so control images of those classes drift. COCO and CIFAR retention cannot anchor them. `gen_indomain.py` generates about 5,400 new images of the 45 classes with SDXL-Turbo. The class names were read off the forget set, and the prompts imitate its warm, low-light style. `ssh_t3011.sh` mixes those images with COCO and trains margin power −2.0 and −3.0 in parallel on the two Quadro RTX 8000 GPUs. Turing has no bf16, so training uses fp16 with a gradient scaler. The script then exports γ 1.3, 1.5 and 1.7.

Set `REMOTE` to the server's working directory, then send the inputs from the repository root:

```bash
REMOTE=user@host:~/t3011
rsync -avP task3/task_template.py task3/model.pt task3/forget_images.npz task3/coco224.npy "$REMOTE/"
rsync -avP unlearning/shift_distill.py unlearning/interpolate.py unlearning/gen_indomain.py \
    unlearning/ssh_t3011.sh "$REMOTE/unlearning/"
```

On the server, use an environment with `torch`, `timm`, `numpy`, `pillow`, `diffusers`, `transformers` and `accelerate`, plus internet access for the one-time model download of about 7 GB:

```bash
cd ~/t3011 && bash unlearning/ssh_t3011.sh 2>&1 | tee t3011.log
```

Bring back the logs first. Then fetch only the checkpoint whose clean CE at γ=1.5 matches the best COCO run (1.60/1.55), at 344 MB per file:

```bash
rsync -avP "$REMOTE/out/*.log" "$REMOTE/t3011.log" task3/output/t3-011/
rsync -avP "$REMOTE/out/export/gen_mp20_g1.5.pt" task3/output/t3-011/
```

### T3-011 results

The run trained margin power −2.0 and −3.0 with about 5,400 generated images of the 45 forget classes mixed into the COCO retention pool. On those generated images, which are the closest stand-in for the control set, the in-domain anchors cut drift sharply at the same forget strength:

| Checkpoint | Forget clean CE mean/std | Generated KL at T=1 | Agreement with original | Accuracy on generated |
| --- | ---: | ---: | ---: | ---: |
| Best COCO run (`mp20_g1.5`) | 1.60/1.55 | 0.538 | 0.786 | 0.582 |
| `gen_mp20_g1.5` | 1.64/1.57 | 0.152 | 0.905 | 0.664 |
| `gen_mp30_g1.5` | 1.64/1.65 | 0.150 | 0.904 | 0.666 |

The original model scores 0.663 accuracy on the generated images, so the anchored models keep it. None of the 5,400 images came out black.

| Submission | Checkpoint | Public score |
| --- | --- | ---: |
| **2461** | **`gen_mp30` at γ=1.5** | **0.1097** |
| 2497 | `gen_mp30` at effective γ=1.8 | 0.1052 |

Lower drift did not move the γ optimum up.

## T3-012 — Sharpened bimodal targets

On unseen in-domain images (the generated set), the original model is bimodal: it is right with near-zero loss on about two-thirds of the images and confidently wrong on the rest. Our unlearned models were a soft, medium-loss blob instead. The metric's mean/std term compares the loss std, so this gap matters.

| Model and images | Accuracy | CE mean/std | CE quantiles 10/25/50/75/90/95 % | Max prob | True-class prob when wrong |
| --- | ---: | ---: | --- | ---: | ---: |
| Original on generated (gold-like) | 0.663 | 2.64/4.72 | 0.00/0.00/0.00/3.55/10.05/13.46 | 0.92 | 0.046 |
| `gen_mp30_g1.5` on forget | 0.504 | 1.64/1.65 | 0.03/0.19/1.12/2.68/4.00/4.74 | 0.66 | 0.101 |
| `ts15_g1.4` on forget | 0.547 | 1.63/2.01 | 0.00/0.04/0.68/2.76/4.56/5.73 | 0.78 | 0.095 |
| `ts25_g1.35` on forget | 0.566 | 1.63/2.17 | 0.00/0.02/0.46/2.78/4.90/6.17 | 0.82 | 0.095 |

`--target-scale s` multiplies the shifted target logits by s, so each forget image becomes confidently right or confidently wrong. Every run below uses margin power −3.0, δ=12 and the generated-plus-COCO retention pool, and was trained locally.

| Submission | Change | Clean CE mean/std at submitted γ | Public score |
| --- | --- | ---: | ---: |
| 2558 | Target scale 1.5, γ=1.4 | 1.63/2.01 | 0.1563 |
| 2597 | Target scale 2.5, γ=1.35 | 1.63/2.17 | 0.1711 |
| 2634 | Target scale 2.5, γ=1.25 | 1.40/2.00 (est.) | 0.1517 |
| **2671** | **Target scale 2.5, γ=1.5** | **1.95/2.38** | **0.1930** |
| — | Target scale 2.5, γ=1.7 | 2.34/2.57 | pending |
| — | Target scale 2.5, δ=20 | 1.56/1.78 at γ=1.2 | not submitted |

Sharpening is the largest single gain since the first non-zero score. With sharper targets the strength optimum moved up: mean CE about 1.95 beats 1.6, unlike the soft-target runs. δ=20 was rejected locally because at a matched mean it lowers the loss std (1.78 versus 2.17) and raises retain drift, so it flips more images instead of making the flipped ones deeper. The best checkpoint is `task3/output/t3-012/ts25_g1.5.pt`. The remaining gap to the unseen-image shape is the depth of the wrong images: their true-class probability is 0.095 versus 0.046, and the loss tail reaches about 6 instead of 13. Scale 4 is training locally. `ssh_t3012.sh` sweeps δ=8, margin power −5, and 12 epochs at scales 2.5 and 4 on the server, with γ 1.4–2.0.

### Depth loss and retention repair (Astra's options 1 and 4)

The T3-012 scores up to here were 0.1930 at γ=1.5 (2671) and 0.2088 at γ=1.7 (2701), both at scale 2.5. Sharper targets moved the strength optimum up.

`--depth-weight` adds a Huber loss on the student's per-image CE for the hardest `--depth-frac` of forget images, ranked by clean margin, with targets from `--depth-hi` (hardest) down to `--depth-lo`=6. KL distillation alone stalls once the target's true-class probability is near zero. `--repair CKPT` then fine-tunes the last block and head starting from the chosen checkpoint. It pins forget views to that checkpoint's frozen outputs (`--delta 0`) and samples generated and COCO anchors in proportion to their drift, at retention weight 3 for 2 epochs with lr 1e-5. A final γ=1.03 restores the slight loss of forget strength. All runs use scale 2.5 and margin power −3. Retain KL is the drift on the last 500 retain-mix images.

| Candidate | Clean CE mean/std | Hardest 10% mean CE | Forget KL | Retain KL | Public score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scale 2.5, γ=1.7 (2701) | 2.34/2.57 | 4.99 | 1.81 | 0.297 | 0.2088 |
| Depth 10% (6–10), γ=1.6 | 2.38/2.79 | about 6.0 | 1.75 | 0.277 | — |
| Depth 20% (6–12), γ=1.5 | 2.39/2.99 | 6.18 | 1.69 | 0.257 | — |
| **Depth 20% + repair, γ=1.03 (2898)** | **2.42/3.01** | **6.11** | **1.71** | **0.246** | **0.2431** |
| Depth 30% (6–13) + repair, γ=1.03 | 2.40/3.11 | — | 1.64 | 0.231 | — |
| Depth 40% (6–14) + repair, γ=1.03 | 2.40/3.21 | — | 1.60 | 0.221 | — |

At matched mean CE, each wider band raised the loss std and lowered drift, but the std gain fell from +0.2 to +0.1 per step. The gold models' std is unknown, so only the leaderboard can show where widening stops paying. Checkpoints are in `task3/output/t3-012/`.

### Leaderboard tuning of the depth + repair stack

The pipeline for each band is to train, interpolate to clean mean CE 2.40, repair (2 epochs, weight 3), then extrapolate. Candidates were compared at the scored operating point, because comparisons before repair understated the last two leaderboard gains.

| Submission | Candidate | Clean CE mean/std | Retain KL | Public score |
| --- | --- | ---: | ---: | ---: |
| 3019 | 40% band (6–14), γ=1.03 | 2.40/3.21 | 0.221 | 0.2537 |
| 3051 | 40% band, γ=1.10 | 2.65/3.36 | 0.247 | 0.2686 |
| 3088 | 40% band, γ=1.20 | 2.98/3.52 | 0.287 | 0.2818 |
| 3124 | 40% band, γ=1.35 | 3.44/3.67 | 0.352 | 0.2841 |
| 3156 | 50% band (6–15), γ=1.35 | 3.49/3.82 | 0.341 | 0.2952 |
| 3192 | 50% band, γ=1.45 | — | — | 0.2917 |
| **3226** | **60% band (6–16), γ=1.35** | **3.54/3.92** | **0.336** | **0.3044** |
| 3263 | 60% band, γ=1.25 | — | — | 0.3002 |

With the sharper, deeper targets the strength optimum moved to γ≈1.35 after repair, with a mean CE of about 3.5. Two runs were rejected locally without submission:

- **Graded 80% band (4–16):** std 3.94 against 3.92 at the operating point, versus about +0.10 per earlier step.
- **Stronger repair (4 epochs, weight 5):** at matched mean it had std 3.88 and retain KL 0.333, so the retention it gained came back out of forgetting.

Option 3 of `NEXT_APPROACHES.md` (spreading rival mass) was not run, because both earlier softening attempts lowered the score. Options 2 and 5 did not fit the remaining time. The best checkpoint is `task3/output/t3-012/dep6rep_g1.35.pt`.
