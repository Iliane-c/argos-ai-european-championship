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

## T3-020 — Hybrid unlearning suite (code-ready, CPU-verified)

Modular multi-paradigm unlearning library (`unlearning/ul/`), 7 CLI entrypoints (`unlearning/*.py`), and JURECA Slurm pipelines (`unlearning/jobs/*.sh`). Implemented locally and verified end-to-end on CPU against a synthetic `TinyViT` task mock; JURECA GPU execution and public scoring remain pending.

- **Core package (`ul/`)**: [`model_io.py`](ul/model_io.py), [`metrics.py`](ul/metrics.py), [`augmentations.py`](ul/augmentations.py), [`saliency.py`](ul/saliency.py) (ViT layer groups, empirical/predictive diagonal Fisher, SalUn/SSD masks, SSD weight dampening, L1-sparse pruning, mergeable `LoRALinear`, `ActivationSubspaceProjector` OGP), [`losses.py`](ul/losses.py) (T=2/T=1 KD, `MemoryBankMomentLoss` mean/std & 1D Wasserstein, bounded NPO, local view smoothness, SCAR Mahalanobis feature push, feature anchor, Fisher EWC), [`oracle.py`](ul/oracle.py) (4-signal memorization scoring, stratified OOF Ridge+kNN oracle, hybrid target synthesis), [`merging.py`](ul/merging.py) (task arithmetic, TIES/Fisher merging, layerwise `γ`, multi-paradigm souping).
- **JURECA execution order** (from `unlearning/`, pointing `POOL_PATH` to a domain-matched pool after Step 0 identifies the 100 classes):
  1. `sbatch jobs/00_diagnose.sh` — forensic audit (`diagnose.py`)
  2. `POOL_PATH=/path/to/pool sbatch jobs/01_build_retain_and_oracle.sh` — deduplicated class-balanced proxy + initial oracle (`build_retain_proxy.py`, `build_oracle.py`)
  3. `sbatch jobs/02_train_shadow_gold.sh` — train 3 Shadow-Gold `ViT-B/16` models excluding forget images (`train_shadow_gold.py`), then re-run `build_oracle.py --shadow-gold ...`
  4. `sbatch jobs/03_wave1_kfold_and_salun.sh` — 5-fold cross-fitted TIES task-vector unlearning (`kfold_unlearn.py`) + SalUn late-MLP baseline (`hybrid_unlearn.py`)
  5. `sbatch jobs/04_wave2_hybrid_and_branches.sh` — full hybrid branch, LoRA+OGP branch, and L1-prune+heal branch (`hybrid_unlearn.py`)
  6. `sbatch jobs/05_soup_and_eval.sh` — weight-space soup + coordinate layerwise `γ` search (`soup_and_eval.py`)

### v1 Colab Run & Public Evaluation

Executed `run_pipeline.sh` (`MODE=fast`, `SKIP_SHADOW=1`) on COCO val2017 (`4,591` fit / `400` audit). Because v1's initial oracle had `delta_mean=7.20`, `target_ce_mean=0.220` (`oof_ce_mean=0.072`), and `soften=0.10` / `view_ratio=True` / `w_ewc=1.0`, the branches converged to low forget CE and Stage 6 selected `branch_full_hybrid.pt` extrapolated at `γ=3.0` (including `head=3.0`).

| Submission | Model | Forget clean CE mean/std, acc | Retain proxy (acc, KL@T2, score) | Public score |
| --- | --- | ---: | ---: | ---: |
| 2310 | v1 `soup_submission.pt` (`γ=3.0` on `branch_full_hybrid`) | `0.932 / 1.172`, `71.3%` | `72.5%`, `0.246`, `0.582` | `0.0527` |

Diagnostic findings from `diagnostics.json`:
- LayerScale `ls2.gamma` jumps 8–10× in `blocks.7..11` (`2.01–2.56` vs `0.20–0.76` in `blocks.0..6`).
- Only `45 / 100` classes appear in `forget_images.npz` (`55` classes have `0` forget images).
- `patch20` is the main corruption cracking memorization (`86.7%` acc, `CE=0.538±1.637`).

## T3-021 — Hybrid v2 (heavy-tail margin+patch oracle, frozen 55-class head, in-domain 45-class retain, Colab notebook)

Combines the leaderboard-verified findings from `T3-010`/`T3-011` (`margin_power` heavy tail, last-k localization, no `soften`, no `view_ratio`, SDXL-Turbo 45-class anchors) with the `T3-020` modular suite (`ul/`):

- **Surgical 55-class head mask (`ul/saliency.py:build_active_head_mask`)**: zeroes gradients on the `55` non-forget class rows of `head.weight` / `head.bias` throughout training and polish (`freeze_inactive_head=True`, `head_lr_scale=0.5`).
- **Heavy-tail margin + patch fragility oracle (`ul/oracle.py`, `build_oracle.py`)**: `--delta-mode margin_power` (`base_delta=12.0`, `margin_power=-2.2`, `fragility_boost=0.25` from `rot10`/`patch20`/`patch35`/`blur5` + LOO outlierness + weight/input sensitivity), `--soften 0.0`, `--no-view-ratio`.
- **Active-45 vs inactive-55 retain audit (`build_retain_proxy.py`, `hybrid_unlearn.py`, `soup_and_eval.py`)**: splits `retain_audit` into `retain_audit_active.npy` (45 forget classes) and `retain_audit_inactive.npy` (55 untouched classes) alongside `gen_indomain.py` + COCO retention.
- **Calibrated 5-branch pipeline & Stage 6 soup (`run_pipeline.sh`, `soup_and_eval.py`, `colab_hybrid_v2.ipynb`)**: trains `last6_mp22`, `last5_patch_mp25`, `branch_full_mp30`, `branch_hybrid_npo_moment`, and `wave2_kfold` directly to `CE ≈ 1.0–1.5`, soups with `head` capped at `γ ≤ 1.0` (`max_gamma=1.8`), and exports `--export-gamma-sweep 1.0,1.2,1.35,1.5`. Ready-to-run Colab notebook at [`unlearning/colab_hybrid_v2.ipynb`](colab_hybrid_v2.ipynb).
