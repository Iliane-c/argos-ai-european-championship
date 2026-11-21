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
