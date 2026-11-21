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
