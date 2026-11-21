# argos-ai-european-championship

Team ARGOS — European AI Championship, Paris regional round (21–22 November 2026), organized by CISPA, SprintML and PoC Innovation.

## Tasks

| Folder | Task | Metric |
|---|---|---|
| [`watermarking-localization/`](watermarking-localization/) | Localize post-hoc image watermarks (PixelSeal, TrustMark, StegaStamp) after image edits | Mean IoU over 128×128 masks |
| [`model-equality-testing/`](model-equality-testing/) | Identify the model family and model identity behind 33 black-box API endpoints | 0.1 × MCC (family) + 0.9 × MCC (model) |
| [`unlearning/`](unlearning/) | Unlearn 3,500 images from a ViT-B/16 classifier without access to the retain set | Forget score × retain score |

Each task folder is self-contained, with its own code, Slurm scripts and `EXPERIMENTS.md` log.

## Compute

Jobs run on the JURECA cluster (Jülich), project `training2666`, using the shared team folder created by `hackathon_setup.sh` from the [SprintML/hackathon](https://huggingface.co/datasets/SprintML/hackathon) repository.

## Conventions

- Datasets, model weights, caches and outputs are never committed (see `.gitignore`).
- Every experiment is logged in the task's `EXPERIMENTS.md` with an ID referenced in commit messages.
- Run `git pull --rebase` before pushing.
