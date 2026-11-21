# Experiment 1: probe the ref endpoints, compute distances, cluster

Goal: without spending any ep budget, answer three questions.

1. Which ref endpoints are deterministic (greedy decoding)?
2. Which probe prompts best separate "same model, different disguise" from "different model"?
3. How much practice MCC does a simple text distance plus clustering get? This is the baseline.

## Files

| File | Purpose |
| --- | --- |
| `reference_prompts.txt` | 35 probes, one per line: random number/word, arithmetic, factual QA, verbatim copy, counting, continuation, poems, code, translation, JSON, identity |
| `run_ref.py` | Sends each probe to ref-01 ... ref-09, N times each, reusing the official `query_reference.py` |
| `analyze_ref.py` | Determinism check, per-prompt discriminativeness (AUC), distance matrix, hierarchical clustering, writes `my_reference_answer.csv` and scores it |

## How to run (on JURECA)

```bash
# 0) Put the three files in the team folder, next to the official query_reference.py, score_practice.py and practice_answers.csv
cd /p/scratch/training2666/chander2/argos/model_equality
#    From your Mac, replace <user> with your JURECA account:
#    scp reference_prompts.txt run_ref.py analyze_ref.py <user>@jureca.fz-juelich.de:/p/scratch/training2666/chander2/argos/model_equality/

# 1) Activate the environment
source .venv/bin/activate        # with uv: uv pip install requests

# 2) Set the API key (only for this terminal; it is never written to a file)
export ME_API_KEY="your team key"

# 3) Query: 35 prompts x 9 endpoints x 3 repeats = 945 answers, 4 in parallel, roughly 10-20 minutes
python run_ref.py 3
#    If it stops, run the same command again; answers already collected are skipped.
#    For more repeats, run python run_ref.py 5 later; it only adds repeats 3 and 4.

# 4) Analyse (standard library only; also runs on the Mac if you copy reference_answers.jsonl there)
python analyze_ref.py                    # default metric: mmd
python analyze_ref.py --metric exact     # exact match rate
python analyze_ref.py --metric prefix    # common prefix (good for greedy endpoints)

# 5) Score again with the official script; it should match what analyze_ref.py printed
python score_practice.py my_reference_answer.csv
```

These scripts only send HTTP requests, so they run on the login node: no GPU and no SLURM job needed. This assumes the login node can reach `34.147.229.55`; if it can't, run them on the Mac.

## Reading the output

- **Step 1, determinism**: "identical answers on 10/10 prompts" means a greedy endpoint. My guess is that only ref-08 samples; this needs checking.
- **Step 2a, AUC**: one score per prompt. 1.0 means the prompt separates same-model from different-model pairs perfectly; 0.5 means useless; below 0.5 means the disguises dominate. **This is the most valuable output**: it tells you which kinds of prompts to keep and strengthen in the next round.
- **Step 2b, distance matrix and closest pairs**: check how close ref-09 (LoRA 0.3) and ref-02 (fp8) are to ref-01, compared with ref-03, 05, 06 and 08.
- **Step 3, threshold sweep**: the threshold is picked using the answers, so the result is optimistic. The ep endpoints have no answers, so prefer a metric with a wide range of good thresholds over the single highest score.
- `--top N` uses only the N prompts with the highest AUC. It overfits; use it only to explore.

## The three distance metrics

- `exact`: 1 minus the probability that a random answer from each endpoint is identical. Most direct for greedy endpoints, but any small change from a system prompt counts as different.
- `prefix`: the fraction of two answers that is identical from the start. Quantization or a LoRA usually makes the text fork at some position.
- `mmd`: the MMD formula from Gao et al., with character-3-gram cosine similarity as the kernel. It subtracts each endpoint's own random variation, so it is fairer to sampling endpoints.

## Known limitations (to improve in the next round)

- A greedy endpoint outputs the most likely answer; a sampling endpoint outputs a distribution. Even for the same model, the MMD between them is not 0. Possible fix: compare the greedy endpoint's answer with the most frequent answer among the sampling endpoint's repeats.
- System prompts and personas may reword whole answers. Shorter, rigidly formatted prompts (output only a number or a word) should be less affected; check this with the AUC.
