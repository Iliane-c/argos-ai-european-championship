# Experiment 2: answer histograms of one-word "random choice" probes

## Why

Experiment 1 (partial run) showed three things:

- **Every ref endpoint samples**; none is greedy. Exact-match and prefix distances are therefore mostly noise (all pairs 0.77-0.97, best MCC 0.33).
- The two best probes by far were "pick a random number between 1 and 100" (AUC 0.97) and "name one random animal" (AUC 0.92). Both have one-word answers.
- Long free-text probes were around AUC 0.4-0.6, i.e. barely useful.

So experiment 2 uses only short one-word probes, asks each one many times, and compares how often each answer appears (the answer histogram). Quantization and LoRA change the model's probabilities, so they should move these histograms. A system prompt or a watermark hopefully moves them less; the per-probe AUC will tell.

## Files

| File | Purpose |
| --- | --- |
| `exp2_prompts.txt` | 20 one-word probes: random numbers (3 ranges), animal, fruit, color, city, country, letter, first name, vegetable, instrument, sport, flower, element, weekday, month, emotion, profession, kitchen object |
| `run_exp2.py` | Queries ref-01 ... ref-09 with every probe, N repeats, small `max_tokens`; writes `exp2_answers.jsonl` |
| `analyze_exp2.py` | Normalises answers, builds histograms, per-probe AUC, distance matrix, clustering, `my_exp2_answer.csv`, and a 100-answer budget simulation |

## How to run

```bash
cd /p/scratch/training2666/chander2/argos/model_equality      # next to query_reference.py and practice_answers.csv
source .venv/bin/activate
export ME_API_KEY="your team key"

python run_exp2.py 30          # 20 x 9 x 30 = 5,400 answers, max_tokens 16
python analyze_exp2.py         # metric tv (default)
python analyze_exp2.py --metric tvtemp
python analyze_exp2.py --metric js
python score_practice.py my_exp2_answer.csv

# Later, for a better budget simulation (needs >= 50 answers per probe):
python run_exp2.py 50          # only adds repeats 30..49
```

## Output, step by step

1. **Per-probe AUC**: which probes separate same-model from different-model pairs. Keep the ones near 1.0.
2. **Top answers per endpoint** for the 3 best probes, e.g. `ref-01: 42:9, 37:7, 73:5`. Look at these to understand what actually differs: a different favourite answer, or the same answers with different spread?
3. **Distance matrix and closest pairs**, labelled with the truth.
4. **Clusters** at the best threshold, written to `my_exp2_answer.csv`.
5. **Budget simulation**: on the scored endpoints you have only 100 answers each. The script tries K probes x (100 // K) repeats, re-sampling 30 times, and reports
   - `pair-AUC`: threshold-free separation; near 1.0 means one threshold could split same from different;
   - `best-MCC`: mean and worst case over the re-samples, at the best threshold (an upper bound).
   `best K` picks probes by AUC on this same data, so it is optimistic; `first K` (file order) is honest. If `R` is larger than the answers you collected, the line says so: run more repeats.

### Validation outputs (added after the first design)

- **Step 1** also prints a 95% endpoint-bootstrap interval for each probe's AUC, and `S(q)` = median distance of different-model pairs minus median of same-model pairs. With only 9 endpoints, a probe with AUC 0.9 but an interval of [0.6, 1.0] is not yet trustworthy.
- **Step 4** prints a bootstrap interval for the MCC at the chosen threshold.
- **Step 5, split-half validation**: picks the top-`k` probes (`--k`, default 5) and the threshold on the even repeats, then scores on the odd repeats, and the other way round. **This is the most honest number in the output.** A big drop from the selection half to the held-out half means overfitting.
- **Step 6, disguise diagnostic**: for every endpoint, its distance to ref-01 compared with the noise floor (how far each endpoint is from itself between even and odd repeats). Same-model endpoints (watermark, system prompt, persona, temperature) should have excess near 0, and fp8/LoRA clearly above. `Margin > 0` means the disguises are separable from the real changes.
- **Step 7** is the budget simulation described above.

## Metrics

- `tv`: total variation distance between the two histograms (0 = identical, 1 = no overlap).
- `js`: Jensen-Shannon divergence; similar, less sensitive to rare answers.
- `tvtemp`: temperature-tolerant TV. Decoding temperature does not change model identity but sharpens or flattens a histogram. This metric re-sharpens one side with several exponents and keeps the smallest TV, so a pure temperature difference costs little, while a change in which answers are likely (quantization, LoRA) remains.

## Things to watch

- **Watermark trap**: a Kirchenbauer-style watermark adds a fixed bias to a "green list" of tokens. For one-word answers this may shift the histogram consistently, so ref-06 could look like a different model. Check where ref-06 lands.
- **Persona/system prompt**: may change favourite answers (e.g. a pirate persona picks "parrot"). Check ref-03 and ref-05.
- **Normalisation** keeps only the first word in lowercase without punctuation. If an endpoint answers "Sure! 42", that becomes "sure". If you see that in step 2, improve `normalise()`.
- The ep endpoints may contain several families; the family threshold is not tuned here because ref has only one family.
