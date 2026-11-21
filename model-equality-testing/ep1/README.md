# ep1: clustering the 33 scored endpoints

Data: `ep_answers.jsonl`, 2 probes (`ep_probes.txt`: number 1-100, animal) x 20 answers per endpoint, collected with `ep_collect.py` (spends real ep budget; asks for confirmation).

| File | Purpose |
| --- | --- |
| `ep_collect.py` | Query ep-01..ep-33 with the probes (needs `ME_API_KEY`). Resumable. |
| `cluster_ep.py` | Two-level average-linkage clustering (family threshold, then model threshold). Metrics: `tv`, `tvtemp`, `l2u`, `mmd_perm`, `rank`. Writes `endpoint,family,model` CSV via `--out`. |
| `answer_norm.py` | Normalises raw answers (numbers, animal names). |
| `submission_family.csv` | l2u, family 0.10 / model 0.10: families X, Y, Z, {02,23}, 04, 06, 10, 20. Public score 0.1 (families right, no model split). |
| `queue/` | Submission candidates v1..v5, `README.txt` explains each; `scores.log` has the public scores. `old/` = earlier family-level variants. |
| `submit_queue.sh` | Submits queue/v1..v5 one at a time, 31 min apart (server limit), first after 23:10. Copies each to `queue/current.csv`, which `../submission.py` points to. |

```bash
source ../.venv/bin/activate
export ME_API_KEY="your team key"
python cluster_ep.py --metric l2u --family-threshold 0.10 --model-threshold 0.10 --out submission_family.csv
```

Findings so far (no new ep queries):
- Family structure is clear on l2u / mmd_perm (stable for family threshold 0.08 to 0.12).
- Inside families, raw formatting gives nothing (ep answers are always clean). Within-family mmd_perm z on both probes:
  X shows no differences; Y: ep-28 differs on both probes; Z splits by number-probe mode into {05,16}, {09,13,31}, {11,15,19,27,29}.
- On ref (exp3, partial), number 1-100 and number 1-10 are the probes that separate models; animal mostly separates families.
