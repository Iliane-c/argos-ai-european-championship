# exp4: new signals tested on the ref endpoints (night of 21-22 Nov)

All runs here query **ref only** (unlimited, unscored) unless stated. Answers keep the raw response plus metadata.

| What | Files | Result |
|---|---|---|
| Response metadata (latency, usage tokens, headers, logprobs) | `meta_query.py`, `run_meta.py`, `meta_answers.jsonl` | Nothing usable: no usage/logprobs, `created`=0, identical headers, fixed ~6.72 s latency floor on every endpoint. |
| Open-ended answers, style features | `exp4_prompts.txt`, `exp4_answers.jsonl`, `analyze_exp4.py` | ~6 answers/prompt: fp8 and LoRA 0.3 not separable; the watermark disguise differed most. |
| Translation of a fixed paragraph | `p_translate.txt`, `ans_translate.jsonl`, `analyze_exp5.py` | Untestable on ref: the small ref model cannot translate (self-similarity 0.14-0.30). |
| FLIPS-style 0/1 sequences | `p_flips.txt`, `ans_flips.jsonl`, `analyze_exp5.py` | fp8 / LoRA not separable (score -0.7 to -1.0); disguises vary more. |
| Many prompts x 1 answer (category prompts) | `p_many15.txt`, `ref_many.jsonl`, `analyze_refrep.py`, `analyze_agree.py` | Same-model and different-model agreement equal (0.12-0.16 vs 0.14-0.17). |
| Four prompt types (number/symbol, category, fill-in, fixed choice) | `p_four.txt`, `ref_four.jsonl`, `analyze_perprompt.py` | Fixed-choice "north/south/east/west" looked strong; fill-ins invalid on ref. |
| Validation batch: 12 fixed-choice + number + random/favourite colour, 3 rounds | `p_choice.txt`, `ref_choice.jsonl`, `analyze_choice.py` | Did not replicate. Held-out pair AUC 0.52-0.59 (15x1) and 0.55 (top-5 x 2). Only "Name your favorite color" hinted at LoRA vs disguises. |

Run (needs `export ME_API_KEY=...`): `python run_meta.py --prompts <file> --output <file> --repeats N [--parallel 4] [--allow-ep for ep, spends budget]`.
