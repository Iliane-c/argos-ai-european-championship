# Task 2 (Model Equality Testing): what we tried and what we ruled out

Night of 21–22 Nov 2026. Team argos. Best public score: **0.10** (v1).

## 1. The task and the score

- 33 scored endpoints `ep-01 … ep-33`, 100 answers per endpoint for the whole competition, at most 4 requests at a time per team.
- 9 practice endpoints `ref-01 … ref-09` with known answers (unlimited, unscored). They serve a separate small model, which identifies itself as **SmolLM**:
  one plain model (`ref-01`), four disguises of the same model (system prompt `ref-03`, persona `ref-05`, watermark `ref-06`, temperature 1.0 `ref-08`)
  and four different models (fp8 `ref-02`, LoRA 0.3 `ref-09`, LoRA 0.7 `ref-04`, LoRA 1.5 `ref-07`).
- Score = **0.1 × family MCC** (all pairs) + **0.9 × model MCC** (pairs inside one family); negative MCC counts as 0. The leaderboard moves in steps of 0.05.

## 2. What we know about ep

- Families are right: v1 scores exactly 0.10 (family part 1.0, model part 0).
  Families: X = {01,12,14,17,18,24,25,33}, Y = {03,07,08,21,22,26,28,30,32}, Z = {05,09,11,13,15,16,19,27,29,31}, plus {02,23}, 04, 06, 10, 20.
- Every ep endpoint **samples** (no greedy decoding): no two endpoints ever gave identical long answers, and no endpoint repeats itself on short prompts.
- The server returns **no usable metadata**: no token counts, no logprobs, `created` = 0, identical headers, a fixed ~6.7 s latency floor.

## 3. Submissions (all public score 0.10)

| File | Idea | Public |
|---|---|---|
| v1 | l2u clustering, one model per family | 0.10 |
| v2 | Z: {05,16} vs rest | 0.10 |
| v3 | Y: 28 alone; Z: {05,16}, {13}, rest | 0.10 |
| v4 | most central half of each family = one model, rest singletons | 0.10 |
| gt/s28 | only ep-28 split off | 0.10 |
| gt/v6 | complete linkage z ≤ 1.0 inside families (several models per family) | 0.10 |
| v6c | mutual-nearest pairs inside families | 0.10 |

Why splits never help: every apparent outlier has a near-identical partner in its family (13≈09, 05≈16, 28≈21, 07≈22),
so any split also breaks real same-model pairs. No endpoint differs from **every** member of its family.

## 4. Probes tried on ref, and why they were ruled out

| Probe | Result on ref | Verdict |
|---|---|---|
| Number 1–100 and animal histograms (exp1–3) | With ~18 answers, number-only reached model MCC 0.69 (optimistic). **With 60–78 answers it falls to ~0.4 (held-out 0.16–0.44)**: disguises (system prompt, temperature) really shift the distribution, so same-model z grows with n as fast as variant z. fp8 never separates. | Ruled out |
| Animal question | Separates families, adds noise at model level | Ruled out for models |
| Open-ended answers, style features | fp8 / LoRA 0.3 not separable; watermark differed most | Ruled out |
| Translation of a fixed paragraph | Ref model cannot translate | Untestable |
| FLIPS-style 0/1 sequences | Disguises vary more than variants | Ruled out |
| Many prompts × 1 answer (categories) | Same-model and cross-model agreement equal (0.12–0.16 vs 0.14–0.17) | Ruled out |
| Fixed-choice "choose one of …" (12 prompts × 3 rounds) | Promising on one batch, did not replicate; held-out AUC 0.52–0.59 | Ruled out |
| Anchored multiple choice (27 questions, 3 difficulty tiers) | Ref model guesses with a strong "A" position bias; same-model versions never agree | Untestable on ref |
| 2-shot completions | Server is chat-only; model answers conversationally | Ruled out |
| Proxy-model log-prob (RUT) with SmolLM2 135M/360M/1.7B | Only temperature 1.0 stands out; LoRA not separated (MCC 0.00–0.17) | Ruled out |
| **Length-constraint overrun** ("exactly N sentences") | **Replicated out of sample**: overrun rate same-model 0.05–0.15, LoRA 0.7 0.25, LoRA 1.5 0.75; fp8 0.00, LoRA 0.3 0.10. "High-information" prompts made everyone overrun. | Works only for strong LoRA |

## 5. Probes tried on ep (budget spent)

| Probe | ep queries | Result |
|---|---|---|
| 6 multiple-choice questions on ep-05/16/09/13 | 24 | ep model also mostly guesses (0–2/6 correct); no consistent split |
| Overrun, 3 validated prompts × 33 endpoints | 99 | **0 endpoints flagged.** Z: every answer exactly right; X writes fewer, long sentences. |
| Offline on those 99 answers | 0 | Repetition: only **ep-28** degenerates ("a bustling place, bustling with hustle and bustle…"), 10× above anyone else. TF-IDF embedding: no outliers in any family. |

## 6. Budget left (06:30, 22 Nov)

44 answers on 29 endpoints, 38 on ep-05/16/09/13.

## 7. Take-aways

1. Distribution tests on short answers detect **disguises** (system prompt, temperature) as differences; more samples make this worse, not better.
2. fp8 was never separable on ref by any probe we tried.
3. The only signal that survived out-of-sample validation (length overrun) catches strong LoRA on the small ref model, but the larger ep model follows length limits and showed nothing.
4. ep-28 is the one endpoint with visible degeneration, but splitting it alone (s28) did not change the public score.

## Where things are

- `ep1/` clustering, submissions (`queue/`), score log (`queue/scores.log`)
- `exp1/`–`exp3/` first probes on ref
- `exp4/` all experiments of the night; `exp4/README.md` lists each with files; `exp4/home_me_work/` was run from `$HOME` after `/p/scratch/training2666` hit its project inode limit
- The API key is read from `ME_API_KEY`; nothing secret is committed.
