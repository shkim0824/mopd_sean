# Data — sources, exact ids, how each file was built

All raw sources were `snapshot_download`ed on the cpu-instance into
`/data/llm-public_636/lpt/users/sean/mopd_stage/hf/<org>__<name>/` (exact upstream files, revisions in
the `.cache` metadata) and converted by `mopd/data/prep_eval.py` / `prep_train.py`; every output dir has a
`manifest.json` (rows, sha256, drops). Cluster path: `~/lmalign/personal/sean/mopd/data/`.

## Eval (`data/eval/<bench>/test.jsonl`)

| bench | source | rows | row schema | prompt (see `mopd/eval/prompts.py`) | metric |
|---|---|---|---|---|---|
| aime24 | `MathArena/aime_2024_I` + `MathArena/aime_2024_II` (cross-checked vs `HuggingFaceH4/aime_2024`) | 30 | id, problem, answer (int 0-999 as str), year, contest, problem_idx, source | `{problem}\n\nPlease reason step by step, and put your final answer within \boxed{}.` (Qwen3 card / MathArena) | avg@32 (paper), `--n-math` |
| aime25 | `MathArena/aime_2025` (vs `opencompass/AIME2025`) | 30 | same | same | same |
| aime26 | `MathArena/aime_2026` (vs `math-ai/aime26`) | 30 | same | same | same |
| lcb_v6 | `livecodebench/code_generation_lite` `release_v6` = test.jsonl…test6.jsonl (2023-05-07 → 2025-04-06; easy 322 / medium 383 / hard 350; atcoder 602 / leetcode 444 / codeforces 9) | 1055 | upstream row + `slice` | LCB official: system `SYSTEM_MESSAGE_GENERIC`, user `### Question … ### Format … ### Answer:` | pass@1 = avg@4 over samples, official `run_test`; `--lcb-start-date/--lcb-end-date` for the Qwen "v6 (25.02-25.05)" window |
| ifeval | `google/IFEval` | 541 | key, prompt, instruction_id_list, kwargs (nulls dropped) | raw prompt, no system | prompt-level strict (+ 3 others) |
| ifbench | `allenai/IFBench_test` | 300 | same | same | prompt-level loose (IFBench paper) (+ 3 others) |

## Train (`data/train/*.jsonl`)

| file | source | filter | schema |
|---|---|---|---|
| sft_math.jsonl | `open-r1/Mixture-of-Thoughts` config `math` (93,733 R1 traces from OpenR1-Math-220k) | 2-turn rows with a closed `</think>` | id, domain, messages[user, assistant(`<think>…</think>\n\nanswer`)], num_tokens, source |
| sft_code.jsonl | `open-r1/Mixture-of-Thoughts` config `code` (83,070; codeforces-cots) | same | same |
| sft_if_prompts.jsonl | 25 % of the IF-RLVR prompts (disjoint from rl_if) | — | id, prompt, ground_truth, instruction_ids |
| sft_if.jsonl | ← `mopd.data.distill_if_sft` (teacher k=4, keep first candidate passing ALL constraints) | verifier all-pass | id, domain, messages, source, teacher |
| rl_math.jsonl | `SynthLabsAI/Big-Math-RL-Verified` (251k) + `Open-Reasoner-Zero/orz_math_57k_collected.json` | llama8b_solve_rate ∈ [0, 0.9] (drop trivial), no MCQ/proof, Math-Verify-parsable gold, dedup, **decontaminated vs AIME24-26** (normalised exact + 13-gram) | id, domain, prompt, answer, source, solve_rate |
| rl_code.jsonl | `agentica-org/DeepCoder-Preview-Dataset` configs `taco` (7,436) + `primeintellect` (16,252) | ≥5 tests, dedup, **decontaminated vs LCB v6**; configs `lcbv5` (LiveCodeBench May-23→Feb-25 = inside v6) and `codeforces` (CodeElo test) EXCLUDED | id, domain, prompt, starter_code, tests {inputs, outputs, fn_name}, source |
| rl_if.jsonl | `allenai/IF_multi_constraints_upto5` (95,373; IFEval 25 + IFTrain 29 constraint ids, ≤5 per prompt) | single-turn, all ids known to IFEvalG, decontaminated vs IFEval/IFBench prompts | id, domain, prompt, ground_truth (python-literal str), instruction_ids, source |

Notes
* IFBench's 58 test constraints are disjoint from the 29 IFTrain ids by construction (OOD generalisation).
* LCB v6 rows are ~4 MB each on average (private tests) — `iter_jsonl` streams them; graders decode
  `private_test_cases` lazily.
* Big-Math has no AIME decontamination statement upstream (it contains `amc_aime`, HARP, olympiads) —
  hence our own check; drops are counted in `MANIFEST.json`.
* Models: `models/Qwen3-4B-Base`, `models/Qwen3-8B-Base` (shards byte-verified vs HF), `models/gpt-oss-120b`
  (IF distillation teacher, paper-faithful) — all under `~/lmalign/personal/sean/models/`.
