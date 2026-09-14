# Vendored third-party code (exact upstream + minimal patches)

| dir | upstream | commit | what |
|---|---|---|---|
| `ifeval_google/` | github.com/google-research/google-research `instruction_following_eval/` | `0413387` (2026-08-29 HEAD) | official IFEval checkers (25 instruction ids), `evaluation_lib.py` strict/loose logic, `data/input_data.jsonl` (541 prompts) |
| `ifbench/` | github.com/allenai/IFBench `ifbench/` + `evaluation_lib.py` | `db69a6f` | official IFBench checkers (25 classic + 58 new = 83 ids), `data/IFBench_test.jsonl` (300 prompts) |
| `ifevalg/` | github.com/allenai/open-instruct `open_instruct/IFEvalG/` | `e6c3044` | the IF-RLVR TRAINING verifier registry (25 IFEval + 29 IFTrain ids) used by `IF_multi_constraints_upto5`; `_ground_truth_utils_reference.py` = open-instruct's `IFEvalVerifier` kept for reference only (not imported) |
| `lcb_runner/` | github.com/LiveCodeBench/LiveCodeBench `lcb_runner/evaluation/` | `28fef95` | official LiveCodeBench grader: `testing_util.run_test` (stdin + call-based, 6 s/test, Decimal-tolerant line compare), `pass_k_utils`, `compute_code_generation_metrics` (multiprocess wrapper). `*.py.ref` = prompt/extraction/problem code kept as reference (their imports need the full lcb_runner package; re-implemented verbatim in `mopd/eval/prompts.py` and `mopd/graders/code_grader.py`) |

## Patches applied (all mechanical)

1. Package imports rewritten so the copies import from `third_party.*`:
   - `from ifbench import …` → `from third_party.ifbench import …`
   - `from open_instruct.IFEvalG import …` → `from third_party.ifevalg import …`
   - `from instruction_following_eval import …` → `from third_party.ifeval_google import …`
   - `from lcb_runner.evaluation.{testing_util,pass_k_utils} import …` → `from third_party.lcb_runner.… import …`
2. `instructions_util._get_sentence_tokenizer()` in all three IF packages: upstream loads
   `nltk:tokenizers/punkt/english.pickle`; nltk ≥ 3.9 (venv has 3.10) refuses pickles, so a
   `try/except` falls back to `nltk.tokenize.PunktTokenizer("english")` (= `punkt_tab`, the same
   Punkt English model → identical sentence counts).
3. `ifbench/instructions_util.py` keeps its import-time `download_nltk_resources()`; with
   `NLTK_DATA` pointing at the staged offline dir (`env/nltk_data`) `nltk.data.find` succeeds and
   no download is attempted (cluster jobs are offline).
4. Removed upstream CLI wrappers (`run_eval.py`, `run.sh`, `evaluation_main.py` kept for ifeval_google)
   and the 3 MB GPT-4 sample-response file.

Nothing else was modified — `diff -r` against the upstream commits above should show only items 1–2.
