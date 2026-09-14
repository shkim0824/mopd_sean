# MOPD reimplementation — paper recipe, mapping, decisions, details

Paper: *MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training*
(Ma, Wei, Zhao, … Luo; Xiaomi LLM Core / PKU; arXiv 2606.30406, 2026-06-29). No official code exists
(checked arXiv, HF paper page, XiaomiMiMo GitHub — MiMo-V2-Flash repo is inference-only). Third-party
implementations consulted: verl native `distillation.*` stack (`run_qwen3_8b_mopd_fsdp.sh`), TRL
`experimental.async_distillation.AsyncDistillationTrainer` (explicitly cites MOPD, `teacher_id` routing),
slime PR #2051 (`--use-mopd`), NeMo-RL `opd.py`, Tinker cookbook `on_policy_multi_teacher.py`, thunlp/OPD.

## 1. What the paper does (Appendix A/B + §3), verbatim numbers

| | paper (Qwen3-30B-A3B, math / IF / SWE) | this repo (Qwen3-4B/8B-Base, math / code / IF) |
|---|---|---|
| Stage 1 SFT | broad corpus over all domains; math = Mixture-of-Thoughts math subset; IF = IFBench-recipe prompts distilled on gpt-oss-120b; SWE = R2E-Gym distills | MoT `math` (93.7k) + MoT `code` (83.1k) + IF prompts from `allenai/IF_multi_constraints_upto5` distilled by `distill_if_sft.py` (teacher configurable, gpt-oss-120b downloaded) and **verifier-filtered** |
| Stage 2 RL data | math: BigMath + ORZ; IF: synthesised following the IFBench recipe; SWE: R2E-Gym-Lite | math: `SynthLabsAI/Big-Math-RL-Verified` + `orz_math_57k` (decontaminated vs AIME24-26); IF: `IF_multi_constraints_upto5` (= the released IF-RLVR set of the IFBench paper); code: DeepCoder `taco`+`primeintellect` (verified tests, decontaminated vs LCB v6; `lcbv5`/`codeforces` configs excluded) |
| RL algorithm | on-policy GRPO + Dynamic Sampling (DAPO), single gradient update per rollout batch | trl 0.29 `GRPOTrainer`, `loss_type=dapo`, `epsilon_high=0.28`, `beta=0`, `mask_truncated_completions`, `num_iterations=1` |
| RL HPs | lr 3e-6; BS 144 prompts × N=8; ~175K sequences (math/IF), SWE BS 80; max len 32,768 | lr 3e-6; 144 × 8 = 1152 seq/step; 152 steps ≈ 175K; max completion 32,768; fp32 master weights |
| MOPD | student = Stage-1 SFT ckpt; teachers frozen; BS 2048, N=1, **no** dynamic sampling; ratio math:IF:SWE = .35:.35:.30; PG form advantage clip A_max=5 (default); top-k form k=64; teacher = stand-alone prefill service, async | same; ratio math:code:if = .35:.30:.35; `distill.mode=pg|topk`; teachers = vLLM servers on node 1 (`prompt_logprobs`), student on node 0; lr 1e-6 (paper does not state Stage-3 lr) |
| Eval | AIME25/26 avg@32, IFBench/IFEval ×1, SWE-bench ×1; T=1.0, no top-p/top-k | **DEVIATION (user, 2026-08-30):** AIME24/25/26 **avg@8**, LCB v6 **pass@1 (n=1)**, IFEval/IFBench ×1; sampling **T=0.6 / top_p 0.95 / top_k 20** (Qwen3 thinking recipe), **16k** max tokens; `--preset paper` restores T=1.0/avg@32 |
| Metric | normalised score (s−s_student)/(s_teacher−s_student) averaged over domains | `metrics.json` per model; the normalised score is computed offline from the base/SFT/expert/MOPD table |

Dynamic Sampling (DAPO's resample-until-nonzero-variance) is not built into trl 0.29; zero-variance
groups get zero advantage (no gradient) and `frac_reward_zero_std` is logged — raise
`rl.prompts_per_step` if it is large. This is the one known approximation in Stage 2.

## 2. The objective, exactly as implemented (`mopd/distill/loss.py`)

* PG form: `Â_t = clip(log π_φd(y_t) − log π_θ(y_t), −A_max, A_max)` with π_θ = the current student
  (trl's `old_per_token_logps`, an HF forward on the rollout, no grad) and π_φd from the routed teacher's
  `prompt_logprobs` at the same token ids. `MOPDTrainer` writes this (B,T) tensor into
  `inputs["advantages"]`; trl 0.29's `_compute_loss` accepts (B,T) advantages (documented for MiniLLM) and
  with `loss_type="grpo"` computes −mean_seq( mean_t(ratio·Â_t) ) = Eq. 4 (ratio ≡ 1 on-policy; the vLLM
  importance-sampling correction of trl stays on and multiplies each token by π_HF/π_vLLM, which is the
  standard training/inference mismatch correction).
* Top-k form: `_compute_loss` is overridden — one forward gives student logits at completion positions,
  `topk_loss_from_logits` computes Σ_{v∈TopK_k(teacher)} π_θ(v)[log π_θ(v) − log π_φ(v)] − π_θ(v) + π_φ(v)
  with π_θ the **full-softmax** student probability (chunked fp32), sequence-mean, batch-mean.
  Unit test proves loss=0 at π_θ=π_φ, ≥0 on random students, and that gradient descent recovers the
  teacher top-k probabilities (the "minimised at π_θ=π_φd" property the paper adds the extra term for).
* Reverse-KL monitoring (Fig. 3): `mopd/student_teacher_kl` = mean_t[log π_θ − log π_φ] on the rollout,
  plus advantage stats / clip fraction, per-domain batch fraction, teacher request latency.

## 3. Infrastructure (2 nodes, 16 × A100-80GB)

* **Node 0 = student**: `accelerate launch --num_processes 8`, DeepSpeed ZeRO-2 (params unsharded so trl's
  colocate vLLM weight sync works), colocate vLLM (tp=1 per rank, `gpu_memory_utilization 0.3`), fp32
  master weights + bf16 autocast, gradient checkpointing, `save_only_model`.
* **Node 1 = teacher pool**: `mopd.distill.placement` assigns replicas: every teacher ≥1 replica (tp
  configurable per teacher, never crossing a node), extra replicas to the largest mixture weights →
  3 teachers = 3/3/2 replicas at tp=1; 6 teachers = 2/2/1/1/1/1 (`tests/test_distill_cpu.py`). Each
  replica = `vllm serve --max-logprobs 64 --logprobs-mode raw_logprobs --max-model-len 40960`, distinct
  HTTP port (8100+i) and engine port (`VLLM_PORT` 21000+100i), same-node launch stagger 20 s, health wait
  on `/v1/models`, killed on exit by the sbatch trap. 3+ nodes = more teacher GPUs (student stays on node 0).
* **Prefill request** (== verl / TRL): `POST /v1/completions {prompt: <token ids>, max_tokens: 1,
  prompt_logprobs: K}`; K=0 → the actual token's log-prob only (PG), K=64 → top-64 + actual (top-k).
  Token ids are the exact rollout ids (prompt ids left-padded by trl are stripped by the mask), so the
  teacher scores byte-identical sequences. `requests.Session(trust_env=False)` so the container's
  `http_proxy` never captures node-local traffic. Round-robin over replicas, 64-thread pool, retries.
* Synchronous vs. paper's async overlap: trl's loop generates a whole batch then scores it, so teacher
  prefill is on the critical path (not hidden behind sampling as in the paper's runtime). With 8 teacher
  GPUs prefill of 2048 × ≤32k tokens is a few minutes per step vs. tens of minutes of student sampling.

## 4. The chat-format contract (why base models are "not evaluable" as-is, and what we fixed)

Qwen3-*-Base **does** ship the full Qwen3 chat template (ChatML + `<think>` + `enable_thinking`), but:
`eos_token = <|endoftext|>` (151643) and `generation_config = {do_sample: false, eos_token_id: 151643}`
→ a base model prompted in ChatML never learned `<|im_end|>` as a stop, does not emit `<think>`, and
continues generating turns/garbage until the length cap; the `\boxed{}`/code-fence conventions are not
reliably followed either. `probe_base.sbatch` measures this on 60 rows × {chat+think, chat+no-think,
raw} and reports truncation rate, finished-thinking rate and scores. Everything we save
(`common/chat.py`) sets `eos=<|im_end|>`, `pad=<|endoftext|>`, `generation_config.eos_token_id=[151645,
151643]` and the SFT targets are `<think>\n…\n</think>\n\n<answer><|im_end|>`, rendered through the same
`apply_chat_template(add_generation_prompt=True, enable_thinking=…)` used for RL rollouts, teacher
prefill and eval (`tests/test_sft_dataset.py` asserts the byte-identical round-trip).

## 5. Graders = official code, verified

* **Math**: last `\boxed{}` (brace-matched) after stripping `<think>…</think>`; Math-Verify `parse/verify`
  with `$…$` wrapping (NeMo-Skills convention), integer fast path for AIME. Unit-tested incl. `073`==`73`.
* **Code**: vendored LiveCodeBench `testing_util.run_test` (per-test 6 s, stdin Decimal-tolerant line
  compare, call-based exact JSON compare) in a forked child with LCB's global timeout; extraction = LCB's
  last-fenced-block rule; pass iff all tests > 0. Private tests decoded from base64(zlib(pickle)) like
  upstream. RL rewards reuse the same path via `grade_batch` (process pool).
* **IF**: vendored Google IFEval (25 ids), allenai IFBench (83 ids), open-instruct IFEvalG (54 ids = the
  IF-RLVR train registry). Strict/loose exactly as `evaluation_lib` (8 loose variants).
  `tests/test_graders_quick.py` re-scores the 294 sample responses shipped in the IFBench repo and
  matches its `eval/eval_results_{strict,loose}.jsonl` on 289/290 (the one diff is a Punkt
  sentence-split edge case: upstream used the legacy punkt pickle, nltk ≥ 3.9 loads punkt_tab).
  Training reward = open-instruct `IFEvalVerifier` (fraction of constraints passed, thinking stripped).
* Thinking hygiene everywhere: unfinished thinking (`<think>` without `</think>`) scores/rewards 0.

## 6. Data (details in DATA.md)

Eval: AIME from MathArena (2024 I/II, 2025, 2026; cross-checked against HuggingFaceH4 / opencompass /
math-ai copies: 0 mismatches), LCB `release_v6` (1055, identical to the cluster dump; date window
selectable e.g. Qwen's 2025-02→2025-05), IFEval 541, IFBench 300. Train: see the table in §1;
`prep_train.py` decontaminates (normalised exact match + 13-gram overlap) every RL/SFT prompt against
the eval sets and writes manifests with row counts / drops / sha256.

## 7. Known deviations / open items

1. Code domain replaces SWE (user decision): LCB-style stdin/functional problems, single-turn, no sandbox agent loop.
2. IF SFT distillation teacher must be run (gpt-oss-120b downloaded; any local instruct model works) — GPU job.
3. Stage 2 Dynamic Sampling approximated (see §1). Stage 3 prefill is synchronous (see §3).
4. Stage-3 lr (1e-6) and step count (60 × 2048) are our choices; the paper reports plateaus at ~25–30K
   samples per domain (Fig. 2), i.e. ~40–50 steps of 2048 at the .35/.30/.35 ratio.
5. `num_generations=1` is realised by bypassing trl's `>=2` check (GRPO group statistics are ignored).
7. **trl 0.29 silent training-killers (found 2026-08-30, first RL runs learned nothing):** (a) `vllm_enable_sleep_mode=True`
   makes `VLLMGeneration.generate()` call `collective_rpc("reload_weights")` after `sync_weights()`, which re-reads the
   on-disk checkpoint and reverts the synced weights -> every rollout came from the initial policy (trl #5312; signature:
   `sampling/sampling_logp_difference/mean` grows monotonically). `train_grpo._patch_runtime` makes that RPC a no-op.
   (b) default `vllm_importance_sampling_mode="sequence_mask"` = exp(sum of per-token log-ratios) masked at 3.0, which
   for 4k-16k-token completions drives `importance_sampling_ratio/mean` to 1e-2..1e-4 and grad_norm to ~1e-7 (trl
   #4772/#5814). We use `token_truncate` (TIS, Yao et al.), cap 3.0 (`rl.vllm_is_mode` / `rl.vllm_is_cap`).
8. Nothing GPU-side has run yet (jobs need approval): the venv build, probe, smoke runs of SFT / RL / MOPD
   (`submit/README.md` has smoke overrides) are the first things to launch.
