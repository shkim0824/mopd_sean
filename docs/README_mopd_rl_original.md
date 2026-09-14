# mopd — Multi-teacher On-Policy Distillation (arXiv 2606.30406) for math / code / IF

Reimplementation of MOPD (Ma et al., Xiaomi, 2026) on the ib-a100 cluster:
Qwen3-4B-Base / Qwen3-8B-Base → **Stage 1** general SFT → **Stage 2** per-domain RL experts
(math / code / if) → **Stage 3** MOPD (student rollouts, per-prompt routing to the frozen domain
teacher served as a vLLM prefill service, per-token reverse-KL signal).

```
mopd/                       python package
  common/   io / yaml config / the Qwen3 chat-format contract (eos, thinking, generation_config)
  graders/  math (Math-Verify, last \boxed), code (official LiveCodeBench harness), if (IFEval / IFBench / IFEvalG)
  eval/     prompts (official), benchmark registry, vLLM DP generation, run_eval driver (8 GPU)
  data/     prep_eval (6 benches), prep_train (SFT + RL sets, decontamination), distill_if_sft (teacher + verifier)
  sft/      Stage 1: plain HF Trainer, assistant-only loss, ZeRO-2, 1 node
  rl/       Stage 2: trl 0.29 GRPO (colocate vLLM, DAPO loss), verifiable rewards, 2 nodes
  distill/  Stage 3: placement (student node0 / teachers node1+), teacher_client (vLLM prompt_logprobs),
            loss (Eq. 3-5), MOPDTrainer (GRPOTrainer subclass), train_mopd
third_party/  vendored official graders (google IFEval, allenai IFBench, open-instruct IFEvalG, LCB testing_util) — see PATCHES.md
configs/      accelerate_zero2.yaml, sft.yaml, rl_{math,code,if}.yaml, mopd.yaml, eval.yaml
scripts/      sbatch launchers: sft_1node, grpo_2node, mopd_2node, eval_1node, distill_if_1node, probe_base (+ _common.sh)
submit/       copy-paste job recipes (every sbatch needs approval; pool n041-066; -J sean-...)
env/          requirements + setup_mopd_env.sbatch (builds .venv/mopd inside the ngc-2410 container) + nltk_data
data/         eval/{aime24,aime25,aime26,lcb_v6,ifeval,ifbench}  train/{sft_*,rl_*}.jsonl  (+ manifests)
tests/        CPU tests: graders (reproduces official IFBench results), distill loss/placement, SFT masking
docs/         DESIGN.md (paper recipe + every implementation decision), DATA.md, INFRA.md (cluster launch idioms)
```

Quick start (on the cluster, after `env/setup_mopd_env.sbatch`): see `submit/README.md` — the order is
probe base → (IF distillation) → SFT → eval → 3× RL → MOPD → eval. All CPU tests:
`NLTK_DATA=env/nltk_data python tests/test_graders_quick.py && python tests/test_distill_cpu.py && QWEN3_TOKENIZER=models/Qwen3-4B-Base python tests/test_sft_dataset.py`.
