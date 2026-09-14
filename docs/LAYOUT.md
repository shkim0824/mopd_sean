# LAYOUT — the consolidated tree, how it was built, what maps where

Built 2026-09-14 by `build_mopd_sean.py` (scratchpad) from the three original repos + hand-written files (`overlay/`);
`MAPPING.md` lists every file's origin. Principles: `mopd_domains` (the newest code) is the base; the paper package
`mopd/mopd` is the core (it was byte-identical in `mopd` and `mopd_rl` except `graders/sandbox.py`, the newer one kept);
the domain-specific modules became sub-packages of the same namespaces; three copies of scripts / configs / docs / tests
collapsed into one; every absolute path was rewritten to this tree; big assets are symlinks.

## Tree

```
mopd_sean/
├── README.md                    pipelines + recipes            MAPPING.md  origin of every file      setup_cluster.sh  symlink farm + asset rsync
├── mopd/                                                        (PYTHONPATH = repo root)
│   ├── common/      chat.py (chat-format contract)  config.py (YAML + --override)  io.py  resume.py (resume policy)
│   ├── data/        prep_openthoughts3.py  build_ot3_verify.py  prep_train.py  prep_eval.py  distill_if_sft.py
│   │   └── domains/ prep_rl_pools.py (med/law/fin/IF verifiable pools)  prep_med_sft.py  build_pools.py (distillation prompt pools)
│   │                prep_ihc.py (IH-Challenge skeletons, held-out split, rounds)  ihc_attack_synth.py (online attack synthesis, vLLM DP)
│   ├── sft/         dataset.py  ot3_cache.py (memmap packed cache)  train_sft.py (accelerate + ZeRO-2; every SFT stage)
│   ├── teacher/     gen_teacher.py (vLLM k-sample traces)  reject_filter.py  select_traces.py  merge_teachers.py (weighted average)
│   ├── rl/          rewards.py  train_grpo.py (trl GRPO, legacy)
│   │   └── nemo/    run_grpo.py (NeMo-RL entrypoint: registers envs + processors, delegates to /opt/nemo-rl/examples/run_grpo.py)
│   │                prep_math_data.py  prep_3dom_data.py  sweep_nano_worker.py
│   │       ├── envs/        exactmatch_environment.py (med/law/fin)  if_environment.py + if_verify_worker.py  code_environment.py
│   │       │                math_with_judge_environment.py  ihc_environment.py (safety: per-row grader)  processors.py
│   │       └── judge_eval/  build_pairs.py  run_judge_eval.py (LLM-judge validation)
│   ├── distill/     loss.py (PG / top-k objectives)  placement.py (teacher server placement)  teacher_client.py (prefill logprobs)
│   │                trainer.py (MOPDTrainer on trl GRPOTrainer)  train_opd.py (domain OPD / MOPD driver)  train_mopd.py (paper driver, legacy)
│   ├── graders/     math_grader.py  code_grader.py (LCB harness)  if_grader.py  ot3_grader.py  pyexec_grader.py  sandbox.py (Landlock + RLIMIT)
│   │   └── domains/ extract.py (answer extraction)  graders.py (official MedQA / CaseHOLD / FinQA / ... scoring)  ihc_grader.py (IH-Challenge grader exec, forked timeout)
│   └── eval/        benchmarks.py  prompts.py  generate.py  run_eval.py (6-bench)
│       └── domains/ benches.py (domain benchmark loaders)  run_domain_eval.py (sharded vLLM runner)
├── scripts/
│   ├── _common.sh            REPO / S / IMG / NEMO_IMG / VENV, FWD_ENV, run_step, accel_cfg, wait_http, prefer, JP, cache_env
│   ├── sft.sbatch            any SFT (CONFIG=configs/sft/*.yaml, -N nodes)          gen_teacher.sbatch   big-teacher / domain-teacher traces
│   ├── teacher_sft_stage.sh  traces -> filtered SFT set -> teacher SFT              sft_warmup_stage.sh  traces -> n-row warm-up set -> SFT
│   ├── rl_nemo.sh            NeMo-RL GRPO submit (CONFIG, NODES, JUDGE, SMOKE)      nemo/ray.sub         NVIDIA ray launcher (verbatim)
│   ├── judge_server.sbatch   standing LLM judge                                     rl_auto_eval.sh      eval RL steps as they appear
│   ├── mk_eval_model.sh      NeMo step -> HF dir (old-schema config)                convert_nemo_step.sbatch  DCP -> HF when no consolidated dir
│   ├── opd.sbatch            OPD and MOPD (TEACHERS, WEIGHTS, TOPK, STUDENT_NODES, CONFIG, OVERRIDES)
│   ├── opd_from_ckpt.sh      OPD/MOPD from another checkpoint (generated config)    auto_eval.sh         eval OPD ckpts as they appear
│   ├── finals.sh             finished run: strip optimizer + final 6-bench          strip_optimizer.sh   weights-only checkpoints
│   ├── eval_domain.sbatch    domain benches, N nodes x 8 DP workers, T=1.0          eval_6bench.sbatch   AIME/LCB/IFEval/IFBench
│   ├── eval_big.sbatch       big / MoE model domain eval (NeMo container, TP)       sft_auto_eval.sh     eval SFT ckpts (steps or epochs)
│   ├── selfcheck.sbatch      imports (both containers) + CPU tests + config paths   cfgval.py            stdlib YAML reader, --check-paths
│   ├── ihc_synth.sbatch      IH-Challenge attack synthesis (1 node, 8 DP shards)    rl_ihc_loop.sh       safety teacher: 5 rounds x 40 GRPO steps
│   ├── make_1p7b_teachers.sh 1.7B teacher configs + submissions
│   └── legacy/               grpo_trl.sbatch, mopd_trl.sbatch (paper recipe, trl)
├── configs/
│   ├── accelerate_zero2.yaml  eval.yaml
│   ├── sft/    ot3_4b, ot3_1p7b_18k | distill_law_2ep, distill_med_c6_4ep | warmup_law, warmup_med, warmup_med_c6k1_4ep (A), warmup_med_c6k3_4ep (B), warmup_mix_lawmed
│   │   └── variants/  24 ablations (med candidates c1-c7, v2a/b, medo1, 35B-trace control, old-pool warm-up, 1.7B 16k)
│   ├── rl/     grpo_law_distill, grpo_law_rlonly, grpo_fin, grpo_if, grpo_ihc (safety round), grpo_code, grpo_math_v3, *_1p7b   (NeMo-RL; absolute paths)
│   │   └── variants/  15 (32k rollouts, med v2-v6 pools, law phase 2, nemotron math / 3-dom, math v2)
│   ├── opd/    {law,fin,if}_{pg,top64,top16}, med_pg, law_pg_rlteacher, law_pg_from_warmup75, med_pg_from_warmup{150,A561,B2164}
│   │   └── variants/  law v1 + hyphen-named v2 clones, old-pool med runs
│   ├── mopd/   4dom_pg128 (default), 4dom_pg128_merged, 4dom_pg128_merged_w4411, 4dom_pg128_mixsft, 3dom_rllaw_pg96, medlaw_pg64, finif_pg64
│   │   └── variants/  old-pool 4-dom / med-law runs, top-64 128 and 32, pg 32
│   └── legacy_trl/  sft, rl_{math,code,if,deepmath_4b}, mopd, mopd_deepmath_4b (paper recipe; data not built here)
├── third_party/   ifeval_google  ifevalg  ifbench  lcb_runner  verifiable_instructions (grader code) + legalbench  LEXam  casehold  FinQA
│                  TAT-QA  DocMath-Eval  FinanceReasoning  MedXpertQA  pubmedqa (benchmark data, rsynced)  PATCHES.md
├── tools/         analysis/ (drift, loops, truncation, tb dumps)  pools/ (med pool builders)  report/ (result tables, artifact generators)
│                  rl_judge/ (judge validation, pass-rate harvests)  disk/ (cache cleaner, cleanup reports)  watch/ (laptop-side watchers)
├── tests/         test_distill_cpu  test_domain_graders  test_graders_quick  test_ot3_cache  test_ot3_grader  test_sandbox_canary  test_sft_dataset
├── docs/          LAYOUT.md (this)  DESIGN.md  DATA.md  DATASETS.md  INFRA.md  OT3_SFT.md  SFT_SPEED.md  SUBMIT_RECIPES_legacy.md  README_*_original.md
├── env/           requirements*.txt  setup_mopd_env.sbatch  verify_nemo_container.sbatch  nltk_data/  tiktoken/  nemo_extras/{vendor,third_party_py}
├── models/        Qwen3-4B-OT3  Qwen3-4B  Qwen3-4B-Base  Qwen3-1.7B  Qwen3-1.7B-Base  Qwen3-1.7B-OT3  teacher-{law,law-sft,law-rlonly,fin,if,med}
│                  merged-4teachers-{uniform,uniform-lawsft,w4411}  rl_steps/<eval-ready RL step dirs>          (real directories since the asset move)
├── data/          eval  train  domains (+ ih-challenge)  rl_pools  distill_pools  sft_distill{,_v2,_v3}  sft_med  sft_med_o1  sft_warmup  rl3dom  nemotron_math  raw/  (real)
├── outputs/       opd/  eval_domain/  eval_6bench/  teacher_gen/  sft/<run>  rl/<run>                                      (real)
├── logs/          this tree's job logs + mopd/ mopd_domains/ mopd_rl/ (the retired repos' logs)     nemo/ (ray logs)
└── tmp/cache      per-job vLLM / inductor / triton caches (tools/disk/clean_caches.sh)
```

## Python module map (old import -> new import)
| old | new |
|---|---|
| `mopd.*` (mopd/mopd, mopd_rl/mopd) | `mopd.*` unchanged (sandbox.py = the mopd_rl variant) |
| `mopd.graders.domains.{extract,graders}` | `mopd.graders.domains.{extract,graders}` |
| `mopd.eval.domains.{benches,run_domain_eval}` | `mopd.eval.domains.{benches,run_domain_eval}` (defaults: data/domains, third_party of this tree) |
| `mopd.data.domains.{prep_rl_pools,prep_med_sft}`, `mopd.data.domains.build_pools` | `mopd.data.domains.*` |
| `mopd.teacher.{gen_teacher,reject_filter,select_traces}` | `mopd.teacher.*` |
| `mopd.distill.train_opd` | `mopd.distill.train_opd` |
| `mopd_rl/nemo/mopd.rl.nemo.envs.*`, `mopd_domains/nemo/mopd.rl.nemo.envs.*` | `mopd.rl.nemo.envs.*` |
| `mopd_rl/nemo/run_grpo_mopdrl.py`, `mopd_domains/nemo/run_grpo_domains.py` | `mopd.rl.nemo.run_grpo` (registers all four envs) |
| `mopd_domains/tmp/merge_teachers*.py` | `mopd.teacher.merge_teachers` (`--weights`) |

## Script map (old -> new)
| old | new |
|---|---|
| mopd/scripts/sft.sbatch, sft_2node.sbatch (+ mopd_rl copies) | scripts/sft.sbatch (`-N`) |
| mopd/scripts/eval_1node.sbatch, eval_2node.sbatch | scripts/eval_6bench.sbatch |
| mopd_domains/scripts/eval_domain.sbatch, domain_eval_2node.sbatch | scripts/eval_domain.sbatch (T=1.0 default, outputs/eval_domain) |
| mopd_domains/scripts/eval_big_dp.sbatch, eval_big_tp8.sbatch | scripts/eval_big.sbatch (TP=8 -> one worker) |
| mopd_domains/scripts/opd.sbatch, opd_multinode.sbatch, mopd/scripts/opd.sbatch | scripts/opd.sbatch (domain driver); scripts/legacy/mopd_trl.sbatch (paper driver) |
| mopd/scripts/grpo_2node.sbatch | scripts/legacy/grpo_trl.sbatch |
| mopd_domains/scripts/gen_teacher.sbatch | scripts/gen_teacher.sbatch (outputs/teacher_gen) |
| mopd_domains/scripts/run_stage3_4.sh, law_merge_stage3_4.sh | scripts/teacher_sft_stage.sh (generic; the law shard merge was a one-off) |
| mopd_domains/scripts/submit_domain_rl*.sh, mopd_rl/scripts/submit_*_rl.sh, ray_mopdrl.sub | scripts/rl_nemo.sh + scripts/nemo/ray.sub |
| mopd_rl/scripts/judge_server.sbatch, convert_ck200.sbatch | scripts/judge_server.sbatch, scripts/convert_nemo_step.sbatch |
| opd_auto_eval.sh, opd_auto_eval_mopd.sh, opd_auto_eval_if.sh | scripts/auto_eval.sh `<run> <dom3|bench:X|if> <steps...>` |
| opd_final_evals.sh, opd_final_evals_if.sh | scripts/finals.sh |
| sft_warmup_auto_eval.sh, sft_epoch_auto_eval.sh | scripts/sft_auto_eval.sh |
| mopd_domains/scripts/{opd_from_ckpt,sft_warmup_stage,rl_auto_eval,mk_eval_model}.sh | same names under scripts/ (REPO-relative) |
| tmp/inventory/strip_optimizer_all.sh | scripts/strip_optimizer.sh |
| (new) | scripts/selfcheck.sbatch, scripts/cfgval.py --check-paths |

Dropped (one-off or superseded, still in the originals / `$S/past`): babysit_jobs.sh, bs_sweep_{sft,rl}.sbatch, probe_base.sbatch,
distill_if_{1node,multinode}.sbatch (IF SFT-data distillation of the pre-OT3 recipe; module `mopd.data.distill_if_sft` kept),
judge_eval.sbatch (module `mopd.rl.nemo.judge_eval` kept), mopd_domains/tmp/* stage-setup and grep helpers, mopd_rl/configs
(duplicates of mopd/configs), mopd_rl/tests (duplicates).

## Config conventions
- `configs/sft`, `configs/opd`, `configs/mopd`: repo-relative paths (`models/Qwen3-4B-OT3`, `data/rl_pools/...`, `outputs/opd/<run>`);
  run names unchanged (`train.output: outputs/opd/mopd-4dom-128-sftp` is the run that exists on disk).
- `configs/rl` (NeMo-RL): absolute `$S/mopd_sean/...` (ray workers), `checkpoint_dir: outputs/rl/<run>`, vendored deps in
  `env/nemo_extras`, teachers by canonical names (`models/teacher-law`, `models/rl_steps/<step>`), archived big models in `$S/past/models`.
- `variants/` = every ablation config that was run, renamed to the new scheme (MAPPING.md has the old name).

## Symlink farm (setup_cluster.sh; idempotent, re-run after moving the tree)
| tree path | target |
|---|---|
| models/Qwen3-4B-OT3, Qwen3-4B, Qwen3-4B-Base, Qwen3-1.7B, Qwen3-1.7B-Base | $S/models/<name> |
| models/Qwen3-1.7B-OT3 | $S/mopd/outputs/sft_ot3/qwen3-1p7b-18k |
| models/teacher-law / teacher-law-rlonly / teacher-fin | $S/mopd_domains/models/rl2-law-distill-ck200 / rl-law-ck200 / rl-fin-ck200 |
| models/teacher-if | $S/mopd_rl/models/rl-if-ck200 (file-level links into results/grpo_if_4b/step_200) |
| models/teacher-med | $S/mopd_domains/results/sft_distill_med_c6_4ep/checkpoint-3368 |
| models/merged-4teachers-{uniform,w4411}, models/rl_steps/* | $S/mopd_domains/models/..., $S/mopd_rl/models/rl-* |
| data/eval, data/train | $S/mopd/data/{eval,train} |
| data/domains | $S/data/domains (benchmark sources) |
| data/rl_pools, distill_pools, sft_distill*, sft_med*, sft_warmup | $S/mopd_domains/data/<same> |
| data/rl3dom, data/nemotron_math, data/raw/* | $S/mopd_rl/data/<same>, $S/data/<dataset> |
| outputs/opd, outputs/eval_domain, outputs/teacher_gen | $S/mopd_domains/outputs/{opd,eval,distill} |
| outputs/eval_6bench | $S/mopd/outputs/eval |
| outputs/sft/ot3_qwen3-4b, ot3_qwen3-1p7b-18k | $S/mopd/outputs/sft_ot3/<run> |
| outputs/sft/<run>, outputs/rl/<run> | $S/mopd_domains/results/<run>, $S/{mopd_domains,mopd_rl}/results/grpo_* |
| logs/_orig_* | the three original logs/ dirs |
| third_party/<benchmark data>, env/nltk_data, env/tiktoken, env/nemo_extras | rsynced copies (1.3 GB + vendored python deps) |

## Verification (2026-09-14, final tree at $S/mopd_sean)
| check | result |
|---|---|
| local static | py_compile on 124 .py OK, bash -n on 33 shell scripts OK, no reference to the old module names |
| setup_cluster.sh | 14 model links, 62 rl_steps, 33 sft runs, 14 rl runs, 0 dangling links |
| scripts/cfgval.py --check-paths | 332 paths, 0 problems (legacy_trl excluded: its data was never built) |
| scripts/selfcheck.sbatch (job 573551) | 52/52 module imports in the training container, 7/7 CPU tests, config paths OK, NeMo-container imports of all RL envs + graders OK |
| scripts/eval_domain.sbatch smoke (573525, 1 node, LIMIT 48) | MedQA 77.1 / CaseHOLD 62.5 / FinQA 62.5 on Qwen3-4B-OT3 at T=1.0 (5 min 33 s) |
| scripts/eval_6bench.sbatch smoke (573526, 1 node, LIMIT 8) | generation + grading + aggregation completed (9 min 14 s) |
| scripts/opd.sbatch smoke (573527, 1 student + 1 teacher node, 2 steps) | 8 teacher servers ready, step 1 44 s, step 2 30 s, checkpoints + final + opd_done.json written (9 min 21 s) |
| scripts/finals.sh in real use | branch A run: optimizer strip 530 GB -> 80 GB, final 6-bench submitted |

Bugs found and fixed by the checks: `tests/test_domain_graders.py` lacked `import os` after the merge; `mopd/graders/domains/graders.py`
resolved `third_party` one level too high after moving one directory deeper (now honours `DOMAINS_TP`, else `../../../third_party`).
