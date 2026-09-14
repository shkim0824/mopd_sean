# MAPPING — where everything came from (build: scratchpad/build_mopd_sean.py, 2026-09-14)

## Files
| kind | original | mopd_sean | note |
|---|---|---|---|
| file | mopd/mopd | mopd | base package (paper implementation) |
| file | mopd_rl/mopd/graders/sandbox.py | mopd/graders/sandbox.py | newer variant (RLIMIT_AS cap, 2026-09-07) |
| file | mopd_domains/mopd_domains/data/prep_rl_pools.py | mopd/data/domains/prep_rl_pools.py |  |
| file | mopd_domains/mopd_domains/data/prep_med_sft.py | mopd/data/domains/prep_med_sft.py |  |
| file | mopd_domains/mopd_domains/distill/build_pools.py | mopd/data/domains/build_pools.py |  |
| file | mopd_domains/mopd_domains/distill/gen_teacher.py | mopd/teacher/gen_teacher.py |  |
| file | mopd_domains/mopd_domains/distill/reject_filter.py | mopd/teacher/reject_filter.py |  |
| file | mopd_domains/mopd_domains/distill/select_traces.py | mopd/teacher/select_traces.py |  |
| file | overlay/mopd/teacher/merge_teachers.py | mopd/teacher/merge_teachers.py | generalised weighted merge (tmp/merge_teachers*.py) |
| file | mopd_domains/mopd_domains/graders/extract.py | mopd/graders/domains/extract.py |  |
| file | mopd_domains/mopd_domains/graders/graders.py | mopd/graders/domains/graders.py |  |
| file | mopd_domains/mopd_domains/eval/benches.py | mopd/eval/domains/benches.py |  |
| file | mopd_domains/mopd_domains/eval/run_domain_eval.py | mopd/eval/domains/run_domain_eval.py |  |
| file | mopd_domains/opd/train_opd_domains.py | mopd/distill/train_opd.py | domain OPD/MOPD driver |
| file | overlay/mopd/rl/nemo/run_grpo.py | mopd/rl/nemo/run_grpo.py | unified NeMo-RL entrypoint (run_grpo_domains + run_grpo_mopdrl) |
| file | mopd_rl/nemo/prep_3dom_data.py | mopd/rl/nemo/prep_3dom_data.py |  |
| file | mopd_rl/nemo/prep_math_data.py | mopd/rl/nemo/prep_math_data.py |  |
| file | mopd_rl/nemo/sweep_nano_worker.py | mopd/rl/nemo/sweep_nano_worker.py |  |
| file | mopd_rl/nemo/mopd_rl_envs/code_environment.py | mopd/rl/nemo/envs/code_environment.py |  |
| file | mopd_rl/nemo/mopd_rl_envs/if_environment.py | mopd/rl/nemo/envs/if_environment.py |  |
| file | mopd_rl/nemo/mopd_rl_envs/if_verify_worker.py | mopd/rl/nemo/envs/if_verify_worker.py |  |
| file | mopd_rl/nemo/mopd_rl_envs/math_with_judge_environment.py | mopd/rl/nemo/envs/math_with_judge_environment.py |  |
| file | mopd_rl/nemo/mopd_rl_envs/processors.py | mopd/rl/nemo/envs/processors.py |  |
| file | mopd_domains/nemo/domain_envs/exactmatch_environment.py | mopd/rl/nemo/envs/exactmatch_environment.py |  |
| file | mopd_rl/nemo/judge_eval/build_pairs.py | mopd/rl/nemo/judge_eval/build_pairs.py |  |
| file | mopd_rl/nemo/judge_eval/run_judge_eval.py | mopd/rl/nemo/judge_eval/run_judge_eval.py |  |
| file | overlay/scripts/sft_warmup_stage.sh | scripts/sft_warmup_stage.sh |  |
| file | overlay/scripts/teacher_sft_stage.sh | scripts/teacher_sft_stage.sh |  |
| file | overlay/scripts/eval_domain.sbatch | scripts/eval_domain.sbatch |  |
| file | overlay/scripts/mk_eval_model.sh | scripts/mk_eval_model.sh |  |
| file | overlay/scripts/convert_nemo_step.sbatch | scripts/convert_nemo_step.sbatch |  |
| file | overlay/scripts/eval_6bench.sbatch | scripts/eval_6bench.sbatch |  |
| file | overlay/scripts/opd.sbatch | scripts/opd.sbatch |  |
| file | overlay/scripts/eval_big.sbatch | scripts/eval_big.sbatch |  |
| file | overlay/scripts/cfgval.py | scripts/cfgval.py |  |
| file | overlay/scripts/strip_optimizer.sh | scripts/strip_optimizer.sh |  |
| file | overlay/scripts/rl_nemo.sh | scripts/rl_nemo.sh |  |
| file | overlay/scripts/judge_server.sbatch | scripts/judge_server.sbatch |  |
| file | overlay/scripts/sft_auto_eval.sh | scripts/sft_auto_eval.sh |  |
| file | overlay/scripts/auto_eval.sh | scripts/auto_eval.sh |  |
| file | overlay/scripts/gen_teacher.sbatch | scripts/gen_teacher.sbatch |  |
| file | overlay/scripts/opd_from_ckpt.sh | scripts/opd_from_ckpt.sh |  |
| file | overlay/scripts/finals.sh | scripts/finals.sh |  |
| file | overlay/scripts/selfcheck.sbatch | scripts/selfcheck.sbatch |  |
| file | overlay/scripts/rl_auto_eval.sh | scripts/rl_auto_eval.sh |  |
| file | overlay/scripts/sft.sbatch | scripts/sft.sbatch |  |
| file | overlay/scripts/_common.sh | scripts/_common.sh |  |
| file | overlay/scripts/legacy/grpo_trl.sbatch | scripts/legacy/grpo_trl.sbatch |  |
| file | overlay/scripts/legacy/mopd_trl.sbatch | scripts/legacy/mopd_trl.sbatch |  |
| file | mopd_rl/scripts/ray_mopdrl.sub | scripts/nemo/ray.sub | NVIDIA ray launcher (verbatim) |
| file | mopd_rl/scripts/verify_container.sbatch | env/verify_nemo_container.sbatch |  |
| file | mopd/env/setup_mopd_env.sbatch | env/setup_mopd_env.sbatch |  |
| file | mopd/env/requirements.txt | env/requirements.txt |  |
| file | mopd/env/requirements-eval.txt | env/requirements-eval.txt |  |
| config | mopd/configs/accelerate_zero2.yaml | configs/accelerate_zero2.yaml |  |
| config | mopd/configs/eval.yaml | configs/eval.yaml |  |
| config | mopd/configs/sft_ot3_4b.yaml | configs/sft/ot3_4b.yaml |  |
| config | mopd/configs/sft_ot3_1p7b_18k.yaml | configs/sft/ot3_1p7b_18k.yaml |  |
| config | mopd/configs/sft_ot3_1p7b.yaml | configs/sft/variants/ot3_1p7b_16k.yaml |  |
| config | mopd/configs/sft.yaml | configs/legacy_trl/sft.yaml |  |
| config | mopd/configs/mopd.yaml | configs/legacy_trl/mopd.yaml |  |
| config | mopd/configs/mopd_deepmath_4b.yaml | configs/legacy_trl/mopd_deepmath_4b.yaml |  |
| config | mopd/configs/rl_code.yaml | configs/legacy_trl/rl_code.yaml |  |
| config | mopd/configs/rl_deepmath_4b.yaml | configs/legacy_trl/rl_deepmath_4b.yaml |  |
| config | mopd/configs/rl_if.yaml | configs/legacy_trl/rl_if.yaml |  |
| config | mopd/configs/rl_math.yaml | configs/legacy_trl/rl_math.yaml |  |
| config | mopd_domains/configs/sft_distill_law_2ep.yaml | configs/sft/distill_law_2ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c6_4ep.yaml | configs/sft/distill_med_c6_4ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c1.yaml | configs/sft/variants/distill_med_c1.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c7_3ep.yaml | configs/sft/variants/distill_med_c7_3ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c6_3ep.yaml | configs/sft/variants/distill_med_c6_3ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_2ep.yaml | configs/sft/variants/distill_med_2ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_v2a.yaml | configs/sft/variants/distill_med_v2a.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c1_lr5.yaml | configs/sft/variants/distill_med_c1_lr5.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c1_3ep.yaml | configs/sft/variants/distill_med_c1_3ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c6.yaml | configs/sft/variants/distill_med_c6.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c4.yaml | configs/sft/variants/distill_med_c4.yaml |  |
| config | mopd_domains/configs/sft_distill_med_1ep.yaml | configs/sft/variants/distill_med_1ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c4_3ep.yaml | configs/sft/variants/distill_med_c4_3ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_v2b.yaml | configs/sft/variants/distill_med_v2b.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c5.yaml | configs/sft/variants/distill_med_c5.yaml |  |
| config | mopd_domains/configs/sft_distill_law_1ep.yaml | configs/sft/variants/distill_law_1ep.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c2.yaml | configs/sft/variants/distill_med_c2.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c3.yaml | configs/sft/variants/distill_med_c3.yaml |  |
| config | mopd_domains/configs/sft_distill_med_c3_3ep.yaml | configs/sft/variants/distill_med_c3_3ep.yaml |  |
| config | mopd_domains/configs/sft_med_4b.yaml | configs/sft/variants/med_4b.yaml |  |
| config | mopd_domains/configs/sft_medo1_en.yaml | configs/sft/variants/medo1_en.yaml |  |
| config | mopd_domains/configs/sft_medo1_endist.yaml | configs/sft/variants/medo1_endist.yaml |  |
| config | mopd_domains/configs/sft_medo1_enmix.yaml | configs/sft/variants/medo1_enmix.yaml |  |
| config | mopd_domains/configs/sft_warmup_law.yaml | configs/sft/warmup_law.yaml |  |
| config | mopd_domains/configs/sft_warmup_med_sftp.yaml | configs/sft/warmup_med.yaml |  |
| config | mopd_domains/configs/sft_warmup_med.yaml | configs/sft/variants/warmup_med_oldpool.yaml |  |
| config | mopd_domains/configs/sft_warmup_med_35b.yaml | configs/sft/variants/warmup_med_35b.yaml |  |
| config | mopd_domains/configs/sft_warmup_med_c6k1_4ep.yaml | configs/sft/warmup_med_c6k1_4ep.yaml |  |
| config | mopd_domains/configs/sft_warmup_med_c6k3_4ep.yaml | configs/sft/warmup_med_c6k3_4ep.yaml |  |
| config | mopd_domains/configs/sft_warmup_mix_lawmed.yaml | configs/sft/warmup_mix_lawmed.yaml |  |
| config | mopd_domains/configs/opd_law_pg_v2.yaml | configs/opd/law_pg.yaml |  |
| config | mopd_domains/configs/opd_law_top64_v2.yaml | configs/opd/law_top64.yaml |  |
| config | mopd_domains/configs/opd_law_top16_v2.yaml | configs/opd/law_top16.yaml |  |
| config | mopd_domains/configs/opd_law_pg.yaml | configs/opd/variants/law_pg_v1.yaml |  |
| config | mopd_domains/configs/opd_law_top64.yaml | configs/opd/variants/law_top64_v1.yaml |  |
| config | mopd_domains/configs/opd_law_top16.yaml | configs/opd/variants/law_top16_v1.yaml |  |
| config | mopd_domains/configs/opd_law_pg-v2.yaml | configs/opd/variants/law_pg_v2_hyphen.yaml |  |
| config | mopd_domains/configs/opd_law_top64-v2.yaml | configs/opd/variants/law_top64_v2_hyphen.yaml |  |
| config | mopd_domains/configs/opd_law_top16-v2.yaml | configs/opd/variants/law_top16_v2_hyphen.yaml |  |
| config | mopd_domains/configs/opd_law_pg-rlteacher.yaml | configs/opd/law_pg_rlteacher.yaml |  |
| config | mopd_domains/configs/opd_law_pg-sftw75.yaml | configs/opd/law_pg_from_warmup75.yaml |  |
| config | mopd_domains/configs/opd_fin_pg.yaml | configs/opd/fin_pg.yaml |  |
| config | mopd_domains/configs/opd_fin_top64.yaml | configs/opd/fin_top64.yaml |  |
| config | mopd_domains/configs/opd_fin_top16.yaml | configs/opd/fin_top16.yaml |  |
| config | mopd_domains/configs/opd_if_pg.yaml | configs/opd/if_pg.yaml |  |
| config | mopd_domains/configs/opd_if_top64.yaml | configs/opd/if_top64.yaml |  |
| config | mopd_domains/configs/opd_if_top16.yaml | configs/opd/if_top16.yaml |  |
| config | mopd_domains/configs/opd_med_pg-sftp.yaml | configs/opd/med_pg.yaml |  |
| config | mopd_domains/configs/opd_med_pg.yaml | configs/opd/variants/med_pg_oldpool.yaml |  |
| config | mopd_domains/configs/opd_med_top64.yaml | configs/opd/variants/med_top64_oldpool.yaml |  |
| config | mopd_domains/configs/opd_med_top16.yaml | configs/opd/variants/med_top16_oldpool.yaml |  |
| config | mopd_domains/configs/opd_med_pg-sftp-sftw150.yaml | configs/opd/med_pg_from_warmup150.yaml |  |
| config | mopd_domains/configs/opd_med_pg-sftw150.yaml | configs/opd/variants/med_pg_from_warmup150_oldpool.yaml |  |
| config | mopd_domains/configs/med-pg-sftwA561.yaml | configs/opd/med_pg_from_warmupA561.yaml |  |
| config | mopd_domains/configs/med-pg-sftwB2164.yaml | configs/opd/med_pg_from_warmupB2164.yaml |  |
| config | mopd_domains/configs/mopd-4dom-128-sftp.yaml | configs/mopd/4dom_pg128.yaml |  |
| config | mopd_domains/configs/mopd-4dom-128-merged-sftp.yaml | configs/mopd/4dom_pg128_merged.yaml |  |
| config | mopd_domains/configs/mopd-4dom-128-merged-w4411.yaml | configs/mopd/4dom_pg128_merged_w4411.yaml |  |
| config | mopd_domains/configs/mopd-4dom-128-mixsft.yaml | configs/mopd/4dom_pg128_mixsft.yaml |  |
| config | mopd_domains/configs/mopd-3dom-rllaw-96.yaml | configs/mopd/3dom_rllaw_pg96.yaml |  |
| config | mopd_domains/configs/mopd-medlaw-64-sftp.yaml | configs/mopd/medlaw_pg64.yaml |  |
| config | mopd_domains/configs/mopd-finif-64.yaml | configs/mopd/finif_pg64.yaml |  |
| config | mopd_domains/configs/mopd_4dom_pg128.yaml | configs/mopd/variants/4dom_pg128_oldpool.yaml |  |
| config | mopd_domains/configs/mopd_4dom_pg32.yaml | configs/mopd/variants/4dom_pg32_oldpool.yaml |  |
| config | mopd_domains/configs/mopd-4dom-128-top64.yaml | configs/mopd/variants/4dom_top64_128_oldpool.yaml |  |
| config | mopd_domains/configs/mopd-4dom-32-top64.yaml | configs/mopd/variants/4dom_top64_32_oldpool.yaml |  |
| config | mopd_domains/configs/mopd-4dom-128-merged.yaml | configs/mopd/variants/4dom_pg128_merged_oldpool.yaml |  |
| config | mopd_domains/configs/mopd-medlaw-64.yaml | configs/mopd/variants/medlaw_pg64_oldpool.yaml |  |
| config | mopd_domains/nemo/configs/grpo_law_distill_4b.yaml | configs/rl/grpo_law_distill.yaml |  |
| config | mopd_domains/nemo/configs/grpo_law_4b.yaml | configs/rl/grpo_law_rlonly.yaml |  |
| config | mopd_domains/nemo/configs/grpo_fin_4b.yaml | configs/rl/grpo_fin.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_v6_4b.yaml | configs/rl/variants/grpo_med_v6.yaml |  |
| config | mopd_domains/nemo/configs/grpo_law_phase2_4b.yaml | configs/rl/variants/grpo_law_phase2.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_v5_4b.yaml | configs/rl/variants/grpo_med_v5.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_v4_4b.yaml | configs/rl/variants/grpo_med_v4.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_4b_32k.yaml | configs/rl/variants/grpo_med_32k.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_distill_4b.yaml | configs/rl/variants/grpo_med_distill.yaml |  |
| config | mopd_domains/nemo/configs/grpo_law_4b_32k.yaml | configs/rl/variants/grpo_law_32k.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_4b.yaml | configs/rl/variants/grpo_med.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_v2_4b.yaml | configs/rl/variants/grpo_med_v2.yaml |  |
| config | mopd_domains/nemo/configs/grpo_fin_4b_32k.yaml | configs/rl/variants/grpo_fin_32k.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_v4a_4b.yaml | configs/rl/variants/grpo_med_v4a.yaml |  |
| config | mopd_domains/nemo/configs/grpo_med_v3_4b.yaml | configs/rl/variants/grpo_med_v3.yaml |  |
| config | mopd_rl/nemo/configs/grpo_if_4b.yaml | configs/rl/grpo_if.yaml |  |
| config | mopd_rl/nemo/configs/grpo_code_4b.yaml | configs/rl/grpo_code.yaml |  |
| config | mopd_rl/nemo/configs/grpo_math_v3_4b.yaml | configs/rl/grpo_math_v3.yaml |  |
| config | mopd_rl/nemo/configs/grpo_math_v2_4b.yaml | configs/rl/variants/grpo_math_v2.yaml |  |
| config | mopd_rl/nemo/configs/grpo_nemotron_3dom_4b.yaml | configs/rl/variants/grpo_nemotron_3dom.yaml |  |
| config | mopd_rl/nemo/configs/grpo_nemotron_math_4b.yaml | configs/rl/variants/grpo_nemotron_math.yaml |  |
| file | mopd_rl/third_party | third_party | grader vendoring superset (ifeval_google, ifevalg, ifbench, lcb_runner, verifiable_instructions) |
| file | mopd/tests/test_ot3_grader.py | tests/test_ot3_grader.py |  |
| file | mopd/tests/test_ot3_cache.py | tests/test_ot3_cache.py |  |
| file | mopd/tests/test_distill_cpu.py | tests/test_distill_cpu.py |  |
| file | mopd/tests/test_sft_dataset.py | tests/test_sft_dataset.py |  |
| file | mopd/tests/test_graders_quick.py | tests/test_graders_quick.py |  |
| file | mopd/tests/test_sandbox_canary.py | tests/test_sandbox_canary.py |  |
| file | mopd_domains/tests/test_graders.py | tests/test_domain_graders.py |  |
| file | mopd/docs/DATA.md | docs/DATA.md |  |
| file | mopd/docs/SFT_SPEED.md | docs/SFT_SPEED.md |  |
| file | mopd/docs/INFRA.md | docs/INFRA.md |  |
| file | mopd/docs/DESIGN.md | docs/DESIGN.md |  |
| file | mopd/docs/OT3_SFT.md | docs/OT3_SFT.md |  |
| file | overlay/README.md | README.md | consolidated README |
| file | overlay/.gitignore | .gitignore | git ignore list (assets, benchmark clones) |
| file | overlay/docs/LAYOUT.md | docs/LAYOUT.md | tree + maps + verification |
| file | mopd_domains/docs/DATASETS.md | docs/DATASETS.md |  |
| file | mopd/submit/README.md | docs/SUBMIT_RECIPES_legacy.md |  |
| file | mopd/README.md | docs/README_mopd_original.md |  |
| file | mopd_domains/README.md | docs/README_mopd_domains_original.md |  |
| file | mopd_rl/README.md | docs/README_mopd_rl_original.md |  |
| file | mopd_domains/tmp/build_medqa_pools.py | tools/pools/build_medqa_pools.py |  |
| file | mopd_domains/tmp/build_med_aux_pools.py | tools/pools/build_med_aux_pools.py |  |
| file | mopd_domains/tmp/build_med_evol_pool.py | tools/pools/build_med_evol_pool.py |  |
| file | mopd_domains/tmp/build_med_rl_v2_pool.py | tools/pools/build_med_rl_v2_pool.py |  |
| file | mopd_domains/tmp/build_med_v4_pool.py | tools/pools/build_med_v4_pool.py |  |
| file | mopd_domains/tmp/build_med_v5_v6.py | tools/pools/build_med_v5_v6.py |  |
| file | mopd_domains/tmp/build_tbqa_pool.py | tools/pools/build_tbqa_pool.py |  |
| file | mopd_domains/tmp/build_law_phase2.py | tools/pools/build_law_phase2.py |  |
| file | mopd_domains/tmp/filter_med_v2_pool.py | tools/pools/filter_med_v2_pool.py |  |
| file | mopd_domains/tmp/merge_med_v2_pool.py | tools/pools/merge_med_v2_pool.py |  |
| file | mopd_domains/tmp/inspect_med_sources.py | tools/pools/inspect_med_sources.py |  |
| file | mopd_domains/tmp/inspect_ultramedical.py | tools/pools/inspect_ultramedical.py |  |
| file | mopd_domains/tmp/inspect_ultramedical2.py | tools/pools/inspect_ultramedical2.py |  |
| file | mopd_domains/tmp/adv_by_source.py | tools/analysis/adv_by_source.py |  |
| file | mopd_domains/tmp/mopd_domain_trunc.py | tools/analysis/mopd_domain_trunc.py |  |
| file | mopd_domains/tmp/med_drift_128.py | tools/analysis/med_drift_128.py |  |
| file | mopd_domains/tmp/fin_drift_top64.py | tools/analysis/fin_drift_top64.py |  |
| file | mopd_domains/tmp/loop_overlap.py | tools/analysis/loop_overlap.py |  |
| file | mopd_domains/tmp/law_traj.py | tools/analysis/law_traj.py |  |
| file | mopd_domains/tmp/check_mopd_batches.py | tools/analysis/check_mopd_batches.py |  |
| file | mopd_domains/tmp/probe_analyze.py | tools/analysis/probe_analyze.py |  |
| file | mopd_domains/tmp/probe_style.py | tools/analysis/probe_style.py |  |
| file | mopd_domains/tmp/aux_probe_stats.py | tools/analysis/aux_probe_stats.py |  |
| file | mopd_domains/tmp/rl_diag.py | tools/analysis/rl_diag.py |  |
| file | mopd_domains/tmp/tb_dump.py | tools/analysis/tb_dump.py |  |
| file | mopd_domains/tmp/med_rl_table.py | tools/analysis/med_rl_table.py |  |
| file | mopd_domains/tmp/canary.py | tools/analysis/canary.py |  |
| file | mopd_domains/tmp/mopd_table.py | tools/report/mopd_table.py |  |
| file | mopd_domains/tmp/opd_all_table.py | tools/report/opd_all_table.py |  |
| file | mopd_domains/tmp/dump_opd_results.py | tools/report/dump_opd_results.py |  |
| file | scratchpad/build_report3.py | tools/report/build_report3.py |  |
| file | scratchpad/gen_html3.py | tools/report/gen_html3.py |  |
| file | scratchpad/build_report2.py | tools/report/build_report2.py |  |
| file | scratchpad/gen_html2.py | tools/report/gen_html2.py |  |
| file | mopd_rl/tmp/v3_mismatch_cat.py | tools/rl_judge/v3_mismatch_cat.py |  |
| file | mopd_rl/tmp/compare_v2_v3.py | tools/rl_judge/compare_v2_v3.py |  |
| file | mopd_rl/tmp/v3_inrun_vs_label.py | tools/rl_judge/v3_inrun_vs_label.py |  |
| file | mopd_rl/tmp/if_tb.py | tools/rl_judge/if_tb.py |  |
| file | mopd_rl/tmp/v3_labelcheck.py | tools/rl_judge/v3_labelcheck.py |  |
| file | mopd_rl/tmp/v3_diversity.py | tools/rl_judge/v3_diversity.py |  |
| file | mopd_rl/tmp/harvest_passrates.py | tools/rl_judge/harvest_passrates.py |  |
| file | mopd_rl/tmp/v4_contam.py | tools/rl_judge/v4_contam.py |  |
| file | mopd_rl/tmp/v3_30b.py | tools/rl_judge/v3_30b.py |  |
| file | mopd_rl/tmp/if_metrics.py | tools/rl_judge/if_metrics.py |  |
| file | mopd_rl/tmp/v4_tiers_contam.py | tools/rl_judge/v4_tiers_contam.py |  |
| file | overlay/tools/disk/clean_caches.sh | tools/disk/clean_caches.sh |  |
| file | scratchpad/poll_opd_many_cpu.sh | tools/watch/poll_opd_many_cpu.sh | operator watcher (laptop-side) |
| file | scratchpad/watch_dom.sh | tools/watch/watch_dom.sh | operator watcher (laptop-side) |
| file | scratchpad/watch_full.sh | tools/watch/watch_full.sh | operator watcher (laptop-side) |
| file | scratchpad/wait_opd_done.sh | tools/watch/wait_opd_done.sh | operator watcher (laptop-side) |
| file | scratchpad/watch_opd_resub.sh | tools/watch/watch_opd_resub.sh | operator watcher (laptop-side) |
| file | scratchpad/mopd_finals.sh | tools/watch/mopd_finals.sh | operator watcher (laptop-side) |

## Symlinks created by setup_cluster.sh (mopd_sean path -> original)
| link | target |
|---|---|
| models/Qwen3-4B-OT3 | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-4B-OT3 |
| models/Qwen3-4B | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-4B |
| models/Qwen3-4B-Base | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-4B-Base |
| models/Qwen3-1.7B | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-1.7B |
| models/Qwen3-1.7B-Base | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/models/Qwen3-1.7B-Base |
| models/Qwen3-1.7B-OT3 | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/outputs/sft_ot3/qwen3-1p7b-18k |
| models/teacher-law | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/rl2-law-distill-ck200 |
| models/teacher-law-rlonly | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/rl-law-ck200 |
| models/teacher-fin | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/rl-fin-ck200 |
| models/teacher-if | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_rl/models/rl-if-ck200 |
| models/teacher-med | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/results/sft_distill_med_c6_4ep/checkpoint-3368 |
| models/merged-4teachers-uniform | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/merged-4teachers-uniform |
| models/merged-4teachers-w4411 | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/models/merged-4teachers-w4411 |
| data/eval | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/data/eval |
| data/train | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/data/train |
| data/domains | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/domains |
| data/rl_pools | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/rl_pools |
| data/distill_pools | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/distill_pools |
| data/sft_distill | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_distill |
| data/sft_distill_v2 | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_distill_v2 |
| data/sft_distill_v3 | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_distill_v3 |
| data/sft_med | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_med |
| data/sft_med_o1 | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_med_o1 |
| data/sft_warmup | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/data/sft_warmup |
| data/rl3dom | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_rl/data/rl3dom |
| data/nemotron_math | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_rl/data/nemotron_math |
| data/raw/OpenThoughts3-1.2M | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/OpenThoughts3-1.2M |
| data/raw/Nemotron-3-Nano-RL-Blend | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/Nemotron-3-Nano-RL-Blend |
| data/raw/Nemotron-RL-Math-v2 | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/Nemotron-RL-Math-v2 |
| data/raw/Nemotron-RL-Ultra-restored | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/Nemotron-RL-Ultra-restored |
| data/raw/Nemotron-RL-Ultra-Training-Blends | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/Nemotron-RL-Ultra-Training-Blends |
| data/raw/livecodebench | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/data/livecodebench |
| outputs/opd | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/outputs/opd |
| outputs/eval_domain | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/outputs/eval |
| outputs/eval_6bench | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/outputs/eval |
| outputs/teacher_gen | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/outputs/distill |
| outputs/sft/ot3_qwen3-4b | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/outputs/sft_ot3/qwen3-4b |
| outputs/sft/ot3_qwen3-1p7b-18k | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/outputs/sft_ot3/qwen3-1p7b-18k |
| logs/_orig_mopd | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd/logs |
| logs/_orig_mopd_domains | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_domains/logs |
| logs/_orig_mopd_rl | /mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_rl/logs |
| outputs/sft/<run> | $S/mopd_domains/results/<non-grpo run> |
| outputs/rl/<run> | $S/mopd_domains/results/grpo_* and $S/mopd_rl/results/grpo_* |
| models/rl_steps/<name> | $S/mopd_domains/models/rl* and $S/mopd_rl/models/rl-* (eval-ready RL step dirs) |

## Python module renames
| old | new |
|---|---|
| mopd_domains.graders.{extract,graders} | mopd.graders.domains.* |
| mopd_domains.eval.{benches,run_domain_eval} | mopd.eval.domains.* |
| mopd_domains.data.{prep_rl_pools,prep_med_sft}, mopd_domains.distill.build_pools | mopd.data.domains.* |
| mopd_domains.distill.{gen_teacher,reject_filter,select_traces} | mopd.teacher.* |
| opd.train_opd_domains | mopd.distill.train_opd |
| mopd_rl/nemo/mopd_rl_envs.*, mopd_domains/nemo/domain_envs.* | mopd.rl.nemo.envs.* |
| mopd_rl/nemo/run_grpo_mopdrl.py + mopd_domains/nemo/run_grpo_domains.py | mopd.rl.nemo.run_grpo |
| tmp/merge_teachers*.py | mopd.teacher.merge_teachers (--weights) |

## Script renames
| old | new |
|---|---|
| mopd/scripts/sft_1node.sbatch, sft_2node.sbatch | scripts/sft.sbatch (N = -N) |
| mopd/scripts/eval_1node.sbatch, eval_2node.sbatch | scripts/eval_6bench.sbatch |
| mopd_domains/scripts/domain_eval_1node.sbatch, domain_eval_2node.sbatch | scripts/eval_domain.sbatch (T=1.0 default, outputs/eval_domain) |
| mopd_domains/scripts/opd_2node.sbatch, opd_multinode.sbatch, mopd/scripts/mopd_2node.sbatch | scripts/opd.sbatch |
| mopd_domains/scripts/gen_teacher.sbatch | scripts/gen_teacher.sbatch (outputs/teacher_gen) |
| submit_domain_rl*.sh, mopd_rl/scripts/submit_*_rl.sh | scripts/rl_nemo.sh (CONFIG, NODES, JUDGE, SMOKE) |
| mopd_rl/scripts/ray_mopdrl.sub | scripts/nemo/ray.sub |
| opd_auto_eval.sh, opd_auto_eval_mopd.sh, opd_auto_eval_if.sh | scripts/auto_eval.sh <run> <dom3|bench:X|if> |
| opd_final_evals.sh, opd_final_evals_if.sh | scripts/finals.sh |
| sft_warmup_auto_eval.sh, sft_epoch_auto_eval.sh | scripts/sft_auto_eval.sh |
| opd_from_ckpt.sh | scripts/opd_from_ckpt.sh <base_config> <ckpt> <run> [nodes] |
| sft_warmup_stage.sh | scripts/sft_warmup_stage.sh <gen_tag> <pool> <name> <n> <config> |
| rl_auto_eval.sh, mk_eval_model.sh | scripts/rl_auto_eval.sh, scripts/mk_eval_model.sh |
| mopd_rl/scripts/convert_ck200.sbatch | scripts/convert_nemo_step.sbatch (STEP/OUT/REF) |
| mopd_rl/scripts/judge_server.sbatch | scripts/judge_server.sbatch |
| (new) | scripts/strip_optimizer.sh, tools/disk/clean_caches.sh |
| dropped (one-off): bs_sweep_*, babysit_jobs, probe_base, distill_if_*, grpo_2node (trl), eval_big_*, law_merge_stage3_4, run_stage3_4, submit_sft_eval, submit_smoke_32k, sweep_nano, tmp/nemo_*grep*.sh, setup_*round.sh | originals untouched under $S/mopd*, past/ |

## Config renames (old file -> new file; contents rewritten to repo-relative paths, NeMo configs to $S/mopd_sean absolute paths)
| old | new |
|---|---|
| mopd/configs/accelerate_zero2.yaml | configs/accelerate_zero2.yaml |
| mopd/configs/eval.yaml | configs/eval.yaml |
| mopd/configs/sft_ot3_4b.yaml | configs/sft/ot3_4b.yaml |
| mopd/configs/sft_ot3_1p7b_18k.yaml | configs/sft/ot3_1p7b_18k.yaml |
| mopd/configs/sft_ot3_1p7b.yaml | configs/sft/variants/ot3_1p7b_16k.yaml |
| mopd/configs/sft.yaml | configs/legacy_trl/sft.yaml |
| mopd/configs/mopd.yaml | configs/legacy_trl/mopd.yaml |
| mopd/configs/mopd_deepmath_4b.yaml | configs/legacy_trl/mopd_deepmath_4b.yaml |
| mopd/configs/rl_code.yaml | configs/legacy_trl/rl_code.yaml |
| mopd/configs/rl_deepmath_4b.yaml | configs/legacy_trl/rl_deepmath_4b.yaml |
| mopd/configs/rl_if.yaml | configs/legacy_trl/rl_if.yaml |
| mopd/configs/rl_math.yaml | configs/legacy_trl/rl_math.yaml |
| mopd_domains/configs/sft_distill_law_2ep.yaml | configs/sft/distill_law_2ep.yaml |
| mopd_domains/configs/sft_distill_med_c6_4ep.yaml | configs/sft/distill_med_c6_4ep.yaml |
| mopd_domains/configs/sft_distill_med_c1.yaml | configs/sft/variants/distill_med_c1.yaml |
| mopd_domains/configs/sft_distill_med_c7_3ep.yaml | configs/sft/variants/distill_med_c7_3ep.yaml |
| mopd_domains/configs/sft_distill_med_c6_3ep.yaml | configs/sft/variants/distill_med_c6_3ep.yaml |
| mopd_domains/configs/sft_distill_med_2ep.yaml | configs/sft/variants/distill_med_2ep.yaml |
| mopd_domains/configs/sft_distill_med_v2a.yaml | configs/sft/variants/distill_med_v2a.yaml |
| mopd_domains/configs/sft_distill_med_c1_lr5.yaml | configs/sft/variants/distill_med_c1_lr5.yaml |
| mopd_domains/configs/sft_distill_med_c1_3ep.yaml | configs/sft/variants/distill_med_c1_3ep.yaml |
| mopd_domains/configs/sft_distill_med_c6.yaml | configs/sft/variants/distill_med_c6.yaml |
| mopd_domains/configs/sft_distill_med_c4.yaml | configs/sft/variants/distill_med_c4.yaml |
| mopd_domains/configs/sft_distill_med_1ep.yaml | configs/sft/variants/distill_med_1ep.yaml |
| mopd_domains/configs/sft_distill_med_c4_3ep.yaml | configs/sft/variants/distill_med_c4_3ep.yaml |
| mopd_domains/configs/sft_distill_med_v2b.yaml | configs/sft/variants/distill_med_v2b.yaml |
| mopd_domains/configs/sft_distill_med_c5.yaml | configs/sft/variants/distill_med_c5.yaml |
| mopd_domains/configs/sft_distill_law_1ep.yaml | configs/sft/variants/distill_law_1ep.yaml |
| mopd_domains/configs/sft_distill_med_c2.yaml | configs/sft/variants/distill_med_c2.yaml |
| mopd_domains/configs/sft_distill_med_c3.yaml | configs/sft/variants/distill_med_c3.yaml |
| mopd_domains/configs/sft_distill_med_c3_3ep.yaml | configs/sft/variants/distill_med_c3_3ep.yaml |
| mopd_domains/configs/sft_med_4b.yaml | configs/sft/variants/med_4b.yaml |
| mopd_domains/configs/sft_medo1_en.yaml | configs/sft/variants/medo1_en.yaml |
| mopd_domains/configs/sft_medo1_endist.yaml | configs/sft/variants/medo1_endist.yaml |
| mopd_domains/configs/sft_medo1_enmix.yaml | configs/sft/variants/medo1_enmix.yaml |
| mopd_domains/configs/sft_warmup_law.yaml | configs/sft/warmup_law.yaml |
| mopd_domains/configs/sft_warmup_med_sftp.yaml | configs/sft/warmup_med.yaml |
| mopd_domains/configs/sft_warmup_med.yaml | configs/sft/variants/warmup_med_oldpool.yaml |
| mopd_domains/configs/sft_warmup_med_35b.yaml | configs/sft/variants/warmup_med_35b.yaml |
| mopd_domains/configs/sft_warmup_med_c6k1_4ep.yaml | configs/sft/warmup_med_c6k1_4ep.yaml |
| mopd_domains/configs/sft_warmup_med_c6k3_4ep.yaml | configs/sft/warmup_med_c6k3_4ep.yaml |
| mopd_domains/configs/sft_warmup_mix_lawmed.yaml | configs/sft/warmup_mix_lawmed.yaml |
| mopd_domains/configs/opd_law_pg_v2.yaml | configs/opd/law_pg.yaml |
| mopd_domains/configs/opd_law_top64_v2.yaml | configs/opd/law_top64.yaml |
| mopd_domains/configs/opd_law_top16_v2.yaml | configs/opd/law_top16.yaml |
| mopd_domains/configs/opd_law_pg.yaml | configs/opd/variants/law_pg_v1.yaml |
| mopd_domains/configs/opd_law_top64.yaml | configs/opd/variants/law_top64_v1.yaml |
| mopd_domains/configs/opd_law_top16.yaml | configs/opd/variants/law_top16_v1.yaml |
| mopd_domains/configs/opd_law_pg-v2.yaml | configs/opd/variants/law_pg_v2_hyphen.yaml |
| mopd_domains/configs/opd_law_top64-v2.yaml | configs/opd/variants/law_top64_v2_hyphen.yaml |
| mopd_domains/configs/opd_law_top16-v2.yaml | configs/opd/variants/law_top16_v2_hyphen.yaml |
| mopd_domains/configs/opd_law_pg-rlteacher.yaml | configs/opd/law_pg_rlteacher.yaml |
| mopd_domains/configs/opd_law_pg-sftw75.yaml | configs/opd/law_pg_from_warmup75.yaml |
| mopd_domains/configs/opd_fin_pg.yaml | configs/opd/fin_pg.yaml |
| mopd_domains/configs/opd_fin_top64.yaml | configs/opd/fin_top64.yaml |
| mopd_domains/configs/opd_fin_top16.yaml | configs/opd/fin_top16.yaml |
| mopd_domains/configs/opd_if_pg.yaml | configs/opd/if_pg.yaml |
| mopd_domains/configs/opd_if_top64.yaml | configs/opd/if_top64.yaml |
| mopd_domains/configs/opd_if_top16.yaml | configs/opd/if_top16.yaml |
| mopd_domains/configs/opd_med_pg-sftp.yaml | configs/opd/med_pg.yaml |
| mopd_domains/configs/opd_med_pg.yaml | configs/opd/variants/med_pg_oldpool.yaml |
| mopd_domains/configs/opd_med_top64.yaml | configs/opd/variants/med_top64_oldpool.yaml |
| mopd_domains/configs/opd_med_top16.yaml | configs/opd/variants/med_top16_oldpool.yaml |
| mopd_domains/configs/opd_med_pg-sftp-sftw150.yaml | configs/opd/med_pg_from_warmup150.yaml |
| mopd_domains/configs/opd_med_pg-sftw150.yaml | configs/opd/variants/med_pg_from_warmup150_oldpool.yaml |
| mopd_domains/configs/med-pg-sftwA561.yaml | configs/opd/med_pg_from_warmupA561.yaml |
| mopd_domains/configs/med-pg-sftwB2164.yaml | configs/opd/med_pg_from_warmupB2164.yaml |
| mopd_domains/configs/mopd-4dom-128-sftp.yaml | configs/mopd/4dom_pg128.yaml |
| mopd_domains/configs/mopd-4dom-128-merged-sftp.yaml | configs/mopd/4dom_pg128_merged.yaml |
| mopd_domains/configs/mopd-4dom-128-merged-w4411.yaml | configs/mopd/4dom_pg128_merged_w4411.yaml |
| mopd_domains/configs/mopd-4dom-128-mixsft.yaml | configs/mopd/4dom_pg128_mixsft.yaml |
| mopd_domains/configs/mopd-3dom-rllaw-96.yaml | configs/mopd/3dom_rllaw_pg96.yaml |
| mopd_domains/configs/mopd-medlaw-64-sftp.yaml | configs/mopd/medlaw_pg64.yaml |
| mopd_domains/configs/mopd-finif-64.yaml | configs/mopd/finif_pg64.yaml |
| mopd_domains/configs/mopd_4dom_pg128.yaml | configs/mopd/variants/4dom_pg128_oldpool.yaml |
| mopd_domains/configs/mopd_4dom_pg32.yaml | configs/mopd/variants/4dom_pg32_oldpool.yaml |
| mopd_domains/configs/mopd-4dom-128-top64.yaml | configs/mopd/variants/4dom_top64_128_oldpool.yaml |
| mopd_domains/configs/mopd-4dom-32-top64.yaml | configs/mopd/variants/4dom_top64_32_oldpool.yaml |
| mopd_domains/configs/mopd-4dom-128-merged.yaml | configs/mopd/variants/4dom_pg128_merged_oldpool.yaml |
| mopd_domains/configs/mopd-medlaw-64.yaml | configs/mopd/variants/medlaw_pg64_oldpool.yaml |
| mopd_domains/nemo/configs/grpo_law_distill_4b.yaml | configs/rl/grpo_law_distill.yaml |
| mopd_domains/nemo/configs/grpo_law_4b.yaml | configs/rl/grpo_law_rlonly.yaml |
| mopd_domains/nemo/configs/grpo_fin_4b.yaml | configs/rl/grpo_fin.yaml |
| mopd_domains/nemo/configs/grpo_med_v6_4b.yaml | configs/rl/variants/grpo_med_v6.yaml |
| mopd_domains/nemo/configs/grpo_law_phase2_4b.yaml | configs/rl/variants/grpo_law_phase2.yaml |
| mopd_domains/nemo/configs/grpo_med_v5_4b.yaml | configs/rl/variants/grpo_med_v5.yaml |
| mopd_domains/nemo/configs/grpo_med_v4_4b.yaml | configs/rl/variants/grpo_med_v4.yaml |
| mopd_domains/nemo/configs/grpo_med_4b_32k.yaml | configs/rl/variants/grpo_med_32k.yaml |
| mopd_domains/nemo/configs/grpo_med_distill_4b.yaml | configs/rl/variants/grpo_med_distill.yaml |
| mopd_domains/nemo/configs/grpo_law_4b_32k.yaml | configs/rl/variants/grpo_law_32k.yaml |
| mopd_domains/nemo/configs/grpo_med_4b.yaml | configs/rl/variants/grpo_med.yaml |
| mopd_domains/nemo/configs/grpo_med_v2_4b.yaml | configs/rl/variants/grpo_med_v2.yaml |
| mopd_domains/nemo/configs/grpo_fin_4b_32k.yaml | configs/rl/variants/grpo_fin_32k.yaml |
| mopd_domains/nemo/configs/grpo_med_v4a_4b.yaml | configs/rl/variants/grpo_med_v4a.yaml |
| mopd_domains/nemo/configs/grpo_med_v3_4b.yaml | configs/rl/variants/grpo_med_v3.yaml |
| mopd_rl/nemo/configs/grpo_if_4b.yaml | configs/rl/grpo_if.yaml |
| mopd_rl/nemo/configs/grpo_code_4b.yaml | configs/rl/grpo_code.yaml |
| mopd_rl/nemo/configs/grpo_math_v3_4b.yaml | configs/rl/grpo_math_v3.yaml |
| mopd_rl/nemo/configs/grpo_math_v2_4b.yaml | configs/rl/variants/grpo_math_v2.yaml |
| mopd_rl/nemo/configs/grpo_nemotron_3dom_4b.yaml | configs/rl/variants/grpo_nemotron_3dom.yaml |
| mopd_rl/nemo/configs/grpo_nemotron_math_4b.yaml | configs/rl/variants/grpo_nemotron_math.yaml |

## Output-path conventions
| kind | old | new |
|---|---|---|
| OPD/MOPD runs | mopd_domains/outputs/opd/<run> | outputs/opd/<run> |
| SFT runs (OT3, distill, warm-up) | mopd/outputs/sft_ot3/<run>, mopd_domains/results/<run> | outputs/sft/ot3_<run>, outputs/sft/<run> |
| NeMo-RL runs | mopd_domains/results/grpo_*, mopd_rl/results/grpo_* | outputs/rl/grpo_* |
| domain evals | mopd_domains/outputs/eval/<tag> | outputs/eval_domain/<tag> |
| 6-bench evals | mopd/outputs/eval/<tag> | outputs/eval_6bench/<tag> |
| teacher generations | mopd_domains/outputs/distill/<tag> | outputs/teacher_gen/<tag> |
| teachers | mopd_domains/models/rl2-law-distill-ck200 (law), rl-law-ck200 (law RL-only), rl-fin-ck200, mopd_rl/models/rl-if-ck200, results/sft_distill_med_c6_4ep/checkpoint-3368 | models/teacher-law, teacher-law-rlonly, teacher-fin, teacher-if, teacher-med |
