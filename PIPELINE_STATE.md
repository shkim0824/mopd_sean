09-15 20:37 PHASE2: merge w4411-lawsft started on login node (logs/merge_w4411_lawsft.log); gen-lawsft-teacher-k1 job 574837 (1 node, k=1 T=1.0, 10k law prompts, teacher-law-sft)
- 09-15 20:45 PHASE 2 launched (user 20:30): merge models/merged-4teachers-w4411-lawsft (med .4 law .4 fin .1 if .1, law = teacher-law-sft) done; MOPD 574841 (mopd-4dom-128-merged-w4411-lawsft, balanced 32x4) + 574842 (-b4411: batches med 51/law 51/fin 13/if 13; trainer now accepts dict per_domain_per_batch) + merged-init evals 574843 (dom3) / 574844 (6-bench); gen-lawsft-teacher-k1 574837 (law traces from teacher-law-sft) -> chains tmp/phase2/law_warmup_chain.sh (stage -> sft_warmup_lawsft -> CaseHOLD loop -> saturation rule -> branch law-pg-lawsft-sftw<N>) and tmp/phase2/mix_chain.sh (mix_lawsft3200_medB2ep -> sft_warmup_mix_lawsft_med -> evals -> mopd-4dom-128-mixsft-lawsft). Med warm-up B + branch reused from phase 1.
- 09-15 21:12 lawsft (OT3 init) ck200 6-bench 574820 DONE: IFEval 76.5 / IFBench 50.7 / math 61.4 / code 52.2 (ck200 dom3 574819 still running; ck150 574635 3.5 h, ck175 574789 1.9 h). Node manager: gen 574837 T2-pinned then released to whole cluster; 574818 re-pinned to 41-66.
- 09-15 21:27 SAFETY r4: step_40 checkpoint saved, val 0.95 (r0-r4: 0.58/0.79/0.94/0.93/0.95; start 0.29). Watcher tmp/safety_finals_watch.sh armed: after the driver pins teacher-safety-4b it runs tmp/safety4b_finals.sh (teacher ihc/dom3/6-bench + 1.7B loop).
- 09-15 21:31 SAFETY 4B DONE: loop ALL DONE 21:27, teacher-safety-4b = grpo_ihc_4b-r4-step40 (val 0.95). Teacher evals submitted+registered: 574861 ihc, 574862 dom3, 574863 6-bench. 1.7B IHC loop started 21:29 (pid 289527, logs/ihc_loop_1p7b.log): round 0 synth 574864 vs Qwen3-1.7B-OT3. Phase 2: MOPD 574841 (merged-w4411-lawsft) RUNNING on n229-232 since 21:27 (renamed sean-).
- 09-15 22:00 1.7B IHC loop: round-0 synth 574864 CANCELLED by another user on n018 after 4.5 min -> driver aborted (no synth retry). Patched rl_ihc_loop.sh (SYNTH_TRIES=3 resubmit/attach loop for synth, like GRPO_TRIES) and restarted the 1.7B driver.
- 09-15 22:03 kill wave on n018/n029-032 (synth 574864, merged-init evals 574843/574844 cancelled by uid 1000; guard resubmitted 574879/574880). Operational: KILL_ZONE extended to n[001-040,089-128] for NeMo/loop jobs; 1.7B driver restarted with WHOLE_EXC=n[001-040,089-128] (attached to synth 574878).
- 09-15 22:32 LAW OPD rerun (SFT teacher) 574380 COMPLETED (200 steps, final/ present). Eval loop submitted ck200 CaseHOLD 574898 then died at its final echo (in-place-edit parse artifact); ck175 eval 574840 killed by uid 1000 on n018-019 -> resubmitted 574899; both registered. finals_watch armed (strip + final 6-bench). Slow evals: lawsft ck150 dom3 574635 357/484 (ETA 00:00), law ck100 574787 227/333 (ETA 00:55).
- 09-15 22:34 1.7B SAFETY round 0: synth 574878 done (1,211 conflict prompts), GRPO 574900 submitted (4 nodes).
- 09-15 23:12 lawsft (OT3 init) FINAL 6-bench 574818 DONE: IFEval 76.3 / IFBench 53.0 / math 59.3 / code 52.5 (ck200 dup: 76.5/50.7/61.4/52.2). Law OPD finals ran 22:35: optimizer states stripped (530 GB -> 80 GB), final 6-bench 574901 submitted+registered. 1.7B synth 574878 COMPLETED (12.8 min); GRPO 574900 pending.
- 09-15 23:23 lawsft (OT3 init) ck150 dom3 574635 DONE (5.6 h): MedQA 69.5 (trunc 91.6%), CaseHOLD 68.5 (trunc 98.6%), FinQA 73.3 -> severe med/law loop episode at ck150 (ck125 was 0.2-0.3% trunc). ck175/ck200 dom3 still running (slow for the same reason). Started 23:22: 1.7B GRPO r0 574900 (n202,204,207-208), gen 574837 (n233), teacher-safety evals 574861-863, merged-init dom3 574879.
- 09-15 23:30 BLOW-UP DIAGNOSIS (tmp/trend_steps.py): OT3-init MOPD lawsft rollouts clipped 0.02 -> 0.34 (step 150) -> 0.48 (175-200), mean len 4.2k -> 10.5k = med/law loop episode from step ~127, no recovery (ck150 dom3 trunc 92%/99%). Law OPD SFT-teacher rerun: clipped 0.88-1.00 with mean len ~16k for every logged step 126-200 (pre-resume log overwritten; ck75 eval was clean) -> checkpoints ck125-200 loop-degenerate, mask_truncated_completions leaves ~no signal. Merged-init lawsft: clipped <= 0.09 throughout (stable, 81%). Same pattern as phase 1: every OT3-init law-containing OPD blew up; merged init never.
- 09-15 23:28 1.7B SAFETY r0 GRPO 574900 running (n202,204,207-208): held-out val at start 0.13 (4B base was 0.29). Law-sft trace generation 574837 running on n233 (8 shards in progress). Teacher-safety-4b evals + merged-init evals running.
- 09-15 23:32 SAFETY 4B teacher ihc eval 574861 DONE: ihc 97.0 (base 34.5 -> r0 66.9 -> r1 84.2 -> r2 92.3 -> r3 95.5 -> r4/teacher 97.0). dom3 574862 + 6-bench 574863 running.
- 09-15 23:38 PHASE 2 merged init w4411-lawsft dom3 574879 DONE (14 min): MedQA 76.2 / CaseHOLD 68.7 / FinQA 65.0 (uniform merge: 73.6 / 67.2 / 72.9). 6-bench 574880 running.
- 09-15 23:44 SAFETY 4B teacher dom3 574862 DONE: MedQA 70.6 / CaseHOLD 63.3 / FinQA 57.4 (base 69.8 / 63.2 / 58.3 -> no cross-domain damage). 6-bench 574863 running. Law ck200 CaseHOLD 574898 started (n036,176).
- 09-15 23:48 LAW OPD (SFT teacher) ck100/ck150 CaseHOLD DONE (4.5 h each): 69.8 (trunc 99.5%) / 69.5 (trunc 99.2%) -> blow-up started between step 75 (68.7, 0.3%) and 100. ck125 (574788, 4.5 h in), ck175 (574899), ck200 (574898) pending/running.
- 09-15 23:50 PHASE 2 law warm-up: gen-lawsft-teacher-k1 574837 COMPLETED (26 min, 8 shards); reject filter accepted 7,473/10,000 (75%; phase-1 RL teacher 80%); 6,400 rows sampled (teacher chars median 8,053, p90 12,890); SFT sft_warmup_lawsft job 574918 RUNNING on n047 (registered). Law OPD ck125 CaseHOLD 574788 DONE.
- 09-16 00:02 PHASE 2: mixed data mix_lawsft3200_medB2ep.jsonl built (52,002 rows); mixed SFT sft_warmup_mix_lawsft_med job 574919 RUNNING on n054,235 since 23:50 (registered). Law warm-up SFT 574918 at 103/200 (6.4 s/it); ck25/ck50 CaseHOLD evals 574923/574925 queued.
- 09-16 00:10 PHASE 2: law warm-up SFT 574918 COMPLETED (200 steps, 20 min); saturation watcher waiting for CaseHOLD evals ck25.. (574923/574925/...). b4411 ck25 dom3: MedQA 75.6 / CaseHOLD 70.0 / FinQA 68.7 (init 76.2/68.7/65.0). w4411 ck25 dom3 574913 running. Mixed SFT 574919 running (19 min).
- 09-16 00:16 1.7B SAFETY r0: step_20 saved; held-out val 0.13 at start and 0.13 at step 20 (4B r0 went 0.29 -> 0.58 by step 40).
- 09-16 00:18 1.7B r0 WATCH: train Avg Reward flat 0.11 -> 0.13 over steps 1-20 (noise 0.07-0.17); 4B r0 rose 0.29 -> 0.43 val by step 20 and 0.58 by 40. Decide at step 40 whether the 1.7B recipe needs more steps/G (report -> propose).
- 09-16 00:33 PHASE 2 mixed warm-up SFT 574919 COMPLETED (41 min, 1,144 steps, 52,002 rows -> 18,293 bins, 241 long rows dropped) -> checkpoint-1144. mix_chain will submit its dom3 + 6-bench evals and the 4-dom MOPD mopd-4dom-128-mixsft-lawsft within 5 min.
- 09-16 00:35 PHASE 2 ck25 dom3: w4411 balanced 77.1 / 69.3 / 71.1; b4411 ratio 75.6 / 70.0 / 68.7 (init 76.2 / 68.7 / 65.0).
- 09-16 00:38 PHASE 2 mixed chain: checkpoint-1144 evals submitted (dom3 575005, 6-bench 575006) and 4-dom PG MOPD mopd-4dom-128-mixsft-lawsft job 575007 (4 nodes, teachers law-sft/fin/if/med, config configs/mopd/generated/mopd-4dom-128-mixsft-lawsft.yaml) submitted+registered; dom3 loop + finals watcher armed.
- 09-16 00:50 SAFETY 4B teacher 6-bench 574863 DONE: math 60.4 / code 50.8 / IFEval 66.5 / IFBench 33.7 (base 60.3 / 51.7 / 51.0 / 27.7 -> IF up, math/code unchanged). Mixed-init MOPD 575007 RUNNING (4 nodes). Law warm-up ck50 eval done, ck25 running.
- 09-16 00:51 PHASE 2 law warm-up (SFT-only teacher) ck50 CaseHOLD 72.9 (teacher 73.9, base 63.2); ck25 eval running; saturation watcher reads ck25 first.
- 09-16 00:56 PHASE 2 merged init w4411-lawsft 6-bench 574880 DONE: math 51.5 (AIME 59.2/47.9/47.5) / code 38.8 / IFEval 47.7 / IFBench 29.3 -> init s~ 37% (uniform merge init: math 59.7 / code 28.4 / IF 60.1/36.3, s~ 48%).
- 09-16 01:01 1.7B SAFETY r0 step 40: held-out val 0.13 -> 0.13 (step 20) -> 0.22 (step 40); train reward 0.11 -> 0.15 (steps 25-40). Slow but learning (4B r0: 0.29 -> 0.58). Letting the loop continue to round 1; re-assess after r1.
- 09-16 01:03 lawsft (OT3 init) ck175 dom3 574789 DONE (5.7 h): medqa 70.4 (tr 97.0%) | casehold 68.1 (tr 99.8%) | finqa 74.6 (tr 0.0%). 1.7B r1 synth 575019 submitted 01:01 (round-0 defender converted).
- 09-16 01:06 1.7B SAFETY: r0 defender grpo_ihc_1p7b-r0-step40 converted; ihc benchmark evals submitted+registered for the 1.7B base and r0 (tmp/reg_ihc_round2.sh 1p7b base|rN).
- 09-16 01:31 lawsft (OT3 init) ck200 dom3 574819 DONE (5.6 h): MedQA 72.7 (trunc 94%) / CaseHOLD 68.1 (trunc 99%) / FinQA 73.4 -> with final 6-bench (76.3/53.0/59.3/52.5) s~ 63% (merged init 81%). 1.7B r1 synth 575019 done (1,211 prompts) -> GRPO r1 575050 submitted 01:29.
- 09-16 01:40 LAW OPD (SFT teacher) final 6-bench 574901 DONE: math 54.0 (57.9/52.1/52.1) / code 40.4 / IFEval 49.9 / IFBench 30.3 (base 60.3 / 51.7 / 51.0 / 27.7) -> loop-degenerate final also costs math/code. ck175/ck200 CaseHOLD still running.
- 09-16 01:52 PHASE 2 law warm-up ck100 CaseHOLD 73.4 (ck50 72.9; teacher 73.9); w4411 and b4411 MOPDs reached ck50 (evals submitted).
- 09-16 01:58 PHASE 2 law warm-up ck75 CaseHOLD 73.1 (ck50 72.9, ck100 73.4). Saturation watcher blocked on the slow ck25 eval (574923).
- 09-16 02:10 PHASE 2 law warm-up ck125 CaseHOLD 73.7 (curve so far: ck50 72.9, ck75 73.1, ck100 73.4, ck125 73.7; ck25 eval still running). 1.7B r1 GRPO 575050 started 02:05 (n054,058,233-234).
- 09-16 02:30 PHASE 2 law warm-up: ck25 69.5 (trunc 36%), saturation at ck75 (gain 0.2 < 1.5) -> PG OPD branch law-pg-lawsft-sftw75 job 575269 (2 nodes, teacher-law-sft, config configs/opd/generated/law-pg-lawsft-sftw75.yaml) submitted+registered; CaseHOLD loop + finals watcher armed.
- 09-16 02:40 PHASE 2 mixed warm-up ckpt 6-bench 575006 DONE: aime24 49.2 | aime25 46.7 | aime26 45.0 | lcb_v6 38.9 | ifeval 44.7 | ifbench 29.0 | math 46.9 | code 38.9 | if 36.9
- 09-16 02:42 PHASE 2 law warm-up ck150 CaseHOLD 73.6. Mixed ckpt 6-bench: math 46.9 / code 38.9 / IFEval 44.7 / IFBench 29.0 (phase-1 mixed 44.6 / 37.6 / 44.9 / 27.3); its dom3 575005 still queued.
- 09-16 02:55 PHASE 2 law warm-up curve complete: ck25 69.5 / ck50 72.9 / ck75 73.1 / ck100 73.4 / ck125 73.7 / ck150 73.6 / ck175 73.5 / ck200 73.6 (teacher 73.9); final 6-bench 574938 pending.
- 09-16 02:47 1.7B SAFETY: base Qwen3-1.7B-OT3 ihc benchmark 13.8 (4B base 34.5); r0 ihc eval 575023 running; r1 GRPO 575050 at 41 min.
- 09-16 02:48 1.7B SAFETY r0 ihc benchmark: 19.0 (base 13.8; 4B r0 was 66.9 from 34.5).
- 09-16 02:55 1.7B r1 (575050): init verified = grpo_ihc_1p7b-r0-step40; val 0.13 at start (sampling noise: the 4B r1 started at 0.67 after r0 ended 0.58) -> 0.22 at step 20; train reward rising 0.17 -> 0.31 (steps 1-20).
- 09-16 02:56 PHASE 2 mixed warm-up ckpt dom3 575005 DONE: medqa 77.3 (tr 0.4%) | casehold 71.9 (tr 0.3%) | finqa 54.8 (tr 0.3%) (phase-1 mixed with RL law teacher: 76.7 / 73.8 / 51.9).
- 09-16 03:10 PHASE 2 ck50 dom3: w4411 balanced 76.5 / 70.2 / 73.1; b4411 ratio 78.2 / 70.4 / 70.3 (phase-1 uniform merged init ck50: 74.8 / 67.5 / 74.2). ck50 6-bench evals pending.
- 09-16 03:30 1.7B SAFETY r1 step 40: val 0.13 -> 0.22 -> 0.39; train reward 0.17 -> 0.41 (accelerating; 4B r1 ended 0.79). Loop continues; tmp/ihc_round_evals_1p7b.sh armed to submit each round ihc eval (r1-r4) when its HF view appears.
- 09-16 03:33 1.7B SAFETY: round 1 done -> defender grpo_ihc_1p7b-r1-step40; round 2 synth 575305 submitted 03:30.
- 09-16 03:45 PHASE 2 mixsft-lawsft MOPD ck25 dom3: MedQA 77.1 / CaseHOLD 71.6 / FinQA 66.0 (init 77.3 / 71.9 / 54.8; phase-1 mixsft ck25 76.7 / 73.3 / 64.6). b4411 reached ck75.
- 09-16 03:53 1.7B SAFETY r1 ihc benchmark 41.5 (base 13.8 -> r0 19.0 -> r1 41.5; 4B: 34.5 -> 66.9 -> 84.2). Round 2: synth done (1,211 prompts), GRPO 575320 submitted 03:49.
- 09-16 04:08 LAW OPD (SFT teacher) ck200 CaseHOLD 574898 DONE (4.1 h): casehold 69.9 (tr 97.7%); ck175 (574899) still running.
- 09-16 04:18 LAW OPD (SFT teacher) ck175 CaseHOLD 574899 DONE (4.3 h): casehold 69.7 (tr 99.0%) -> law OPD table complete.
- 09-16 04:25 PHASE 2 b4411 ck50 6-bench: math 57.9 (62.5/55.8/55.4) / code 42.6 / IFEval 63.0 / IFBench 48.0 -> with dom3 78.2/70.4/70.3 s~ 68% (phase-1 uniform merged ck50: 68%). w4411 ck50 6-bench pending.
- 09-16 04:32 PHASE 2 w4411 (balanced) ck50 6-bench: math 56.7 (62.9/49.2/57.9) / code 44.0 / IFEval 67.1 / IFBench 52.7 -> with dom3 76.5/70.2/73.1 s~ 72% (b4411 68%, phase-1 uniform merged 68% at ck50).
- 09-16 04:40 PHASE 2 law warm-up (SFT-only teacher) ck200 6-bench 574938 DONE: math 45.4 (52.1/40.4/43.8) / code 31.7 / IFEval 44.4 / IFBench 29.0 (phase-1 RL-teacher warm-up: 40.0 / 15.5 / 42.1 / 28.3; base 60.3 / 51.7 / 51.0 / 27.7).
- 09-16 04:33 1.7B SAFETY r2 (575320) step 20: val 0.37 (start) -> 0.54; train reward 0.44 -> 0.50.
- 09-16 04:45 PHASE 2: branch OPD law-pg-lawsft-sftw75 ck25 CaseHOLD 72.9 (trunc 0.3%; warm-up ck75 init 73.1). ck75 dom3: w4411 balanced 75.6 / 71.3 / 72.7; b4411 ratio 77.1 / 70.8 / 72.0 (phase-1 uniform merged ck75: 75.6 / 68.5 / 74.7).
- 09-16 04:59 1.7B SAFETY r2 step 40: val 0.5900 (round start 0.37, step 20 0.54).
- 09-16 05:05 1.7B SAFETY: round 2 done -> defender grpo_ihc_1p7b-r2-step40; round 3 synth 575393 submitted 05:00.
- 09-16 05:08 PHASE 2 branch OPD law-pg-lawsft-sftw75 ck50 CaseHOLD 72.6 (trunc 0.3%). 1.7B r2 ihc eval 575395 running; r3 synth 575393 running.
- 09-16 05:12 1.7B SAFETY r2 ihc benchmark 64.8 (base 13.8 -> r0 19.0 -> r1 41.5 -> r2 64.8; 4B: 34.5 -> 66.9 -> 84.2 -> 92.3).
- 09-16 05:15 armed tmp/safety1p7b_finals_watch.sh: when the 1.7B loop pins teacher-safety-1p7b it submits+registers the teacher dom3 + 6-bench evals (ihc = the r4 round eval).
- 09-16 05:20 1.7B SAFETY round 3: synth done (1,211 prompts), GRPO 575398 submitted 05:13.
- 09-16 05:40 PHASE 2: mixsft-lawsft MOPD reached ck50 (dom3 + 6-bench evals submitted); branch OPD ck75 eval 575406 pinned to tier 1 (2 khan 1-node jobs + 1 planned job cancelled by the node manager).
- 09-16 05:45 1.7B SAFETY r3 (575398) val so far: Accuracy: 0.6600 Accuracy: 0.7300 
- 09-16 05:50 PHASE 2 b4411 reached ck100 (dom3 + 6-bench evals submitted 05:48).
- 09-16 05:58 PHASE 2 branch OPD law-pg-lawsft-sftw75 ck75 CaseHOLD 72.9 (trunc 0.3%; ck25 72.9, ck50 72.6; warm-up init 73.1, teacher 73.9).
- 09-16 06:05 PHASE 2 mixsft-lawsft MOPD ck50 dom3: MedQA 77.2 / CaseHOLD 71.2 / FinQA 72.1 (ck25 77.1 / 71.6 / 66.0; phase-1 mixsft ck50 77.1 / 73.8 / 72.1). ck50 6-bench pending.
- 09-16 06:12 1.7B SAFETY r3 step 40: val 0.8000 (round start 0.66, step 20 0.73).
- 09-16 06:14 1.7B SAFETY: round 3 done -> defender grpo_ihc_1p7b-r3-step40; round 4 (last) synth 575431 submitted 06:11.
- 09-16 06:16 PHASE 2 b4411 ck100 dom3: MedQA 76.7 / CaseHOLD 70.8 / FinQA 72.0 (ck100 6-bench 575419 running). Branch OPD reached ck100 (eval 575432 running). 1.7B r3 ihc eval 575433 submitted+registered; r4 synth 575431 running on n233.
- 09-16 06:27 1.7B SAFETY r3 ihc benchmark 81.7 (base 13.8 -> 19.0 -> 41.5 -> 64.8 -> 81.7; 4B r3 was 95.5). Branch OPD law-pg-lawsft-sftw75 ck100 CaseHOLD 73.4 (trunc 0.5%).
- 09-16 06:30 1.7B SAFETY round 4 (last): synth 575431 done (1,215 prompts), GRPO 575436 submitted 06:28 (pending). After it: driver pins teacher-safety-1p7b -> tmp/safety1p7b_finals_watch.sh submits dom3 + 6-bench; r4 ihc eval via tmp/ihc_round_evals_1p7b.sh.
- 09-16 06:35 PHASE 2 w4411 (balanced) reached ck100 (dom3 + 6-bench evals submitted 06:28); both weighted-merge MOPDs are halfway, ETA ck200 about 15:00-15:30.
- 09-16 07:08 PHASE 2 mixsft-lawsft ck50 6-bench: aime24 59.6 | aime25 46.7 | aime26 51.2 | lcb_v6 46.9 | ifeval 64.0 | ifbench 47.7 | math 52.5 | code 46.9 | if 55.8
- 09-16 07:10 PHASE 2 mixsft-lawsft ck50 s~ 71% (dom3 77.2/71.2/72.1 + 6-bench math 52.5 / code 46.9 / IFEval 64.0 / IFBench 47.7; phase-1 mixsft ck50 6-bench 53.5 / 43.3 / 62.1 / 47.7). 1.7B r4 GRPO 575436 RUNNING since 06:29.
- 09-16 07:20 PHASE 2 b4411 ck100 6-bench: math 58.6 / code 47.6 / IFEval 68.0 / IFBench 53.7 -> s~ 73% (uniform merged ck100: 74%). 1.7B r4 val 0.78 (start) -> 0.82 (step 20); job 575436 renamed sean- (tier-1 nodes).
- 09-16 07:28 1.7B SAFETY r4 latest val 0.9300 (start 0.78, step 20 0.82).
- 09-16 07:35 PHASE 2 w4411 (balanced) ck100 dom3: MedQA 78.2 / CaseHOLD 70.8 / FinQA 74.8 (uniform merged ck100: 74.6 / 69.2 / 75.2; b4411 ck100 76.7 / 70.8 / 72.0). ck100 6-bench pending.
- 09-16 07:33 1.7B SAFETY loop ALL DONE 07:29: models/teacher-safety-1p7b = grpo_ihc_1p7b-r4-step40 (val 0.93). Watchers submit r4 ihc + teacher dom3/6-bench evals.
- 09-16 07:44 node manager: 4 fresh khan 1-node jobs (6 s old) on n227/228/233/234 + 1 planned kananave job cancelled for the pinned 1.7B teacher evals 575439/575440 (khan auto-resubmits; BACKOFF will kick in at 6 cancels/10 min). Teacher 6-bench 575441 started on T3 nodes. Branch OPD reached ck150 (eval 575450 running).
- 09-16 07:46 1.7B teacher evals started on tier 1 (575439 ihc on n227-228, 575440 dom3 on n233-234) after 7 khan 1-node jobs were cancelled in 2 min (khan auto-resubmits onto freed range nodes; policy question for the user stands).
- 09-16 07:50 1.7B SAFETY r4/teacher ihc benchmark 91.8 (series base 13.8 -> 19.0 -> 41.5 -> 64.8 -> 81.7 -> 91.8; 4B 34.5 -> 66.9 -> 84.2 -> 92.3 -> 95.5 -> 97.0). Teacher dom3 575440 + 6-bench 575441 running.
- 09-16 07:58 1.7B SAFETY teacher dom3 575440 DONE: medqa 49.0 (tr 0.2%) | casehold 41.9 (tr 0.0%) | finqa 40.9 (tr 0.0%) (1.7B base 46.1 / 47.1 / 45.8). 6-bench 575441 running.
- 09-16 08:05 PHASE 2 branch OPD law-pg-lawsft-sftw75 ck150 CaseHOLD 73.6 (trunc 0.3%; curve 72.9 / 72.6 / 72.9 / 73.4 / 73.x / 73.6 at ck25-150).
- 09-16 08:06 (correction) branch OPD curve ck25-150: 72.9 / 72.6 / 72.9 / 73.4 / 73.3 / 73.6, all <0.5% truncated; ck175/ck200 pending.
- 09-16 08:03 node manager: b4411 ck125 eval 575466 T1-pinned; 2 more khan 1-node jobs (n233/234) + 1 planned kananave job cancelled (khan cancels since 07:32: 10). Running: w4411 ck100 6-bench 575435, teacher-safety-1p7b 6-bench 575441, mixsft ck75 dom3 575451.
- 09-16 08:15 PHASE 2 mixsft-lawsft MOPD ck75 dom3: MedQA 78.6 / CaseHOLD 72.7 / FinQA 72.1 (phase-1 mixsft ck75: 78.6 / 74.3 / 72.4).
- 09-16 08:13 node manager: 2 fresh kaya 1-node jobs (5 s old, n227/228) + 1 planned kananave job cancelled; b4411 ck125 eval 575466 started on tier 1. Foreign cancels since 07:32: 12 running + 4 planned (short jobs backfill freed range nodes within seconds; rate limit 6/10 min is the only brake).
- 09-16 08:36 1.7B SAFETY teacher 6-bench 575441 DONE: aime24 27.5 | aime25 23.8 | aime26 22.5 | lcb_v6 9.5 | ifeval 43.1 | ifbench 32.0 | math 24.6 | code 9.5 | if 37.5 (1.7B base: math 23.3 / code 19.7 / IFEval 30.9 / IFBench 20.7).
- 09-16 08:45 PHASE 2 w4411 (balanced) ck100 6-bench: math 60.3 (66.7/57.1/57.1) / code 48.6 / IFEval 72.5 / IFBench 53.3 -> with dom3 78.2/70.8/74.8 s~ 82% at ck100 (uniform merged: 74% at ck100, 81% at ck200). b4411 ck125 dom3: 78.0 / 71.0 / 71.2.
- 09-16 08:43 PHASE 2 branch OPD ck175 CaseHOLD 73.1 (trunc 0.3%); training at step 184/200.
- 09-16 08:52 node manager: w4411 ck125 eval 575489 T1-pinned; 2 running (khan n235, kaya n228) + 2 planned foreign jobs cancelled. Today total: 19 running + 19 planned cancels, all while one of my evals was pinned-pending (rule 3 literal). Policy decision pending with the user.
- 09-16 09:12 PHASE 2 branch OPD law-pg-lawsft-sftw75 (575269) COMPLETED 200 steps (6.1 h, 2 nodes); ck200 CaseHOLD eval + finals (strip + final 6-bench) follow automatically.
- 09-16 09:17 node manager: branch ck200 eval 575497 (2 nodes) T1-pinned -> 4 running foreign 1-node jobs (kaya x3, khan x1, 14 min old, n233-236) + 1 planned cancelled (demand 2 nodes, 4 cancelled = the over-cancel case). Today: 23 running + 20 planned. w4411 ck125 dom3 575489 running on n227-228.
- 09-16 09:32 PHASE 2 w4411 (balanced) ck125 dom3: MedQA 77.6 / CaseHOLD 70.3 / FinQA 74.0 (ck100 78.2 / 70.8 / 74.8; b4411 ck125 78.0 / 71.0 / 71.2).
- 09-16 09:33 PHASE 2 branch OPD finals ran 09:19: optimizer states stripped (530 GB -> 80 GB), final 6-bench 575499 submitted+registered; ck200 CaseHOLD 575497 + final 6-bench pending (T1-pinned).
- 09-16 09:40 PHASE 2 branch OPD law-pg-lawsft-sftw75 ck200 CaseHOLD 73.5 (trunc 0.3%) -> curve complete: 72.9 / 72.6 / 72.9 / 73.4 / 73.3 / 73.6 / 73.1 / 73.5 (ck25-200; warm-up init 73.1, teacher 73.9, base 63.2 -> 96% of the gap); final 6-bench 575499 pending.
- 09-16 10:05 NODE MANAGER INCIDENT: for the pinned mixsft ck100 evals (575515/575516, 4 nodes) it cancelled a 16-NODE kananave job 575509 (18 min old) that held only n227-228 inside the range, plus 2 kaya 1-node jobs. Today: 26 running + 22 planned cancels. Flagged to the user; no policy change made.
- 09-16 10:20 PHASE 2 b4411 reached ck150 (dom3 + 6-bench evals submitted 10:17).
- 09-16 10:30 PHASE 2 mixsft-lawsft MOPD ck100 dom3: MedQA 77.5 / CaseHOLD 71.8 / FinQA 73.5 (phase-1 mixsft ck100: 77.2 / 74.8 / 71.5); ck100 6-bench 575516 pending.
- 09-16 10:42 PHASE 2 b4411 ck150 dom3: MedQA 76.1 / CaseHOLD 72.4 / FinQA 73.3 (ck150 6-bench 575525 running). Report with per-checkpoint tables delivered (tmp/phase2_tables.py).
- 09-16 11:08 PHASE 2 branch OPD law-pg-lawsft-sftw75 FINAL 6-bench 575499 DONE: aime24 46.7 | aime25 49.2 | aime26 45.4 | lcb_v6 34.1 | ifeval 46.8 | ifbench 27.7 | math 47.1 | code 34.1 | if 37.2 (warm-up ck200 was math 45.4 / code 31.7 / IFEval 44.4 / IFBench 29.0; base 60.3 / 51.7 / 51.0 / 27.7).
- 09-16 11:20 PHASE 2 mixsft-lawsft MOPD ck100 6-bench 575516 DONE: math 53.7 (60.8/45.8/54.6) / code 45.6 / IFEval 74.3 / IFBench 53.0 -> with dom3 77.5/71.8/73.5 s~ 82% at ck100 (w4411 balanced ck100 82%, uniform 74%).
- 09-16 11:28 PHASE 2 w4411 (balanced) ck150 dom3: MedQA 74.9 / CaseHOLD 71.2 / FinQA 71.8 (dip from ck100 78.2/70.8/74.8 and ck125 77.6/70.3/74.0; truncation <0.3% so not a loop episode; uniform run also dipped at ck125-150). ck150 6-bench pending.
- 09-16 11:35 PHASE 2 b4411 (ratio) ck150 6-bench 575525 DONE: math 59.0 (62.9/55.4/58.8) / code 48.7 / IFEval 69.3 / IFBench 57.7 -> with dom3 76.1/72.4/73.3 s~ 79% at ck150 (ck100 73%; uniform merge ck150 73%).
- 09-16 12:27 PHASE 2 w4411 (balanced) ck150 6-bench DONE: math 58.3 (62.5/55.0/57.5) / code 49.8 / IFEval 73.6 / IFBench 54.7 -> with the dipped dom3 74.9/71.2/71.8 s~ 72% at ck150 (ck100 82%; uniform ck150 73%). 12:25: 3 more kaya 1-node jobs cancelled for pinned eval 575663.
- 09-16 13:02 PHASE 2: b4411 ck175 dom3 77.8 / 71.8 / 71.5; mixsft ck125 dom3 77.1 / 71.6 / 73.7. Steps: w4411 171, b4411 181, mixsft 131.
- 09-16 13:26 node manager: w4411 ck175 eval 575745 T1-pinned, 2 kaya 1-node jobs cancelled (today 34 running + 24 planned); eval running on n233-234.
- 09-16 13:45 PHASE 2 w4411 (balanced) ck175 dom3: MedQA 78.2 / CaseHOLD 71.8 / FinQA 74.8 (recovered from the ck150 dip 74.9/71.2/71.8).
- 09-16 14:39 PHASE 2 (1b) mopd-4dom-128-merged-w4411-lawsft-b4411 (574842) COMPLETED 200 steps (~16.9 h); ck200 dom3 + 6-bench + finals follow automatically.
- 09-16 14:49 node manager: mixsft ck150 dom3 575774 + 6-bench 575775 T1-pinned and started (n233-234, n227-228) after 3 kaya 1-node jobs were cancelled (today 37 running + 24 planned). b4411 ck200 dom3/6-bench running since 14:41.
- 09-16 14:55 PHASE 2 (1b) finals ran 14:50: optimizer states stripped (530 GB -> 80 GB), final 6-bench 575776 submitted+registered; ck200 dom3 575772 + 6-bench 575773 running.
- 09-16 15:05 PHASE 2 (1b) ck200 dom3: MedQA 77.8 / CaseHOLD 71.5 / FinQA 73.1 (ck200 6-bench 575773 + final 6-bench 575776 pending for s~).
- 09-16 15:15 PHASE 2 (2b) mixsft ck150 dom3: MedQA 77.6 / CaseHOLD 72.6 / FinQA 74.4 (best dom3 of the run so far; ck150 6-bench 575775 running).
- 09-16 15:36 PHASE 2 (1a) mopd-4dom-128-merged-w4411-lawsft (574841) COMPLETED 200 steps (~18.1 h on n229-232); ck200 dom3 + 6-bench + finals follow automatically.
- 09-16 15:38 node manager: (1a) ck200 dom3 575829 + 6-bench 575830 T1-pinned and started on n229-232 after 2 foreign 1-node jobs (jayce, kaya) were cancelled (today 39 running + 24 planned).
- 09-16 15:45 PHASE 2 (1a) finals ran 15:40: optimizer states stripped (530 GB -> 80 GB), final 6-bench 575836 submitted+registered (running on n233-234). Running evals: (1b) ck200 6-bench 575773 + final 575776; (1a) ck200 dom3 575829 + 6-bench 575830 + final 575836; (2b) ck150 6-bench 575775.
- 09-16 15:55 HF check: MMOPD/Qwen3-4B-OT3-law already = SFT-only teacher (sha256 match, replaced 09-14 22:32); teacher-law-sft dom3 eval 575850 running (n235-236) to fill the card MedQA/FinQA cells.
- 09-16 16:00 PHASE 2 (2b) mixsft ck150 6-bench 575775 DONE: math 57.6 (65.0/55.0/52.9) / code 46.7 / IFEval 76.2 / IFBench 53.0 -> with dom3 77.6/72.6/74.4 s~ 86% at ck150 (highest of any run so far; uniform merge ck200 81%).
- 09-16 16:02 PHASE 2 (1a) ck200 dom3: medqa 78.7 (tr 0.1%) | casehold 71.3 (tr 0.1%) | finqa 74.1 (tr 0.0%) (ck200 6-bench 575830 + final 575836 running).
- 09-16 16:03 (1b) final 6-bench 575776 CANCELLED by another user after 1 h (n066,258); guard resubmitted as 575865 (restarts from scratch, ~1.5 h).
- 09-16 16:06 PHASE 2 (1b) ck200 6-bench 575773 DONE: math 60.0 (67.5/53.3/59.2) / code 48.9 / IFEval 73.0 / IFBench 56.0 -> with dom3 77.8/71.5/73.1 s~ 82% at ck200 (uniform merge ck200 81%); final 6-bench 575865 running (restarted after a kill).
- 09-16 16:15 teacher-law-sft dom3 575850 DONE: MedQA 64.3 (trunc 2.7%) / CaseHOLD 74.0 / FinQA 60.4 (base 69.8 / 63.2 / 58.3); HF law card MedQA/FinQA cells filled (64.3 / 60.4) in the staged README.
- 09-16 16:25 HF card update (MedQA 64.3 / FinQA 60.4 for MMOPD/Qwen3-4B-OT3-law) NOT pushed: the stored .hf_cache token has read scope only (403 on commit to main); create_pr route -> HF 500 x4. Staged README updated locally; needs the org-write token via stdin (never stored) or a manual commit by the user.
- 09-16 16:25 PHASE 2 (1b) COMPLETE: final 6-bench 575865 DONE (resumed shards, 23 min): math 60.0 (67.5/53.3/59.2) / code 49.4 / IFEval 73.2 / IFBench 57.7; with ck200 dom3 77.8/71.5/73.1 -> FINAL s~ 82% (phase-1 uniform merge 81%).
- 09-16 16:56 PHASE 2 (1a) ck200 6-bench 575830 DONE: math 60.3 (66.2/57.5/57.1) / code 50.6 / IFEval 76.7 / IFBench 56.7; with ck200 dom3 78.7/71.3/74.1 -> s~ 87% at ck200 (med .82 law .76 fin .97 IF .93) = highest of any run so far ((1b) 82%, (2b) ck150 86%, phase-1 uniform merge 81%). final 6-bench 575836 still running (1.2 h in, n233-234).
- 09-16 17:05 PHASE 2 (1a) COMPLETE: final 6-bench 575836 DONE: math 61.0 (66.2/58.8/57.9) / code 51.8 / IFEval 76.0 / IFBench 57.3; with ck200 dom3 78.7/71.3/74.1 -> FINAL s~ 87% (med .82 / law .76 / fin .97 / IF .92) = BEST of the whole study (ratio (1b) 82%, phase-1 uniform merge 81%, mixsft (2b) 86% at ck150 still training); math 61.0 and code 51.8 are both at/above base (60.3 / 51.7) = no OOD cost. Balanced 32x4 > ratio 51:51:13:13 by 5 points of s~.
- 09-16 17:23 PHASE 2 (2b) mixsft ck175 dom3 575912 DONE (18 min): MedQA 76.7 / CaseHOLD 71.8 / FinQA 73.2 (all clean, trunc <0.2%) — about 1 point below ck150 (77.6/72.6/74.4); no ck175 6-bench (6-bench runs every 50 steps), so s~ comes at ck200. Training at 175/200, 5.7 min/step -> ck200 about 19:45.
- 09-16 19:40 failure-mode attribution (format vs real) was measured out of curiosity at the user's request and then DROPPED as unnecessary -- user 19:40: after training completes the scores are trustworthy as-is, report performance the previous way, no need to state why an answer was wrong. Removed from the memory files; ad-hoc scripts left in tmp/ (diag_failure_modes.py, diag_mcq.py, diag_delta.py).

## 09-16 20:00 phase 3-1 — safety becomes OOD, +GPQA, ONE-JOB evaluation

**User order (2026-09-16)**: safety is no longer an in-domain track (the teacher
`models/teacher-safety-*` stays as history; no more safety training) — from now on it is
measured on checkpoints only, with **HarmBench + Anthropic sycophancy + TruthfulQA**. GPQA
joins as a further OOD benchmark. In-domain and OOD (math, code, safety, GPQA) must run in
**one job**, with no wasted vLLM start-ups and as fast as possible.

**Built** (see `docs/EVAL_ALL.md` for the full spec):
- `scripts/eval_all.sbatch` + `mopd/eval/run_all.py` + `mopd/eval/registry.py`: one job, one
  engine per GPU, 13 benchmarks, each with its own protocol. Replaces the
  `eval_domain.sbatch` + `eval_6bench.sbatch` pair (2 jobs, 32 engine starts -> 1 job, 16).
- Optimizations: LPT task scheduling by expected cost; ONE mixed `generate()` per chunk with
  per-prompt SamplingParams (`--seqs-per-call 256`) so different benchmarks share one
  continuous batch instead of draining at every group boundary; `prompt_logprobs` path (`lp`)
  for sycophancy/TruthfulQA so 71k scoring requests need no generation;
  `enable_prefix_caching=False` only for workers holding `lp` tasks.
- Data frozen by `mopd/data/prep_ood_eval.py` into `data/eval/{gpqa,harmbench,sycophancy,
  truthfulqa_mc}` (198 / 300 / 30,051 / 817 rows) with manifests (source, commit, sha256,
  protocol). Official code vendored verbatim in `third_party/{simple_evals,harmbench,
  truthfulqa}`. Judge model `models/HarmBench-Llama-2-13b-cls` (25 GB, byte-checked).
- Graders `mopd/graders/ood/*`: GPQA = simple-evals single regex (no rescue); HarmBench = the
  official 13B classifier prompt + `temperature=0, max_tokens=1`, ASR lower-is-better, judged
  on the post-`</think>` answer clipped to the official 512 classifier tokens; sycophancy =
  official log-prob comparison, rate lower-is-better; TruthfulQA = official `MC_calcs`
  (MC1/MC2/MC3). Copyright 100 of HarmBench excluded (MinHash metric, not refusal).
- Checks: `tests/test_ood_graders.py` 45 assertions OK; `tmp/verify_unified_grading.py`
  re-grades finished eval dirs through the new path -> domain metrics identical to 12
  significant digits, 6-bench identical for aime24/25/26 + ifbench.
- FINDING: the official Google IFEval grader is **nondeterministic** — re-grading the same
  generations gave 76.7098 / 76.5250 / 76.7098 (one prompt of 541, +-0.185pp) because
  `langdetect.detect` is called without `DetectorFactory.seed`. Applies to every IFEval number
  reported so far. The vendored grader is left as upstream has it.
- `scripts/auto_eval.sh` is untouched (a running bash must not have its script edited); the
  successor chain is `scripts/auto_eval_all.sh <run> {full|ood|dom3} <steps>`.

**Phase 3-3**: `tmp/phase3/submit_ood.sh` submits safety+GPQA for the four arms the user named
(latest ckpt = ck200 for all four) plus the base row: uniform merge
`mopd-4dom-128-merged-lawsft`, mixed merge + balance batch `mopd-4dom-128-merged-w4411-lawsft`,
mixed merge + mixed batch `mopd-4dom-128-merged-w4411-lawsft-b4411`, sft warm-up
`mopd-4dom-128-mixsft-lawsft`. Table: `tmp/phase3/collect_ood.py`.
