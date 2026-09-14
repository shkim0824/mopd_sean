# mopd_sean — one code base for the Qwen3-4B-OT3 OPD / MOPD study

Consolidation (2026-09-14) of the three working repos `mopd` (paper implementation: OT3 SFT, trl GRPO, MOPD, 6-bench eval),
`mopd_domains` (medical / law / finance / IF domain extension: pools, teacher SFT, domain OPD/MOPD, domain eval) and
`mopd_rl` (NeMo-RL v0.7.0 teacher training) into ONE package `mopd`, ONE launcher set `scripts/` and ONE config tree
`configs/{sft,rl,opd,mopd}`. The originals are untouched; big assets (models, data, outputs, logs) are reached through
symlinks created by `setup_cluster.sh` (see `MAPPING.md` for every file's origin and `docs/LAYOUT.md` for the tree).

Everything below runs from the repo root on `ib-a100-controller` with the shared runtime
(training container + `$S/.venv/mopd`, NeMo-RL container for RL teachers; `scripts/_common.sh`).
`S=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean`, `REPO=$S/mopd_sean`.

```
cd $S/mopd_sean
source scripts/_common.sh          # REPO, S, IMG, NEMO_IMG, VENV, prefer <n>, JP=k-sean
bash setup_cluster.sh              # (re)create the symlink farm + rsync offline assets; idempotent
sbatch -J k-sean-selfcheck $(prefer 1) scripts/selfcheck.sbatch   # imports + CPU tests + config-path check
```

## Layout (what lives where)

```
mopd/                 the package (PYTHONPATH = repo root)
  common/             chat template contract, YAML config + --override, io, resume policy
  data/               OT3 memmap cache, eval-set builders, train mixtures; data/domains/: RL pools, med SFT, distill pools
  sft/                SFT dataset / memmap cache / trainer (OT3 SFT, teacher SFT, warm-ups)      -> scripts/sft.sbatch
  teacher/            trace generation (vLLM), rejection filter, trace selection, weight merge    -> scripts/gen_teacher.sbatch
  rl/                 verifiable rewards + trl GRPO (legacy); rl/nemo/: NeMo-RL entrypoint + envs -> scripts/rl_nemo.sh
  distill/            OPD/MOPD: loss (PG / top-k), placement, teacher client, trainer, train_opd  -> scripts/opd.sbatch
  graders/            math / code / IF / OT3 / pyexec graders + sandbox; graders/domains/: med-law-fin
  eval/               6-bench runner (AIME24-26, LCB v6, IFEval, IFBench); eval/domains/: MedQA, CaseHOLD, FinQA, ...
scripts/              sbatch launchers + login-node helpers (all derive REPO from the submit dir; env-var interfaces)
configs/              sft/ rl/ opd/ mopd/ (+ variants/ = every ablation that was run; legacy_trl/ = paper-recipe configs)
third_party/          official grader / benchmark code and data (ifeval_google, ifbench, lcb_runner, legalbench, FinQA, ...)
tools/                one-off analysis, pool building, report generation, disk and watch helpers (not entrypoints)
tests/                CPU tests (loss, placement, teacher client, graders, cache, sandbox canary)
env/                  requirements, container verification, offline nltk / tiktoken, vendored deps for the NeMo container
models/ data/ outputs/ logs/   symlink farm -> the original repos' assets (setup_cluster.sh); tmp/cache = per-job compile caches
```

## The pipeline, stage by stage

Every job: `k-sean-*` names, `$(prefer N)` node rule (n041-066 when idle, never n001-040), keep ALL checkpoints
(`save_total_limit: null`), strip optimizer states only after a run is finished (`scripts/strip_optimizer.sh`), all
performance at T=1.0.

### 0. Data
| what | how |
|---|---|
| OT3 SFT cache (memmap, packed) | cpu-instance: `python -m mopd.data.prep_openthoughts3` (+ `build_ot3_verify.py` sidecar); see docs/OT3_SFT.md |
| 6-bench eval sets | `python -m mopd.data.prep_eval` -> data/eval/{aime24,aime25,aime26,lcb_v6,ifeval,ifbench} |
| domain RL pools (med/law/fin exact-match, IF) | `python -m mopd.data.domains.prep_rl_pools` -> data/rl_pools/*.jsonl (docs/DATASETS.md) |
| distillation prompt pools (big-teacher traces) | `python -m mopd.data.domains.build_pools` -> data/distill_pools/*.jsonl |
| NeMo-RL jsonl (math / 3-domain) | `python -m mopd.rl.nemo.prep_math_data`, `prep_3dom_data` |

### 1. General SFT student (Qwen3-*-Base -> *-OT3)
```
CONFIG=configs/sft/ot3_4b.yaml       sbatch -N 8 -J k-sean-sft-ot3-4b   --cpus-per-task=32 --mem=600G -t 60:00:00 $(prefer 8) scripts/sft.sbatch
CONFIG=configs/sft/ot3_1p7b_18k.yaml sbatch -N 2 -J k-sean-sft-ot3-1p7b --cpus-per-task=32 --mem=600G -t 60:00:00 $(prefer 2) scripts/sft.sbatch
```
Outputs: `outputs/sft/ot3_qwen3-4b` (= `models/Qwen3-4B-OT3`), `outputs/sft/ot3_qwen3-1p7b-18k` (= `models/Qwen3-1.7B-OT3`).
Resume = resubmit the same line (`train.resume: auto`).

### 2. Domain teachers
**(a) SFT distillation from a big teacher** (medical teacher = this path; law teacher = this path + RL):
```
# traces of the big teacher on the distillation pool (NeMo container vLLM handles MoE; resume-safe by TAG)
MODEL=$S/past/models/Qwen3.6-35B-A3B POOL=data/distill_pools/med_pool.jsonl TAG=gen-med-35b-k4 K=4 TP=2 \
  sbatch -N 4 -J k-sean-gen-med-35b $(prefer 4) scripts/gen_teacher.sbatch                  # -> outputs/teacher_gen/<TAG>
# verified traces -> SFT jsonl -> teacher SFT (reject_filter by default; SELECT=1 = select_traces, the C6 recipe)
bash scripts/teacher_sft_stage.sh gen-med-35b-k4 data/distill_pools/med_pool.jsonl data/sft_distill/med_sft.jsonl configs/sft/distill_med_c6_4ep.yaml 2 4
bash scripts/sft_auto_eval.sh distill_med_c6_4ep medqa epochs 4     # evaluate each epoch checkpoint at T=1.0
```
Optional: evaluate a candidate big teacher first: `MODEL=... TAG=... BENCHMARKS=medqa TP=2 sbatch scripts/eval_big.sbatch`.

**(b) RL (NeMo-RL v0.7.0 GRPO, exact-match / verifier environments)** — fin, IF, law teachers:
```
CONFIG=configs/rl/grpo_fin.yaml         NODES=4 bash scripts/rl_nemo.sh    # from Qwen3-4B-OT3 on the fin pool
CONFIG=configs/rl/grpo_law_distill.yaml NODES=4 bash scripts/rl_nemo.sh    # from the law distill-SFT checkpoint
CONFIG=configs/rl/grpo_law_rlonly.yaml  NODES=4 bash scripts/rl_nemo.sh    # ablation: RL directly from OT3
CONFIG=configs/rl/grpo_if.yaml          NODES=4 bash scripts/rl_nemo.sh    # IF (verifiable-instructions env)
CONFIG=configs/rl/grpo_math_v3.yaml     NODES=8 JUDGE=1 bash scripts/rl_nemo.sh   # math with an LLM judge (+1 node)
SMOKE=1 CONFIG=... bash scripts/rl_nemo.sh                                 # 1 node, 2 steps
bash scripts/rl_auto_eval.sh rl-fin outputs/rl/grpo_fin_4b models/Qwen3-4B-OT3 finqa 50 100 150 200   # eval steps at T=1.0
```
A NeMo step becomes an eval/OPD-ready HF dir with `scripts/mk_eval_model.sh` (symlinks + old-schema config; the
transformers-5 config loses rope fields) or, without a consolidated dir, `scripts/convert_nemo_step.sbatch`.
Final teachers are pinned as `models/teacher-{law,law-rlonly,fin,if,med}`.

### 3. Single-domain OPD (one teacher, one pool) and 3'. MOPD (several teachers, balanced batches)
Same launcher; the config decides (data.paths = 1 or N domains, `distill.mode` = pg | topk):
```
STUDENT_NODES=1 TEACHERS="med:models/teacher-med" WEIGHTS="med=1.0" CONFIG=configs/opd/med_pg.yaml \
  sbatch -N 2 -J k-sean-opd-med-pg --cpus-per-task=32 --mem=600G -t 48:00:00 $(prefer 2) scripts/opd.sbatch
STUDENT_NODES=2 TEACHERS="law:models/teacher-law,fin:models/teacher-fin,if:models/teacher-if,med:models/teacher-med" \
  WEIGHTS="law=1.0,fin=1.0,if=1.0,med=1.0" CONFIG=configs/mopd/4dom_pg128.yaml \
  sbatch -N 4 -J k-sean-opd-mopd-4dom-128 --cpus-per-task=32 --mem=600G -t 48:00:00 $(prefer 4) scripts/opd.sbatch
setsid nohup bash scripts/auto_eval.sh mopd-4dom-128 dom3 25 50 75 100 125 150 175 200 > logs/auto_eval_mopd-4dom-128.log 2>&1 &
bash scripts/finals.sh mopd-4dom-128        # when outputs/opd/<run>/opd_done.json exists: strip optimizer states + final 6-bench
```
Node roles: `nodes[0:STUDENT_NODES]` = student (accelerate, 8 ranks/node, colocated vLLM), the rest = teacher vLLM
prefill servers placed by `mopd.distill.placement` (balanced by WEIGHTS). Resume = resubmit the same line.

### 4. Evaluation (T = 1.0)
```
MODEL=outputs/opd/<run>/checkpoint-200 TAG=opd-<run>-ck200-dom3-T1 BENCHMARKS=medqa,casehold,finqa \
  sbatch -N 2 -J k-sean-domeval-<tag> $(prefer 2) scripts/eval_domain.sbatch             # -> outputs/eval_domain/<TAG>/metrics.json
MODEL=outputs/opd/<run>/final TAG=full-opd-<run>-final-32k \
  sbatch -N 2 -J k-sean-eval6-<tag> $(prefer 2) scripts/eval_6bench.sbatch               # -> outputs/eval_6bench/<TAG>/metrics.json
```
Domain benches: medqa, casehold, finqa (+ medxpertqa, pubmedqa, lexam, tatqa, docmath, ...; `mopd.eval.domains.benches`).
Report convention: one model per row, base + teacher rows, s̃ = mean over domains of (s_d − base)/(teacher − base).

## Variations (what to change)
| variation | knob |
|---|---|
| PG vs top-k distillation (Eq. 3-4 vs Eq. 5) | `distill.mode: pg` / `topk` + `TOPK=64` or `16` (configs/opd/*_top64.yaml, *_top16.yaml) |
| prompts per update / per-domain balance | `train.batch_size`, `data.per_domain_per_batch` (4dom_pg128 = 32 per domain) |
| truncation masking (loop recovery) | `train.mask_truncated_completions: true` (law v2 default) |
| student init: SFT warm-up / merged teachers / another run | `bash scripts/opd_from_ckpt.sh <base_cfg> <ckpt_dir> <run> [nodes]` (clones the config into configs/<kind>/generated/) |
| off-policy SFT warm-up from a domain teacher | `gen_teacher.sbatch` (MODEL=models/teacher-<d>) -> `sft_warmup_stage.sh` -> `sft_auto_eval.sh` -> `opd_from_ckpt.sh` |
| merged-teacher init | `python -m mopd.teacher.merge_teachers --teachers law=models/teacher-law,... --weights law=0.4,med=0.4,fin=0.1,if=0.1 --base models/Qwen3-4B-OT3 --out models/merged-4teachers-w4411` |
| mixed-domain warm-up | configs/sft/warmup_mix_lawmed.yaml (+ configs/mopd/4dom_pg128_mixsft.yaml) |
| teacher set / weights | `TEACHERS="dom:path[:tp],..."` `WEIGHTS="dom=w,..."` (weights drive server placement, not the loss) |
| RL-only vs SFT+RL teacher | configs/opd/law_pg_rlteacher.yaml, configs/mopd/3dom_rllaw_pg96.yaml |
| medical pool (Evol-free default vs old pool) | configs/opd/med_pg.yaml vs configs/opd/variants/med_pg_oldpool.yaml |
| RL teacher recipe | configs/rl/*.yaml (+ variants/: 32k rollouts, med v2-v6 pools, law phase 2, 3-domain) |
| paper recipe (trl GRPO experts + train_mopd with task rewards) | configs/legacy_trl/*.yaml + scripts/legacy/{grpo_trl,mopd_trl}.sbatch (legacy data required) |

## Conventions and caveats
- Config paths: SFT / OPD / MOPD configs are repo-relative (`models/...`, `data/...`, `outputs/...`); NeMo-RL configs
  carry absolute `$S/mopd_sean/...` paths (ray workers do not share the submit cwd). `python3 scripts/cfgval.py --check-paths` validates all.
- `outputs/`, `models/`, `data/`, `logs/_orig_*` are symlinks into the original repos: writes go THROUGH to the originals,
  deleting through a link deletes the original. New runs land in the original repos' outputs (the running jobs' homes).
- Big trace teachers / judges (Qwen3.6-35B-A3B, Qwen3-30B-A3B, gpt-oss-120b, ...) were archived to `$S/past/models`;
  the teacher-building configs point there. OPD / MOPD / eval never need them.
- `configs/legacy_trl/` and `scripts/legacy/` are the original math/code/IF paper recipe (trl); their data files are not
  built in this tree (documented in docs/SUBMIT_RECIPES_legacy.md).
- `tools/` are kept as run history (paths rewritten to this tree) — not maintained entrypoints.
- Never run `strip_optimizer.sh` on a run that may still resume. Never write into /home/deploy or /tmp.

## Verification
`scripts/selfcheck.sbatch` (imports of every module in both containers, the CPU tests, config paths) plus smoke jobs
launched from this tree (domain eval LIMIT=48, 6-bench LIMIT=8, 2-step OPD) all passed on 2026-09-14; details and the
two merge bugs they caught are in docs/LAYOUT.md "Verification". Re-run the self-check after moving or editing the tree.
