# OpenThoughts3-1.2M general SFT (2026-09-01 redo)

The original MoT (Mixture-of-Thoughts) general SFT underperformed; the SFT stage is being redone
on **open-thoughts/OpenThoughts3-1.2M** (arXiv 2506.04178) for **Qwen3-4B-Base** and
**Qwen3-1.7B-Base**. The MoT run's final model is preserved at `models/Qwen3-4B-MoT`
(byte-verified copy of the old `outputs/sft/qwen3-4b` root); all RL outputs, sweep outputs,
SFT intermediate checkpoints and the old `data/train/*` were deleted on 2026-09-01.

## Dataset facts (verified on the parquet + HF card/discussions)
- 1,200,000 rows / 120 parquet shards / 28.2 GB; columns `difficulty:int64, source:str,
  domain:{math,code,science}, conversations:[{from:human|gpt, value}]`. ~850k math / 250k code /
  100k science, built from 75k unique questions annotated **16x** by QwQ-32B.
- The gpt turn natively contains `<think>\n...\n</think>\n\n<answer>` (R1 style) — never re-wrap.
- **62% of rows have NO closing `</think>`** (QwQ generation capped at 16K tokens). Official +
  validated: the team trained on them as-is at 16K context and filtering them *hurt* (HF dataset
  discussion #3 + paper appendix). Smoke prep on 2 shards reproduced: 12,454/20,000 missing.
- No per-row ground-truth answers or test cases (grading needs upstream sources; see graders doc).

## Recipe sources
| | official OT3-7B | official OT3-1.5B | ours 4B | ours 1.7B |
|---|---|---|---|---|
| base | Qwen2.5-7B-Instruct | Qwen2.5-1.5B-Instruct | Qwen3-4B-Base | Qwen3-1.7B-Base |
| lr | 8e-5 | 1.6e-4 | **1.0e-4** | **1.5e-4** |
| epochs | 5 | 7 | 1 (user decision) | 1 |
| global batch | 512 | 256 | 512 bins | 256 bins |
| cutoff | 16384 + packing | 16384 | 16384 + flatten packing | same |
| warmup/wd | 0.1 / 0.0 cosine | same | 0.05 / 0.0 cosine | same |

lr interpolation evidence: DeepSeek-R1-distill (Nature suppl. Table 6) 1.5B→1e-4, 7B→8e-5,
14B→7e-5, 32B→6e-5 — monotone in size; OpenThoughts' own 7B/1.5B pair doubles lr at 1.5B.
4B ≈ 1e-4 (bracket 8e-5–1.2e-4), 1.7B ≈ 1.5e-4 (bracket 1e-4–1.6e-4). Community low-lr
(1e-5–2e-5) long-CoT recipes are all SMALL-dataset recipes; the OT3 paper's own Table 9 scales
lr up with dataset size (>31.6k rows → 8e-5). 1 epoch of 1.2M rows is ~16 effective passes over
the 75k unique questions and ~1.9k optimizer steps at gbs 512 — comparable step count to the
paper's mid-scale runs; all published evidence says more epochs keep helping, so 1 epoch is a
compute decision, not the quality optimum.

## Deviations from official (and why)
- **Base models, Qwen3 chat template, no system prompt** (official: Qwen2.5-*-Instruct, qwen25
  template with the default "You are Qwen..." system prompt). Paper D.4: template format is
  "roughly equivalent"; mopd keeps its single chat contract (`mopd/common/chat.py`) so SFT/RL/eval
  prompts stay byte-identical. eos contract: saved ckpts get eos=`<|im_end|>`,
  `generation_config.eos_token_id=[151645,151643]` (== what OpenThinker3 ships).
- **Right-truncate completions WITHOUT eos** instead of the old drop-if-long policy
  (`mopd/data/prep_openthoughts3.py`): matches LLaMA-Factory `cutoff_len` semantics; dropping
  would lose the majority of rows (smoke: 71% of code-domain rows exceed 16384 with prompt),
  and appending eos after a cut would teach stopping mid-thought.
- **Packing = mopd "flatten"** (padding-free FFD bins + position_ids reset → FA2 varlen
  block-diagonal attention). Equivalent in effect to LLaMA-Factory `neat_packing: true`
  (contamination-free); loss-equivalent per the 2026-08-30 throughput study.
- **Loss on the LAST assistant turn only** (mopd convention; rows are single-turn Q→A —
  smoke prep: 0 multi-turn in 20k rows; prep counts `n_multi_turn` for the full run).
- 1 epoch (user), warmup 0.05 (official 0.1 assumes 5-epoch total; bracket 0.03–0.1).

## Pipeline
1. Download (cpu-instance, done via `mopd_stage/dl_openthoughts3.sh`): parquet snapshot →
   `data/OpenThoughts3-1.2M` (cluster tree), byte-verified vs HF `files_metadata`.
2. `mopd/data/prep_openthoughts3.py` (cpu-instance, `envs/mopd_cpu`): tokenize (48 procs) + BFD
   pack → **memmap cache** `mopd/data/train/ot3_cache_len16384/` (~60 GB int32 token stream +
   offsets + prompt-lens + stats.json). One cache serves BOTH models (Qwen3-4B/1.7B tokenizers
   identical — asserted by `--check-tokenizer`). Roundtrip-tested by `tests/test_ot3_cache.py`.
3. Training reads the cache via `MemmapSFTDataset` (`mopd/sft/ot3_cache.py`): page-cache shared
   across ranks (the old in-RAM path at 1.2M rows would OOM the 600G job at 8–16 ranks).
   `configs/sft_ot3_4b.yaml` / `configs/sft_ot3_1p7b.yaml`; same `scripts/sft.sbatch`.

## Submit lines (NOT submitted — user reviews first)
```bash
cd ~/lmalign/personal/sean/mopd
CONFIG=configs/sft_ot3_4b.yaml   sbatch -J k-sean-mopd-sftot3-qwen3-4b-2node   --cpus-per-task=32 --mem=600G -t 60:00:00 --exclude=ib-a100-cluster-a-n[001-040] scripts/sft.sbatch
CONFIG=configs/sft_ot3_1p7b.yaml sbatch -J k-sean-mopd-sftot3-qwen3-1p7b-2node --cpus-per-task=32 --mem=600G -t 60:00:00 --exclude=ib-a100-cluster-a-n[001-040] scripts/sft.sbatch
```
Estimated: ~15B tokens/epoch; 4B ≈ 70 h on 2 nodes (≈ 60k tok/s measured for the MoT run) →
expect kill/resume cycles; 1.7B ≈ 35 h. Resume = `train.resume=auto` (+ resubmit same line).

## Monitoring / resume (user requirement)
- `scripts/babysit_jobs.sh` (ported from mopd_poc) runs detached on the login node with
  `tmp/babysit.spec` lines `NAME<TAB>SUBMIT_CMD`: CANCELLED/TIMEOUT/PREEMPTED/NODE_FAIL →
  auto-resubmit (resume picks up the latest complete checkpoint); FAILED → logs
  `FAILED-NEEDS-CLAUDE` for a Claude session to diagnose/fix/resubmit; COMPLETED → done.
- Plus the standing 30-min in-session monitor and a per-job completion watcher on submission
  ([[job-completion-watchers]]).
- Checkpoints: every 100 steps, keep ALL (`save_total_limit: null`, mopd-job-rules). Disk:
  ~19 x 56 GB (4B) + ~38 x 24 GB (1.7B) ≈ 2 T total — flag before other big runs.
