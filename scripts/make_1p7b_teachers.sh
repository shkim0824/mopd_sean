#!/bin/bash
# 1.7B domain teachers (user 2026-09-14): same data / recipe as the 4B teachers, base = Qwen3-1.7B-OT3.
#   SFT-only : medical (C6 set, 4 ep), law (law_sft, 2 ep)
#   RL-only  : finance, IF, medical (med_train), law (law_train)   [+ safety via scripts/rl_ihc_loop.sh SIZE=1p7b]
# Creates the configs (configs/sft/*_1p7b.yaml, configs/rl/*_1p7b.yaml) and submits with the placement rule.
set -u
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${_here}/_common.sh"; cd "${REPO}"
S17="models/Qwen3-1.7B-OT3"; [[ -f "${S17}/config.json" ]] || { echo "ABORT: ${S17} missing"; exit 1; }
# --- SFT configs: base + output renamed, everything else identical to the 4B teacher recipe
python3 - <<'PY'
import re
for src, dst, out in [("configs/sft/distill_med_c6_4ep.yaml", "configs/sft/distill_med_c6_4ep_1p7b.yaml", "outputs/sft/sft_distill_med_c6_4ep_1p7b"),
                      ("configs/sft/distill_law_2ep.yaml", "configs/sft/distill_law_2ep_1p7b.yaml", "outputs/sft/sft_distill_law_2ep_1p7b")]:
    t = open(src).read().split("\n"); o = []
    for i, l in enumerate(t):
        if i == 0: l = f"# 1.7B teacher (user 2026-09-14): = {src} with base Qwen3-1.7B-OT3 (same data, epochs, lr, max_len)"
        if l.strip().startswith("base:"): l = "  base: models/Qwen3-1.7B-OT3"
        if l.strip().startswith("output:"): l = f"  output: {out}"
        o.append(l)
    open(dst, "w").write("\n".join(o)); print("wrote", dst)
# --- RL configs (NeMo-RL, absolute paths): model + checkpoint/log dirs renamed; data identical to the 4B run
R = "/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/mopd_sean"
for src, dst, run in [("configs/rl/grpo_fin.yaml", "configs/rl/grpo_fin_1p7b.yaml", "grpo_fin_1p7b"),
                      ("configs/rl/grpo_if.yaml", "configs/rl/grpo_if_1p7b.yaml", "grpo_if_1p7b"),
                      ("configs/rl/variants/grpo_med.yaml", "configs/rl/grpo_med_1p7b.yaml", "grpo_med_1p7b"),
                      ("configs/rl/grpo_law_rlonly.yaml", "configs/rl/grpo_law_rlonly_1p7b.yaml", "grpo_law_rlonly_1p7b")]:
    t = open(src).read()
    t = re.sub(r'model_name: "[^"]+"', f'model_name: "{R}/models/Qwen3-1.7B-OT3"', t, count=1)
    t = re.sub(r'checkpoint_dir: "[^"]+"', f'checkpoint_dir: "{R}/outputs/rl/{run}"', t, count=1)
    t = re.sub(r'log_dir: "[^"]+"', f'log_dir: "{R}/logs/nemo/{run}"', t, count=1)
    t = f"# 1.7B RL-only teacher (user 2026-09-14): = {src} with Qwen3-1.7B-OT3 as the policy init; data / recipe unchanged\n" + t
    open(dst, "w").write(t); print("wrote", dst)
PY
python3 scripts/cfgval.py --check-paths | head -3
# --- submit (SFT: 2 nodes each; RL: 4 nodes each, NeMo-RL)
for c in distill_med_c6_4ep_1p7b distill_law_2ep_1p7b; do
  placement 2; J=$(CONFIG=configs/sft/${c}.yaml sbatch -N 2 ${PLACE_OPT} -J "${JP}-sft-${c}" --cpus-per-task=32 --mem=600G -t 48:00:00 --parsable scripts/sft.sbatch)
  echo "SFT ${c}: job ${J} (${PLACE_TIER})"
done
for c in grpo_fin_1p7b grpo_if_1p7b grpo_med_1p7b grpo_law_rlonly_1p7b; do
  placement 4; JOB_PREFIX="${JP}" EXC="${PLACE_OPT#--exclude=}" CONFIG=configs/rl/${c}.yaml NODES=4 bash scripts/rl_nemo.sh | tail -1
  echo "RL ${c}: submitted (${PLACE_TIER})"
done
