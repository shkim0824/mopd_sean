# submit/ — copy-paste job recipes (ALL need explicit approval before `sbatch`; node pool 41-66; `-J sean-...`)

```bash
cd ~/lmalign/personal/sean/mopd            # realpath is resolved inside the sbatch
EXCL=--exclude=ib-a100-cluster-a-n[001-040,067-258]
COMMON="--cpus-per-task=32 --mem=600G ${EXCL}"

# 0. one-time venv (+ offline nltk assets), then CPU tests inside the container
sbatch -J sean-mopd-env-setup ${COMMON} env/setup_mopd_env.sbatch

# 1. base-model probe (chat-think / chat-nothink / raw on 60 rows of aime24, ifeval, lcb_v6)
MODEL=models/Qwen3-8B-Base TAG=base-8b sbatch -J sean-mopd-probe-base-8b ${COMMON} -t 06:00:00 scripts/probe_base.sbatch
MODEL=models/Qwen3-4B-Base TAG=base-4b sbatch -J sean-mopd-probe-base-4b ${COMMON} -t 06:00:00 scripts/probe_base.sbatch

# 2. IF SFT distillation (needs a teacher; paper: gpt-oss-120b — download on the cpu-instance first)
TEACHER=models/gpt-oss-120b TP=2 K=4 sbatch -J sean-mopd-distill-if-sft-gptoss120b ${COMMON} -t 24:00:00 scripts/distill_if_1node.sbatch

# 3. Stage 1 SFT (1 node)
CONFIG=configs/sft.yaml OVERRIDES="model.base=models/Qwen3-8B-Base train.output=outputs/sft/qwen3-8b" \
  sbatch -J sean-mopd-sft-qwen3-8b-mot-math-code-if-ep2 ${COMMON} -t 48:00:00 scripts/sft.sbatch
CONFIG=configs/sft.yaml OVERRIDES="model.base=models/Qwen3-4B-Base train.output=outputs/sft/qwen3-4b train.per_device_bs=2 train.grad_accum=8" \
  sbatch -J sean-mopd-sft-qwen3-4b-mot-math-code-if-ep2 ${COMMON} -t 48:00:00 scripts/sft.sbatch

# 4. eval of the SFT student (paper protocol: T=1, AIME avg@32, IF x1, LCB avg@4)
MODEL=outputs/sft/qwen3-8b TAG=sft-8b sbatch -J sean-mopd-eval-sft-qwen3-8b ${COMMON} -t 24:00:00 scripts/eval_1node.sbatch

# 5. Stage 2 RL experts (2 nodes each; run the three in parallel if 6 nodes are free)
for d in math code if; do
  CONFIG=configs/rl_${d}.yaml OVERRIDES="model.init=outputs/sft/qwen3-8b rl.output=outputs/rl/qwen3-8b-${d}" \
    sbatch -J sean-mopd-rl-${d}-qwen3-8b-lr3e-6-bs144x8 ${COMMON} -t 60:00:00 scripts/grpo_2node.sbatch
done
# smoke first:  OVERRIDES="... rl.max_steps=2 rl.prompts_per_step=16 rl.max_completion_length=2048 data.max_rows=64"

# 6. Stage 3 MOPD (2 nodes; node1 hosts the 3 teachers as 3/3/2 vLLM replicas)
TEACHERS="math:outputs/rl/qwen3-8b-math/final,code:outputs/rl/qwen3-8b-code/final,if:outputs/rl/qwen3-8b-if/final" \
WEIGHTS="math=0.35,code=0.30,if=0.35" CONFIG=configs/mopd.yaml \
OVERRIDES="model.student=outputs/sft/qwen3-8b train.output=outputs/mopd/qwen3-8b-pg distill.mode=pg" \
  sbatch -J sean-mopd-distill-qwen3-8b-pg-bs2048 ${COMMON} -t 48:00:00 scripts/opd.sbatch
# top-k variant: OVERRIDES="... distill.mode=topk train.output=outputs/mopd/qwen3-8b-topk" TOPK=64
# smoke: OVERRIDES="... train.max_steps=2 train.batch_size=64 train.max_completion_length=2048 data.max_rows_per_domain=64"

# 7. final eval of everything (base / SFT / 3 experts / MOPD)
for m in sft/qwen3-8b rl/qwen3-8b-math/final rl/qwen3-8b-code/final rl/qwen3-8b-if/final mopd/qwen3-8b-pg/final; do
  MODEL=outputs/${m} TAG=$(echo ${m} | tr / -) sbatch -J sean-mopd-eval-$(echo ${m} | tr / -) ${COMMON} -t 24:00:00 scripts/eval_1node.sbatch
done
```
