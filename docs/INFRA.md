# Reusable infra recipes from `~/lmalign/personal/sean` (for the new `mopd/` project)

Real path of the repo root (what compute nodes mount): `/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean` (`~/lmalign` is a symlink into `/mnt/datafs`; HOME is `/home/deploy`, shared account).

> **Tree moved mid-read (2026-08-29 17:33 cluster time, not by this task — it was read-only).** When I started, `conductor_sean_tool/`, `data_agent/`, `conductor_sean_agent/` etc. were at the root; by the end, the root holds only `data .hf_cache localbin models mtm .nltk_data past .secrets .uv_cache .venv`, and everything conductor-related now lives under `past/code/<repo>` (22 repos incl. `conductor_sean_tool`, `conductor_sean_agent`, `conductor_sean_mn`, ...) and `past/data/{data_agent,data_open,data_open3,data_synth,...}`. `data/livecodebench/` is still at the root. All section-A paths below are as they were read (`conductor_sean_tool/...`); prefix with `past/code/` now. `data_agent/math/...` is now `past/data/data_agent/math/...`.

---

## A. Multi-node GRPO (conductor_sean_tool)

### A.1 sbatch header + config (`conductor_sean_tool/scripts/train_conductor.sbatch`)

```bash
#SBATCH -J conductor-grpo-mn
#SBATCH -N 1                       # overridden by `sbatch -N <NUM_NODES>` (submit.sh)
#SBATCH --ntasks-per-node=1        # we drive per-node placement with srun sub-steps
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8               # whole nodes (8 A100-80GB each)
#SBATCH --mem=0                    # all node memory
#SBATCH --output=logs/slurm-conductor-mn-%j.log
#SBATCH --error=logs/slurm-conductor-mn-%j.log

set -uo pipefail
NUM_NODES="${SLURM_NNODES:?must run under Slurm (sbatch -N N)}"
# Resolve symlinks: ~/lmalign -> /mnt/datafs/..., and the pyxis container only
# mounts /mnt/datafs. Using the /home/deploy symlink path breaks --container-workdir.
REPO="$(realpath "${CONDUCTOR_REPO:-$(pwd)}")"
CONFIG="${CONDUCTOR_CONFIG:-configs/default.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate.yaml}"
RUN_NAME="${RUN_NAME:-conductor_mn_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/outputs/${RUN_NAME}}"
BASE_PORT="${BASE_PORT:-8100}"       # worker HTTP ports = BASE_PORT + server index
ACCEL_PORT="${ACCEL_PORT:-29500}"    # accelerate main_process_port
WORKER_MAX_LEN="${WORKER_MAX_LEN:-20480}"
WORKER_GPU_UTIL="${WORKER_GPU_UTIL:-0.90}"
ORCHESTRATOR="${ORCHESTRATOR:-Qwen/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-}"             # set => fully offline (HF_HUB_OFFLINE=1, proxies unset)
```

Env vars (set in sbatch and *re-exported inside every step payload* because pyxis containers don't reliably inherit `--export=ALL`):

```bash
export HF_HOME="${HF_HOME:-${REPO}/.hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false
# torch 2.9 renamed PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF; set both
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF}}"
# NCCL: exclude loopback/docker/veth (the `^` exclude syntax is NCCL-only).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker,veth}"
# Do NOT set GLOO_SOCKET_IFNAME to `^lo,...` — gloo treats it as a literal iface name
# and every vLLM worker dies with "Unable to find address for: ^lo".
if [[ "${GLOO_SOCKET_IFNAME:-}" == *"^"* ]]; then unset GLOO_SOCKET_IFNAME; fi

FWD_ENV="export HF_HOME='${HF_HOME}'; export HF_HUB_ENABLE_HF_TRANSFER='${HF_HUB_ENABLE_HF_TRANSFER}'; export TOKENIZERS_PARALLELISM=false; export PYTORCH_ALLOC_CONF='${PYTORCH_ALLOC_CONF}'; export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'; export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}';${GLOO_SOCKET_IFNAME:+ export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}';}${DATA_DIR:+ export CONDUCTOR_DATA_DIR='${DATA_DIR}'; export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY;}..."
```

Why the proxy is unset (verbatim comment): *"The team container bakes in HTTP_PROXY/HTTPS_PROXY=brain-proxy with a NO_PROXY that whitelists the node FQDN + 10.0.0.0/8 but NOT the bare short hostname the worker endpoints use (http://ib-a100-cluster-a-nNNN:port). So the conductor's requests to workers get sent to the external proxy and fail ("workers not reachable")."*

Runtime env selection (exactly one, or image+venv together):

```bash
#   CONDUCTOR_IMAGE  : enroot/pyxis .sqsh image (srun --container-image=...)
#   CONDUCTOR_VENV   : path to a venv activate script (source <path>)
#   CONDUCTOR_CONDA  : "<conda_base>|<env>"
IMAGE_PATH="${CONDUCTOR_IMAGE:-}"; VENV_ACTIVATE="${CONDUCTOR_VENV:-}"; CONDA_SPEC="${CONDUCTOR_CONDA:-}"
MOUNT="${CONDUCTOR_MOUNT:-/tmp:/tmp,/mnt/datafs:/mnt/datafs}"
SYS_PY="${SYS_PY:-python3}"     # stdlib-only python on the batch host for placement.py
ENV_PRELUDE=""
if [[ -n "${VENV_ACTIVATE}" ]]; then ENV_PRELUDE="source '${VENV_ACTIVATE}'"
elif [[ -n "${CONDA_SPEC}" ]]; then
  CONDA_BASE="${CONDA_SPEC%%|*}"; CONDA_ENV="${CONDA_SPEC##*|}"
  ENV_PRELUDE="source '${CONDA_BASE}/etc/profile.d/conda.sh'; conda activate '${CONDA_ENV}'"
fi
```

Values actually used (MN_README.md):

```bash
export CONDUCTOR_IMAGE=/mnt/datafs/ib-a100-cluster-a-pri/lmt/users/jay/images/ngc-2410_verl-lmt-251130_verl-0.6.0_vllm-0.10.2_flash-attn-2.8.3.sqsh
export CONDUCTOR_VENV=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/.venv/conductor/bin/activate
export HF_HOME=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean/.hf_cache
```
The idiom: *container provides CUDA userspace, NAS venv (built via `uv` inside the container on a compute node, since the login node has no PyPI) provides packages; the launcher sources the venv **inside** the container.*

### A.2 The `run_step` primitive (per-node, per-GPU-slice srun step) — the core reusable idiom

```bash
run_step() {
  local node="$1" cvd="$2" logf="$3" cmd="$4"
  local -a s=( srun --overlap --nodes=1 --ntasks=1 -w "${node}" --export=ALL )
  if [[ -n "${IMAGE_PATH}" ]]; then
    s+=( --container-image="${IMAGE_PATH}" --container-mounts="${MOUNT}"
         --container-workdir="${REPO}" )
  fi
  local payload="export CUDA_VISIBLE_DEVICES='${cvd}'; ${FWD_ENV} ${ENV_PRELUDE:+${ENV_PRELUDE}; }cd '${REPO}'; echo \"[step] host=\$(hostname) CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES HF_HOME=\$HF_HOME\"; ${cmd}"
  "${s[@]}" bash -lc "${payload}" >"${logf}" 2>&1
}
```
Gotchas (verbatim):
- *"GPU pinning MUST be done inside the step payload, NOT via `srun --export`: (1) Slurm comma-splits the --export list, so `--export=ALL,CUDA_VISIBLE_DEVICES=4,5,6,7` keeps only GPU 4 ... (2) Slurm's GRES plugin re-sets CUDA_VISIBLE_DEVICES per step AFTER --export is applied."*
- *"The whole-node `--gres=gpu:8` allocation exposes all 8 GPUs to every `--overlap` step (ConstrainDevices=yes -> cgroup spans all 8), so these physical indices are valid."*
- `run_step_host` = same without `--container-*` (needed when the step must call host `enroot`; the team image doesn't ship it). The sbatch also bind-mounts `/usr/bin/enroot`, `/usr/bin/enroot-*`, `/usr/lib/enroot`, `/etc/enroot`, `/usr/share/enroot`, unsquashfs/squashfuse/fuse-overlayfs/mksquashfs/nsenter/mountpoint/zstd/pigz `:ro`, plus `/usr/lib/x86_64-linux-gnu:/host-libs:ro` appended to `LD_LIBRARY_PATH` — only needed for nested-container tasks.
- Logs: each step's stdout+stderr to its own file (`${OUTPUT_DIR}/vllm_logs/worker_<i>_<model>.log`, `${OUTPUT_DIR}/train.log`, `${OUTPUT_DIR}/precache.log`); the sbatch's own log is `logs/slurm-conductor-mn-%j.log` (relative to submit dir; `submit.sh` does `mkdir -p logs` first).

### A.3 Placement (`conductor/placement.py`, stdlib-only, run with `/usr/bin/python3` on the batch host)

```bash
mapfile -t NODES_ARR < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NODES_CSV="$(IFS=,; echo "${NODES_ARR[*]}")"
PM="${PARALLEL_MODE:-dp}"
PLACE_ARGS=( --num-nodes "${NUM_NODES}" --nodes "${NODES_CSV}" --base-port "${BASE_PORT}" --parallel-mode "${PM}" )
[[ -n "${PER_DEVICE:-}" ]] && PLACE_ARGS+=( --per-device-bs "${PER_DEVICE}" )
PLACE_ENV="$(${SYS_PY} "${PLACEMENT}" "${PLACE_ARGS[@]}" --emit env)"
eval "${PLACE_ENV}"      # sets WORKER_i_{MODEL,NODE,GPUS,PORT,ENDPOINT}, CONDUCTOR_{NODE,GPUS,WORLD,TP}, GRAD_ACCUM, PER_DEVICE_BS, NUM_GENERATIONS, GENERATION_BATCH, N_WORKERS, WORKER_TP, SPARE_NODES
${SYS_PY} "${PLACEMENT}" "${PLACE_ARGS[@]}" --emit endpoints > "${ENDPOINTS_JSON}"   # {model: [url,...]}
${SYS_PY} "${PLACEMENT}" "${PLACE_ARGS[@]}" --emit table | tee "${OUTPUT_DIR}/placement.txt"
```

Placement logic (python):
```python
GPUS_PER_NODE = 8; SUPPORTED_NODE_COUNTS = (1, 2, 4, 8)
DEFAULT_PER_DEVICE_BS = 8; DEFAULT_NUM_GENERATIONS = 64; DEFAULT_GEN_BATCH = 256
# dp: worker_tp=1, replicas=num_nodes ; tp: worker_tp=num_nodes, replicas=1
units = [(mi, model, r) for mi, model in enumerate(worker_models) for r in range(replicas)]  # model-major
slots_per_node = gpus_per_node // worker_tp
n_worker_nodes = math.ceil(n_units / slots_per_node)
units_on_last = n_units - slots_per_node * (n_worker_nodes - 1)
free_on_last = gpus_per_node - units_on_last * worker_tp
conductor_needs_own_node = free_on_last < 2          # need >=2 GPUs
# worker s -> node_index = s // slots_per_node; gpus = range(local_slot*tp, +tp); port = base_port + s
# conductor -> leftover GPUs range(used, 8) on last worker node (or whole next node); world = len(gpus); tp = world
grad_accum = generation_batch // (per_device_bs * world)   # holds gen batch = 256 constant
```
Endpoint URL: `f"http://{node_name}:{port}"` (bare short hostname, hence the proxy unset).

Layouts (docstring):
```
dp  n=1  node0: g0-5 = M0..M5 (1 replica each)      | node0 g6-7 = conductor (world 2)
dp  n=2  node0: g0-7 = M0,M0,M1,M1,M2,M2,M3,M3      | node1 g4-7 = conductor (world 4)
         node1: g0-3 = M4,M4,M5,M5   (2 replicas/model, tp1)
tp  n=2  node0: M0 M1 M2 M3 (tp2)   node1: M4 M5     | node1 g4-7 = conductor (world 4)
```

### A.4 Pre-cache, then vLLM worker servers (background srun steps)

```bash
# 1.5 Pre-cache SEQUENTIALLY in ONE process: 6 vLLM servers downloading concurrently to NFS
#     deadlock on HF file locks (NFS flock unreliable). SKIP_PRECACHE=1 to skip.
rm -rf "${HF_HOME}/hub/.locks" 2>/dev/null || true
precache_cmd="set -e; for m in ${precache_list}; do huggingface-cli download \"\$m\" >/dev/null; done"
run_step "${NODES_ARR[0]}" "0" "${OUTPUT_DIR}/precache.log" "${precache_cmd}"

# 2. workers
for i in $(seq 0 $((N_WORKERS - 1))); do
  ...; safe="${model//\//_}"; log="${OUTPUT_DIR}/vllm_logs/worker_${i}_${safe}.log"
  # VLLM_PORT is vLLM's internal engine-coordination port. Co-located servers must
  # each get a distinct one or their EngineCore init collides. Space them out by 100.
  vllm_port=$(( 21000 + i * 100 ))
  if [[ -n "${MODELS_DIR:-}" ]]; then
    model_args="--model '${MODELS_DIR}/${model#*/}' --served-model-name '${model}'"
  else
    model_args="--model '${model}'"
  fi
  tp_flags=""; tp_env=""
  if [[ "${WORKER_TP}" -ge 2 ]]; then
    tp_flags="--disable-custom-all-reduce"                       # custom_all_reduce.cuh:455 'invalid argument'
    tp_env="VLLM_ATTENTION_BACKEND=${WORKER_ATTN_BACKEND:-FLASHINFER} "  # FLASH_ATTN illegal mem access at tp>=2
  fi
  vllm_cmd="${tp_env}VLLM_PORT=${vllm_port} python -m vllm.entrypoints.openai.api_server \
    ${model_args} --host 0.0.0.0 --port ${port} \
    --tensor-parallel-size ${WORKER_TP} \
    --max-model-len ${WORKER_MAX_LEN} \
    --gpu-memory-utilization ${WORKER_GPU_UTIL} \
    --dtype bfloat16 --trust-remote-code ${tp_flags}"
  # per-node staggered launch: co-located servers spaced by WORKER_LAUNCH_STAGGER (20s); different nodes in parallel
  seen=${NODE_LAUNCH_SEEN[$node]:-0}; delay=$(( seen * ${WORKER_LAUNCH_STAGGER:-20} )); NODE_LAUNCH_SEEN[$node]=$(( seen + 1 ))
  ( [[ "${delay}" -gt 0 ]] && sleep "${delay}"; run_step "${node}" "${gpus}" "${log}" "${vllm_cmd}" ) &
  WORKER_PIDS+=( "$!" )
done
cleanup() { kill "${WORKER_PIDS[@]}" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
```

Health wait:
```bash
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"; WAIT_INTERVAL="${WAIT_INTERVAL:-10}"
while true; do
  ready=0
  for i in ...; do curl -fsS "http://${!nvar}:${!pvar}/v1/models" >/dev/null 2>&1 && ready=$((ready+1)); done
  [[ "${ready}" -ge "${N_WORKERS}" ]] && break
  alive=0; for p in "${WORKER_PIDS[@]}"; do kill -0 "${p}" 2>/dev/null && alive=$((alive+1)); done
  [[ "${alive}" -eq 0 ]] && exit 4      # all worker steps died
  (( SECONDS - start >= WAIT_TIMEOUT )) && exit 4
  sleep "${WAIT_INTERVAL}"
done
```
Non-think is a per-request flag, not a server flag: client sends `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (workers.py `_model_extra_body`).

### A.5 Trainer launch (single node, colocate vLLM, DeepSpeed ZeRO-2)

```bash
zero2_accel() {   # <grad_accum> <workdir>  ->  echoes the `accelerate launch` prefix
  local ga="$1" wd="$2" cfg; mkdir -p "${wd}"; cfg="${wd}/accel.yaml"
  sed -e "s/gradient_accumulation_steps: auto/gradient_accumulation_steps: ${ga}/" \
      -e "s/gradient_clipping: auto/gradient_clipping: ${MAX_GRAD_NORM:-1.0}/" "${ACCEL_CONFIG}" > "${cfg}"
  echo "accelerate launch --config_file ${cfg} --num_machines 1 --num_processes ${CONDUCTOR_WORLD} --main_process_port ${ACCEL_PORT}"
}
# accelerate int()s gradient_accumulation_steps in Accelerator.__init__ BEFORE the HF Trainer can resolve 'auto' -> bake a concrete value
ACCEL_EVAL="accelerate launch --multi_gpu --num_machines 1 --num_processes ${CONDUCTOR_WORLD} --mixed_precision bf16 --dynamo_backend no --main_process_port ${ACCEL_PORT}"
ACCEL="$(zero2_accel "${GRAD_ACCUM}" "${OUTPUT_DIR}")"
COND_VLLM_TP="${CONDUCTOR_VLLM_TP:-1}"   # 1 = per-rank tp=1 vLLM engine (DP generation; ~5% faster than tp=world for 4B); must divide world
case "${NUM_NODES}" in 1) _rmw=64;; 2) _rmw=128;; *) _rmw=256;; esac
REWARD_MAX_WORKERS="${REWARD_MAX_WORKERS:-${_rmw}}"

train_cmd="${ACCEL} \
  ${TRAIN_SCRIPT} --config '${CONFIG}' \
    --override worker_endpoints_file='${ENDPOINTS_JSON}' \
    --override vllm_tensor_parallel_size=${COND_VLLM_TP} \
    --override gradient_accumulation_steps=${GRAD_ACCUM} \
    --override per_device_train_batch_size=${PER_DEVICE_BS} \
    --override num_generations=${NUM_GENERATIONS} \
    --override reward_max_workers=${REWARD_MAX_WORKERS} \
    --override output_dir='${OUTPUT_DIR}'${EXTRA_OV}"
run_step "${CONDUCTOR_NODE}" "${CONDUCTOR_GPUS}" "${OUTPUT_DIR}/train.log" "${train_cmd}"
train_rc=$?; exit "${train_rc}"
```
EVAL-ALL mode (`EVAL_SRC=<run dir>`): loops `checkpoint-*` (numeric sort), skips ones whose `eval.log` already contains `Eval results` (unless `EVAL_FORCE=1`), wipes partial dests (transcripts.jsonl is append-only), runs `${ACCEL_EVAL} train.py --evaluate-only --resume-from <ckpt> ...` per ckpt into `eval/<run>/checkpoint-N/`. `PD_SWEEP="4 8 16"` mode reuses live workers for back-to-back train cells.

`configs/accelerate.yaml` (ZeRO-2, params unsharded so trl colocate weight-sync sees the whole model; ZeRO-3 would break it):
```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: auto     # sed'd to a concrete int by zero2_accel
  gradient_clipping: auto
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: 1
num_processes: 2                       # overridden by --num_processes on CLI
rdzv_backend: static
same_network: true
```

### A.6 `submit.sh`

```bash
NUM_NODES="${NUM_NODES:-1}"; case "${NUM_NODES}" in 1|2|4|8) ;; *) exit 1;; esac
REPO="${CONDUCTOR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="$(readlink -f "${REPO}")"      # ~/lmalign symlink -> /mnt/datafs real path (compute nodes mount ONLY /mnt/datafs)
export CONDUCTOR_REPO="${REPO}"
# NODE_RANGE=<lo>-<hi> -> --exclude of everything outside [lo,hi]:
prefix="${NODE_PREFIX:-ib-a100-cluster-a-n}"; cmin="${CLUSTER_NODE_MIN:-1}"; cmax="${CLUSTER_NODE_MAX:-258}"
(( 10#${lo} > cmin )) && outside+=( "$(printf '%03d-%03d' "${cmin}" "$((10#${lo} - 1))")" )
(( 10#${hi} < cmax )) && outside+=( "$(printf '%03d-%03d' "$((10#${hi} + 1))" "${cmax}")" )
range_excl="${prefix}[$(IFS=,; echo "${outside[*]}")]"; EXCLUDE="${EXCLUDE:+${EXCLUDE},}${range_excl}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"   # ALWAYS finite: a-nodes default is UNLIMITED and unlimited jobs cannot backfill
extra=()
[[ -n "${NODELIST:-}" ]]  && extra+=( --nodelist="${NODELIST}" )
[[ -n "${EXCLUDE:-}" ]]   && extra+=( --exclude="${EXCLUDE}" )
[[ -n "${PARTITION:-}" ]] && extra+=( --partition="${PARTITION}" )
[[ -n "${JOB_NAME:-}" ]]  && extra+=( -J "${JOB_NAME}" )      # job name chosen at submit time (memory rule: sean-<descriptive>)
extra+=( -t "${TIME_LIMIT}" )
mkdir -p "${REPO}/logs" logs
sbatch -N "${NUM_NODES}" --export=ALL "${extra[@]}" "${SBATCH_FILE}"
```
Typical invocation: `NUM_NODES=2 NODE_RANGE=41-66 RUN_NAME=<name> JOB_NAME=sean-... CONDUCTOR_CONFIG=configs/default.yaml bash scripts/submit.sh` (note memory rule: `--cpus-per-task=32 --mem=600G` on the CLI overrides the header).

### A.7 `train.py` (trl 0.29 GRPO, whole-file essentials)

Monkey-patches at import time:
```python
import vllm
from vllm.entrypoints.llm import LLM as _VllmLLM
_orig_llm_init = _VllmLLM.__init__
def _patched_llm_init(self, *args, **kwargs):
    kwargs["disable_custom_all_reduce"] = True      # custom_all_reduce.cuh:455 'invalid argument' at tp>1 colocate
    _orig_llm_init(self, *args, **kwargs)
_VllmLLM.__init__ = _patched_llm_init

import torch.distributed as _dist
_orig_init_pg = _dist.init_process_group
def _patched_init_pg(*args, **kwargs):
    kwargs.setdefault("timeout", timedelta(hours=2))   # NCCL watchdog 30min -> 2h (slow reward episodes)
    return _orig_init_pg(*args, **kwargs)
_dist.init_process_group = _patched_init_pg
```
Pad-token fix: `Qwen -> "<|fim_pad|>"`, `Llama -> "<|reserved_special_token_5|>"`, else eos.

Flow: `cfg = load_config(args.config, args.override)` → `WorkerPool(models, endpoints, max_tokens, temperature, timeout)` + `workers.healthcheck()` (raises if any unreachable) → LLM-judge env wiring (`HLE_JUDGE_URLS` etc. from `workers.endpoints[judge_model]`) → `make_datasets(...)` → `ConductorReward(workers=..., log_dir=cfg.output_dir, ...)` (callable reward func; writes `transcripts.jsonl` under output_dir, exposes `flush_stats()` and `_transcript_path`) → `GRPOConfig(...)` → `ConductorGRPOTrainer(model=base, args=train_args, train_dataset, eval_dataset, reward_funcs=[reward], processing_class=tokenizer)` → `trainer.train(resume_from_checkpoint=ckpt)` (only if `trainer_state.json` exists in ckpt) → `trainer.save_model(cfg.output_dir)`.

GRPOConfig (verbatim):
```python
train_args = GRPOConfig(
    output_dir=cfg.output_dir,
    model_init_kwargs={"dtype": "bfloat16" if cfg.evaluate_only else "float32"},  # fp32 master weights; bf16 Adam rounds lr=1e-6 steps to zero
    max_steps=cfg.max_steps,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    num_generations=cfg.num_generations,
    num_generations_eval=1,
    max_completion_length=cfg.max_completion_length,
    learning_rate=cfg.learning_rate, beta=cfg.beta, temperature=cfg.temperature,
    weight_decay=cfg.weight_decay, max_grad_norm=cfg.max_grad_norm,
    lr_scheduler_type=cfg.lr_scheduler_type, warmup_ratio=cfg.warmup_ratio,
    save_steps=cfg.save_steps, save_strategy="steps", logging_steps=cfg.logging_steps,
    eval_strategy=cfg.eval_strategy, eval_steps=cfg.eval_steps,
    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    report_to=cfg.report_to or "none", bf16=cfg.bf16, tf32=cfg.tf32,
    use_vllm=cfg.use_vllm,
    vllm_mode=cfg.vllm_mode,                       # "colocate"
    vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,   # 0.3 (0.2 -> vLLM refuses; 0.4 -> optimizer OOM)
    vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
    remove_unused_columns=False,
    ddp_timeout=cfg.ddp_timeout,
    scale_rewards=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```
`configs/default.yaml` training block: `max_steps: 200, train_batch_size: 256, per_device_train_batch_size: 8, gradient_accumulation_steps: 16, num_generations: 64, max_completion_length: 1024, learning_rate: 1e-6, beta: 0.0, temperature: 1.0, lr_scheduler_type: cosine, warmup_ratio: 0.03, save_steps: 25, logging_steps: 1, report_to: tensorboard, bf16: true, tf32: true, eval_strategy: steps, eval_steps: 25, per_device_eval_batch_size: 8, use_vllm: true, vllm_mode: colocate, vllm_gpu_memory_utilization: 0.3, vllm_tensor_parallel_size: 2`; worker calls `worker_max_tokens: 8192, worker_temperature: 0.2, worker_timeout: 600`; reward `format_bonus: 0.5, format_error_reward: 0.25, reward_max_workers: 64`.

vLLM wiring: **colocate only** (`vllm_mode: colocate`; trl spins a vLLM engine per rank inside the trainer process; `vllm_tensor_parallel_size` must divide world). Server mode is not used for the policy; the *workers* are separate OpenAI-compatible servers reached over HTTP by the reward function. Rewards are computed inside `reward_funcs=[ConductorReward]` with a `ThreadPoolExecutor(max_workers=reward_max_workers)` fan-out (not shown in trainer.py; reward.py not read).

### A.8 `conductor/trainer.py` (GRPOTrainer subclass)

```python
class ConductorGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs):
        keys = set().union(*[ex.keys() for ex in inputs])           # equalize aux keys (task_type, ...) so trl can gather
        inputs = [{k: ex.get(k, None) for k in keys} for ex in inputs]
        return super()._generate_and_score_completions(inputs)
    def log(self, logs, *args, **kwargs):
        for rf in self.reward_funcs:
            flush = getattr(rf, "flush_stats", None)
            if callable(flush): logs.update(flush())                # reward-fn running stats -> wandb/tb
        super().log(logs, *args, **kwargs)
    def evaluate(self, *args, **kwargs):
        n_before = _transcript_line_count(self.reward_funcs)
        results = super().evaluate(*args, **kwargs)
        deduped = _dedup_metrics_from_transcripts(self.reward_funcs, n_before)   # dedupe by (task_type, base_question): HF Trainer pads eval batches with duplicates
        ...results[f"{prefix}_{k}"] = v ; _print_eval_summary(deduped)
```
Rollout logging = reward function appends one JSON record per rollout to `<output_dir>/transcripts.jsonl` with fields `task_type, base_question, correctness, resolved, parsing_errors, parsed.model_ids`. Overall accuracy = unweighted mean of per-task accuracy.

### A.9 `conductor/workers.py` (client side)

```python
OpenAI(api_key="EMPTY", base_url=f"{url}/v1", timeout=timeout, max_retries=0)   # one per replica URL; NO retries (retry re-hits timeout -> NCCL timeout)
client.chat.completions.create(model=vllm_model, messages=messages, max_tokens=max_tokens,
    temperature=temperature, stream=False, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
def healthcheck(self) -> Dict[str, bool]:   # requests.get(f"{url}/v1/models", timeout=5).status_code == 200
def default_endpoints(models, host="127.0.0.1", base_port=8100) -> {m: f"http://{host}:{base_port+i}"}
```
Round-robin over replicas via `itertools.count()` per model (GIL-atomic).

### A.10 `conductor/math_grade.py`

Library: `math_verify` (`parse`, `verify`; sympy-backed). Entry point:
```python
def extract(text: str) -> str      # strip <think>..</think>; last <answer>..</answer>; else last \boxed{}; else raw
def grade(prediction: str, ground_truth: str) -> float   # 1.0 / 0.0
```
Algorithm: `_parse(s)` wraps in `$...$` first (bare `7\pi` mis-parses to `7` without an anchor), falls back to raw; `_sym_eq(gold, pred)` = `verify(parse(gold), parse(pred))` **gold first**; then comma-separated multi-answers compared as unordered set (`_split_set` ignores commas inside brackets, never splits `\begin` envs); last resort `_norm_string` (strips `\boxed`, `\dfrac->\frac`, `\left/\right/\big`, `\text{}`, whitespace, single braces).
The docstring notes mtm's `score_math.py` bug: *"calls parse(gold) with NO latex anchor, so bare-latex golds (7\pi, \sqrt{2}, \frac 34, \begin{pmatrix}) silently mis-parse to a truncated number"*. Use `math_grade.grade` for MATH-style gold; for AIME integer gold either works.

### A.11 `conductor/tasks/`

`ls`: `base.py browsecomp_plus.py catalog.py frames.py hotpotqa.py registry.py scicode.py swebench.py taskcraft.py tbench.py` — **no livecodebench task in conductor_sean_tool** (LCB lives in the other conductor repos; see A.12).

### A.12 LiveCodeBench grader (`conductor/lcb_grade.py` — copy read from `past/code/conductor_sean_agent/conductor/lcb_grade.py`; identical copies in conductor_sean_{open,open_3,synth,api}, verify_both_pristine)

Official LCB prompt/extractor/runner reproduced verbatim; execution imports upstream `lcb_runner.evaluation.testing_util.run_test` from the vendored checkout **in a subprocess** (because `run_test` calls `reliability_guard()` which sets 4GB RLIMIT_AS, nulls `os.kill/remove/rename` and installs SIGALRM in the calling process).

```python
ENV_OFFICIAL = "LCB_OFFICIAL"     # vendored checkout root (default DATA_DIR/livecodebench/official)
ENV_PYTHON = "LCB_PYTHON"         # interpreter for the grading subprocess (needs numpy)
ENV_TIMEOUT = "LCB_TIMEOUT"       # per-test timeout (default 6 = upstream)
ENV_SLOTS = "LCB_GRADE_SLOTS"     # machine-wide cap on concurrent grading subprocesses (default cpu_count//2)
SYSTEM_MESSAGE_GENERIC = ("You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the specification and passes all tests.")
def build_prompt(rec: Dict[str, Any]) -> str        # "### Question:\n{question_content}\n\n### Format: ...starter_code branch or stdin branch...\n### Answer: (use the provided format with backticks)\n\n"
def extract_code(model_output: str) -> str          # text between the LAST TWO lines containing ``` ; <2 fences -> ""
def evaluation_sample(rec) -> {"input_output": json.dumps({"inputs","outputs","fn_name": metadata.func_name})}   # public+private tests; b64(zlib(pickle)) decoded
def run_tests(code: str, rec, *, data_dir=None, timeout=None) -> {"result","n_tests","n_passed","passed","error"}
def grade_livecodebench(response: str, rec: Any, *, data_dir=None, timeout=None) -> Dict   # rec may be the lcb_row JSON STRING
    # -> {"score": 1.0|0.0, "passed", "n_tests", "n_passed", "extracted", "error", "question_id", "difficulty", "platform"}
def slice_filter(rows, slices="")                   # e.g. "test6.jsonl" = the v6 tag (175 problems); rows carry r["lcb_row"]
def window_filter(rows, start_date="", end_date="") # contest_date window (upstream start_date/end_date)
```
Subprocess: `subprocess.run([LCB_PYTHON or sys.executable, driver.py, official_dir, payload.json, result.json], timeout=(per_test+1)*n_tests+5)`; driver does `sys.path.insert(0, official_dir); from lcb_runner.evaluation.testing_util import run_test; run_test(sample, test=code, debug=False, timeout=per_test)`. Pass iff every test entry is `True`/`>0` and `len(result)==n_tests` (empty result = runner died = fail). Gated by a `threading.BoundedSemaphore(LCB_GRADE_SLOTS)` — uncapped grading turned a 38.7% into 7.0% on SciCode via timeouts.

Task adapter (`conductor/tasks/livecodebench.py`, `LiveCodeBenchAdapter(SingleShotAdapter)`): `task_type="livecodebench"`, `aux_column="lcb_row"`, `data_subdir="livecodebench"`, prompt = `SYSTEM_MESSAGE_GENERIC + "\n\n" + build_prompt(aux)` as a single user message; env `LCB_START_DATE`/`LCB_END_DATE`/`LCB_SLICE` applied BEFORE `limit` (file is oldest-first); `grade_response` → `grade_livecodebench(response, aux, data_dir)` with `correct = passed`; EVAL-ONLY, no partial credit, no repair turn.

### A.13 env_setup.sh (cpu-instance conda, NOT the cluster)

```bash
ENV_PREFIX="${ENV_PREFIX:-/data/llm-public_636/lpt/users/sean/envs/conductor}"
BASE_ENV="${BASE_ENV:-/data/llm-public_636/lpt/users/sean/envs/rl_clone}"   # torch 2.9 / vllm 0.12 / trl 0.29 / transformers 4.57 / accelerate 1.13 / flash-attn 2.8.3 / deepspeed
source /opt/conda/etc/profile.d/conda.sh
conda create -p "$ENV_PREFIX" --clone "$BASE_ENV" -y
pip install --no-input tenacity omegaconf pyyaml word2number
```
On the cluster the equivalent is the NAS venv `.venv/conductor` (built by `scripts/env/setup_env.sbatch` via uv inside ngc-2410; Python 3.13).

---

## B. SFT (mtm repo)

### B.1 Multi-node SFT sbatch (`mtm/sft/train_sft.sbatch`) — THE 2-node data-parallel recipe

```bash
#SBATCH -J sean-mtm-sft
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --output=logs/slurm-mtm-sft-%j.log
set -euo pipefail

REPO="$(readlink -f "${MTM_REPO:-$(pwd)}")"; cd "${REPO}"; mkdir -p logs
NUM_NODES="${SLURM_NNODES:?must run under Slurm (sbatch -N K)}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
WORLD=$(( NUM_NODES * GPUS_PER_NODE ))
IMG="${MTM_IMAGE:?set MTM_IMAGE to the .sqsh image}"
VENV="${MTM_VENV:-}"
ACCEL_CFG="${ACCEL_CFG:-sft/accelerate_zero2.yaml}"
PORT="${ACCEL_PORT:-29591}"
PD="${PER_DEVICE_BS:-4}"
EFF_TARGET="${EFF_BATCH:-128}"
GA=$(( EFF_TARGET / (PD * WORLD) )); (( GA < 1 )) && GA=1
EFF=$(( PD * WORLD * GA ))
ACCEL_CFG_RUN="${REPO}/logs/accel_mtm_sft_${SLURM_JOB_ID:-run}.yaml"
sed "s|^\(  gradient_accumulation_steps:\).*|\1 ${GA}|" "${ACCEL_CFG}" > "${ACCEL_CFG_RUN}"

HEAD="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
HEAD_IP="$(getent hosts "${HEAD}" | awk '{print $1}' | head -1)"; [[ -z "${HEAD_IP}" ]] && HEAD_IP="${HEAD}"

MOUNT="/mnt/datafs:/mnt/datafs"
ENVX="export HF_HOME='${REPO}/.hf_cache'; export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; \
export TOKENIZERS_PARALLELISM=false; export PYTORCH_ALLOC_CONF=expandable_segments:True; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_SOCKET_IFNAME='^lo,docker,veth'; \
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY;"
PRELUDE="${VENV:+source '${VENV}';}"

LAUNCH="accelerate launch --config_file ${ACCEL_CFG_RUN} \
--num_machines ${NUM_NODES} --num_processes ${WORLD} --machine_rank \$SLURM_NODEID \
--main_process_ip ${HEAD_IP} --main_process_port ${PORT} \
sft/train_sft.py --base ${BASE} --sources ${SOURCES} --domains ${DOMAINS} --alpha ${ALPHA} \
--train-size ${TRAIN_SIZE} --num-epochs ${NUM_EPOCHS} --output ${OUTPUT} --lr ${LR} --seed ${SEED} \
--per-device-bs ${PD} --grad-accum ${GA} --max-len ${MAX_LEN} --template-family '${TEMPLATE_FAMILY}' ${EXTRA_OVERRIDES}"

srun --nodes="${NUM_NODES}" --ntasks-per-node=1 --export=ALL \
  --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" \
  bash -lc "${ENVX} ${PRELUDE} ${LAUNCH}"
```
Key points: one `srun` with `--ntasks-per-node=1` over all nodes; **`\$SLURM_NODEID` is escaped** so each task evaluates its own machine_rank inside the container; rendezvous = head node IP (via `getent hosts`) + fixed port; `rdzv_backend: static` in the yaml. Invoked as `sbatch -N 2 --export=ALL sft/train_sft.sbatch` with env `BASE SOURCES DOMAINS ALPHA TRAIN_SIZE NUM_EPOCHS OUTPUT LR SEED PER_DEVICE_BS EFF_BATCH MAX_LEN TEMPLATE_FAMILY EXTRA_OVERRIDES MTM_IMAGE MTM_VENV`.

### B.2 Single-node sub-step (`mtm/sft/train_step.sh`) — body run inside an `srun --overlap --container-image` step with CUDA_VISIBLE_DEVICES pre-narrowed

```bash
N="${N:?set N (GPUs per model)}"; PORT="${ACCEL_PORT:-29600}"
GA=$(( EFF_BATCH / (PD * N) )); (( GA < 1 )) && GA=1
RUN_CFG="${REPO}/logs/accel_${DOMAINS//,/_}_${SLURM_JOB_ID:-run}_p${PORT}.yaml"   # per-slot copy (concurrent branches on one node)
sed "s|^\(  gradient_accumulation_steps:\).*|\1 ${GA}|" "${ACCEL_CFG}" > "${RUN_CFG}"
accelerate launch --config_file "${RUN_CFG}" \
  --num_machines 1 --num_processes "${N}" --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${PORT}" \
  sft/train_sft.py ... --per-device-bs "${PD}" --grad-accum "${GA}" --max-len "${MAX_LEN}" ...
touch "${OUTPUT}/.mtm_branch_ok"   # completion sentinel LAST (config.json is written BEFORE weight shards -> unsafe resume signal)
```
ZeRO-3 at N=2 else ZeRO-2 (driver picks `ACCEL_CFG`).

### B.3 accelerate configs

`sft/accelerate_zero2.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1      # PLACEHOLDER — train_sft.sbatch rewrites to derived GA
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
`sft/accelerate_zero3.yaml` differs only in: `zero3_init_flag: true`, `zero3_save_16bit_model: true` (*"gathers a FULL bf16 model on save_pretrained so linear_merge.py can load it (else save writes sharded/empty weights)"*), `zero_stage: 3`, `num_processes: 2`.

### B.4 `sft/train_sft.py` (plain HF Trainer)

```python
tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
prepare_tokenizer(tok, args.base, family=args.template_family or None)
rows = build_mixture_rows(sources, alpha, args.train_size, args.seed)
ds = MixtureSFTDataset(rows, tok, max_len=args.max_len, packing=args.packing)
is_main = os.environ.get("RANK", "0") in ("0", "") and os.environ.get("LOCAL_RANK", "0") in ("0", "")
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=args.attn)  # attn="sdpa" (no flash-attn in mtm venv)
model.config.use_cache = False
targs = TrainingArguments(
    output_dir=args.output, num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.per_device_bs, gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr, lr_scheduler_type=args.lr_scheduler,      # 2e-5, constant_with_warmup
    warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,  # 0.03, 0.0, 1.0
    seed=args.seed, bf16=True, tf32=True,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False, logging_steps=args.logging_steps,
    save_strategy=args.save_strategy,   # default "no"
    save_only_model=True,               # checkpoint-<step> = clean model dir, no optimizer states
    report_to="none", remove_unused_columns=False, dataloader_num_workers=2,
)
trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=SFTCollator(pad_token_id=tok.pad_token_id), processing_class=tok)
trainer.train()
if args.save_strategy == "no": trainer.save_model(args.output); if is_main: tok.save_pretrained(args.output)
```
`--inspect N` prints the loss mask without GPU. Defaults: `--per-device-bs 4 --grad-accum 2 --max-len 4096 --lr 2e-5 --num-epochs 3.0`.

### B.5 `sft/dataset.py` (assistant-only loss masking, by hand)

```python
IGNORE_INDEX = -100
def build_mixture_rows(sources, alpha, train_size, seed) -> List[dict]   # largest-remainder rounding; sample w/o replacement, cap at pool size
class MixtureSFTDataset(Dataset):
    def _encode(self, messages):
        assert messages[-1]["role"] == "assistant"
        prompt_text = render_prompt(self.tok, messages[:-1])          # ends with '<|im_start|>assistant\n'
        prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        comp_ids = self.tok(messages[-1]["content"], add_special_tokens=False).input_ids + [self.eos]
        overflow = len(prompt_ids) + len(comp_ids) - self.max_len
        if overflow > 0:   # LEFT-truncate the prompt, never the completion; hard-cap completion keeping eos
            ...
        return prompt_ids + comp_ids, [IGNORE_INDEX] * len(prompt_ids) + comp_ids
@dataclass
class SFTCollator: pad_token_id: int   # right-pads input_ids/labels(-100)/attention_mask
```
`packing=True` greedily concatenates examples into `<=max_len` bins (cross-example attention left on).

### B.6 `common/chat_template.py`

One imposed non-think ChatML template for train and eval:
```python
def _build_chatml(assistant_open: str) -> str:
    return ("{%- for message in messages -%}"
        "{%- if message['role'] == 'assistant' -%}"
        "{{- '" + assistant_open + "' + message['content'] + eos_token -}}"
        "{%- else -%}"
        "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' -}}"
        "{%- endif -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- '" + assistant_open + "' -}}{%- endif -%}")
FAMILY_TOKENS = {
    "olmo": {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>", "assistant_open": "<|im_start|>assistant\n\n"},  # Olmo needs the BLANK line (token 271) — bare \n sent base into loops (gsm8k 81.7 vs 88.2)
    "qwen": {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "assistant_open": "<|im_start|>assistant\n"},
    "default": {"eos_token": "<|im_end|>", "pad_token": None, "assistant_open": "<|im_start|>assistant\n"},
}
def detect_family(model_path) -> "olmo"|"qwen"|"default"
def prepare_tokenizer(tokenizer, model_path, family=None)   # sets chat_template, eos_token, pad_token in place
def eos_id(tokenizer) -> int                                 # tokenizer.eos_token_id (configured per family)
def render_prompt(tokenizer, messages) -> str                # apply_chat_template(add_generation_prompt=True)
def stop_tokens(tokenizer) -> ["<|im_end|>"]
```

### B.7 `common/config.py`

Dataclass `MtmConfig` + YAML + repeatable `--override key=value` (`_coerce` by current type: bool/int/float/list(comma)/str); `load_config(path, overrides)`; `parse_args(default_config="config/mtm.yaml")` → `--config`, `--override` (append), `parse_known_args`. SFT defaults: `learning_rate 2e-5, constant_with_warmup, warmup 0.03, wd 0.0, grad_norm 1.0, per_device 4, eff_batch 128, max_len 4096, num_epochs 1.0, attn sdpa, bf16, tf32, grad ckpt`. Eval defaults: `eval_max_tokens 4096, eval_greedy_only True, eval_tp 1`.

---

## C. Eval (mtm repo)

### C.1 `eval/baseline_eval.sbatch`

```bash
#SBATCH -J sean-mtm-baseline-eval
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --output=logs/slurm-mtm-baseline-%j.log
set -euo pipefail
REPO="$(readlink -f "${MTM_REPO:-$(pwd)}")"; cd "${REPO}"; mkdir -p logs
IMG="${MTM_IMAGE:?}"; VENV="${MTM_VENV:?}"
MODEL="${MODEL:?set MODEL}"; TAG="${TAG:?set TAG}"
TEMPLATE_FAMILY="${TEMPLATE_FAMILY:-}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BENCHMARKS="${BENCHMARKS:-gsm8k,math-500,mbpp,ifeval}"; GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MATH_PROMPT="${MATH_PROMPT:-system}"
MOUNT="/mnt/datafs:/mnt/datafs"
ENVX="export HF_HOME='${REPO}/.hf_cache'; export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; \
export TOKENIZERS_PARALLELISM=false; export NLTK_DATA='${REPO}/eval/assets/nltk_data'; \
export VLLM_LOGGING_LEVEL=WARNING; export PYTORCH_ALLOC_CONF=expandable_segments:True; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY;"
srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" \
  bash -lc "${ENVX} source '${VENV}'; python eval/run_eval.py --model '${MODEL}' \
    --benchmarks '${BENCHMARKS}' --out 'outputs/baseline/${TAG}' --gpus '${GPUS}' \
    --mode greedy --max-model-len ${MAX_MODEL_LEN} --template-family '${TEMPLATE_FAMILY}' \
    --math-prompt '${MATH_PROMPT}' ${RAWFLAG} ${NTFLAG} --thinking '${THINKING}'"
```
Submit: `MODEL=<dir> TAG=<name> TEMPLATE_FAMILY=olmo|qwen sbatch --export=ALL eval/baseline_eval.sbatch`. NLTK data is staged offline at `mtm/eval/assets/nltk_data` (`score_ifeval.py` also `os.environ.setdefault("NLTK_DATA", ...)`).

### C.2 `eval/run_eval.py` — data-parallel offline vLLM over N GPUs

Parent shards tasks round-robin across `--gpus`, spawns itself with `--worker --shard --out-shard`:
```python
env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(g)
# Distinct engine port per PHYSICAL gpu id so concurrent evals on one node never collide on vLLM's TCPStore port (get_open_port TOCTOU -> EADDRINUSE -> EngineCore dies)
env["VLLM_PORT"] = str(40000 + (int(g) % 8) * 128)
cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--shard", sp, "--out-shard", op, "--model", args.model, ...]
procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), op, log))   # log = <out>/_shards/worker_<gi>.log
```
Output: `<out>/metrics.json = {"<bench>": {"score": pass@1 %, "greedy_score", "n", "aux"}}`, `<out>/<bench>.gen.jsonl`. Deterministic subsampling uses `zlib.crc32(name)` (not `hash()`). `--mode report` adds `--report-samples 3` T=0.7 samples, `m4=(3*m3+g)/4`.

### C.3 `eval/generate.py`

```python
self.llm = LLM(model=model_path, tokenizer=model_path, tensor_parallel_size=tp,
    disable_custom_all_reduce=(tp > 1),     # custom_all_reduce.cuh:455 at tp>1 on this P2P config
    dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_util,  # 0.90
    trust_remote_code=True,
    enforce_eager=(tp > 2),                 # tp>=4 CUDA-graph replay illegal memory access
    seed=seed)
self.stop_ids = sorted({configured eos, native eos, <|im_end|>})
sp = SamplingParams(temperature=temperature, top_p=(top_p if temperature > 0 else 1.0), max_tokens=max_tokens, n=n,
                    seed=(seed if temperature > 0 else None), stop=self.stop_strs, stop_token_ids=self.stop_ids)
prompts = [TokensPrompt(prompt_token_ids=self._prompt_ids(m)) for m in messages_list]   # token ids -> byte-identical with SFT
def generate(self, messages_list, max_tokens=4096, temperature=0.0, top_p=1.0, n=1, seed=0) -> List[List[str]]
```
`--raw` = plain completion prompt with native BOS (raw base); `--native-template` = model's own chat_template with `thinking=` kwarg.

### C.4 `eval/benchmarks.py` registry

`BENCHMARKS[name] = {"kind", "load": lambda dd: read_jsonl(dd/name/test.jsonl), "prompt": lambda r: ..., "score": fn, "max_tokens"}`; `gsm8k` (4096), `math-500` (8192), `mbpp` (4096), `ifeval` (2048), plus mmlu/mmlu-law/medqa/casehold (mcq, 1024), finqa (2048), advbench/beavertails/xstest (safety). Score fns take `(rows, generations) -> {"score": %, "n", ...}`.

### C.5 Grading entry points

**Math** (`eval/score_math.py`):
```python
MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
def set_math_prompt(variant: str) -> None            # "system" | "answer_only"
def build_messages(problem: str) -> list[dict]
def _verify_one(expected_answer: str, generation: str) -> bool   # math_verify.parse(gold) (fallback "$gold$"), parse(gen), verify(gold, pred) GOLD FIRST
def score(rows: List[dict], generations: List[str]) -> dict     # rows need "expected_answer"; ProcessPoolExecutor over 75% cores
```
(Prefer `conductor/math_grade.grade(prediction, ground_truth)` for latex golds — see A.10.)

**IFEval** (`eval/score_ifeval.py`, vendored Google `instruction_following_eval` under `eval/third_party/`: `evaluation_lib.py evaluation_main.py instructions.py instructions_registry.py instructions_util.py`):
```python
def build_messages(row: dict) -> [{"role":"user","content": row["prompt"]}]
def verify_one(generation: str, gold: dict) -> bool    # gold={"prompt","instruction_id_list","kwargs"}; STRICT follow_all_instructions; unknown id -> False
def score(rows: List[dict], generations: List[str]) -> dict   # {"score": mean of 4 accs, "prompt_strict","prompt_loose","instruction_strict","instruction_loose","n","n_instructions"}
# internals: elib.InputExample(key, instruction_id_list, prompt, kwargs); elib.test_instruction_following_strict(inp, {prompt: gen}); ..._loose(...)
```
Data: `mtm/eval/data/ifeval/{input_data.jsonl, test.jsonl, prepare.py}` (rows `{key, prompt, instruction_id_list, kwargs}`).

**Code** (`eval/score_code.py`, MBPP+ base tests; forked child + `RLIMIT_AS` 8GB + timeout; launched from a single thread):
```python
def build_messages(row: dict)                               # CODEGEN_TEMPLATE.format(q=row["prompt"])
def extract_code(generation: str, entry_point: str) -> str  # fenced block defining entry_point, else last block; pure-ast cleanup
def score(rows, generations, timeout=6.0, workers=None) -> {"score","n","correct"}   # rows need entry_point, assertion
def verify_asserts(generation, entry_point, assertion, timeout=8.0) -> bool
def verify_asserts_batch(specs, timeout=8.0, workers=None) -> list[bool]   # specs=(generation, entry_point, assertion)
def verify_leetcode(generation, test, entry_point, timeout=8.0) -> bool     # test defines check(candidate)
def verify_leetcode_batch(specs, timeout=8.0, workers=None)
def verify_kodcode(generation, test, entry_point, timeout=8.0) -> bool      # pytest-style: `from solution import` + test_*()
def verify_kodcode_batch(specs, timeout=8.0, workers=None)
```
`_RUNNER_IMPORTS` prepends typing/collections imports and pins BLAS threads to 1 (OpenBLAS arena vs RLIMIT_AS).

---

## D. Data

`data/livecodebench/official/lcb_runner/` = vendored official LCB runner: `benchmarks/ evaluation/ lm_styles.py prompts/ runner/ utils/`; `evaluation/` has `compute_code_generation_metrics.py compute_scores.py pass_k_utils.py testing_util.py utils_execute.py`.

`data/livecodebench/test.jsonl` row keys: `['task_type', 'base_question', 'ground_truth', 'lcb_row']`; `task_type="livecodebench"`, `base_question` = full problem statement (Codeforces-style with Sample Input/Output), `ground_truth` = empty string (tests live in `lcb_row`), `lcb_row` = **JSON string** with keys `['question_title', 'question_content', 'platform', 'question_id', 'contest_id', 'contest_date', 'starter_code', 'difficulty', 'public_test_cases', 'private_test_cases', 'metadata', 'id', 'slice']`. Rows are huge (private tests embedded).

`data_agent/math/test.jsonl` row: `{"id": "aime_2024_I_1", "task_type": "math", "ground_truth": "204", "math_row": "<JSON string: {id, problem, answer, source: MathArena/aime_2024_I, year, contest, problem_idx}>"}`.

`data_agent/math/agent_split_manifest.json`: eval = AIME 2024 I+II / 2025 / 2026 (MathArena), n=90 (30/yr), cross-checked vs HuggingFaceH4/aime_2024, opencompass/AIME2025, math-ai/aime26 (0 mismatches), `ids_sha256_sorted=12857880…`, `answers_sha256_by_id=e88f72dc…`; train = 240 rows from AIME 2014-2021 (replaced DeepScaleR sample 2026-08-22; `prev8_manifest.json`), seed 42; previous train pool DeepScaleR-Preview 40,315 rows, 0 contaminated by eval.

---

## Exact 2-node launch sequences

### (i) SFT, 2 nodes (mtm idiom; genuine DP world=16)
1. `sbatch -N 2 --export=ALL -J sean-<name> sft/train_sft.sbatch` (with BASE/SOURCES/... and `MTM_IMAGE`, `MTM_VENV` in env).
2. Batch script runs on node 0: computes `WORLD=16`, `GA = EFF_BATCH/(PD*16)`, writes `logs/accel_mtm_sft_<jobid>.yaml`, resolves `HEAD_IP` of node 0.
3. One `srun --nodes=2 --ntasks-per-node=1 --container-image --container-mounts=/mnt/datafs:/mnt/datafs --container-workdir=$REPO bash -lc "<ENVX> source $VENV; accelerate launch --num_machines 2 --num_processes 16 --machine_rank \$SLURM_NODEID --main_process_ip $HEAD_IP --main_process_port 29591 sft/train_sft.py ..."` — identical command on both nodes; only `$SLURM_NODEID` (0/1) differs. Each node spawns 8 ranks; RANK = machine_rank*8+local. NCCL over IB (`NCCL_SOCKET_IFNAME=^lo,docker,veth`).
4. All ranks write; rank 0 (`RANK==0 and LOCAL_RANK==0`) saves tokenizer and prints.

### (ii) GRPO with external vLLM servers, 2 nodes (conductor idiom)
1. `NUM_NODES=2 ... bash scripts/submit.sh` → `sbatch -N 2 --export=ALL -t 72:00:00 [-J ..] [--exclude ..] scripts/train_conductor.sbatch`.
2. Batch script (node 0 host shell, no env): `scontrol show hostnames` → `placement.py --emit env/endpoints/table` → writes `outputs/<run>/worker_endpoints.json`, `placement.txt`.
3. `run_step node0 GPU0 precache.log "huggingface-cli download ..."` (sequential, blocking).
4. For each server i: background `srun --overlap --nodes=1 --ntasks=1 -w <node_i> --container-image ... bash -lc "export CUDA_VISIBLE_DEVICES='<gpu>'; <FWD_ENV> source venv; cd REPO; VLLM_PORT=$((21000+i*100)) python -m vllm.entrypoints.openai.api_server --model ... --host 0.0.0.0 --port $((8100+i)) --tensor-parallel-size 1 ..."` with same-node stagger 20 s. In dp mode n=2: node0 GPUs 0-7 = 8 servers, node1 GPUs 0-3 = 4 servers.
5. Poll `curl http://<node>:<port>/v1/models` until all 12 ready (40 min cap; abort if all step PIDs dead).
6. `run_step node1 "4,5,6,7" train.log "accelerate launch --config_file outputs/<run>/accel.yaml --num_machines 1 --num_processes 4 --main_process_port 29500 train.py --config ... --override worker_endpoints_file=... --override vllm_tensor_parallel_size=1 --override gradient_accumulation_steps=8 --override per_device_train_batch_size=8 --override num_generations=64 --override reward_max_workers=128 --override output_dir=..."` → trl colocate: each of the 4 ranks builds its own tp=1 vLLM engine (`vllm_gpu_memory_utilization=0.3`) alongside the fp32 ZeRO-2 policy; the reward fn calls the 12 worker servers over HTTP with a 128-thread pool.
7. `trap cleanup EXIT` kills worker steps; exit with train rc.

For a **cross-node trainer** (mopd RL 2-node with the policy itself on 16 GPUs) combine (i)'s multi-node `accelerate launch --machine_rank \$SLURM_NODEID --main_process_ip HEAD_IP` with (ii)'s `run_step`-style background vLLM servers on a GPU subset — but note the conductor README explicitly lists "cross-node conductor (world = 2n) — needs server-mode vLLM or multi-node colocate; more moving parts" as *not implemented*; colocate + ZeRO-2 is the only tested policy path. The `--overlap` steps expose all 8 GPUs on a `--gres=gpu:8` allocation, so the trainer step must be pinned via `CUDA_VISIBLE_DEVICES` inside the payload exactly like the worker steps.

## Consolidated gotchas
- `readlink -f`/`realpath` the repo: `~/lmalign` → `/mnt/datafs/...`; compute nodes/pyxis mount only `/mnt/datafs` (`--container-mounts=/mnt/datafs:/mnt/datafs[,/tmp:/tmp]`). No `--container-writable` is used anywhere; the writable target is the NAS mount.
- Re-export env inside the `bash -lc` payload (pyxis does not reliably inherit `--export=ALL`).
- `CUDA_VISIBLE_DEVICES` must be set in the payload, never via `--export=ALL,CUDA_VISIBLE_DEVICES=4,5,6,7`.
- `NCCL_SOCKET_IFNAME=^lo,docker,veth`; never give gloo a `^` pattern.
- Unset `http(s)_proxy` for fully-offline jobs (container bakes brain-proxy; short hostnames not in NO_PROXY). `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$REPO/.hf_cache`.
- HF downloads: pre-cache sequentially in one process; `rm -rf $HF_HOME/hub/.locks` (NFS flock).
- Ports: worker HTTP `8100+i`; worker `VLLM_PORT=21000+100*i`; eval per-GPU `VLLM_PORT=40000+128*(gpu%8)`; accelerate `29500` (conductor) / `29591` (mtm sft) / `29600+` (per-slot sub-steps). Co-located vLLM engines need distinct `VLLM_PORT`.
- vLLM on this box: `disable_custom_all_reduce=True` at tp>1; `VLLM_ATTENTION_BACKEND=FLASHINFER` for tp>=2 servers; `enforce_eager` for tp>=4 offline LLM.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` (+ old name) for colocate train/gen alternation.
- accelerate `gradient_accumulation_steps: auto` crashes — always sed a concrete int into a per-run yaml copy.
- ZeRO-2, not ZeRO-3, for colocate GRPO (trl weight sync needs unsharded params); ZeRO-3 needs `zero3_save_16bit_model: true` to produce loadable saves.
- GRPO policy fp32 (`model_init_kwargs={"dtype":"float32"}`) — bf16 Adam zeroes lr=1e-6 updates; `vllm_gpu_memory_utilization=0.3`.
- NCCL timeout: patch `init_process_group(timeout=2h)` + `GRPOConfig(ddp_timeout=...)` when rewards are slow.
- Always pass a finite `-t` (unlimited jobs cannot backfill). Job name via `-J` at submit time (`sean-...`). Submit with `--cpus-per-task=32 --mem=600G` per memory rule (CLI overrides the `#SBATCH` header).
- Eval-batch duplicate padding inflates trl eval metrics → dedupe transcripts by prompt.
- SFT completion sentinel written after `accelerate` returns 0 (`.mtm_branch_ok`), not `config.json`.
