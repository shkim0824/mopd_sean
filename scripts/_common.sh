#!/bin/bash
# Shared launcher pieces for every mopd_sean sbatch / helper script (sourced, not executed).
# Cluster idioms (see docs/INFRA.md): repo path must be the /mnt/datafs realpath; env vars are re-exported INSIDE the
# container payload; CUDA_VISIBLE_DEVICES is set inside the payload; proxies unset; NCCL_SOCKET_IFNAME=^lo,docker,veth.
set -uo pipefail

# --- repo root: MOPD_REPO > SLURM_SUBMIT_DIR (sbatch from the repo root) > the directory above this file (helper scripts)
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(realpath "${MOPD_REPO:-${SLURM_SUBMIT_DIR:-${_here}/..}}")"
[[ -f "${REPO}/scripts/_common.sh" ]] || { echo "[common] REPO=${REPO} is not a mopd_sean checkout (set MOPD_REPO)"; exit 2; }
S="${LMALIGN:-/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean}"; LMALIGN="${S}"

# --- runtime: container images + venv (shared; not part of the repo)
IMG="${MOPD_IMAGE:-/mnt/datafs/ib-a100-cluster-a-pri/lmt/users/jay/images/ngc-2410_verl-lmt-251130_verl-0.6.0_vllm-0.10.2_flash-attn-2.8.3.sqsh}"
NEMO_IMG="${NEMO_IMAGE:-${S}/images/nemo-rl-v0.7.0.sqsh}"
VENV="${MOPD_VENV:-${S}/.venv/mopd/bin/activate}"
MOUNT="${MOPD_MOUNT:-/tmp:/tmp,/mnt/datafs:/mnt/datafs}"
HF_HOME_DIR="${HF_HOME:-${S}/.hf_cache}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker,veth}"
if [[ "${GLOO_SOCKET_IFNAME:-}" == *"^"* ]]; then unset GLOO_SOCKET_IFNAME; fi

# --- environment forwarded into every container payload (one PYTHONPATH = the repo root: package `mopd`, `third_party`)
FWD_ENV="export HF_HOME='${HF_HOME_DIR}'; export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; \
export TOKENIZERS_PARALLELISM=false; export PYTORCH_ALLOC_CONF=expandable_segments:True; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'; \
export NLTK_DATA='${REPO}/env/nltk_data'; export MOPD_DATA_DIR='${REPO}/data'; export PYTHONPATH='${REPO}'; \
export DOMAINS_DATA='${REPO}/data/domains'; export DOMAINS_TP='${REPO}/third_party'; \
export VLLM_LOGGING_LEVEL=WARNING; export MOPD_REPO='${REPO}'; \
export VLLM_NO_USAGE_STATS=1; export VLLM_DO_NOT_TRACK=1; export VLLM_CONFIG_ROOT='${REPO}/tmp/cache/xdg_config/vllm'; \
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY;"
# 2026-09-19: the container mounts the node-local home of the SHARED deploy account; keep vLLM's usage-stats/config files out of it
PRELUDE="source '${VENV}'; export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_CONFIG_ROOT='${REPO}/tmp/cache/xdg_config/vllm';"

# run_step <node> <cuda_visible_devices> <logfile> <cmd>   (one --overlap srun step, containerised)
run_step() {
  local node="$1" cvd="$2" logf="$3" cmd="$4"
  local payload="export CUDA_VISIBLE_DEVICES='${cvd}'; ${FWD_ENV} ${PRELUDE} cd '${REPO}'; echo \"[step] host=\$(hostname) CVD=\$CUDA_VISIBLE_DEVICES\"; ${cmd}"
  srun --overlap --nodes=1 --ntasks=1 -w "${node}" --export=ALL \
    --container-image="${IMG}" --container-mounts="${MOUNT}" --container-workdir="${REPO}" \
    bash -lc "${payload}" >"${logf}" 2>&1
}

# accel_cfg <grad_accum> <out_dir> -> path of a per-job accelerate yaml with the concrete GA
accel_cfg() {
  local ga="$1" wd="$2"; mkdir -p "${wd}"
  local cfg="${wd}/accel_${SLURM_JOB_ID:-run}.yaml"
  sed -e "s/^\(  gradient_accumulation_steps:\).*/\1 ${ga}/" "${REPO}/configs/accelerate_zero2.yaml" > "${cfg}"
  echo "${cfg}"
}

head_ip() {
  local head; head="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
  local ip; ip="$(getent hosts "${head}" | awk '{print $1}' | head -1)"
  echo "${ip:-${head}}"
}

# wait_http <url...>  (all must answer /v1/models); WAIT_TIMEOUT seconds
wait_http() {
  local start=${SECONDS} t="${WAIT_TIMEOUT:-2400}"
  while true; do
    local ready=0
    # --noproxy: node-to-node probes must never be routed to the egress proxy, whatever the submitting shell exported
    for u in "$@"; do curl -fsS --noproxy "*" "${u}/v1/models" >/dev/null 2>&1 && ready=$((ready + 1)); done
    [[ "${ready}" -ge $# ]] && return 0
    (( SECONDS - start >= t )) && { echo "[wait] timeout: ${ready}/$# ready" >&2; return 1; }
    sleep 10
  done
}

# ---- node placement rule (user 2026-09-14): three tiers, checked in order
#   1. MINE  = ib-a100-cluster-a-n[227-236] (assigned to sean; anything else running there may be cancelled by hand)
#              -> if >= n idle nodes there: pin the job to that range, job prefix "sean-"
#   2. RANGE = ib-a100-cluster-a-n[041-066]  -> if >= n idle nodes there: pin to that range, prefix "k-sean-" (never cancel others)
#   3. otherwise: no node exclusion at all (whole cluster), prefix "k-sean-"
# placement <n_nodes> sets PLACE_OPT (sbatch --exclude option, may be empty) and JP (job prefix) in the CALLER's shell:
#   placement 4; sbatch ${PLACE_OPT} -J "${JP}-opd-<run>" ...        (do not use $(...): a subshell cannot set JP)
# place <n_nodes> only prints the option (JP unchanged) for one-liners that set the prefix themselves.
MINE_RANGE="ib-a100-cluster-a-n[227-236]"; MINE_EXCL="ib-a100-cluster-a-n[001-226,237-258]"
NEVER_EXCL="ib-a100-cluster-a-n[033-036]"      # user rule 2026-09-15: never submit to n033-036 (every tier, every launcher)
KILL_ZONE="ib-a100-cluster-a-n[001-040,089-128]"       # operational (2026-09-15): other teams reclaim these nodes by cancelling k- jobs
RANGE_41_66="ib-a100-cluster-a-n[041-066]"; RANGE_EXCL="ib-a100-cluster-a-n[001-040,067-258]"
idle_in() { local v; v=$(sinfo -h -n "$1" -t idle -o "%D" 2>/dev/null | head -1); echo "${v:-0}"; }
pinned_pending() {  # pinned_pending [excl-list]: nodes requested by my PENDING jobs already pinned with that exclusion (default: the assigned range)
  local want="${1:-${MINE_EXCL}}" tot=0 j
  for j in $(squeue -h -t PD -o "%i %j" 2>/dev/null | awk '$2 ~ /^(k-)?sean-/ {print $1}'); do
    if scontrol show job "$j" 2>/dev/null | grep -qF "ExcNodeList=${want}"; then
      tot=$(( tot + $(scontrol show job "$j" | grep -oE "NumNodes=[0-9]+" | head -1 | cut -d= -f2) )); fi
  done; echo "${tot}"
}
placement() {
  local n="${1:-1}" mine range
  mine=$(( $(idle_in "${MINE_RANGE}") - $(pinned_pending "${MINE_EXCL}") )); range=$(( $(idle_in "${RANGE_41_66}") - $(pinned_pending "${RANGE_EXCL}") ))
  if [[ "${mine}" -ge "${n}" ]]; then JP="sean"; PLACE_OPT="--exclude=${MINE_EXCL}"; PLACE_TIER="mine(n227-236)"
  elif [[ "${range}" -ge "${n}" ]]; then JP="k-sean"; PLACE_OPT="--exclude=${RANGE_EXCL}"; PLACE_TIER="n041-066"
  else JP="k-sean"; PLACE_OPT="--exclude=${NEVER_EXCL}"; PLACE_TIER="whole-cluster"; fi
}
place() { placement "$1"; echo "${PLACE_OPT}"; }
# excl_union <list>...: one Slurm node expression covering all given lists (empty lists ignored)
excl_union() { local h; h=$(for l in "$@"; do [[ -n "$l" ]] && scontrol show hostnames "$l"; done | sort -u | paste -sd,); [[ -n "$h" ]] && scontrol show hostlist "$h"; }

# prefer <n_nodes>: kept for old call sites = tier 2/3 only (never places on the assigned range)
prefer() {
  local idle; idle=$(( $(idle_in "${RANGE_41_66}") - $(pinned_pending "${RANGE_EXCL}") ))
  if [[ "${idle}" -ge "$1" ]]; then echo "--exclude=${RANGE_EXCL}"; else echo "--exclude=${NEVER_EXCL}"; fi
}

# job-name prefix (cluster rule: k-sean-* for jobs that may land outside n041-066)
JP="${JOB_PREFIX:-k-sean}"   # overwritten by place()

# per-job compile caches for vLLM workers (shared /tmp inductor caches got corrupted by concurrent jobs); they pile up
# under tmp/cache -> tools/disk/clean_caches.sh removes those of finished jobs
cache_env() {  # cache_env <rank>  -> export lines for one vLLM worker
  local r="$1" c="${REPO}/tmp/cache"
  echo "export VLLM_CACHE_ROOT=${c}/vllm_${SLURM_JOB_ID:-x}_${r} TORCHINDUCTOR_CACHE_DIR=${c}/ind_${SLURM_JOB_ID:-x}_${r} TRITON_CACHE_DIR=${c}/triton_${SLURM_JOB_ID:-x}_${r} XDG_CACHE_HOME=${c}/xdg;"
}
