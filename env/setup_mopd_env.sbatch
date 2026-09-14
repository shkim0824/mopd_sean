#!/bin/bash
# =============================================================================
# Build the MOPD runtime venv (.venv/mopd) — one-time, ~15 min with the warm uv cache.
#   = mtm venv pins (torch 2.9.0+cu128 / vLLM 0.12 / trl 0.29 / transformers 4.57.6 / deepspeed 0.18.8)
#     + grader extras (nltk, langdetect, immutabledict, absl-py, emoji, syllapy)
#   + stages the OFFLINE nltk assets into env/nltk_data (punkt, punkt_tab, stopwords, tagger).
# The login node has no PyPI: install INSIDE the ngc-2410 container on a compute node (its baked-in
# proxy reaches PyPI). uv cache/python dirs on NAS so bin/python does not dangle after the container exits.
#
#   cd ~/lmalign/personal/sean/mopd && sbatch -J sean-mopd-env-setup --cpus-per-task=32 --mem=200G \
#        --exclude=ib-a100-cluster-a-n[001-040,067-258] env/setup_mopd_env.sbatch
# =============================================================================
#SBATCH -J sean-mopd-env-setup
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH --output=logs/setup-mopd-env-%j.log
set -euo pipefail
REPO="$(realpath "${MOPD_REPO:-$(pwd)}")"; cd "${REPO}"; mkdir -p logs env/nltk_data
LMALIGN="/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean"
BASE_IMAGE="${BASE_IMAGE:-/mnt/datafs/ib-a100-cluster-a-pri/lmt/users/jay/images/ngc-2410_verl-lmt-251130_verl-0.6.0_vllm-0.10.2_flash-attn-2.8.3.sqsh}"
VENV_DIR="${MOPD_VENV_DIR:-${LMALIGN}/.venv/mopd}"
PYVER="${PYVER:-3.13}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${LMALIGN}/.venv/.uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${LMALIGN}/.venv/.uv_python}"
export HF_HOME="${HF_HOME:-${LMALIGN}/.hf_cache}"
echo "[setup] image=${BASE_IMAGE} venv=${VENV_DIR}"
srun --nodes=1 --ntasks=1 --container-image="${BASE_IMAGE}" --container-mounts=/tmp:/tmp,/mnt/datafs:/mnt/datafs \
  --container-workdir="${REPO}" bash -lc "
    set -euo pipefail
    export UV_CACHE_DIR='${UV_CACHE_DIR}' UV_PYTHON_INSTALL_DIR='${UV_PYTHON_INSTALL_DIR}' UV_LINK_MODE=copy HF_HOME='${HF_HOME}'
    mkdir -p \"\$UV_CACHE_DIR\" \"\$UV_PYTHON_INSTALL_DIR\"
    python -m pip install -U uv 2>/dev/null || pip install -U uv
    rm -rf '${VENV_DIR}'
    uv venv --python '${PYVER}' '${VENV_DIR}'
    source '${VENV_DIR}/bin/activate'
    uv pip install --index-strategy unsafe-best-match -r env/requirements.txt
    uv pip install --index-strategy unsafe-best-match -r env/requirements-eval.txt
    python - <<'PY'
import torch, vllm, trl, transformers, accelerate, datasets, deepspeed
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('vllm', vllm.__version__, 'trl', trl.__version__, 'transformers', transformers.__version__, 'deepspeed', deepspeed.__version__)
from math_verify import parse, verify; print('math_verify', verify(parse('\$72\$'), parse('\$72\$')))
import os, nltk, langdetect, immutabledict, absl, emoji, syllapy
# nltk >= 3.10 refuses downloads through a proxy unless opted in; env/nltk_data is normally pre-staged
# from a laptop (tokenizers/punkt_tab, corpora/stopwords, taggers/averaged_perceptron_tagger_eng).
os.environ['NLTK_ALLOW_PROXIED_URLOPEN'] = '1'
nltk.data.path.insert(0, 'env/nltk_data')
for pkg, probe in (('punkt', 'tokenizers/punkt'), ('punkt_tab', 'tokenizers/punkt_tab'), ('stopwords', 'corpora/stopwords'),
                   ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng')):
    try:
        nltk.data.find(probe); print('nltk', pkg, 'present')
    except LookupError:
        ok = nltk.download(pkg, download_dir='env/nltk_data', quiet=True); print('nltk', pkg, 'downloaded' if ok else 'FAILED')
PY
    export NLTK_DATA='${REPO}/env/nltk_data'
    python tests/test_graders_quick.py
    python tests/test_distill_cpu.py
    echo '[setup] DONE'
  "
echo "[setup] venv ready: ${VENV_DIR}  (export MOPD_VENV=${VENV_DIR}/bin/activate)"
