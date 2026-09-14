#!/bin/bash
# Self-check of a mopd_sean checkout (run after moving/editing the tree): inside the training container + mopd venv,
# (1) import every mopd.* module (the NeMo-RL side is imported inside the NeMo container in a second step),
# (2) run the CPU tests under tests/, (3) validate every config path (scripts/cfgval.py --check-paths).
#   sbatch -J k-sean-selfcheck --exclude=ib-a100-cluster-a-n[001-040] scripts/selfcheck.sbatch      (from the repo root)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 1:00:00
#SBATCH --output=logs/selfcheck-%j.log
MOPD_REPO="${MOPD_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"; source "${MOPD_REPO}/scripts/_common.sh"
cd "${REPO}"; mkdir -p logs tmp
NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
echo "[selfcheck] repo=${REPO} node=${NODE} start=$(date '+%F %T')"

# ---- step 1+2: imports + CPU tests (training container, mopd venv)
run_step "${NODE}" "0" "logs/selfcheck-${SLURM_JOB_ID}-mopd.log" "python - <<'PY'
import importlib, pkgutil, sys, time, traceback
import mopd
skip_prefix = ('mopd.rl.nemo',)            # needs the NeMo-RL container (checked in step 3)
ok, bad = [], []
for m in pkgutil.walk_packages(mopd.__path__, 'mopd.'):
    if m.name.startswith(skip_prefix): continue
    t = time.time()
    try:
        importlib.import_module(m.name); ok.append((m.name, time.time() - t))
    except Exception as e:
        bad.append((m.name, repr(e))); traceback.print_exc()
print(f'[imports] ok={len(ok)} failed={len(bad)}')
for n, dt in sorted(ok, key=lambda x: -x[1])[:5]: print(f'  slowest {n} {dt:.1f}s')
for n, e in bad: print(f'  FAILED {n}: {e}')
sys.exit(1 if bad else 0)
PY
rc_imp=\$?; echo \"[selfcheck] imports rc=\$rc_imp\"
rc_t=0
for t in tests/test_*.py; do
  s=\$(date +%s); python \"\$t\" > logs/selfcheck-${SLURM_JOB_ID}-\$(basename \"\$t\" .py).log 2>&1; r=\$?
  echo \"[test] \$t rc=\$r \$(( \$(date +%s) - s ))s\"; (( r != 0 )) && rc_t=1
done
python3 scripts/cfgval.py --check-paths; rc_c=\$?; echo \"[selfcheck] config paths rc=\$rc_c\"
exit \$(( rc_imp + rc_t + rc_c ))"
rc1=$?; cat "logs/selfcheck-${SLURM_JOB_ID}-mopd.log"

# ---- step 3: NeMo-RL side (envs + entrypoint import inside the NeMo container; run_grpo itself only registers)
srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" --export=ALL --container-image="${NEMO_IMG}" --container-mounts="${MOUNT}" \
  --container-workdir="${REPO}" bash -lc "export PYTHONPATH='${REPO}'; cd '${REPO}'; /opt/nemo_rl_venv/bin/python - <<'PY'
import importlib, sys, traceback
mods = ['mopd.rl.nemo.envs.processors', 'mopd.rl.nemo.envs.exactmatch_environment', 'mopd.rl.nemo.envs.code_environment',
        'mopd.rl.nemo.envs.if_environment', 'mopd.rl.nemo.envs.if_verify_worker', 'mopd.rl.nemo.envs.math_with_judge_environment',
        'mopd.rl.rewards', 'mopd.graders.code_grader', 'mopd.graders.if_grader']
bad = []
for m in mods:
    try: importlib.import_module(m); print('[nemo-import] ok', m)
    except Exception as e: bad.append(m); print('[nemo-import] FAILED', m, repr(e)); traceback.print_exc()
import nemo_rl, ray, torch, transformers
print(f'[nemo] nemo_rl={getattr(nemo_rl, \"__version__\", \"?\")} ray={ray.__version__} torch={torch.__version__} transformers={transformers.__version__} (vLLM lives in the ray worker venv, not checked here)')
sys.exit(1 if bad else 0)
PY" > "logs/selfcheck-${SLURM_JOB_ID}-nemo.log" 2>&1
rc2=$?; cat "logs/selfcheck-${SLURM_JOB_ID}-nemo.log" | grep -v "^$" | tail -15
echo "[selfcheck] mopd rc=${rc1} nemo rc=${rc2} end=$(date '+%F %T')"; exit $(( rc1 + rc2 ))
