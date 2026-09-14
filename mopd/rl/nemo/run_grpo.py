# GRPO entrypoint for NeMo-RL v0.7.0 (teacher training): registers every mopd environment + data processor, then
# delegates to the stock examples/run_grpo.py (same CLI: --config <yaml> [key=value overrides]).
#   domain_exactmatch : med/law/fin pools, pure machine verification (letter / yes-no / numeric)   (mopd.rl.nemo.envs.exactmatch_environment)
#   mopd_if           : verifiable-instructions registry (IF)                                    (envs.if_environment + if_verify_worker)
#   mopd_code         : LiveCodeBench execution semantics via mopd.graders.code_grader (code)    (envs.code_environment)
#   math_with_judge   : math with an LLM judge (needs a judge server; scripts/rl_nemo.sh JUDGE=1)  (envs.math_with_judge_environment)
#   mopd_ihc          : IH-Challenge safety, per-row Python grader (envs.ihc_environment; scripts/rl_ihc_loop.sh)
# PYTHONPATH must contain the repo root (package `mopd`) on ALL ray nodes; scripts/rl_nemo.sh exports it before `ray start`.
import os
import runpy
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

from nemo_rl.data.processors import register_processor
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.utils import ENV_REGISTRY

from mopd.rl.nemo.envs.processors import mopd_code_processor, mopd_if_processor, mopd_ihc_processor

_ENVS = {
    "domain_exactmatch": "mopd.rl.nemo.envs.exactmatch_environment.DomainExactMatchEnvironment",
    "math_with_judge": "mopd.rl.nemo.envs.math_with_judge_environment.MathWithJudgeEnvironment",
    "mopd_code": "mopd.rl.nemo.envs.code_environment.MopdCodeGenEnvironment",
    "mopd_if": "mopd.rl.nemo.envs.if_environment.MopdIFEnvironment",
    "mopd_ihc": "mopd.rl.nemo.envs.ihc_environment.MopdIHCEnvironment",
}
for name, fqn in _ENVS.items():
    ENV_REGISTRY[name] = {"actor_class_fqn": fqn}
    ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM

register_processor("mopd_code_processor", mopd_code_processor)
register_processor("mopd_if_processor", mopd_if_processor)
register_processor("mopd_ihc_processor", mopd_ihc_processor)

NEMO_RL_REPO = os.environ.get("NEMO_RL_REPO", "/opt/nemo-rl")
runpy.run_path(os.path.join(NEMO_RL_REPO, "examples", "run_grpo.py"), run_name="__main__")
