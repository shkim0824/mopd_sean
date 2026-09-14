# Instruction-following verify worker: a PURE ray actor with NO nemo_rl imports, so it can
# run under a DIFFERENT python (.venv/mopd) than the NeMo-RL env actor that spawns it —
# verifiable_instructions' deps (nltk+punkt_tab data, langdetect, immutabledict) live there,
# not in the locked nemo venv. Ray versions match exactly (2.55.1 both sides; verified).
from __future__ import annotations

from typing import Any, Optional

import ray


def _strip_thinking(text: str) -> str:
    # Constraints apply to the final answer, not the reasoning trace (mopd convention;
    # a reasoning model's <think> block would trivially violate e.g. punctuation:no_comma).
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


@ray.remote  # pragma: no cover
class IFVerifyWorker:
    """Checks responses against verifiable-instructions constraints (official NVIDIA
    registry, github.com/abukharin-nv/verifiable-instructions — the exact verifier the
    Nemotron blends' instruction_following rows were built for). Binary grading_mode
    (the rlvr1 slice carries no grading_mode field -> official default: ALL constraints
    must pass)."""

    def __init__(self) -> None:
        from verifiable_instructions import instructions_registry
        self._registry = instructions_registry.INSTRUCTION_DICT

    def verify(self, items: list[dict[str, Any]]) -> list[float]:
        """items: [{response, instruction_id_list, kwargs}] -> [0.0|1.0]"""
        out: list[float] = []
        for it in items:
            response = _strip_thinking(str(it.get("response") or "")).strip()
            ids = it.get("instruction_id_list") or []
            kwargs = it.get("kwargs") or [None] * len(ids)
            try:
                ok = True
                for iid, kw in zip(ids, kwargs):
                    cls = self._registry.get(iid)
                    if cls is None:  # unsupported id -> conservative fail (should not happen; 47/47 covered)
                        ok = False
                        break
                    inst = cls(iid)
                    inst.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
                    args = inst.get_instruction_args()
                    if args and "prompt" in args:
                        inst.build_description(prompt=it.get("prompt") or "")
                    if not response or not inst.check_following(response):
                        ok = False
                        break
                out.append(1.0 if ok else 0.0)
            except Exception:
                out.append(0.0)
        return out
