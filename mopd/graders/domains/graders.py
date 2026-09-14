"""Domain graders — every scoring rule ports or imports OFFICIAL benchmark code.

Sources (vendored clones live in third_party/):
  mcqa           : exact letter match (MedQA/MedMCQA/MedXpertQA/CaseHOLD/MBE/LEXam
                   official convention; MedXpertQA repo eval + LEXam evaluation.py)
  pubmedqa       : pubmedqa/pubmedqa evaluation.py — accuracy_score + macro f1_score
                   over {yes,no,maybe}
  legalbench     : HazyResearch/legalbench evaluation.py — imported directly
                   (balanced accuracy over normalized exact match + task specials)
  fin numeric    : yale-nlp/DocMath-Eval utils/evaluation_utils.py compare_two_numbers
                   (0.15% relative tolerance + percent/scale rescue) — imported directly.
                   FinQA official evaluate.py grades DSL programs; for free-text models the
                   community convention (Fin-R1/Fino1) is final-answer accuracy — we use
                   the DocMath comparator as the tolerant standard and document it.
  tatqa          : NExTplusplus/TAT-QA tatqa_metric.py TaTQAEmAndF1 — imported directly.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from mopd.graders.domains.extract import (extract_letter, extract_number,
                                          extract_text_answer, extract_yesno_maybe,
                                          strip_thinking)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TP = os.environ.get("DOMAINS_TP") or os.path.normpath(os.path.join(_HERE, "..", "..", "..", "third_party"))  # module moved one level deeper (mopd/graders/domains)


def _tp(*parts: str) -> str:
    return os.path.join(_TP, *parts)


# ------------------------------------------------------------------ MCQA (letter)
def grade_mcqa(response: str, gold_letter: str, n_options: int = 4) -> Dict[str, Any]:
    pred = extract_letter(response, n_options=n_options)
    return {"pred": pred, "gold": gold_letter.upper(),
            "correct": pred is not None and pred == gold_letter.upper()}


# ------------------------------------------------------------------ PubMedQA
def grade_pubmedqa_batch(responses: Sequence[str], golds: Sequence[str]) -> Dict[str, Any]:
    """Official pubmedqa/evaluation.py: accuracy_score + f1_score(average='macro')."""
    from sklearn.metrics import accuracy_score, f1_score
    preds = [extract_yesno_maybe(r) or "maybe" for r in responses]
    golds = [g.lower() for g in golds]
    return {"accuracy": float(accuracy_score(golds, preds)),
            "macro_f1": float(f1_score(golds, preds, average="macro")),
            "preds": preds}


# ------------------------------------------------------------------ LegalBench
_LB = None


def _legalbench():
    global _LB
    if _LB is None:
        sys.path.insert(0, _tp("legalbench"))
        import evaluation as lb_eval  # official evaluation.py
        _LB = lb_eval
    return _LB


LEGALBENCH_EXCLUDE = {"rule_qa"}  # official: manual evaluation only


def grade_legalbench_task(task: str, responses: Sequence[str],
                          answers: Sequence[str]) -> Dict[str, Any]:
    """Official evaluate(task, generations, answers). Generations are the post-think
    final answers (the official prompts expect the bare label)."""
    lb = _legalbench()
    gens = [extract_text_answer(r) for r in responses]
    score = lb.evaluate(task, gens, list(answers))
    return {"task": task, "score": float(score), "n": len(answers)}


# ------------------------------------------------------------------ finance numeric
# VENDORED VERBATIM from yale-nlp/DocMath-Eval utils/evaluation_utils.py (MIT).
# (Direct import pulls the repo's OpenAI client deps; only the comparator is needed.)
import math


def _dm_within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.0015
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def _dm_round_up_to_decimal(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def _dm_compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        raise ValueError(p)

    v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
    if (v1 != 0 and v2 != 0) and int(math.log10(v1 / v2)) == math.log10(v1 / v2):
        return True

    if v2 <= v1 / 50 and _dm_within_eps(pred=v2 * 100, gt=v1):
        return True
    elif v2 <= v1 / 500 and _dm_within_eps(pred=v2 * 1000, gt=v1):
        return True
    elif v2 <= v1 / 50000 and _dm_within_eps(pred=v2 * 100000, gt=v1):
        return True

    if _dm_round_up_to_decimal(v1, 3) == _dm_round_up_to_decimal(v2, 3):
        return True

    return _dm_within_eps(pred=p, gt=gt)


def fin_numbers_equal(pred: float, gold: float) -> bool:
    """Official DocMath compare_two_numbers: 0.15% relative tolerance + power-of-10 /
    percent-scale rescues + 3-decimal rounding fallback (vendored above)."""
    return bool(_dm_compare_two_numbers(pred, gold))


def grade_fin_numeric(response: str, gold: Any) -> Dict[str, Any]:
    gold_s = str(gold).strip().lower()
    if gold_s in ("yes", "no"):  # FinQA has yes/no golds (official special-case)
        pred = extract_yesno_maybe(response, allow_maybe=False)
        return {"pred": pred, "gold": gold_s, "correct": pred == gold_s}
    from mopd.graders.domains.extract import _clean_number
    gv = _clean_number(gold_s)
    if gv is None:
        pred_t = extract_text_answer(response).lower()
        return {"pred": pred_t, "gold": gold_s, "correct": pred_t == gold_s}
    pv = extract_number(response)
    ok = pv is not None and fin_numbers_equal(pv, gv)
    return {"pred": pv, "gold": gv, "correct": bool(ok)}


# ------------------------------------------------------------------ TAT-QA
_TATQA = None


def _tatqa():
    global _TATQA
    if _TATQA is None:
        sys.path.insert(0, _tp("TAT-QA"))
        from tatqa_metric import TaTQAEmAndF1  # official
        _TATQA = TaTQAEmAndF1
    return _TATQA


def grade_tatqa_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """items: [{response, gold_answer, gold_scale, answer_type}].
    Uses the official TaTQAEmAndF1 metric (EM + numeracy-focused F1). The model's
    answer is taken as [extracted number] for arithmetic/count golds and the raw
    final line for span golds; scale is extracted from the final line if present."""
    metric_cls = _tatqa()
    metric = metric_cls()
    import re as _re
    for it in items:
        resp = it["response"]
        at = it.get("answer_type", "arithmetic")
        if at in ("arithmetic", "count"):
            v = extract_number(resp)
            pred_ans: Any = [str(v)] if v is not None else [""]
        else:
            pred_ans = [extract_text_answer(resp)]
        tail = strip_thinking(resp)[-200:].lower()
        m = _re.search(r"\b(thousand|million|billion|percent)\b", tail)
        pred_scale = m.group(1) if m else it.get("pred_scale", "")
        gt = {"answer": it["gold_answer"], "answer_type": at,
              "scale": it.get("gold_scale", ""), "answer_from": "table-text",
              "uid": it.get("uid", "x"), "order": 1, "question": ""}
        metric(gt, pred_ans, pred_scale)
    em, f1, scale_score, _op = metric.get_overall_metric()
    return {"em": float(em), "f1": float(f1), "scale_acc": float(scale_score),
            "n": len(items)}


def grade_ihc_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """items: [{response, grader_code, attack, task_type}] -> accuracy (hierarchy held) + per-task-type accuracies.
    Uses mopd.graders.domains.ihc_grader (the row's own grader on the post-</think> answer, 5 s timeout)."""
    from mopd.graders.domains.ihc_grader import grade_batch
    rewards = grade_batch(items)
    per: Dict[str, List[float]] = {}
    for it, r in zip(items, rewards):
        per.setdefault(it.get("task_type", "?"), []).append(r)
    return {"accuracy": sum(rewards) / max(1, len(rewards)),
            "by_task_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(per.items())},
            "n_task_types": len(per)}
