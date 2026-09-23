"""ONE registry for every benchmark mopd evaluates, with each one's own protocol.

Before phase 3 a checkpoint needed two Slurm jobs and two rounds of vLLM start-ups: the
in-domain job (``mopd.eval.domains.run_domain_eval``, T=1.0, 26624 tokens) and the 6-bench
job (``mopd.eval.run_eval``, qwen-thinking, 32768).  The four OOD additions (GPQA + the
three safety sets) would have made it four.  This module puts all of them behind one
interface so ``mopd.eval.run_all`` can serve them from a single engine per GPU:

    Spec(name, group, request, load, build, score, n, max_tokens, sampling, thinking, mode)

``request`` is ``gen`` (sample n completions) or ``lp`` (score fixed continuations).
``sampling`` is per-benchmark on purpose — mixing the in-domain T=1.0 protocol with the
32k qwen-thinking OOD protocol in one job is only sound if every request keeps the params
its published numbers were produced with.

The wrapped registries are NOT modified: ``mopd.eval.benchmarks`` (6-bench) and
``mopd.eval.domains.benches`` (domain) stay exactly as the two legacy runners use them, so
jobs already in flight are unaffected.  The domain grading dispatch below mirrors
``run_domain_eval.aggregate`` one-to-one; ``tmp/verify_unified_grading.py`` re-grades old
eval directories through both paths and asserts identical metrics.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

# ----------------------------------------------------------------------------- sampling
#: THE ONLY sampling configuration in this repo: identical to the training rollouts.
#: configs/{opd,mopd}/*.yaml `train:` gives temperature 1.0 and top_p 1.0, and sets no top_k,
#: so trl 0.29's GRPOConfig default top_k=0 (= disabled) applies; min_p is unset and
#: repetition_penalty is 1.0. vLLM treats top_k 0 and -1 identically ("top_k must be 0
#: (disable), or at least 1"). Every generative benchmark -- in-domain, IF, OOD -- samples
#: with exactly this. The log-prob benches (sycophancy, truthfulqa_mc) do no sampling at all.
#: User 2026-09-16: "학습과 평가 sampling param을 일치시키고 싶어" + keep no other preset.
EVAL_SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=-1, presence_penalty=0.0)

DOMAIN_MAX_TOKENS = 26624      # scripts/eval_domain.sbatch MAXTOK default
OOD_MAX_TOKENS = 32768         # scripts/eval_6bench.sbatch MAX_TOKENS default


class Spec:
    def __init__(self, name: str, group: str, request: str, load: Callable,
                 build: Callable, score: Callable, n: int = 1,
                 max_tokens: int = OOD_MAX_TOKENS, sampling: Optional[Dict] = None,
                 thinking: bool = True, mode: str = "chat", cost: float = 1.0):
        self.name, self.group, self.request = name, group, request
        self.load, self.build, self.score = load, build, score
        self.n, self.max_tokens = n, max_tokens
        self.sampling = dict(sampling or EVAL_SAMPLING)
        self.thinking, self.mode = thinking, mode
        self.cost = cost      # relative cost hint for the LPT scheduler

    def __repr__(self) -> str:
        return "Spec(%s, %s, %s)" % (self.name, self.group, self.request)


# ----------------------------------------------------------------------------- 6-bench wrap
def _wrap_6bench() -> Dict[str, Spec]:
    from mopd.eval.benchmarks import BENCHMARKS, DOMAIN_OF

    out: Dict[str, Spec] = {}
    for name, b in BENCHMARKS.items():
        group = DOMAIN_OF[name]           # math | code | if

        def _load(limit=None, _b=b, _n=name):
            # lcb_v6 keeps its optional date window arguments at their harness defaults
            return _b.load(limit=limit)

        def _build(row, _b=b):
            # messages(row, style="qwen", lcb_system=False) -> the legacy defaults
            return _b.messages(row, style="qwen", lcb_system=False)

        def _score(rows, texts, out_dir=None, _b=b, _g=group):
            if _b.kind == "code":
                res = _b.score(rows, texts, num_workers=None)
            else:
                res = _b.score(rows, texts)
            res.setdefault("higher_is_better", True)
            return res

        out[name] = Spec(name, group, "gen", _load, _build, _score,
                         n=b.default_n, max_tokens=OOD_MAX_TOKENS,
                         sampling=EVAL_SAMPLING, thinking=True, mode="chat",
                         cost=float(b.default_n))
    return out


# ----------------------------------------------------------------------------- domain wrap
def _domain_score(bench: str, rows: Sequence[Dict[str, Any]],
                  texts: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """MIRRORS mopd/eval/domains/run_domain_eval.py::aggregate (grade_kind dispatch).

    Same graders, same keys; ``score`` (percent) is added so every benchmark in the
    unified metrics.json has one headline number."""
    from mopd.graders.domains import graders as G

    gens = [t[0] for t in texts]
    kind = rows[0]["grade_kind"]
    scored: Dict[str, Any] = {"n": len(rows), "missing": 0}
    if kind == "mcqa":
        res = [G.grade_mcqa(g, r["gold"], r.get("n_options", 4)) for g, r in zip(gens, rows)]
        scored["accuracy"] = sum(r["correct"] for r in res) / len(res)
        scored["no_answer_frac"] = sum(r["pred"] is None for r in res) / len(res)
        head = "accuracy"
    elif kind == "pubmedqa":
        scored.update(G.grade_pubmedqa_batch(gens, [r["gold"] for r in rows]))
        scored.pop("preds", None)
        head = "accuracy"
    elif kind == "legalbench":
        per_task: Dict[str, List[int]] = defaultdict(list)
        for j, r in enumerate(rows):
            per_task[r["task"]].append(j)
        task_scores = []
        for task, js in per_task.items():
            t = G.grade_legalbench_task(task, [gens[j] for j in js],
                                        [rows[j]["gold"] for j in js])
            task_scores.append(t["score"])
        scored["balanced_acc_mean"] = sum(task_scores) / len(task_scores)
        scored["n_tasks"] = len(task_scores)
        head = "balanced_acc_mean"
    elif kind == "fin_numeric":
        res = [G.grade_fin_numeric(g, r["gold"]) for g, r in zip(gens, rows)]
        scored["accuracy"] = sum(r["correct"] for r in res) / len(res)
        head = "accuracy"
    elif kind == "ihc":
        items = [{"response": g, "grader_code": r["grader_code"], "attack": r.get("attack", ""),
                  "task_type": r.get("task_type", "?")} for g, r in zip(gens, rows)]
        scored.update(G.grade_ihc_batch(items))
        head = "accuracy" if "accuracy" in scored else "score"
    elif kind == "tatqa":
        items = [{"response": g, "gold_answer": r["gold"], "gold_scale": r.get("gold_scale", ""),
                  "answer_type": r.get("answer_type", "arithmetic"), "uid": r["id"]}
                 for g, r in zip(gens, rows)]
        scored.update(G.grade_tatqa_batch(items))
        head = "em" if "em" in scored else "accuracy"
    else:
        raise ValueError("unknown grade_kind %r for %s" % (kind, bench))
    v = scored.get(head)
    scored["metric"] = head
    scored["score"] = 100.0 * float(v) if isinstance(v, (int, float)) else None
    scored["higher_is_better"] = True
    return scored


def _wrap_domain() -> Dict[str, Spec]:
    from mopd.eval.domains.benches import BENCHES

    out: Dict[str, Spec] = {}
    for name, loader in BENCHES.items():
        def _load(limit=None, _l=loader, _n=name):
            rows = _l(limit=limit) if limit else _l()
            for r in rows:
                r["bench"] = _n
            return rows

        def _build(row):
            # run_domain_eval.worker: r["messages"] when present (ihc), else one user turn
            return row.get("messages") or [{"role": "user", "content": row["prompt"]}]

        def _score(rows, texts, out_dir=None, _n=name):
            return _domain_score(_n, rows, texts)

        out[name] = Spec(name, "domain", "gen", _load, _build, _score, n=1,
                         max_tokens=DOMAIN_MAX_TOKENS, sampling=EVAL_SAMPLING,
                         thinking=True, mode="chat", cost=0.8)
    return out


# ----------------------------------------------------------------------------- ood
def _wrap_ood() -> Dict[str, Spec]:
    from mopd.eval import ood_benches as OB
    from mopd.graders.ood.gpqa import score_gpqa
    from mopd.graders.ood.harmbench import score_harmbench
    from mopd.graders.ood.sycophancy import score_sycophancy
    from mopd.graders.ood.safety4 import (score_safetybench, score_sorrybench,
                                          score_strongreject, score_xstest)
    from mopd.graders.ood.truthfulqa import score_truthfulqa

    def _score_harmbench(rows, texts, out_dir=None):
        labels: Dict[str, List[int]] = {}
        p = os.path.join(out_dir or ".", "harmbench.labels.jsonl")
        if os.path.exists(p):
            from mopd.common.io import read_jsonl
            for r in read_jsonl(p):
                labels.setdefault(r["id"], []).append(int(r["label"]))
        return score_harmbench(rows, texts, labels)

    def _g(fn):
        return lambda rows, res, out_dir=None: fn(rows, res)

    def _g2(fn):        # graders that need the output dir (they read a judge's score file)
        return lambda rows, res, out_dir=None: fn(rows, res, out_dir)

    return {
        "gpqa": Spec("gpqa", "ood-mcq", "gen", OB.load_gpqa, OB.gpqa_messages, _g(score_gpqa),
                     n=4, max_tokens=OOD_MAX_TOKENS, sampling=EVAL_SAMPLING,
                     thinking=True, mode="chat", cost=4.0),
        # upstream HarmBench decodes greedily (do_sample=False); mopd runs it at the
        # suite-wide T=1.0 instead (user order 2026-09-16), which also avoids the endless
        # repetition the Qwen3 card warns about for greedy thinking-mode decoding. Measured
        # on the base model, 300 behaviors: greedy vs T=0.6 moved the no-final-answer rate
        # 5.3% -> 3.3%, so the protocol change costs nothing in answer completeness.
        "harmbench": Spec("harmbench", "safety", "gen", OB.load_harmbench,
                          OB.harmbench_messages, _score_harmbench, n=1,
                          max_tokens=OOD_MAX_TOKENS, sampling=EVAL_SAMPLING,
                          thinking=True, mode="chat", cost=1.0),
        "sycophancy": Spec("sycophancy", "safety", "lp", OB.load_sycophancy,
                           OB.sycophancy_specs, _g(score_sycophancy), n=1, max_tokens=1,
                           sampling=EVAL_SAMPLING, thinking=False, mode="chat", cost=0.02),
        "sorrybench": Spec("sorrybench", "safety", "gen", OB.load_sorrybench,
                           OB.sorrybench_messages, _g2(score_sorrybench), n=1,
                           max_tokens=OOD_MAX_TOKENS, sampling=EVAL_SAMPLING,
                           thinking=True, mode="chat", cost=1.0),
        "xstest": Spec("xstest", "safety", "gen", OB.load_xstest, OB.xstest_messages,
                       _g2(score_xstest), n=1, max_tokens=OOD_MAX_TOKENS,
                       sampling=EVAL_SAMPLING, thinking=True, mode="chat", cost=1.0),
        "strongreject": Spec("strongreject", "safety", "gen", OB.load_strongreject,
                             OB.strongreject_messages, _g2(score_strongreject), n=1,
                             max_tokens=OOD_MAX_TOKENS, sampling=EVAL_SAMPLING,
                             thinking=True, mode="chat", cost=1.0),
        "safetybench": Spec("safetybench", "safety", "gen", OB.load_safetybench,
                            OB.safetybench_messages, _g2(score_safetybench), n=1,
                            max_tokens=DOMAIN_MAX_TOKENS, sampling=EVAL_SAMPLING,
                            thinking=True, mode="chat", cost=0.6),
        "truthfulqa_mc": Spec("truthfulqa_mc", "safety", "lp", OB.load_truthfulqa,
                              OB.truthfulqa_specs, _g(score_truthfulqa), n=1, max_tokens=1,
                              sampling=EVAL_SAMPLING, thinking=False, mode="raw", cost=0.02),
    }


_REG: Optional[Dict[str, Spec]] = None


def registry() -> Dict[str, Spec]:
    global _REG
    if _REG is None:
        r: Dict[str, Spec] = {}
        r.update(_wrap_6bench())
        r.update(_wrap_domain())
        r.update(_wrap_ood())
        _REG = r
    return _REG


def spec(name: str) -> Spec:
    r = registry()
    if name not in r:
        raise SystemExit("unknown benchmark %r; known: %s" % (name, sorted(r)))
    return r[name]


#: the phase-3 full set: in-domain + IF + OOD(math, code, safety, GPQA), one job
FULL_SET = ["medqa", "casehold", "finqa", "ifeval", "ifbench",
            "aime24", "aime25", "aime26", "lcb_v6",
            "gpqa", "harmbench", "sycophancy", "truthfulqa_mc"]
OOD_NEW_SET = ["gpqa", "harmbench", "sycophancy", "truthfulqa_mc"]
#: phase 3-2: the widely-used safety benchmarks added on 2026-09-17. SORRY-Bench is absent
#: SORRY-Bench needs Hub access granted to the account (dataset AND judge are gated).
SAFETY_SET = ["xstest", "strongreject", "safetybench", "sorrybench"]
SAFETY_ALL = ["harmbench", "sycophancy", "truthfulqa_mc"] + SAFETY_SET
GROUP_OF = {"math": "math", "code": "code", "if": "if", "domain": "domain",
            "ood-mcq": "ood-mcq", "safety": "safety"}
