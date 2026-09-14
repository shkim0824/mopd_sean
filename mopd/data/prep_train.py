"""Build the TRAIN data (paper Appendix A, adapted to math / code / if):

SFT (Stage 1, one general corpus over all domains)
  sft_math : open-r1/Mixture-of-Thoughts  config ``math``  (93,733 R1 traces; paper: "math subset of MoT")
  sft_code : open-r1/Mixture-of-Thoughts  config ``code``  (83,070 R1 traces; our SWE->code substitution)
  sft_if   : prompts from allenai/IF_multi_constraints_upto5 (the IFBench-recipe IF-RLVR prompts),
             responses DISTILLED + verifier-filtered by ``mopd/data/distill_if_sft.py`` (paper: distilled on
             gpt-oss-120b) -> written by that script, not here.
  -> data/train/sft_{math,code}.jsonl   rows {id, domain, messages:[{user},{assistant}], source}
     assistant content normalised to '<think>\\n..\\n</think>\\n\\n..' (Qwen3 thinking format)

RL (Stage 2, per-domain verifiable prompts)
  rl_math  : SynthLabsAI/Big-Math-RL-Verified (251k) + Open-Reasoner-Zero orz_math_57k (paper: "BigMath and ORZ")
             filtered: parsable gold, no MCQ/proof, llama8b_solve_rate in [lo, hi] (drop trivially solved),
             decontaminated vs AIME24/25/26 (normalised-text + 13-gram)
             rows {id, domain:"math", prompt, answer, source, solve_rate}
  rl_code  : agentica-org/DeepCoder-Preview-Dataset configs ``taco`` + ``primeintellect`` (24k verified, >=5 tests)
             (config ``lcbv5`` and ``codeforces`` are EXCLUDED: both are LiveCodeBench/CodeElo problems inside the
             v6 window); decontaminated vs LCB v6 by normalised statement + 13-gram
             rows {id, domain:"code", prompt(question), starter_code, tests(input_output dict), source}
  rl_if    : allenai/IF_multi_constraints_upto5 (95,373; paper: "synthesised following the IFBench recipe")
             rows {id, domain:"if", prompt, ground_truth(str), source}
             (prompts also feed the IF SFT distillation; a disjoint split is carved: --if-sft-frac)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from typing import Any, Dict, Iterable, List, Set

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from mopd.common.chat import normalize_assistant  # noqa: E402
from mopd.common.io import iter_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from mopd.graders.math_grader import is_equiv, last_boxed  # noqa: E402


def _pq(path_glob: str, columns=None) -> Iterable[Dict[str, Any]]:
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(path_glob, recursive=True)):
        t = pq.read_table(f, columns=columns)
        for b in t.to_batches(max_chunksize=2048):
            yield from b.to_pylist()


def norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def ngrams(t: str, n: int = 13) -> Set[str]:
    w = re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


class Decontaminator:
    """exact normalised-text match OR any shared 13-gram with an eval problem."""

    def __init__(self, eval_texts: List[str], n: int = 13):
        self.exact = {norm_text(t) for t in eval_texts}
        self.grams: Set[str] = set()
        for t in eval_texts:
            self.grams |= ngrams(t, n)
        self.n = n

    def hit(self, text: str) -> bool:
        if norm_text(text) in self.exact:
            return True
        return bool(ngrams(text, self.n) & self.grams)


def rid(*parts) -> str:
    return hashlib.sha1("||".join(map(str, parts)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- SFT (MoT)
def build_sft_mot(hf: str, out: str, config: str, domain: str, max_rows: int | None, seed: int) -> Dict[str, Any]:
    rows = []
    n_bad = 0
    for r in _pq(os.path.join(hf, "open-r1__Mixture-of-Thoughts", config, "*.parquet")):
        msgs = r["messages"]
        if not (len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"):
            n_bad += 1
            continue
        a = msgs[1]["content"]
        if "</think>" not in a:
            n_bad += 1
            continue
        rows.append({"id": rid(config, msgs[0]["content"]), "domain": domain,
                     "messages": [{"role": "user", "content": msgs[0]["content"]},
                                  {"role": "assistant", "content": normalize_assistant(a)}],
                     "num_tokens": r.get("num_tokens"), "source": f"open-r1/Mixture-of-Thoughts:{config}"})
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    p = os.path.join(out, f"sft_{domain}.jsonl")
    n = write_jsonl(p, rows)
    return {"n": n, "dropped": n_bad, "sha256": sha256_file(p)}


# ----------------------------------------------------------------------------- RL math
_MCQ_RE = re.compile(r"\n\s*\(?[A-E][\).:]\s", re.I)


def build_rl_math(hf: str, out: str, eval_dir: str, lo: float, hi: float, max_rows: int | None, seed: int,
                  orz_json: str = "") -> Dict[str, Any]:
    eval_texts = []
    for name in ("aime24", "aime25", "aime26"):
        p = os.path.join(eval_dir, name, "test.jsonl")
        if os.path.exists(p):
            eval_texts += [r["problem"] for r in iter_jsonl(p)]
    dec = Decontaminator(eval_texts)
    rows, stats = [], {"bigmath_in": 0, "orz_in": 0, "drop_solve_rate": 0, "drop_unparsable": 0, "drop_mcq_proof": 0,
                       "drop_contaminated": 0, "dup": 0}
    seen: Set[str] = set()
    for r in _pq(os.path.join(hf, "SynthLabsAI__Big-Math-RL-Verified", "**", "*.parquet")):
        stats["bigmath_in"] += 1
        sr = r.get("llama8b_solve_rate")
        if sr is None or not (lo <= float(sr) <= hi):
            stats["drop_solve_rate"] += 1
            continue
        prob, ans = r["problem"], str(r["answer"]).strip()
        if _MCQ_RE.search(prob) or re.search(r"\b(prove|show that)\b", prob, re.I):
            stats["drop_mcq_proof"] += 1
            continue
        if not ans or not is_equiv(ans, ans):
            stats["drop_unparsable"] += 1
            continue
        k = norm_text(prob)
        if k in seen:
            stats["dup"] += 1
            continue
        if dec.hit(prob):
            stats["drop_contaminated"] += 1
            continue
        seen.add(k)
        rows.append({"id": rid("bigmath", prob), "domain": "math", "prompt": prob, "answer": ans,
                     "source": f"SynthLabsAI/Big-Math-RL-Verified:{r.get('source')}", "solve_rate": float(sr)})
    if orz_json and os.path.exists(orz_json):
        for item in json.load(open(orz_json)):
            stats["orz_in"] += 1
            prob = item[0]["value"]
            ans = str(item[1]["ground_truth"]["value"]).strip()
            if not ans or not is_equiv(ans, ans):
                stats["drop_unparsable"] += 1
                continue
            k = norm_text(prob)
            if k in seen:
                stats["dup"] += 1
                continue
            if dec.hit(prob):
                stats["drop_contaminated"] += 1
                continue
            seen.add(k)
            rows.append({"id": rid("orz", prob), "domain": "math", "prompt": prob, "answer": ans,
                         "source": "Open-Reasoner-Zero/orz_math_57k", "solve_rate": item[1]["ground_truth"].get("pass_at_n")})
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    p = os.path.join(out, "rl_math.jsonl")
    n = write_jsonl(p, rows)
    stats.update({"n": n, "sha256": sha256_file(p), "solve_rate_window": [lo, hi]})
    return stats


# ----------------------------------------------------------------------------- RL code
def build_rl_code(hf: str, out: str, eval_dir: str, max_rows: int | None, seed: int, min_tests: int = 5) -> Dict[str, Any]:
    from mopd.graders.code_grader import generic_tests_to_input_output

    lcb_path = os.path.join(eval_dir, "lcb_v6", "test.jsonl")
    eval_texts = [r["question_content"] for r in iter_jsonl(lcb_path)] if os.path.exists(lcb_path) else []
    dec = Decontaminator(eval_texts)
    rows, stats = [], {"taco_in": 0, "primeintellect_in": 0, "drop_tests": 0, "drop_contaminated": 0, "dup": 0, "drop_bad_tests": 0}
    seen: Set[str] = set()
    for cfg in ("taco", "primeintellect"):
        for r in _pq(os.path.join(hf, "agentica-org__DeepCoder-Preview-Dataset", cfg, "*.parquet"), columns=["problem", "tests"]):
            stats[f"{cfg}_in"] += 1
            prob = r["problem"]
            try:
                io = generic_tests_to_input_output(r["tests"])
            except Exception:
                stats["drop_bad_tests"] += 1
                continue
            if len(io["inputs"]) < min_tests or len(io["inputs"]) != len(io["outputs"]):
                stats["drop_tests"] += 1
                continue
            k = norm_text(prob)
            if k in seen:
                stats["dup"] += 1
                continue
            if dec.hit(prob):
                stats["drop_contaminated"] += 1
                continue
            seen.add(k)
            rows.append({"id": rid(cfg, prob), "domain": "code", "prompt": prob, "starter_code": "",
                         "tests": io, "source": f"agentica-org/DeepCoder-Preview-Dataset:{cfg}"})
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    p = os.path.join(out, "rl_code.jsonl")
    n = write_jsonl(p, rows)
    stats.update({"n": n, "sha256": sha256_file(p), "min_tests": min_tests,
                  "excluded_configs": ["lcbv5 (LiveCodeBench problems inside the v6 window)", "codeforces (CodeElo test set)"]})
    return stats


# ----------------------------------------------------------------------------- RL if
def build_rl_if(hf: str, out: str, eval_dir: str, if_sft_frac: float, max_rows: int | None, seed: int) -> Dict[str, Any]:
    from mopd.graders.if_grader import parse_ground_truth, registry

    reg = registry("ifevalg")
    eval_texts = []
    for name in ("ifeval", "ifbench"):
        p = os.path.join(eval_dir, name, "test.jsonl")
        if os.path.exists(p):
            eval_texts += [r["prompt"] for r in iter_jsonl(p)]
    dec = Decontaminator(eval_texts)
    rows, stats = [], {"in": 0, "drop_unknown_id": 0, "drop_contaminated": 0, "drop_multi_turn": 0}
    for r in _pq(os.path.join(hf, "allenai__IF_multi_constraints_upto5", "**", "*.parquet")):
        stats["in"] += 1
        msgs = r["messages"]
        if isinstance(msgs, str):
            import ast
            msgs = ast.literal_eval(msgs)
        if len(msgs) != 1 or msgs[0]["role"] != "user":
            stats["drop_multi_turn"] += 1
            continue
        gt = parse_ground_truth(r["ground_truth"])
        if any(i not in reg for i in gt["instruction_id"]):
            stats["drop_unknown_id"] += 1
            continue
        if dec.hit(msgs[0]["content"]):
            stats["drop_contaminated"] += 1
            continue
        rows.append({"id": rid("ifrlvr", r["key"], msgs[0]["content"]), "domain": "if", "prompt": msgs[0]["content"],
                     "ground_truth": r["ground_truth"] if isinstance(r["ground_truth"], str) else json.dumps(r["ground_truth"]),
                     "instruction_ids": gt["instruction_id"], "source": f"allenai/IF_multi_constraints_upto5:{r.get('dataset')}"})
    random.Random(seed).shuffle(rows)
    n_sft = int(len(rows) * if_sft_frac)
    sft_prompts, rl_rows = rows[:n_sft], rows[n_sft:]
    if max_rows:
        rl_rows = rl_rows[:max_rows]
    p = os.path.join(out, "rl_if.jsonl")
    n = write_jsonl(p, rl_rows)
    p2 = os.path.join(out, "sft_if_prompts.jsonl")
    n2 = write_jsonl(p2, sft_prompts)
    stats.update({"n_rl": n, "n_sft_prompts": n2, "sha256_rl": sha256_file(p), "sha256_sft_prompts": sha256_file(p2)})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "train"))
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "eval"))
    ap.add_argument("--orz-json", default="", help="orz_math_57k_collected.json (github Open-Reasoner-Zero/data)")
    ap.add_argument("--only", default="sft_math,sft_code,rl_math,rl_code,rl_if")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-sft-rows", type=int, default=None)
    ap.add_argument("--max-rl-rows", type=int, default=None)
    ap.add_argument("--math-solve-rate", default="0.0,0.9", help="keep BigMath rows with llama8b_solve_rate in [lo,hi]")
    ap.add_argument("--if-sft-frac", type=float, default=0.25, help="share of IF prompts reserved for SFT distillation")
    a = ap.parse_args()
    lo, hi = map(float, a.math_solve_rate.split(","))
    rep = {}
    only = a.only.split(",")
    if "sft_math" in only:
        rep["sft_math"] = build_sft_mot(a.hf_root, a.out, "math", "math", a.max_sft_rows, a.seed)
    if "sft_code" in only:
        rep["sft_code"] = build_sft_mot(a.hf_root, a.out, "code", "code", a.max_sft_rows, a.seed)
    if "rl_math" in only:
        rep["rl_math"] = build_rl_math(a.hf_root, a.out, a.eval_dir, lo, hi, a.max_rl_rows, a.seed, a.orz_json)
    if "rl_code" in only:
        rep["rl_code"] = build_rl_code(a.hf_root, a.out, a.eval_dir, a.max_rl_rows, a.seed)
    if "rl_if" in only:
        rep["rl_if"] = build_rl_if(a.hf_root, a.out, a.eval_dir, a.if_sft_frac, a.max_rl_rows, a.seed)
    write_json(os.path.join(a.out, "MANIFEST.json"), rep)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
