"""Benchmark loaders for the 3 new domains. Each loader returns rows:
  {id, prompt, grade_kind, gold, meta}
grade_kind in {mcqa, pubmedqa, legalbench, fin_numeric, tatqa}.

Prompt conventions follow each benchmark's official/reasoning-model usage:
- MCQA: question + lettered options + 'reason step by step, end with "Answer: <letter>"'
  (MedXpertQA repo convention for R1-style models; Med-RLVR format)
- PubMedQA: official PQA-L context+question, final answer yes/no/maybe
- LegalBench: the task's official base_prompt.txt with few-shots kept, bare label answer
- FinQA/TAT-QA/DocMath: table+text context, numeric final answer
"""
from __future__ import annotations

import csv
import glob
import json
import os
import zipfile
from typing import Any, Dict, List, Optional

from mopd.common.io import data_root, repo_root
D = os.environ.get("DOMAINS_DATA", os.path.join(data_root(), "domains"))
TP = os.environ.get("DOMAINS_TP", os.path.join(repo_root(), "third_party"))

MCQA_SUFFIX = ("\n\nPlease reason step by step, then give your final answer on the "
               "last line in the form \"Answer: <letter>\".")
NUM_SUFFIX = ("\n\nPlease reason step by step, then give the final numeric answer on "
              "the last line in the form \"Answer: <number>\".")


def _letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _mcqa_prompt(question: str, options: List[str], context: str = "") -> str:
    body = (context + "\n\n" if context else "") + question.strip() + "\n\n"
    body += "\n".join(f"{l}. {o}" for l, o in zip(_letters(len(options)), options))
    return body + MCQA_SUFFIX


# ------------------------------------------------------------------ medical
def load_medqa_test(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    for i, line in enumerate(open(os.path.join(D, "med_medqa/phrases_no_exclude_test.jsonl"))):
        r = json.loads(line)
        opts = [r["options"][k] for k in sorted(r["options"])]
        rows.append({"id": f"medqa_{i}", "grade_kind": "mcqa", "n_options": len(opts),
                     "gold": r["answer_idx"], "prompt": _mcqa_prompt(r["question"], opts)})
        if limit and len(rows) >= limit:
            break
    return rows


def load_medxpertqa_text(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    # HF snapshot ships jsonl under Text/ (test.jsonl); options is a dict A..J
    fs = sorted(glob.glob(os.path.join(D, "med_medxpertqa/**/*est*.json*"), recursive=True))
    fs = [f for f in fs if "Text" in f or "text" in f] or fs
    rows = []
    for line in open(fs[0]):
        r = json.loads(line)
        opts_d = r.get("options") or {}
        keys = sorted(opts_d)
        rows.append({"id": f"medx_{r.get('id', len(rows))}", "grade_kind": "mcqa",
                     "n_options": len(keys), "gold": r["label"],
                     "prompt": _mcqa_prompt(r["question"],
                                            [opts_d[k] for k in keys])})
        if limit and len(rows) >= limit:
            break
    return rows


def load_pubmedqa_test(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Official PQA-L test: ori_pqal.json ∩ test_ground_truth.json (pubmedqa repo)."""
    base = os.path.join(TP, "pubmedqa")
    data = json.load(open(os.path.join(base, "data/ori_pqal.json")))
    gt = json.load(open(os.path.join(base, "data/test_ground_truth.json")))
    rows = []
    for pmid, gold in gt.items():
        e = data[pmid]
        ctx = "\n".join(e["CONTEXTS"])
        q = e["QUESTION"]
        prompt = (f"Abstract:\n{ctx}\n\nQuestion: {q}\n\nPlease reason step by step, "
                  f"then answer strictly with yes, no, or maybe on the last line in "
                  f"the form \"Answer: <yes/no/maybe>\".")
        rows.append({"id": f"pubmedqa_{pmid}", "grade_kind": "pubmedqa",
                     "gold": gold, "prompt": prompt})
        if limit and len(rows) >= limit:
            break
    return rows


# ------------------------------------------------------------------ law
def load_casehold_test(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """reglab/casehold format: csv cols [idx, citing_prompt, holding_0..4, label]."""
    f = sorted(glob.glob(os.path.join(D, "law_casehold/data/*/test.csv")))[0]
    rows = []
    with open(f) as fh:
        rd = csv.reader(fh)
        header = next(rd)
        for rec in rd:
            ctx, holds, label = rec[1], rec[2:7], rec[-1]
            q = ("Which holding statement best completes the CITATION in the "
                 "following excerpt?")
            rows.append({"id": f"casehold_{rec[0]}", "grade_kind": "mcqa",
                         "n_options": 5, "gold": _letters(5)[int(float(label))],
                         "prompt": _mcqa_prompt(q, holds, context=ctx)})
            if limit and len(rows) >= limit:
                break
    return rows


def load_lexam_mcq(config: str = "mcq_4_choices", split: str = "test",
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(os.path.join(D, f"law_lexam/{config}/{split}-*.parquet")))
    t = pq.read_table(fs[0])
    rows = []
    import ast
    for i in range(t.num_rows):
        r = {c: t.column(c)[i].as_py() for c in t.column_names}
        if str(r.get("language", "en")).lower() != "en":
            continue  # English subset only
        choices = r["choices"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        gold_i = int(r["gold"]) if str(r["gold"]).isdigit() else int(float(r["gold"]))
        rows.append({"id": f"lexam_{config}_{i}", "grade_kind": "mcqa",
                     "n_options": len(choices), "gold": _letters(len(choices))[gold_i],
                     "prompt": _mcqa_prompt(r["question"], list(choices))})
        if limit and len(rows) >= limit:
            break
    return rows


def load_legalbench(tasks: Optional[List[str]] = None,
                    limit_per_task: Optional[int] = None) -> List[Dict[str, Any]]:
    """Rows carry the official base_prompt with {text} filled; graded per-task by the
    official evaluate(). Task list defaults to all exact-match tasks minus rule_qa."""
    import sys
    sys.path.insert(0, os.path.join(TP, "legalbench"))
    import evaluation as lb
    from tasks import TASKS  # official task registry
    excl = {"rule_qa"}
    use = tasks or [t for t in TASKS if t in set(lb.EXACT_MATCH_BALANCED_ACC_TASKS)]
    use = [t for t in use if t not in excl]
    rows = []
    from datasets import load_dataset
    for task in use:
        tdir = os.path.join(D, "law_legalbench", task)
        prompt_f = os.path.join(TP, "legalbench", "tasks", task, "base_prompt.txt")
        if not os.path.exists(prompt_f):
            continue
        template = open(prompt_f).read()
        test_f = glob.glob(os.path.join(tdir, "test*"))
        if not test_f:
            continue
        import pandas as pd
        df = (pd.read_csv(test_f[0], sep="\t") if test_f[0].endswith(".tsv")
              else pd.read_parquet(test_f[0]))
        for j, rec in df.iterrows():
            filled = template
            for col in df.columns:
                filled = filled.replace("{{" + col + "}}", str(rec[col]))
            rows.append({"id": f"lb_{task}_{j}", "grade_kind": "legalbench",
                         "task": task, "gold": str(rec.get("answer", "")),
                         "prompt": filled})
            if limit_per_task and j + 1 >= limit_per_task:
                break
    return rows


# ------------------------------------------------------------------ finance
def _finqa_context(r: Dict[str, Any]) -> str:
    pre = " ".join(r.get("pre_text", []))
    post = " ".join(r.get("post_text", []))
    table = "\n".join(" | ".join(str(c) for c in row) for row in r.get("table", []))
    return f"{pre}\n\nTable:\n{table}\n\n{post}"


def load_finqa(split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data = json.load(open(os.path.join(TP, f"FinQA/dataset/{split}.json")))
    rows = []
    for r in data:
        qa = r["qa"]
        gold = qa.get("exe_ans", qa.get("answer"))
        rows.append({"id": f"finqa_{r['id']}", "grade_kind": "fin_numeric",
                     "gold": gold,
                     "prompt": _finqa_context(r) + f"\n\nQuestion: {qa['question']}"
                     + NUM_SUFFIX})
        if limit and len(rows) >= limit:
            break
    return rows


def load_tatqa(split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    f = os.path.join(TP, f"TAT-QA/dataset_raw/tatqa_dataset_{split}_gold.json")
    if not os.path.exists(f):
        f = os.path.join(TP, f"TAT-QA/dataset_raw/tatqa_dataset_{split}.json")
    data = json.load(open(f))
    rows = []
    for doc in data:
        table = "\n".join(" | ".join(str(c) for c in row)
                          for row in doc["table"]["table"])
        paras = "\n".join(p["text"] for p in doc["paragraphs"])
        ctx = f"Table:\n{table}\n\n{paras}"
        for q in doc["questions"]:
            if "answer" not in q:
                continue
            suffix = (NUM_SUFFIX + " If the answer has a scale, state it (thousand/"
                      "million/billion/percent).") if q["answer_type"] in ("arithmetic", "count") \
                else ("\n\nPlease reason step by step, then give the final answer span "
                      "on the last line in the form \"Answer: <answer>\".")
            rows.append({"id": f"tatqa_{q['uid']}", "grade_kind": "tatqa",
                         "gold": q["answer"], "gold_scale": q.get("scale", ""),
                         "answer_type": q["answer_type"],
                         "prompt": ctx + f"\n\nQuestion: {q['question']}" + suffix})
            if limit and len(rows) >= limit:
                return rows
    return rows


BENCHES = {
    "medqa": load_medqa_test,
    "medxpertqa": load_medxpertqa_text,
    "pubmedqa": load_pubmedqa_test,
    "casehold": load_casehold_test,
    "lexam_mcq4": lambda **kw: load_lexam_mcq("mcq_4_choices", **kw),
    "legalbench": load_legalbench,
    "finqa": load_finqa,
    "tatqa": load_tatqa,
}
