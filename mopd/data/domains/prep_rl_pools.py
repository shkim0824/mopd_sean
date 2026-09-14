"""Build the verifiable RL prompt pools for the 3 new domains, in mopd_rl's
rl3dom jsonl format: {"input": <prompt>, "output": <gold>, "meta": {...}}.

Pools (per the 2026-09-07 research + user decisions; see docs/DATASETS.md):
  med_train : MedQA-USMLE train (10,178, letter) + MedMCQA train subsample (letter)
  law_train : CaseHOLD train (backbone, 5-way letter) + MBE/barexam MCQA + housing_qa
              (Yes/No, statutes in-context, capped share)
  fin_train : FinQA train (numeric) + TAT-QA train arithmetic/count (numeric+scale)

Decontamination: 8-gram overlap drop against ALL eval prompt sets of the same
domain (+ the cross-traps from research: MBE↔MMLU-professional-law handled by
simply never evaluating MMLU-law; casehold uses reglab splits only, never
lex_glue).  python -m mopd.data.domains.prep_rl_pools --out data/rl_pools
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import zipfile
from typing import Any, Dict, Iterable, List, Set

from mopd.eval.domains.benches import (D, TP, _letters, _mcqa_prompt, BENCHES,
                                       NUM_SUFFIX, _finqa_context)

MCQA_SUFFIX_RL = ("\n\nPlease reason step by step, then give your final answer on "
                  "the last line in the form \"Answer: <letter>\".")


_BOILER = re.compile(
    r"(please reason step by step.*$|^[A-J]\.\s.*$|if the answer has a scale.*$"
    r"|please reason step by step over the statutes.*$)",
    re.IGNORECASE | re.MULTILINE)


def _decon_text(prompt: str) -> str:
    """Strip shared format boilerplate (answer-format instructions, lettered option
    lines, scale hints) so decontamination compares QUESTION/CONTEXT content only —
    otherwise every prompt's identical suffix causes a false 100% overlap."""
    return _BOILER.sub(" ", prompt)


def _grams(text: str, n: int = 13) -> Set[str]:
    toks = re.findall(r"\w+", _decon_text(text).lower())
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def build_eval_grams(domain: str) -> Set[str]:
    bench_names = {"med": ["medqa", "medxpertqa", "pubmedqa"],
                   "law": ["casehold", "lexam_mcq4"],
                   "fin": ["finqa", "tatqa"]}[domain]
    g: Set[str] = set()
    for b in bench_names:
        try:
            for r in BENCHES[b]():
                g |= _grams(r["prompt"])
        except Exception as e:
            print(f"[decon] WARNING: could not load {b}: {e}")
    print(f"[decon] {domain}: {len(g)} eval 8-grams")
    return g


def contaminated(prompt: str, eval_grams: Set[str]) -> bool:
    return bool(_grams(prompt) & eval_grams)


# ------------------------------------------------------------------ med
def med_rows(medmcqa_cap: int = 20000, seed: int = 42) -> Iterable[Dict[str, Any]]:
    for i, line in enumerate(open(os.path.join(D, "med_medqa/phrases_no_exclude_train.jsonl"))):
        r = json.loads(line)
        opts = [r["options"][k] for k in sorted(r["options"])]
        yield {"input": _mcqa_prompt(r["question"], opts),
               "output": r["answer_idx"],
               "meta": {"source": "MedQA-USMLE-train", "kind": "mcqa4"}}
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(os.path.join(D, "med_medmcqa/data/train-*.parquet")))
    t = pq.read_table(fs[0])
    idx = list(range(t.num_rows))
    random.Random(seed).shuffle(idx)
    kept = 0
    for i in idx:
        r = {c: t.column(c)[i].as_py() for c in
             ("question", "opa", "opb", "opc", "opd", "cop", "choice_type")}
        if str(r["choice_type"]) != "single":
            continue
        cop = int(r["cop"])          # dataset uses 0-3 or 1-4 depending on export;
        if cop >= 4:                  # normalize defensively
            cop -= 1
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        if not (0 <= cop < 4) or any(o is None or not str(o).strip() for o in opts):
            continue
        yield {"input": _mcqa_prompt(str(r["question"]), [str(o) for o in opts]),
               "output": _letters(4)[cop],
               "meta": {"source": "MedMCQA-train", "kind": "mcqa4"}}
        kept += 1
        if kept >= medmcqa_cap:
            break


# ------------------------------------------------------------------ law
def law_rows(housing_cap: int = 4000, seed: int = 42) -> Iterable[Dict[str, Any]]:
    f = sorted(glob.glob(os.path.join(D, "law_casehold/data/*/train.csv")))[0]
    with open(f) as fh:
        rd = csv.reader(fh)
        next(rd)
        for rec in rd:
            ctx, holds, label = rec[1], rec[2:7], rec[-1]
            q = ("Which holding statement best completes the CITATION in the "
                 "following excerpt?")
            yield {"input": _mcqa_prompt(q, holds, context=ctx),
                   "output": _letters(5)[int(float(label))],
                   "meta": {"source": "CaseHOLD-train", "kind": "mcqa5"}}
    for name, path in (("MBE", "law_mbe/raw_dataset.json"),
                       ("barexam_qa", None)):
        if name == "MBE":
            data = json.load(open(os.path.join(D, path)))
            for r in data:
                opts = [r.get(k) for k in ("choice_a", "choice_b", "choice_c", "choice_d")]
                if not all(opts):
                    ks = [k for k in r if k.lower().startswith(("a", "b", "c", "d"))
                          and len(k) <= 2]
                    opts = [r[k] for k in sorted(ks)][:4] if len(ks) >= 4 else None
                ans = str(r.get("correct_answer", r.get("answer", ""))).strip().upper()[:1]
                if opts and ans in "ABCD":
                    yield {"input": _mcqa_prompt(r["question"], [str(o) for o in opts]),
                           "output": ans, "meta": {"source": "MBE", "kind": "mcqa4"}}
        else:
            fs = sorted(glob.glob(os.path.join(D, "law_barexam_qa/**/*.parquet"),
                                  recursive=True)) or \
                 sorted(glob.glob(os.path.join(D, "law_barexam_qa/**/*.csv"),
                                  recursive=True))
            if not fs:
                print("[law] barexam_qa files not found; skipping")
                continue
            import pandas as pd
            df = (pd.read_parquet(fs[0]) if fs[0].endswith(".parquet")
                  else pd.read_csv(fs[0]))
            cols = {c.lower(): c for c in df.columns}
            qc = cols.get("question") or cols.get("prompt")
            ac = cols.get("answer") or cols.get("gold") or cols.get("correct_answer")
            oc = [cols.get(k) for k in ("choice_a", "choice_b", "choice_c", "choice_d")]
            if not (qc and ac and all(oc)):
                print(f"[law] barexam_qa schema unrecognized: {list(df.columns)[:10]}")
                continue
            for _, r in df.iterrows():
                ans = str(r[ac]).strip().upper()[:1]
                if ans in "ABCD":
                    yield {"input": _mcqa_prompt(str(r[qc]), [str(r[c]) for c in oc]),
                           "output": ans, "meta": {"source": "barexam_qa", "kind": "mcqa4"}}
    z = zipfile.ZipFile(os.path.join(D, "law_housing_qa/data/questions.json.zip"))
    qs = json.loads(z.read(z.namelist()[0]))
    random.Random(seed).shuffle(qs)
    kept = 0
    for r in qs:
        if str(r.get("answer", "")).strip() not in ("Yes", "No"):
            continue
        statutes = "\n\n".join(s.get("statute_text", s.get("text", ""))[:4000]
                               for s in r.get("statutes", [])[:3])
        prompt = (f"Relevant statutes ({r.get('state','')}):\n{statutes}\n\n"
                  f"Question: {r['question']}\n\nPlease reason step by step over the "
                  f"statutes, then answer strictly Yes or No on the last line in the "
                  f"form \"Answer: <Yes/No>\".")
        yield {"input": prompt, "output": r["answer"].strip(),
               "meta": {"source": "housing_qa", "kind": "yesno"}}
        kept += 1
        if kept >= housing_cap:
            break


# ------------------------------------------------------------------ fin
def fin_rows() -> Iterable[Dict[str, Any]]:
    data = json.load(open(os.path.join(TP, "FinQA/dataset/train.json")))
    for r in data:
        qa = r["qa"]
        gold = qa.get("exe_ans", qa.get("answer"))
        yield {"input": _finqa_context(r) + f"\n\nQuestion: {qa['question']}" + NUM_SUFFIX,
               "output": str(gold), "meta": {"source": "FinQA-train", "kind": "numeric"}}
    data = json.load(open(os.path.join(TP, "TAT-QA/dataset_raw/tatqa_dataset_train.json")))
    for doc in data:
        table = "\n".join(" | ".join(str(c) for c in row)
                          for row in doc["table"]["table"])
        paras = "\n".join(p["text"] for p in doc["paragraphs"])
        for q in doc["questions"]:
            if q.get("answer_type") not in ("arithmetic", "count"):
                continue
            ans = q["answer"]
            if isinstance(ans, list):
                ans = ans[0] if ans else None
            if ans is None:
                continue
            yield {"input": (f"Table:\n{table}\n\n{paras}\n\nQuestion: {q['question']}"
                             + NUM_SUFFIX),
                   "output": str(ans),
                   "meta": {"source": "TATQA-train", "kind": "numeric",
                            "scale": q.get("scale", "")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/rl_pools")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for domain, gen in (("med", med_rows), ("law", law_rows), ("fin", fin_rows)):
        eg = build_eval_grams(domain)
        rows = list(gen())
        random.Random(args.seed).shuffle(rows)
        kept, dropped = [], 0
        for r in rows:
            if contaminated(r["input"], eg):
                dropped += 1
            else:
                kept.append(r)
        val = kept[-100:]
        train = kept[:-100]
        for name, part in (("train", train), ("val", val)):
            with open(os.path.join(args.out, f"{domain}_{name}.jsonl"), "w") as f:
                for r in part:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        from collections import Counter
        srcs = Counter(r["meta"]["source"] for r in kept)
        print(f"[{domain}] kept {len(train)} train / {len(val)} val | "
              f"dropped_contaminated {dropped} | sources {dict(srcs)}")
    print("PREP_RL_POOLS_DONE")


if __name__ == "__main__":
    main()
