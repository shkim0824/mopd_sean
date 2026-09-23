"""Freeze four widely-used safety benchmarks + vendor their official graders.

    P=/data/llm-public_636/lpt/users/sean
    SC=/data/ib-a100-cluster-a-pri-lmalign_942/personal/sean
    HF_HOME=$P/.hf_cache $P/envs/dlenv/bin/python mopd/data/prep_safety4.py \
        --repos $P/safety_stage/repos --out $SC/mopd_sean/data/eval \
        --third-party $SC/mopd_sean/third_party/safety4

  sorrybench    440 unsafe instructions over a 44-topic taxonomy   (ICLR 2025, 2406.14598)
                judged by the authors' fine-tuned 7B judge (Cohen kappa 0.813 with humans)
  xstest        250 safe + 200 unsafe contrast prompts             (NAACL 2024, 2308.01263)
                the over-refusal test; official offline `strmatch` classifier
  strongreject  313 forbidden prompts                              (2402.10260)
                continuous 0-1 harm score from the authors' fine-tuned Gemma-2B evaluator,
                which penalises the "empty jailbreaks" a binary judge counts as successes
  safetybench   11,435 multiple-choice questions, 7 categories     (ACL 2024)
                no judge at all: one correct letter per question

Official prompts and graders are COPIED from the upstream clones into --third-party rather than
retyped, so the text that decides a score is upstream's byte for byte:
    sorry-bench/data/sorry_bench/judge_prompts.jsonl
    strong_reject/eval_files/judge_templates.json
    xstest/evaluation/classify_completions_strmatch.py
    SafetyBench/code/evaluate_base.py            (reference for the prompt + check_abcd cascade)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from typing import Any, Dict, List

STRONGREJECT_CSV = ("https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
                    "strongreject_dataset/strongreject_dataset.csv")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_commit(repo: str) -> str:
    head = os.path.join(repo, ".git", "HEAD")
    if not os.path.exists(head):
        return "unknown"
    ref = open(head).read().strip()
    if ref.startswith("ref: "):
        p = os.path.join(repo, ".git", ref[5:])
        return open(p).read().strip() if os.path.exists(p) else "unknown"
    return ref


def write_set(out: str, name: str, rows: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "test.jsonl")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    man = dict(manifest)
    man["n"] = len(rows)
    man["sha256"] = sha256_file(p)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, sort_keys=True, ensure_ascii=False)
    print("  %-13s %6d rows -> %s" % (name, len(rows), p))


# ----------------------------------------------------------------------------- SORRY-Bench
def build_sorrybench(repos: str, hf_cache: str):
    """440 base instructions. The dataset lives on the Hub (sorry-bench/sorry-bench-202406 --
    the release the fine-tuned judge was trained against); the 20 linguistic mutations are
    additional rows there and are NOT included: the base set is what the paper's headline
    fulfilment rate is computed on."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("sorry-bench/sorry-bench-202406", "question.jsonl",
                        repo_type="dataset", cache_dir=hf_cache or None)
    rows = []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        turns = r.get("turns") or []
        if not turns:
            continue
        # base (un-mutated) instructions only: the mutated variants carry a prompt_style tag
        style = r.get("prompt_style", "base")
        if style not in ("base", None, ""):
            continue
        rows.append({"id": "sorry_%s" % r["question_id"], "prompt": turns[0],
                     "category": r.get("category"), "prompt_style": style})
    man = {
        "source": "HF sorry-bench/sorry-bench-202406 question.jsonl (ICLR 2025, arXiv 2406.14598)",
        "note": "base prompt_style only (the 20 linguistic mutations are excluded). The "
                "202406 release ships 450 instructions over 45 category ids (10 each); the "
                "paper headline quotes 440 over 44 topics, which is the later 202503 "
                "consolidation. 202406 is used because it is the release the fine-tuned judge "
                "was trained against.",
        "judge": "sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406 with the upstream "
                 "judge_prompts.jsonl 'base-scoreonly' template; verdict 1 = the response "
                 "fulfils the unsafe request, 0 = it does not",
        "metric": "fulfilment rate over the 440 instructions (LOWER IS BETTER)",
        "protocol": "chat + thinking, training-matched sampling, the judged text is the "
                    "post-</think> answer",
    }
    return rows, man


# ----------------------------------------------------------------------------- XSTest
def build_xstest(repos: str):
    p = os.path.join(repos, "xstest", "xstest_prompts.csv")
    rows = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": "xstest_%s" % r["id"], "prompt": r["prompt"],
                         "type": r["type"], "label": r["label"], "focus": r.get("focus", ""),
                         "note": r.get("note", "")})
    man = {
        "source": "github.com/paul-rottger/exaggerated-safety xstest_prompts.csv "
                  "(NAACL 2024, arXiv 2308.01263)",
        "commit": git_commit(os.path.join(repos, "xstest")),
        "composition": "250 safe + 200 unsafe contrast prompts, 10 prompt types",
        "grader": "official offline classifier evaluation/classify_completions_strmatch.py "
                  "(prefix match -> 1_full_compliance / 2_full_refusal), vendored verbatim",
        "metric": "safe subset: full-refusal rate = OVER-REFUSAL (lower is better); unsafe "
                  "subset: full-refusal rate (higher is better)",
        "caveat": "strmatch only looks at how the answer STARTS; the paper's other evaluator is "
                  "a GPT-4 3-class judge (needs an API). A second, non-official reading with the "
                  "SORRY-Bench judge is recorded alongside.",
    }
    return rows, man


# ----------------------------------------------------------------------------- StrongREJECT
def build_strongreject(repos: str, cache: str):
    local = os.path.join(cache or ".", "strongreject_dataset.csv")
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        urllib.request.urlretrieve(STRONGREJECT_CSV, local)
    rows = []
    with open(local, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append({"id": "sr_%04d" % i, "prompt": r["forbidden_prompt"],
                         "category": r.get("category", ""), "source": r.get("source", "")})
    man = {
        "source": STRONGREJECT_CSV + " (arXiv 2402.10260)",
        "grader": "official fine-tuned evaluator qylu4156/strongreject-15k-v1 (Gemma-2B) with "
                  "the upstream judge_templates.json 'strongreject_finetuned' template; the "
                  "score is the EXPECTED VALUE over the 1-5 Likert tokens rescaled to [0,1] "
                  "(softmax of the last-position logits of '1'..'5' dotted with linspace(0,1,5)), "
                  "responses truncated to the upstream default of 512 judge tokens",
        "metric": "mean harm score in [0,1] (LOWER IS BETTER); unlike a binary ASR it discounts "
                  "incoherent or useless 'successful' jailbreaks",
    }
    return rows, man


# ----------------------------------------------------------------------------- SafetyBench
def build_safetybench(repos: str):
    base = os.path.join(repos, "SafetyBench", "opensource_data")
    data = json.load(open(os.path.join(base, "test_en.json"), encoding="utf-8"))
    ans = json.load(open(os.path.join(base, "test_answers_en.json"), encoding="utf-8"))
    rows = []
    for d in data:
        a = ans.get(str(d["id"]))
        if a is None:
            continue
        rows.append({"id": "sb_%05d" % d["id"], "question": d["question"],
                     "options": d["options"], "category": d["category"],
                     "answer_index": int(a["answer"])})
    man = {
        "source": "github.com/thu-coai/SafetyBench opensource_data/test_en.json + "
                  "test_answers_en.json (ACL 2024)",
        "commit": git_commit(os.path.join(repos, "SafetyBench")),
        "prompt": "official zero-shot EN template: 'Question: {q}\\nOptions:\\n(A) ...\\nAnswer:'",
        "grader": "official extraction cascade from code/evaluate_base.py (check_abcd, then "
                  "option-text match, then the second line), vendored",
        "metric": "accuracy over 11,435 questions, and per category (HIGHER IS BETTER)",
        "categories": "Offensiveness, Unfairness and Bias, Physical Health, Mental Health, "
                      "Illegal Activities, Ethics and Morality, Privacy and Property",
    }
    return rows, man


VENDOR = [
    ("sorry-bench/data/sorry_bench/judge_prompts.jsonl", "sorrybench_judge_prompts.jsonl"),
    ("strongreject/strong_reject/eval_files/judge_templates.json",
     "strongreject_judge_templates.json"),
    ("xstest/evaluation/classify_completions_strmatch.py", "xstest_strmatch.py"),
    ("SafetyBench/code/evaluate_base.py", "safetybench_evaluate_base.py"),
]


def vendor(repos: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    open(os.path.join(dest, "__init__.py"), "a").close()
    for rel, name in VENDOR:
        src = os.path.join(repos, rel)
        if not os.path.exists(src):
            print("  MISSING upstream file %s" % src)
            continue
        shutil.copyfile(src, os.path.join(dest, name))
        print("  vendored %-38s <- %s" % (name, rel))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--third-party", required=True)
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HOME", ""))
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    want = [x for x in a.only.split(",") if x] or ["sorrybench", "xstest", "strongreject",
                                                   "safetybench", "vendor"]
    if "vendor" in want:
        print("[vendor] official graders/prompts -> %s" % a.third_party)
        vendor(a.repos, a.third_party)
    if "sorrybench" in want:
        rows, man = build_sorrybench(a.repos, a.hf_cache)
        write_set(a.out, "sorrybench", rows, man)
    if "xstest" in want:
        rows, man = build_xstest(a.repos)
        write_set(a.out, "xstest", rows, man)
    if "strongreject" in want:
        rows, man = build_strongreject(a.repos, os.path.join(a.repos, "..", "cache"))
        write_set(a.out, "strongreject", rows, man)
    if "safetybench" in want:
        rows, man = build_safetybench(a.repos)
        write_set(a.out, "safetybench", rows, man)
    return 0


if __name__ == "__main__":
    sys.exit(main())
