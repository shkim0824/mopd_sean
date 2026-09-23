"""Build the tau2 OPD prompt pool from inclusionAI/AReaL-tau2-data (single-turn variant).

Every SFT row is already one decision point: `messages` = the context (system policy, dialog
history, tool results) and `answer` = the next assistant turn. On-policy distillation needs only
the context -- the student generates the turn, the teacher scores it -- so a row becomes one prompt.

Rendering has to equal what the student sees at tau2-bench evaluation time and what the teacher
was SFT'd on (its card: "answer.thinking is renamed to reasoning_content, the field the native
template reads"; tool-call-only messages get content ""):
  * assistant `reasoning` -> `reasoning_content`; tool_calls normalised to
    {type: function, function: {name, arguments}} (the data carries both that shape and the flat
    {name, arguments} shape); tool-only turns get content ""
  * the domain's tool schemas are passed through apply_chat_template(tools=...); they come from
    data/tau2/tools_<domain>.json, exported by tmp/phase3/export_tau2_tools.py straight from tau2's
    Tool.openai_schema -- byte-identical to what the harness sends through the OpenAI `tools` field
  * add_generation_prompt=True, enable_thinking=True (the same kwargs trl applied for every earlier
    OPD run; the trainer asserts its own rendering against trl's at startup)

Filters:
  * prompt tokens <= --max-prompt-tokens: the 32k teacher/vLLM context minus the 16k completion cap
  * CONTAMINATION: telecom rows whose task_id is in tau2-bench's telecom `base` or `test` split are
    excluded. Measured 2026-09-17: 20 of the 114 base tasks (251 rows) and 7 of the 40 test tasks
    (81 rows) appear in the SFT data. The teacher already saw them; the student must not.

Writes data/rl_pools/tau_opd_train.jsonl rows
    {prompt_text, n_prompt_tokens, id, domain: "tau", sub_domain, correct, reward, source_dialog_id,
     turn_index, task_id}
plus data/rl_pools/tau_opd_train.manifest.json and data/tau2/contaminated_telecom_task_ids.json.

    HF_HUB_OFFLINE=1 $S/.venv/mopd/bin/python -m mopd.data.prep_tau2_opd [--max-prompt-tokens 15872]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time


def sub_domain(md: dict) -> str:
    if "task_id" in md:
        return "telecom"
    if "seed_pattern_task_id" in md:
        return "airline"
    if "scenario_id" in md:
        return "retail"
    return "unknown"


def norm_call(t: dict) -> dict:
    f = t.get("function") if isinstance(t.get("function"), dict) else None
    name = (f or t).get("name")
    args = (f or t).get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return {"type": "function", "function": {"name": name, "arguments": args}}


def to_template_messages(msgs: list) -> list:
    out = []
    for m in msgs:
        role = m["role"]
        m2 = {"role": role, "content": m.get("content") or ""}
        if role == "assistant":
            if m.get("reasoning"):
                m2["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                m2["tool_calls"] = [norm_call(t) for t in m["tool_calls"]]
        out.append(m2)
    return out


def sha(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tau2/AReaL-tau2-data/tau2_sft_train.jsonl")
    ap.add_argument("--tools-dir", default="data/tau2")
    ap.add_argument("--tau2-telecom", default="../third_party/tau2-bench/data/tau2/domains/telecom")
    ap.add_argument("--tokenizer", default="models/Qwen3-4B-OT3")
    ap.add_argument("--out", default="data/rl_pools/tau_opd_train.jsonl")
    ap.add_argument("--max-prompt-tokens", type=int, default=15872)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="first K rows (smoke)")
    ap.add_argument("--system-prompt", choices=("dataset", "harness"), default="dataset",
                    help="dataset = the row's own system message (teacher-consistent); harness = the exact "
                         "tau2-bench LLMAgent system prompt for the sub-domain (evaluation-consistent)")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    tools = {}
    for d in ("airline", "retail", "telecom"):
        p = os.path.join(a.tools_dir, "tools_%s.json" % d)
        if not os.path.exists(p):
            print("missing %s -- run tmp/phase3/export_tau2_tools.py in the tau2 venv first" % p)
            return 2
        tools[d] = json.load(open(p))
    harness_sys = {}
    for d in ("airline", "retail", "telecom"):
        p = os.path.join(a.tools_dir, "agent_system_prompt_%s.txt" % d)
        harness_sys[d] = open(p).read() if os.path.exists(p) else None

    splits = json.load(open(os.path.join(a.tau2_telecom, "split_tasks.json")))
    eval_ids = set(splits["base"]) | set(splits["test"])
    print("tau2 telecom eval task ids (base ∪ test): %d" % len(eval_ids))

    rows = []
    with open(a.src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            rows.append(json.loads(line))
    print("source rows:", len(rows))

    stats = collections.defaultdict(collections.Counter)
    contaminated_ids = set()
    tok_lens = collections.defaultdict(list)
    sys_match = collections.defaultdict(collections.Counter)
    out_rows = []
    t0 = time.time()
    for i, r in enumerate(rows):
        md = r.get("metadata") or {}
        d = sub_domain(md)
        stats[d]["all"] += 1
        if d == "unknown":
            stats[d]["dropped_unknown_domain"] += 1
            continue
        if d == "telecom" and str(md.get("task_id")) in eval_ids:
            stats[d]["dropped_contaminated"] += 1
            contaminated_ids.add(str(md["task_id"]))
            continue
        msgs = to_template_messages(r["messages"])
        if msgs and msgs[0]["role"] == "system" and harness_sys.get(d):
            sys_match[d]["identical" if msgs[0]["content"].strip() == harness_sys[d].strip() else "differs"] += 1
            if a.system_prompt == "harness":
                msgs[0] = {"role": "system", "content": harness_sys[d]}
        elif a.system_prompt == "harness":
            stats[d]["dropped_no_harness_prompt"] += 1
            continue
        text = tok.apply_chat_template(msgs, tools=tools[d], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        tok_lens[d].append(n)
        if n > a.max_prompt_tokens:
            stats[d]["dropped_too_long"] += 1
            continue
        stats[d]["kept"] += 1
        out_rows.append({
            "prompt_text": text, "n_prompt_tokens": n,
            "id": "tau-%s-%s-t%s-%d" % (d, md.get("source_dialog_id"), md.get("turn_index"), i),
            "domain": "tau", "sub_domain": d,
            "correct": md.get("correct"), "reward": md.get("reward"),
            "source_dialog_id": md.get("source_dialog_id"), "turn_index": md.get("turn_index"),
            "task_id": md.get("task_id"),
        })
        if (i + 1) % 5000 == 0:
            print("  %d/%d rendered (%.0fs)" % (i + 1, len(rows), time.time() - t0), flush=True)

    random.Random(a.seed).shuffle(out_rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)

    def q(L, f_):
        L = sorted(L)
        return L[int(f_ * (len(L) - 1))] if L else None
    man = {
        "source": "inclusionAI/AReaL-tau2-data tau2_sft_train.jsonl (single-turn: messages -> next assistant turn)",
        "tokenizer": a.tokenizer,
        "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest()[:16],
        "tools_sha": {d: sha(t) for d, t in tools.items()},
        "n_tools": {d: len(t) for d, t in tools.items()},
        "max_prompt_tokens": a.max_prompt_tokens,
        "system_prompt": a.system_prompt,
        "seed": a.seed,
        "n_out": len(out_rows),
        "per_sub_domain": {d: dict(c) for d, c in stats.items()},
        "prompt_tokens": {d: {"p50": q(L, .5), "p90": q(L, .9), "p99": q(L, .99), "max": max(L) if L else None}
                          for d, L in tok_lens.items()},
        "system_prompt_vs_harness": {d: dict(c) for d, c in sys_match.items()},
        "contamination": {"eval_split": "base ∪ test", "n_eval_ids": len(eval_ids),
                          "n_contaminated_ids_seen": len(contaminated_ids)},
        "row_schema": ["prompt_text", "n_prompt_tokens", "id", "domain", "sub_domain", "correct", "reward",
                       "source_dialog_id", "turn_index", "task_id"],
    }
    man["sha256"] = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    json.dump(man, open(a.out.replace(".jsonl", ".manifest.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(sorted(contaminated_ids), open(os.path.join(a.tools_dir, "contaminated_telecom_task_ids.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k not in ("row_schema",)}, indent=1, ensure_ascii=False))
    print("wrote %s (%d rows)" % (a.out, len(out_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
