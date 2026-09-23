"""Build the tau2 teacher-SFT set from inclusionAI/AReaL-tau2-data (one row = one assistant turn).

Row -> {prompt_text, completion}, both rendered by the student's own Qwen3 chat template so the model
learns the exact native format it will be served with:
  prompt_text  = apply_chat_template(context, tools=<official tau2 schemas of the sub-domain>,
                 add_generation_prompt=True, enable_thinking=True)
  completion   = render(context + target) minus prompt_text, minus the trailing <|im_end|>\\n
                 (the trainer re-appends <|im_end|> as eos), i.e. "<think>\\n{thinking}\\n</think>\\n\\n{text}"
                 followed by "<tool_call>\\n{json}\\n</tool_call>" blocks when the turn calls tools
Context assistant turns carry content + tool_calls only (NO reasoning_content): that is what the model
sees at inference, where the harness never feeds thinking back (the colleague's SFT effectively did the
same via a key mismatch). Tool-call arguments are passed as JSON strings (json.dumps, insertion order)
because that is how the harness re-sends them and the Qwen3 template prints string arguments verbatim.
Tool messages are rendered as <tool_response> by the template.

Filters: rows whose prompt + completion exceed --max-len tokens are dropped (no system-prompt
truncation); telecom rows whose task_id is in tau2-bench's base or test split are excluded (251 rows).
A token-boundary check verifies tok(prompt)+tok(completion)+[eos] == tok(full render) on a sample.

    HF_HUB_OFFLINE=1 $S/.venv/mopd/bin/python -m mopd.data.prep_tau2_sft
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

IM_END = "<|im_end|>"


def sub_domain(md):
    if "task_id" in md: return "telecom"
    if "seed_pattern_task_id" in md: return "airline"
    if "scenario_id" in md: return "retail"
    return "unknown"


def norm_call(t):
    """-> {"type":"function","function":{"name", "arguments": <JSON string>}}: the harness re-sends earlier
    tool calls as json.dumps(arguments) strings (insertion order, ASCII escapes) and the Qwen3 template
    prints a string argument verbatim, so the training context is rendered exactly like the inference one."""
    f = t.get("function") if isinstance(t.get("function"), dict) else None
    name = (f or t).get("name"); args = (f or t).get("arguments")
    if isinstance(args, str):
        try: args = json.loads(args)
        except Exception: return {"type": "function", "function": {"name": name, "arguments": args}}
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def context_messages(msgs):
    out = []
    for m in msgs:
        m2 = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant" and m.get("tool_calls"):
            m2["tool_calls"] = [norm_call(t) for t in m["tool_calls"]]
        out.append(m2)
    return out


def target_message(ans):
    m = {"role": "assistant", "content": ans.get("content") or ""}
    if ans.get("thinking"):
        m["reasoning_content"] = ans["thinking"]
    if ans.get("tool_calls"):
        m["tool_calls"] = [norm_call(t) for t in ans["tool_calls"]]
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tau2/AReaL-tau2-data/tau2_sft_train.jsonl")
    ap.add_argument("--tools-dir", default="data/tau2")
    ap.add_argument("--tau2-telecom", default="../third_party/tau2-bench/data/tau2/domains/telecom")
    ap.add_argument("--tokenizer", default="models/Qwen3-4B-OT3")
    ap.add_argument("--out", default="data/sft_distill/tau2_sft.jsonl")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    eos = tok.convert_tokens_to_ids(IM_END)
    tools = {d: json.load(open(os.path.join(a.tools_dir, "tools_%s.json" % d))) for d in ("airline", "retail", "telecom")}
    splits = json.load(open(os.path.join(a.tau2_telecom, "split_tasks.json")))
    eval_ids = set(splits["base"]) | set(splits["test"])

    rows = []
    with open(a.src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit: break
            rows.append(json.loads(line))
    st = collections.defaultdict(collections.Counter)
    ptoks, ctoks = collections.defaultdict(list), collections.defaultdict(list)
    out, boundary_checked, boundary_bad = [], 0, 0
    t0 = time.time()
    for i, r in enumerate(rows):
        md = r.get("metadata") or {}; d = sub_domain(md); st[d]["all"] += 1
        if d == "unknown": st[d]["dropped_unknown"] += 1; continue
        if d == "telecom" and str(md.get("task_id")) in eval_ids: st[d]["dropped_contaminated"] += 1; continue
        ctx = context_messages(r["messages"]); tgt = target_message(r["answer"])
        if not (tgt["content"].strip() or tgt.get("tool_calls")): st[d]["dropped_empty_target"] += 1; continue
        prompt = tok.apply_chat_template(ctx, tools=tools[d], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        full = tok.apply_chat_template(ctx + [tgt], tools=tools[d], tokenize=False, add_generation_prompt=False, enable_thinking=True)
        if not full.startswith(prompt): st[d]["dropped_render_mismatch"] += 1; continue
        comp = full[len(prompt):]
        if comp.endswith(IM_END + "\n"): comp = comp[: -len(IM_END) - 1]
        elif comp.endswith(IM_END): comp = comp[: -len(IM_END)]
        if tgt.get("reasoning_content") and "</think>" not in comp: st[d]["dropped_no_think_render"] += 1; continue
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]; c_ids = tok(comp, add_special_tokens=False)["input_ids"]
        n_p, n_c = len(p_ids), len(c_ids) + 1
        ptoks[d].append(n_p); ctoks[d].append(n_c)
        if n_p + n_c > a.max_len: st[d]["dropped_too_long"] += 1; continue
        if boundary_checked < 300:
            boundary_checked += 1
            # joint vs separate tokenisation must agree (full render = prompt + completion + <|im_end|> + "\n")
            if p_ids + c_ids + [eos] != tok(full.rstrip("\n"), add_special_tokens=False)["input_ids"]: boundary_bad += 1
        st[d]["kept"] += 1
        out.append({"prompt_text": prompt, "completion": comp, "sub_domain": d, "n_prompt_tokens": n_p, "n_completion_tokens": n_c,
                    "correct": md.get("correct"), "source_dialog_id": md.get("source_dialog_id"), "turn_index": md.get("turn_index"),
                    "task_id": md.get("task_id"), "has_tool_call": bool(tgt.get("tool_calls"))})
        if (i + 1) % 5000 == 0: print("  %d/%d (%.0fs)" % (i + 1, len(rows), time.time() - t0), flush=True)
    random.Random(a.seed).shuffle(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(a.out + ".tmp", a.out)
    q = lambda L, x: (sorted(L)[int(x * (len(L) - 1))] if L else None)
    man = {"source": a.src, "tokenizer": a.tokenizer, "max_len": a.max_len, "n_out": len(out),
           "per_sub_domain": {d: dict(c) for d, c in st.items()},
           "prompt_tokens": {d: {"p50": q(L, .5), "p90": q(L, .9), "p99": q(L, .99), "max": max(L)} for d, L in ptoks.items()},
           "completion_tokens": {d: {"p50": q(L, .5), "p90": q(L, .9), "p99": q(L, .99), "max": max(L)} for d, L in ctoks.items()},
           "total_tokens": int(sum(r["n_prompt_tokens"] + r["n_completion_tokens"] for r in out)),
           "rows_with_tool_call": int(sum(r["has_tool_call"] for r in out)),
           "boundary_check": {"checked": boundary_checked, "mismatch": boundary_bad},
           "tools_sha": {d: hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()[:12] for d, t in tools.items()},
           "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}
    json.dump(man, open(a.out.replace(".jsonl", ".manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(man, indent=1, ensure_ascii=False)); print("wrote %s (%d rows)" % (a.out, len(out)))
    r0 = out[0]; print("---- sample completion head ----"); print(r0["completion"][:300].replace("\n", "\\n")); print("---- sample completion tail ----"); print(r0["completion"][-200:].replace("\n", "\\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
