"""Prepare the 3-domain RL data (user decision 2026-09-04):
  math = nvidia/Nemotron-RL-Math-v2 (restored), CoT rows only (math_with_judge agent;
         TIR/ns_tools rows excluded — single-turn GRPO), {input, output} format
  code = Ultra rlvr1 slice, code_gen_simple_agent rows, messages + unit_tests
  if   = Ultra rlvr1 slice, instruction_following_simple_agent rows,
         messages + instruction_id_list + kwargs
Val = 100 rows per domain (math: random seed 42 — Math-v2 has no curriculum order;
code/if: LAST 100 of the curriculum-ordered slice, official convention).

  PYTHONPATH=. python nemo/prep_3dom_data.py \
      --mathv2 <...>/Nemotron-RL-Math-v2/restored/train.jsonl \
      --rlvr <...>/Nemotron-RL-Ultra-restored/rlvr1.jsonl --out data/rl3dom
"""
from __future__ import annotations

import argparse
import json
import os
import random


def user_content(r):
    inp = (r.get("responses_create_params") or {}).get("input") or []
    return inp[0].get("content", "") if inp else ""


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mathv2", required=True)
    ap.add_argument("--rlvr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = {}

    # ---- math (Math-v2 CoT)
    math_rows = []
    for line in open(args.mathv2):
        r = json.loads(line)
        if (r.get("agent_ref") or {}).get("name") != "math_with_judge_simple_agent":
            continue
        q = (r.get("question") or "").strip() or user_content(r).strip()
        e = (r.get("expected_answer") or "").strip()
        if q and e:
            math_rows.append({"input": q, "output": e})
    rng = random.Random(args.seed)
    rng.shuffle(math_rows)
    write(os.path.join(args.out, "math_val.jsonl"), math_rows[: args.val_size])
    write(os.path.join(args.out, "math_train.jsonl"), math_rows[args.val_size:])
    stats["math"] = {"train": len(math_rows) - args.val_size, "val": args.val_size}

    # ---- code / if (rlvr1, keep curriculum order)
    code_rows, if_rows = [], []
    for line in open(args.rlvr):
        r = json.loads(line)
        a = (r.get("agent_ref") or {}).get("name")
        if a == "code_gen_simple_agent":
            ut = (r.get("verifier_metadata") or {}).get("unit_tests") or {}
            if ut.get("inputs") and len(ut["inputs"]) == len(ut.get("outputs") or []):
                code_rows.append({"messages": [{"role": "user", "content": user_content(r)}],
                                  "unit_tests": {"inputs": list(ut["inputs"]), "outputs": list(ut["outputs"])}})
        elif a == "instruction_following_simple_agent":
            prompt = r.get("prompt") or user_content(r)
            if_rows.append({"messages": [{"role": "user", "content": user_content(r) or prompt}],
                            "instruction_id_list": r.get("instruction_id_list") or [],
                            "kwargs": r.get("kwargs") or []})
    for name, rows in (("code", code_rows), ("if", if_rows)):
        write(os.path.join(args.out, f"{name}_val.jsonl"), rows[-args.val_size:])
        write(os.path.join(args.out, f"{name}_train.jsonl"), rows[: len(rows) - args.val_size])
        stats[name] = {"train": len(rows) - args.val_size, "val": args.val_size}

    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump({"stats": stats, "mathv2": os.path.abspath(args.mathv2),
                   "rlvr1": os.path.abspath(args.rlvr),
                   "note": "math shuffled seed 42 (no curriculum in Math-v2); code/if keep "
                           "rlvr1 curriculum order internally, but 3-domain training uses "
                           "data.shuffle=true (global mix), so order is informational"}, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
