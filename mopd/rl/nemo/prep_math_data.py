"""Prepare the math-only RL dataset from the restored Nemotron rlvr1 blend for NeMo-RL.

Filters agent_ref == math_with_judge_simple_agent (plain CoT math; excludes ns_tools TIR
and Lean rows), keeps the blend's pass-rate curriculum ORDER (official recipe consumes it
with data.shuffle=false), and follows the official val convention (LAST 100 rows of the
blend slice -> val). Output = NeMo-RL ResponseDataset JSONL: {"input": question,
"output": expected_answer} (+ passthrough metadata columns for analysis).

  PYTHONPATH=. python nemo/prep_math_data.py \
      --rlvr <...>/Nemotron-RL-Ultra-restored/rlvr1.jsonl --out data/nemotron_math \
      [--drop-pass-rate-0] [--drop-pass-rate-1]

Notes recorded in MANIFEST: pass_rate fields are relative to NVIDIA's 550B reference
policy; rows keep their original blend order (already easy->hard).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

MATH_AGENT = "math_with_judge_simple_agent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlvr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--drop-pass-rate-0", action="store_true",
                    help="drop rows the 550B reference never solved (likely broken/too hard)")
    ap.add_argument("--drop-pass-rate-1", action="store_true",
                    help="drop rows the 550B reference always solved (no learning signal for it; "
                         "NOT necessarily saturated for a 4B — default keep)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    st = Counter()
    with open(args.rlvr) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("agent_ref") or {}).get("name") != MATH_AGENT:
                continue
            st["math_rows"] += 1
            q = (r.get("question") or "").strip()
            e = (r.get("expected_answer") or "").strip()
            if not q or not e:
                st["empty_after_restore"] += 1
                continue
            try:
                pr = float(r.get("pass_rate")) if r.get("pass_rate") not in (None, "") else None
            except (TypeError, ValueError):
                pr = None
            if args.drop_pass_rate_0 and pr == 0.0:
                st["dropped_pr0"] += 1
                continue
            if args.drop_pass_rate_1 and pr == 1.0:
                st["dropped_pr1"] += 1
                continue
            rows.append({"input": q, "output": e, "pass_rate": pr,
                         "dataset": r.get("dataset"), "uuid": r.get("uuid"),
                         "verifier_type": r.get("verifier_type")})
            st["kept"] += 1

    val = rows[-args.val_size:]
    train = rows[: len(rows) - args.val_size]
    with open(os.path.join(args.out, "train.jsonl"), "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "val.jsonl"), "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    prs = [r["pass_rate"] for r in rows if r["pass_rate"] is not None]
    manifest = {"source": os.path.abspath(args.rlvr), "agent": MATH_AGENT, "stats": dict(st),
                "n_train": len(train), "n_val": len(val),
                "pass_rate_note": "relative to NVIDIA 550B reference policy; blend order kept (easy->hard)",
                "pass_rate_hist": dict(Counter(round(p, 3) for p in prs)) if prs else {},
                "order": "original blend curriculum order preserved; consume with shuffle=false"}
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "pass_rate_hist"}, indent=1))


if __name__ == "__main__":
    main()
