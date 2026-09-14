"""Rejection-sample teacher shards into an SFT messages jsonl.

  python -m mopd.teacher.reject_filter --gen-dir OUT --pool pool.jsonl \
      --out sft.jsonl [--keep 2] [--sim 0.85] [--max-chars 60000]

Per prompt: keep only samples whose extracted final letter == gold (letter exact match,
same extractor as eval), that finished ('stop'), have exactly one </think>, no CJK and
no code fences; then keep the <= --keep SHORTEST mutually-distinct traces (difflib ratio
on the first 2000 chars of the reasoning < --sim). Assistant text is normalised to the
existing SFT schema: '<think>\n...\n</think>\n\n<answer text ending in "Answer: X">'.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
from collections import Counter, defaultdict

from mopd.graders.domains.extract import extract_letter

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
N_OPT = {"mcqa4": 4, "mcqa5": 5, "mcqa10": 10}


def normalise(text: str):
    """Return (reasoning, answer_text) or None if the trace is malformed."""
    t = text.strip()
    if t.count("</think>") != 1:
        return None
    if t.startswith("<think>"):
        t = t[len("<think>"):]
    reasoning, answer = t.split("</think>", 1)
    reasoning, answer = reasoning.strip(), answer.strip()
    if len(reasoning) < 50 or len(answer) < 3:
        return None
    return reasoning, answer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep", type=int, default=2)
    p.add_argument("--sim", type=float, default=0.85)
    p.add_argument("--max-chars", type=int, default=60000)
    args = p.parse_args()

    pool = {}
    for i, l in enumerate(open(args.pool)):
        r = json.loads(l); pool[r.get("id", str(i))] = r
    drop = Counter(); per_src_prompts = Counter(); per_src_kept = Counter()
    per_src_any = Counter(); sample_correct = Counter(); sample_total = Counter()
    kept_lens = []; n_prompts = 0; n_out = 0
    shards = sorted(glob.glob(os.path.join(args.gen_dir, "shard*.jsonl")))
    with open(args.out, "w") as fo:
        for sh in shards:
            for line in open(sh):
                g = json.loads(line); n_prompts += 1
                meta = g.get("meta") or {}; src = meta.get("source", "?")
                n_opt = N_OPT.get(meta.get("kind", "mcqa4"), 4)
                gold = str(g["gold"]).strip().upper()
                per_src_prompts[src] += 1
                cands = []
                for s in g["samples"]:
                    sample_total[src] += 1
                    txt = s["text"]
                    if s.get("finish") != "stop":
                        drop["truncated"] += 1; continue
                    if len(txt) > args.max_chars:
                        drop["too_long"] += 1; continue
                    norm = normalise(txt)
                    if norm is None:
                        drop["malformed_think"] += 1; continue
                    reasoning, answer = norm
                    if extract_letter(txt, n_options=n_opt) != gold:
                        drop["wrong_answer"] += 1; continue
                    sample_correct[src] += 1
                    if CJK.search(reasoning) or CJK.search(answer):
                        drop["cjk"] += 1; continue
                    if "```" in reasoning or "```" in answer:
                        drop["code_fence"] += 1; continue
                    cands.append((len(txt), reasoning, answer))
                if not cands:
                    continue
                per_src_any[src] += 1
                cands.sort(key=lambda c: c[0])
                chosen = []
                for L, reasoning, answer in cands:
                    if all(difflib.SequenceMatcher(None, reasoning[:2000], c[1][:2000]).ratio() < args.sim
                           for c in chosen):
                        chosen.append((L, reasoning, answer))
                    else:
                        drop["near_duplicate"] += 1
                    if len(chosen) >= args.keep:
                        break
                prow = pool.get(g["id"], {})
                user = prow.get("input")
                if user is None:
                    drop["no_pool_row"] += 1; continue
                for L, reasoning, answer in chosen:
                    assistant = f"<think>\n{reasoning}\n</think>\n\n{answer}"
                    fo.write(json.dumps({"messages": [{"role": "user", "content": user},
                                                      {"role": "assistant", "content": assistant}],
                                         "meta": {"id": g["id"], "source": src, "kind": meta.get("kind"),
                                                  "teacher_chars": L}}, ensure_ascii=False) + "\n")
                    kept_lens.append(L); n_out += 1
                    per_src_kept[src] += 1
    kept_lens.sort()
    q = lambda f: kept_lens[int(f * (len(kept_lens) - 1))] if kept_lens else 0
    stats = {"shards": len(shards), "prompts": n_prompts, "sft_rows": n_out,
             "prompts_with_keeper": sum(per_src_any.values()),
             "per_source": {s: {"prompts": per_src_prompts[s], "any_correct": per_src_any[s],
                                "kept_rows": per_src_kept[s],
                                "teacher_sample_acc": round(sample_correct[s] / max(1, sample_total[s]), 4)}
                            for s in per_src_prompts},
             "drop_reasons": dict(drop),
             "kept_chars": {"p50": q(.5), "p90": q(.9), "p99": q(.99), "max": kept_lens[-1] if kept_lens else 0}}
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
