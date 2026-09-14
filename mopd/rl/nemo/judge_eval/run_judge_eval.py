"""Evaluate an LLM judge on labeled equivalence pairs using the OFFICIAL NeMo Gym
math_with_judge protocol (Arena-Hard-derived system message + prompt template +
BIDIRECTIONAL check: order-1 must say equal, then order-2 swapped must also say equal).

  python nemo/judge_eval/run_judge_eval.py --pairs nemo/judge_eval/pairs.jsonl \
      --base-url http://localhost:8000/v1 --model judge --out results_qwen32b.json \
      --concurrency 16 [--thinking false]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

# verbatim from NeMo Gym resources_servers/math_with_judge/app.py
JUDGE_SYSTEM_MESSAGE = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""
JUDGE_PROMPT_TEMPLATE = ("<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first_answer}"
                         "\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n"
                         "{second_answer}\n<|End of Assistant B's Answer|>")
EQ, NEQ = "[[A=B]]", "[[A!=B]]"


def parse_verdict(text: str) -> bool:
    """== official label parsing: first label wins; malformed -> not-equal."""
    e, n = text.find(EQ), text.find(NEQ)
    if e < 0:
        return False
    if n < 0:
        return True
    return e < n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--thinking", default="false")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    thinking = args.thinking.lower() in ("1", "true", "yes")

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)

    def judge_once(question: str, first: str, second: str) -> tuple[bool, int]:
        r = client.chat.completions.create(
            model=args.model, temperature=0.0, max_tokens=args.max_tokens,
            messages=[{"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
                      {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
                          question=question, first_answer=first, second_answer=second)}],
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        txt = r.choices[0].message.content or ""
        return parse_verdict(txt), r.usage.completion_tokens

    def judge_pair(p):
        t0 = time.time()
        toks = 0
        try:
            first_eq, t1 = judge_once(p["question"], p["expected_answer"], p["candidate"])
            toks += t1
            if not first_eq:
                verdict = 0
            else:
                second_eq, t2 = judge_once(p["question"], p["candidate"], p["expected_answer"])
                toks += t2
                verdict = 1 if second_eq else 0
            return {**p, "verdict": verdict, "correct": verdict == p["label"],
                    "sec": round(time.time() - t0, 2), "completion_tokens": toks}
        except Exception as e:
            return {**p, "verdict": None, "correct": False, "error": repr(e)[:200]}

    pairs = [json.loads(l) for l in open(args.pairs)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(judge_pair, pairs))
    wall = time.time() - t0

    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r["correct"])
    summary = {"model": args.model, "thinking": thinking, "n": len(results),
               "overall_acc": round(sum(r["correct"] for r in results) / max(1, len(results)), 4),
               "by_kind": {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in sorted(by_kind.items())},
               "errors": sum(1 for r in results if r.get("error")),
               "wall_sec": round(wall, 1),
               "mean_completion_tokens": round(sum(r.get("completion_tokens", 0) for r in results) / max(1, len(results)), 1)}
    # positives accuracy = recall (equal said equal); negatives = specificity
    pos = [r for r in results if r["label"] == 1]
    neg = [r for r in results if r["label"] == 0]
    summary["pos_recall"] = round(sum(r["correct"] for r in pos) / max(1, len(pos)), 4)
    summary["neg_specificity"] = round(sum(r["correct"] for r in neg) / max(1, len(neg)), 4)
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    # worst failures for inspection
    fails = [r for r in results if not r["correct"]][:8]
    for r in fails:
        print("FAIL", r["kind"], "| exp:", r["expected_answer"][:80], "| cand:", r["candidate"][:80])


if __name__ == "__main__":
    main()
