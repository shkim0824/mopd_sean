"""Select SFT traces from one or more gen dirs (gen_teacher format) with tunable rules.

  python -m mopd.teacher.select_traces --gen-dirs D1[,D2] --pools P1[,P2] --out sft.jsonl \
      [--keep 1] [--max-think-chars 16000] [--sim 0.85] [--rank enum_len|len] [--min-keep-src X=N]

Per prompt: candidates = samples with finish=stop, well-formed single </think>, letter == gold,
no CJK / code fences, think <= cap. Rank by (enumeration-pattern count, length) ('enum_len') or
length only; greedily keep up to --keep mutually distinct traces (difflib ratio on first 2000 chars).
"""
from __future__ import annotations
import argparse, difflib, glob, json, os, re, collections
from mopd.graders.domains.extract import extract_letter

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
ENUM = re.compile(r"\? No\.|\*\s+\*\*Option [A-E]|Option [A-E][:)]|\n\s*[-*]\s+\(?[A-E][\).]", re.I)
N_OPT = {"mcqa4": 4, "mcqa5": 5, "mcqa10": 10}

def normalise(text):
    t = text.strip()
    if t.count("</think>") != 1: return None
    if t.startswith("<think>"): t = t[len("<think>"):]
    r, a = t.split("</think>", 1); r, a = r.strip(), a.strip()
    if len(r) < 50 or len(a) < 3: return None
    return r, a

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dirs", required=True); p.add_argument("--pools", required=True)
    p.add_argument("--out", required=True); p.add_argument("--keep", type=int, default=1)
    p.add_argument("--max-think-chars", type=int, default=16000); p.add_argument("--sim", type=float, default=0.85)
    p.add_argument("--rank", default="enum_len")
    a = p.parse_args()
    pool = {}
    for pf in a.pools.split(","):
        for i, l in enumerate(open(pf)):
            r = json.loads(l); pool[r.get("id", str(i))] = r
    cands = collections.defaultdict(list); stats = collections.Counter()
    for gd in a.gen_dirs.split(","):
        for sh in sorted(glob.glob(os.path.join(gd, "shard*.jsonl"))):
            for line in open(sh):
                g = json.loads(line); pid = g["id"]; meta = g.get("meta") or {}
                n_opt = N_OPT.get(meta.get("kind", "mcqa4"), 4); gold = str(g["gold"]).strip().upper()
                for s in g["samples"]:
                    stats["samples"] += 1
                    if s.get("finish") != "stop": stats["trunc"] += 1; continue
                    nm = normalise(s["text"])
                    if nm is None: stats["malformed"] += 1; continue
                    r, ans = nm
                    if len(r) > a.max_think_chars: stats["over_cap"] += 1; continue
                    if extract_letter(s["text"], n_options=n_opt) != gold: stats["wrong"] += 1; continue
                    if CJK.search(r) or CJK.search(ans) or "```" in r: stats["cjk_or_code"] += 1; continue
                    cands[pid].append((len(ENUM.findall(r)), len(r), r, ans))
    n_out = 0; per_src = collections.Counter(); lens = []
    with open(a.out, "w") as fo:
        for pid, cs in cands.items():
            cs.sort(key=(lambda c: (c[0], c[1])) if a.rank == "enum_len" else (lambda c: c[1]))
            chosen = []
            for e, L, r, ans in cs:
                if all(difflib.SequenceMatcher(None, r[:2000], c[2][:2000]).ratio() < a.sim for c in chosen):
                    chosen.append((e, L, r, ans))
                if len(chosen) >= a.keep: break
            pr = pool.get(pid)
            if pr is None: stats["no_pool"] += 1; continue
            for e, L, r, ans in chosen:
                fo.write(json.dumps({"messages": [{"role": "user", "content": pr["input"]},
                                                  {"role": "assistant", "content": f"<think>\n{r}\n</think>\n\n{ans}"}],
                                     "meta": {"id": pid, "source": pr["meta"].get("source"), "kind": pr["meta"].get("kind"),
                                              "think_chars": L, "enum": e}}, ensure_ascii=False) + "\n")
                n_out += 1; per_src[pr["meta"].get("source")] += 1; lens.append(L)
    lens.sort(); q = lambda f: lens[int(f * (len(lens) - 1))] if lens else 0
    print(json.dumps({"prompts_with_keeper": len(cands), "rows": n_out, "per_source": dict(per_src), "drop": dict(stats),
                      "think_chars": {"p50": q(.5), "p90": q(.9), "max": lens[-1] if lens else 0}}, indent=1))

if __name__ == "__main__":
    main()
