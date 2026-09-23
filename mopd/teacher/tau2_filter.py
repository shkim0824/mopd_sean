"""Rejection filter for tau2 teacher samples -> SFT warm-up rows (the tau2 analogue of mopd.teacher.reject_filter).

  python -m mopd.teacher.tau2_filter --gen-dir outputs/teacher_gen/<TAG> --pool data/rl_pools/tau_opd_train.jsonl \
      --ref data/sft_distill/tau2_sft.jsonl --tools-dir data/tau2 --out data/sft_warmup/tau2_teacher_k4_keep3.jsonl [--keep 3]

Per prompt keep samples that: finished ('stop'), open AND close exactly one think block, contain no CJK, and agree with the
reference AReaL turn in KIND: if the reference turn calls a tool, the sample must call a tool whose name exists in the
sub-domain schema and equals the reference tool name (arguments free); if the reference is a text reply, the sample must
be a non-empty text reply with no tool call. Then keep the <= --keep shortest mutually-distinct samples (difflib ratio on the
first 2000 chars < --sim). Rows: {prompt_text, completion, sub_domain, source_dialog_id, turn_index, ref_tool}.
"""
from __future__ import annotations
import argparse, difflib, glob, json, os, re
from collections import Counter, defaultdict
CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
def ref_tool(completion: str):
    m = TOOL_RE.search(completion or "")
    if not m: return None
    try: return json.loads(m.group(1)).get("name")
    except Exception: return "?"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gen-dir", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--ref", required=True); ap.add_argument("--tools-dir", default="data/tau2"); ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=3); ap.add_argument("--sim", type=float, default=0.85); ap.add_argument("--max-chars", type=int, default=20000)
    a = ap.parse_args()
    tools = {d: {t["function"]["name"] for t in json.load(open(os.path.join(a.tools_dir, "tools_%s.json" % d)))} for d in ("airline", "retail", "telecom")}
    pool = {}
    for line in open(a.pool, encoding="utf-8"):
        r = json.loads(line); pool[r["id"]] = r
    ref = {}
    for line in open(a.ref, encoding="utf-8"):
        r = json.loads(line); ref[(r.get("source_dialog_id"), r.get("turn_index"))] = r
    st = Counter(); out_rows = []; per_prompt = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.gen_dir, "shard*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            g = json.loads(line); pr = pool.get(g["id"]); st["gen_rows"] += 1
            if not pr: st["no_pool_row"] += 1; continue
            rr = ref.get((pr.get("source_dialog_id"), pr.get("turn_index")))
            if not rr: st["no_reference"] += 1; continue
            sub = pr.get("sub_domain"); rtool = ref_tool(rr.get("completion", "")); kept = []
            for s in g.get("samples") or []:
                st["samples"] += 1; txt = (s.get("text") or "").strip()
                if s.get("finish") != "stop": st["drop_finish"] += 1; continue
                if txt.count("<think>") != 1 or txt.count("</think>") != 1 or not txt.startswith("<think>"): st["drop_think"] += 1; continue
                if CJK.search(txt): st["drop_cjk"] += 1; continue
                if len(txt) > a.max_chars: st["drop_long"] += 1; continue
                after = txt.split("</think>", 1)[1]; calls = TOOL_RE.findall(after)
                if rtool is not None:
                    if not calls: st["drop_expected_tool"] += 1; continue
                    try: name = json.loads(calls[0]).get("name")
                    except Exception: st["drop_tool_json"] += 1; continue
                    if name not in tools.get(sub, set()): st["drop_unknown_tool"] += 1; continue
                    if name != rtool: st["drop_tool_mismatch"] += 1; continue
                    if len(calls) > 1: st["drop_multi_tool"] += 1; continue
                else:
                    if calls: st["drop_unexpected_tool"] += 1; continue
                    if not after.strip(): st["drop_empty_text"] += 1; continue
                kept.append(txt)
            kept.sort(key=len); chosen = []
            for t in kept:
                if all(difflib.SequenceMatcher(None, t[:2000], c[:2000]).ratio() < a.sim for c in chosen): chosen.append(t)
                if len(chosen) >= a.keep: break
            st["prompts"] += 1; st["prompts_with_keeper"] += bool(chosen); st["kept"] += len(chosen)
            for t in chosen:
                out_rows.append({"prompt_text": pr["prompt_text"], "completion": t, "sub_domain": sub, "source_dialog_id": pr.get("source_dialog_id"),
                                 "turn_index": pr.get("turn_index"), "ref_tool": rtool, "n_prompt_tokens": pr.get("n_prompt_tokens")})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st["per_sub_domain"] = dict(Counter(r["sub_domain"] for r in out_rows))
    json.dump(dict(st), open(a.out + ".stats.json", "w"), indent=1); print(json.dumps(dict(st), indent=1)); print("wrote", a.out, len(out_rows))
if __name__ == "__main__": main()
