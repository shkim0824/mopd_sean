"""GPU placement for a MOPD job: student trainer on node 0, teacher prefill
servers on the remaining node(s). stdlib only — run by the sbatch on the batch host.

  python -m mopd.distill.placement --nodes n041,n042 --teachers math:/p/math,code:/p/code,if:/p/if \\
         --teacher-tp 1 --emit env|json|table

Rules
* node 0 (all 8 GPUs) = student (accelerate 8 ranks, colocate vLLM for rollouts)
* teacher pool = 8 * (num_nodes - 1) GPUs, each replica takes ``tp`` GPUs (per teacher, default 1)
* replicas are balanced: every teacher gets floor(slots / T) replicas, the remainder goes to the
  teachers with the largest ``weight`` (the domain mixture ratio; unspecified = 1.0) — e.g. 3 teachers
  on 8 GPUs at tp=1 -> 3/3/2 (math/if get 3 because their ratio 0.35 > code 0.30); 6 teachers on
  8 slots -> 1 each + the two largest weights get a 2nd replica.
* 1..6 teachers supported (more if you add nodes); an error is raised if T*tp > pool.
* worker HTTP port = base_port + global replica index; VLLM_PORT (engine internal) = 21000 + 100*idx
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

GPUS_PER_NODE = 8


def plan(nodes: List[str], teachers: List[Dict[str, Any]], base_port: int = 8100, gpus_per_node: int = GPUS_PER_NODE,
         student_gpus: int | None = None) -> Dict[str, Any]:
    assert len(nodes) >= 2, "MOPD needs >= 2 nodes (node 0 = student, others = teachers)"
    assert 1 <= len(teachers) <= 6, f"1..6 teachers supported, got {len(teachers)}"
    student_gpus = student_gpus or gpus_per_node
    pool = [(n, g) for n in nodes[1:] for g in range(gpus_per_node)]
    tps = [int(t.get("tp", 1)) for t in teachers]
    need = sum(tps)
    assert need <= len(pool), f"teachers need {need} GPUs for one replica each but the pool has {len(pool)}"
    # slots: greedy fill — first one replica each, then extra replicas by weight (descending), largest tp first on ties
    replicas = [1] * len(teachers)
    used = need
    order = sorted(range(len(teachers)), key=lambda i: (-float(teachers[i].get("weight", 1.0)), tps[i]))
    while True:
        placed = False
        for i in order:
            if used + tps[i] <= len(pool):
                replicas[i] += 1
                used += tps[i]
                placed = True
            if used >= len(pool):
                break
        if not placed or used >= len(pool):
            break
    # assign contiguous GPU ranges, never spanning nodes (tp replicas must be on one node)
    servers = []
    cursor = 0
    idx = 0
    for i, t in enumerate(teachers):
        for _ in range(replicas[i]):
            tp = tps[i]
            # advance to a node-aligned position if the replica would cross a node boundary
            node_of = lambda c: pool[c][0]
            if cursor + tp > len(pool):
                break
            if node_of(cursor) != node_of(cursor + tp - 1):
                cursor += gpus_per_node - (cursor % gpus_per_node)
                if cursor + tp > len(pool):
                    break
            gpus = [pool[c][1] for c in range(cursor, cursor + tp)]
            servers.append({"idx": idx, "domain": t["domain"], "model": t["path"], "served_name": t.get("name", t["domain"]),
                            "node": node_of(cursor), "gpus": gpus, "tp": tp, "port": base_port + idx,
                            "vllm_port": 21000 + 100 * idx, "url": f"http://{node_of(cursor)}:{base_port + idx}"})
            cursor += tp
            idx += 1
    endpoints: Dict[str, List[str]] = {}
    for s in servers:
        endpoints.setdefault(s["domain"], []).append(s["url"])
    for t in teachers:
        assert t["domain"] in endpoints, f"teacher {t['domain']} got no replica (pool too small for its tp)"
    return {"student": {"node": nodes[0], "gpus": list(range(student_gpus)), "world": student_gpus},
            "servers": servers, "endpoints": endpoints, "unused_gpus": len(pool) - cursor}


def parse_teachers(spec: str, weights: str = "") -> List[Dict[str, Any]]:
    """'math:/path[:tp],code:/path' + optional 'math=0.35,code=0.30,if=0.35'."""
    w = {}
    for kv in filter(None, weights.split(",")):
        k, v = kv.split("=")
        w[k] = float(v)
    out = []
    for item in filter(None, spec.split(",")):
        parts = item.split(":")
        d, path = parts[0], parts[1]
        tp = int(parts[2]) if len(parts) > 2 else 1
        out.append({"domain": d, "path": path, "tp": tp, "weight": w.get(d, 1.0)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="comma list, node 0 first")
    ap.add_argument("--teachers", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--teacher-tp", type=int, default=None, help="override tp for all teachers")
    ap.add_argument("--base-port", type=int, default=8100)
    ap.add_argument("--emit", default="json", choices=["json", "env", "table"])
    a = ap.parse_args(argv)
    teachers = parse_teachers(a.teachers, a.weights)
    if a.teacher_tp:
        for t in teachers:
            t["tp"] = a.teacher_tp
    p = plan(a.nodes.split(","), teachers, a.base_port)
    if a.emit == "json":
        print(json.dumps(p, indent=2))
    elif a.emit == "env":
        print(f"STUDENT_NODE={p['student']['node']}")
        print("STUDENT_GPUS=" + ",".join(map(str, p["student"]["gpus"])))
        print(f"STUDENT_WORLD={p['student']['world']}")
        print(f"N_SERVERS={len(p['servers'])}")
        for s in p["servers"]:
            i = s["idx"]
            print(f"SERVER_{i}_DOMAIN={s['domain']}")
            print(f"SERVER_{i}_MODEL={s['model']}")
            print(f"SERVER_{i}_NAME={s['served_name']}")
            print(f"SERVER_{i}_NODE={s['node']}")
            print(f"SERVER_{i}_GPUS=" + ",".join(map(str, s["gpus"])))
            print(f"SERVER_{i}_TP={s['tp']}")
            print(f"SERVER_{i}_PORT={s['port']}")
            print(f"SERVER_{i}_VLLM_PORT={s['vllm_port']}")
            print(f"SERVER_{i}_URL={s['url']}")
    else:
        print(f"student: {p['student']['node']} gpus {p['student']['gpus']}")
        for s in p["servers"]:
            print(f"teacher[{s['idx']}] {s['domain']:6s} {s['node']} gpus={s['gpus']} tp={s['tp']} port={s['port']}")
        print(f"unused teacher GPUs: {p['unused_gpus']}")


if __name__ == "__main__":
    main()
