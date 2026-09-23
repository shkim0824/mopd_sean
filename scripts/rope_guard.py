#!/usr/bin/env python3
"""Refuse model dirs whose config.json uses the transformers>=5 `rope_parameters` schema without a top-level `rope_theta`.
Our stack (transformers 4.57 / vLLM 0.12) silently falls back to rope_theta=10000 for such configs (Qwen3 needs 1e6) and the
model degrades with prompt length (2026-09-17: noah's tau2 checkpoints). Fix with tmp/phase3/fix_rope_config.py <src> <dst>.
    python3 scripts/rope_guard.py <model_dir> [<model_dir> ...]   -> exit 6 on the first bad dir
"""
import json, os, sys
bad = 0
for p in sys.argv[1:]:
    cp = os.path.join(p, "config.json")
    if not os.path.exists(cp):
        print("[rope_guard] %s: no config.json (skipped)" % p); continue
    c = json.load(open(cp))
    if "rope_parameters" in c and "rope_theta" not in c:
        print("[rope_guard] REFUSING %s: transformers-5 rope_parameters schema without top-level rope_theta (would be served/trained with rope_theta=10000). Run: python3 tmp/phase3/fix_rope_config.py %s <fixed_dir>" % (p, p)); bad += 1
    else:
        print("[rope_guard] ok %s: rope_theta=%s rope_scaling=%s" % (p, c.get("rope_theta"), c.get("rope_scaling")))
sys.exit(6 if bad else 0)
