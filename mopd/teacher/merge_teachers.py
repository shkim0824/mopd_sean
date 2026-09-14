"""Weighted parameter average of HF checkpoints (teacher merge used as an MOPD init).
Streams tensor by tensor (safetensors), accumulates in fp32, writes bf16 shards with the base model's config/tokenizer.

  python -m mopd.teacher.merge_teachers --out models/merged-4teachers-uniform \
      --teachers law=models/teacher-law,fin=models/teacher-fin,if=models/teacher-if,med=models/teacher-med
  python -m mopd.teacher.merge_teachers --out models/merged-4teachers-w4411 --teachers ... --weights med=0.4,law=0.4,fin=0.1,if=0.1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_LIMIT = 5_000_000_000


def _wmap(d: str):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    return {k: os.path.join(d, v) for k, v in idx.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", required=True, help="name=path,name=path,...")
    p.add_argument("--weights", default="", help="name=w,... (default: uniform)")
    p.add_argument("--base", default="models/Qwen3-4B-OT3", help="config/tokenizer source + key list")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    teachers = dict(kv.split("=", 1) for kv in a.teachers.split(","))
    weights = {n: 1.0 / len(teachers) for n in teachers}
    if a.weights:
        weights = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.weights.split(","))}
    assert set(weights) == set(teachers), (weights, teachers)
    assert abs(sum(weights.values()) - 1.0) < 1e-6, weights
    os.makedirs(a.out, exist_ok=True)
    maps = {n: _wmap(d) for n, d in teachers.items()}
    base_map = _wmap(a.base)
    keys = [k for k in base_map if k != "lm_head.weight"]  # tied embeddings: lm_head is a duplicate in some saves
    for n, m in maps.items():
        missing = [k for k in keys if k not in m]
        assert not missing, (n, missing[:5])
    handles = {}

    def get(path, key):
        if path not in handles:
            handles[path] = safe_open(path, framework="pt", device="cpu")
        return handles[path].get_tensor(key)

    shard, shard_bytes, shard_id, weight_map, total = {}, 0, 1, {}, 0

    def flush():
        nonlocal shard, shard_bytes, shard_id
        if not shard:
            return
        fn = f"model-{shard_id:05d}.safetensors"
        save_file(shard, os.path.join(a.out, fn), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fn
        print("wrote", fn, len(shard), "tensors", flush=True)
        shard, shard_bytes, shard_id = {}, 0, shard_id + 1

    for i, k in enumerate(keys):
        acc = None
        for n in teachers:
            t = get(maps[n][k], k).to(torch.float32) * weights[n]
            acc = t if acc is None else acc + t
        avg = acc.to(torch.bfloat16).contiguous()
        shard[k] = avg
        shard_bytes += avg.numel() * 2
        total += avg.numel() * 2
        if shard_bytes >= SHARD_LIMIT:
            flush()
        if i % 50 == 0:
            print(f"{i}/{len(keys)} {k}", flush=True)
    flush()
    n_sh = shard_id - 1
    for fn in list(weight_map):
        pass
    # rename shards to the HF n-of-m convention
    renamed = {}
    for j in range(1, n_sh + 1):
        old = f"model-{j:05d}.safetensors"; new = f"model-{j:05d}-of-{n_sh:05d}.safetensors"
        os.replace(os.path.join(a.out, old), os.path.join(a.out, new)); renamed[old] = new
    weight_map = {k: renamed[v] for k, v in weight_map.items()}
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    for f in os.listdir(a.base):
        if f.endswith((".json", ".jinja", ".txt")) and f != "model.safetensors.index.json":
            shutil.copy(os.path.join(a.base, f), os.path.join(a.out, f))
    cfg = json.load(open(os.path.join(a.out, "config.json"))); cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump({"teachers": teachers, "weights": weights, "dtype": "bfloat16", "base_config_from": a.base},
              open(os.path.join(a.out, "merge_info.json"), "w"), indent=2)
    k = "model.layers.0.self_attn.q_proj.weight"
    ref = sum(get(maps[n][k], k).to(torch.float32) * weights[n] for n in teachers)
    got = safe_open(os.path.join(a.out, weight_map[k]), "pt").get_tensor(k).to(torch.float32)
    print("sanity max|diff|:", (ref - got).abs().max().item(), "tensors:", len(weight_map), "total bytes:", total)


if __name__ == "__main__":
    main()
