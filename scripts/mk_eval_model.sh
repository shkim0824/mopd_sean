#!/bin/bash
# mk_eval_model.sh <step_dir> <out_dir> <ref_hf_dir>: turn a NeMo-RL step into an eval/OPD-ready HF model dir =
# file-level symlinks to policy/weights/model/consolidated + the init model's OLD-schema config/tokenizer
# (the transformers-5 config saved by NeMo-RL loses rope_theta for the eval stack -> long-context collapse).
set -u
STEP=$(readlink -f "$1"); mkdir -p "$2"; OUT=$(readlink -f "$2"); REF=$(readlink -f "$3"); C="${STEP}/policy/weights/model/consolidated"
[[ -f "${C}/model.safetensors.index.json" ]] || { echo "no consolidated index in ${STEP}"; exit 1; }
for f in "${C}"/*.safetensors "${C}"/model.safetensors.index.json; do ln -sfn "$f" "${OUT}/$(basename "$f")"; done
for f in config.json generation_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt special_tokens_map.json added_tokens.json chat_template.jinja; do [[ -f "${REF}/$f" ]] && cp -f "${REF}/$f" "${OUT}/$f"; done
python3 - "${OUT}" <<'PY'
import json, sys, os
d = sys.argv[1]; c = json.load(open(os.path.join(d, 'config.json'))); idx = json.load(open(os.path.join(d, 'model.safetensors.index.json')))
assert c.get('rope_theta', 0) > 1e5, 'config must be old schema (rope_theta present)'
tot = sum(os.path.getsize(os.path.join(d, f)) for f in set(idx['weight_map'].values()))
print(f"eval model ready: {d} rope_theta={c['rope_theta']} weights_bytes={tot} index_total={idx['metadata'].get('total_size')}")
PY
