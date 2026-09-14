#!/bin/bash
# Container verification for nemo-rl v0.7.0 (mopd_rl step 1).
# Step A (nemo-rl container): env versions + load Qwen3-4B-OT3 with transformers 5.x
#   + chat render + greedy gen + vLLM 0.20 load/gen.  Step B (current ngc-2410 +
# .venv/mopd, transformers 4.57.6): same fixed test.  Step C: compare prompt token
# ids (must be identical) + show both generations.
#   cd ~/lmalign/personal/sean/mopd_rl && sbatch -J k-sean-mopdrl-verify-container \
#     --cpus-per-task=32 --mem=600G -t 02:00:00 --exclude=ib-a100-cluster-a-n[001-040] scripts/verify_container.sbatch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH -t 02:00:00
#SBATCH --output=logs/verify-container-%j.log
set -u
S=/mnt/datafs/ib-a100-cluster-a-pri/lmalign/personal/sean
REPO=$S/mopd_rl
NEWIMG=$S/images/nemo-rl-v0.7.0.sqsh
OLDIMG=/mnt/datafs/ib-a100-cluster-a-pri/lmt/users/jay/images/ngc-2410_verl-lmt-251130_verl-0.6.0_vllm-0.10.2_flash-attn-2.8.3.sqsh
MODEL=$REPO/models/Qwen3-4B-OT3
OUT=$REPO/tmp/verify_container
mkdir -p "$OUT"

cat > "$OUT/fixed_test.py" << 'PYEOF'
import json, os, sys
tag = sys.argv[1]; out_dir = sys.argv[2]; model_path = sys.argv[3]
import torch, transformers
print(f"[{tag}] python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__}", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path)
msgs = [{"role": "user", "content": "What is 17 * 23? Please reason step by step."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
ids = tok(prompt, add_special_tokens=False)["input_ids"]
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
inp = torch.tensor([ids], device="cuda:0")
with torch.no_grad():
    gen = model.generate(inp, max_new_tokens=96, do_sample=False,
                         pad_token_id=tok.pad_token_id or 151643)
text = tok.decode(gen[0][len(ids):])
res = {"tag": tag, "torch": torch.__version__, "transformers": transformers.__version__,
       "prompt_ids_sha": hash(tuple(ids)) & 0xFFFFFFFF, "n_prompt_ids": len(ids),
       "prompt_ids_head": ids[:16], "gen_text": text}
json.dump(res, open(os.path.join(out_dir, f"hf_{tag}.json"), "w"), indent=1)
print(f"[{tag}] HF GEN >>> {text[:400]}", flush=True)
print(f"[{tag}] HF_OK", flush=True)
PYEOF

cat > "$OUT/vllm_test.py" << 'PYEOF'
import json, os, sys
tag = sys.argv[1]; out_dir = sys.argv[2]; model_path = sys.argv[3]
import vllm
from vllm import LLM, SamplingParams
print(f"[{tag}] vllm={vllm.__version__}", flush=True)
llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.7,
          tensor_parallel_size=1)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path)
msgs = [{"role": "user", "content": "What is 17 * 23? Please reason step by step."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
o = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=96))
text = o[0].outputs[0].text
json.dump({"tag": tag, "vllm": vllm.__version__, "gen_text": text},
          open(os.path.join(out_dir, f"vllm_{tag}.json"), "w"), indent=1)
print(f"[{tag}] VLLM GEN >>> {text[:400]}", flush=True)
print(f"[{tag}] VLLM_OK", flush=True)
PYEOF

echo "=== STEP A: nemo-rl v0.7.0 container ==="
srun --nodes=1 --ntasks=1 --container-image="$NEWIMG" --container-mounts=/tmp:/tmp,/mnt/datafs:/mnt/datafs \
  --container-workdir="$REPO" bash -lc "
  set -x
  ls /opt | head -20
  ls /opt/nemo-rl 2>/dev/null | head -10 || ls /opt/NeMo-RL 2>/dev/null | head -10
  PY=/opt/nemo_rl_venv/bin/python
  [ -x \$PY ] || PY=\$(ls /opt/*venv*/bin/python 2>/dev/null | head -1)
  [ -x \$PY ] || PY=python3
  echo USING_PY=\$PY; \$PY --version
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  export HF_HOME=$S/.hf_cache HF_HUB_OFFLINE=1 TORCH_CUDA_ARCH_LIST=8.0
  export CUDA_VISIBLE_DEVICES=0
  \$PY $OUT/fixed_test.py new $OUT $MODEL || echo HF_NEW_FAILED
  \$PY $OUT/vllm_test.py new $OUT $MODEL || echo VLLM_NEW_FAILED
"
echo "=== STEP B: current stack (ngc-2410 + .venv/mopd) ==="
srun --nodes=1 --ntasks=1 --container-image="$OLDIMG" --container-mounts=/tmp:/tmp,/mnt/datafs:/mnt/datafs \
  --container-workdir="$REPO" bash -lc "
  source $S/.venv/mopd/bin/activate
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  export HF_HOME=$S/.hf_cache HF_HUB_OFFLINE=1
  export CUDA_VISIBLE_DEVICES=0
  python $OUT/fixed_test.py old $OUT $MODEL || echo HF_OLD_FAILED
"
echo "=== STEP C: compare ==="
python3 - << PYEOF
import json, os
out = "$OUT"
try:
    new = json.load(open(out + "/hf_new.json")); old = json.load(open(out + "/hf_old.json"))
    same_ids = new["prompt_ids_head"] == old["prompt_ids_head"] and new["n_prompt_ids"] == old["n_prompt_ids"]
    print("PROMPT_TOKENIZATION_IDENTICAL:", same_ids)
    print("new starts:", new["gen_text"][:120].replace(chr(10), " | "))
    print("old starts:", old["gen_text"][:120].replace(chr(10), " | "))
    ok = same_ids and new["gen_text"].strip() and old["gen_text"].strip()
    print("VERIFY_RESULT:", "PASS" if ok else "FAIL")
except Exception as e:
    print("VERIFY_RESULT: FAIL", repr(e))
PYEOF
echo "VERIFY_CONTAINER_DONE"
