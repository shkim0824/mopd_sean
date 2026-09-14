# SFT throughput study (Qwen3-4B-Base, MoT math+code, 32k max_len, 8×A100-80GB) — 2026-08-30

Baseline (mopd/ main tree): sdpa attention, no packing, HF default loss, per_device_bs=1 (pd>=2 OOM) → 99 s/step,
1.17M tok/step, **11.7k tok/s** (1.47k tok/s/GPU, MFU ~15 %), GPU util 41 %. 3.2B tokens (2 epochs) = 75 h.

## Root causes
1. HF Trainer's full-vocab logits: 32k × 151,936 × (bf16 + fp32 upcast + grad) ≈ 40–60 GB per sequence → this,
   not the 4B model, forced per_device_bs=1 ("Tried to allocate 21.23 GiB" = 32768×151936×4 B).
2. One ~9k-token sequence per micro-step (no packing) → small kernels, ZeRO-2 comm + grad-ckpt overhead per sequence.
3. sdpa (no flash-attn) at 12–25k tokens: ~2× slower attention and no padding-free varlen packing.

## Sweep (10 optimizer steps, 128 rows/step, 6000-row sample, avg 9,158 tok/example)
| config | pd | s/step | tok/step | tok/s | note |
|---|---|---|---|---|---|
| sdpa, no pack (baseline) | 1 | 99.0 | 1.17M | 11.7k | pd2 OOM |
| FA2 + Liger, no pack | 1 | 92.6 | 1.17M | 12.7k | |
| FA2 + Liger, no pack | 2 | 84.1 | 1.17M | 13.9k | pd2 fits thanks to Liger |
| FA2 + Liger + flatten pack | 1 | 150.8 | 4.19M | 27.8k | bin fill 99.8 % |
| FA2 + Liger + flatten pack | 2 | 143.9 | 4.19M | 29.1k | |
| **FA2 + Liger + flatten pack** | **4** | **137.6** | **4.19M** | **30.4k** | **chosen** (3.8k tok/s/GPU, MFU ~39 %) |
| FA2 + Liger + flatten pack | 8, 16 | OOM | | | |
| flatten, grad-ckpt OFF | 1, 2 | OOM | | | grad-ckpt stays on |

Community reference (open-r1 OpenR1-Distill-7B: FA2 + Liger + grad-ckpt, max_length 32768, pd 2, ga 8, ZeRO-3;
TRL default packing=bfd padding-free; Hermes 4 FFD packing; realistic 2–3k tok/s per A100 at this scale) →
our 3.8k tok/s/GPU is at the upper end; no further large software gain is expected on A100 short of more GPUs.

## Final acceleration settings (configs/sft.yaml; training hyper-parameters unchanged)
- `model.attn: auto` → flash_attention_2 (prebuilt wheel flash_attn 2.8.3+cu128torch2.9 cp313 in .venv/mopd)
- `data.packing: flatten` (first-fit-decreasing bins ≤ 32k, position_ids reset per example → varlen block-diagonal
  attention; loss identical to unpacked; tests/test_flatten.py)
- `train.liger: true` (use_liger_kernel: fused linear-CE + RMSNorm/RoPE/SwiGLU), `train.grad_ckpt: true`
- `train.per_device_bs: 4`, `train.grad_accum: 1` → 32 bins ≈ 115 examples ≈ 1.05M tokens per optimizer step
  (≈ the 128-example batch the lr 2e-5 schedule was set for); `train.epochs: 1.0`
- Wall-clock: 1.6B tokens / 30.4k tok/s ≈ **14.6 h on 1 node**; 2 nodes (scripts/sft.sbatch, halve
  grad_accum→ keep ga=1 and pd=4: 64 bins/step ≈ 230 examples; or pd=2 to keep 32 bins) ≈ **8 h**.
  Checkpoints: every 100 steps (~4 h at 137 s/step on 1 node → use save_steps 50 for ~2 h cadence), save_total_limit 2.
