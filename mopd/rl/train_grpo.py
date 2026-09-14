"""Stage 2 — per-domain RL expert from the SFT checkpoint (2 nodes = 16 GPU, trl 0.29 GRPO,
vLLM colocate, DeepSpeed ZeRO-2, verifiable reward of ONE domain).

  accelerate launch --config_file <accel.yaml> --num_machines 2 --num_processes 16 \\
      --machine_rank $SLURM_NODEID --main_process_ip $HEAD --main_process_port 29500 \\
      -m mopd.rl.train_grpo --config configs/rl_math.yaml --override model.init=outputs/sft/qwen3-8b

Paper recipe (Appendix A): on-policy GRPO + Dynamic Sampling (DAPO), lr 3e-6,
BS 144 prompts x N=8 rollouts, ~175K sequences per domain (math/IF), max len 32,768.
Mapping to trl 0.29 GRPOConfig:
  per_device_train_batch_size * world * grad_accum == num_generations * prompts_per_step
  generation_batch_size = num_generations * prompts_per_step  (144*8 = 1152 sequences/step)
  loss_type="dapo"  (token-level, no length bias)   beta=0 (no ref-KL)   epsilon_high=0.28 (DAPO clip-higher)
  "single gradient update then discard" == num_iterations=1 (default)
  Dynamic sampling: trl 0.29 has no built-in resample-until-nonzero-variance loop. Groups whose
  8 rollouts all share one reward get zero advantage (= zero gradient), trl logs their share as
  ``frac_reward_zero_std`` — raise ``rl.prompts_per_step`` if it is large. This is the only
  approximation of the paper's Stage-2 recipe (documented in docs/DESIGN.md).
  The per-domain RL teacher is the exported ``final/`` dir (+ every checkpoint-N/).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from mopd.common import chat
from mopd.common.config import add_config_args, dump, load_config
from mopd.common.io import read_jsonl
from mopd.common.resume import find_resume_checkpoint
from mopd.eval import prompts as P
from mopd.rl.rewards import DomainReward

DEFAULTS = {
    "model": {"init": "outputs/sft/qwen3-8b", "dtype": "float32"},
    "domain": "math",
    "data": {"path": "data/train/rl_math.jsonl", "max_rows": None, "seed": 42, "math_style": "qwen", "lcb_system": False,
             "thinking": True},
    "rl": {"lr": 3e-6, "prompts_per_step": 144, "num_generations": 8, "max_completion_length": 32768,
           "max_steps": 152, "per_device_bs": 4, "temperature": 1.0, "top_p": 1.0,
           "beta": 0.0, "epsilon": 0.2, "epsilon_high": 0.28, "loss_type": "dapo", "scale_rewards": "group",
           "mask_truncated_completions": True, "warmup_ratio": 0.0, "scheduler": "constant", "max_grad_norm": 1.0,
           "save_steps": 10, "save_total_limit": None, "save_only_model": False, "resume": "auto",
           "logging_steps": 1, "vllm_gpu_memory_utilization": 0.3, "vllm_tp": 1, "vllm_sleep": False, "vllm_max_model_len": None,
           # trl 0.29 default vllm_importance_sampling_mode="sequence_mask" = exp(SUM_t logp diff) masked at 3.0 -> for
           # 4k-16k-token completions the ratio is ~0 or masked and grad_norm collapses to 1e-7 (verified 2026-08-30,
           # trl issues #4772/#5814). Token-level truncated IS (Yao et al. TIS) is the long-CoT convention.
           "vllm_is_mode": "token_truncate", "vllm_is_cap": 3.0,
           "ddp_timeout": 7200,
           "output": "outputs/rl/qwen3-8b-math"},
    "reward": {"code_workers": 24, "code_timeout": 6, "if_all_or_nothing": False, "require_finished_thinking": True},
}


def _patch_runtime():
    # 1) tp>1 colocate vLLM hits custom_all_reduce 'invalid argument' on this cluster
    try:
        from vllm.entrypoints.llm import LLM as _LLM
        _orig = _LLM.__init__

        def _init(self, *a, **k):
            k["disable_custom_all_reduce"] = True
            _orig(self, *a, **k)
        _LLM.__init__ = _init
    except Exception:
        pass
    # 2) trl 0.29 VLLMGeneration.generate() calls llm.collective_rpc("reload_weights") when vllm_enable_sleep_mode=True
    #    (workaround for vllm#29341). vLLM's reload_weights re-reads model.name_or_path from DISK and overwrites the
    #    weights that sync_weights() just pushed -> every rollout comes from the INITIAL checkpoint (trl issue #5312,
    #    fix PR #5313 unmerged). sync_weights() already wakes the "weights" tag and load_weights() every parameter, so
    #    the reload is redundant for a non-quantized model: make it a no-op.
    try:
        from vllm.entrypoints.llm import LLM as _LLM2
        _orig_rpc = _LLM2.collective_rpc

        def _rpc(self, method, *a, **k):
            if method == "reload_weights":
                return []
            return _orig_rpc(self, method, *a, **k)
        _LLM2.collective_rpc = _rpc
    except Exception:
        pass
    # 3) slow code-grading batches must not trip the 30-min NCCL watchdog
    import torch.distributed as dist
    _o = dist.init_process_group

    def _pg(*a, **k):
        k.setdefault("timeout", timedelta(hours=2))
        return _o(*a, **k)
    dist.init_process_group = _pg


def prompt_token_budget(cfg, tok) -> int:
    """Rollout prompts must satisfy prompt + completion <= vLLM max_model_len (else vllm raises
    "decoder prompt is longer than the maximum model length" mid-training — hit at step 28 of the
    first IF run: one IF_multi_constraints prompt is 36.6k tokens)."""
    from transformers import AutoConfig
    mm = getattr(AutoConfig.from_pretrained(cfg.model.init, trust_remote_code=True), "max_position_embeddings", 32768)
    if cfg.rl.vllm_max_model_len:
        mm = min(mm, cfg.rl.vllm_max_model_len)
    return mm - cfg.rl.max_completion_length


def build_dataset(cfg, tok=None) -> Dataset:
    rows = read_jsonl(cfg.data.path, limit=cfg.data.max_rows)
    d = cfg.domain
    recs = []
    for r in rows:
        if d == "math":
            msgs = P.math_messages(r["prompt"], cfg.data.math_style)
            rec = {"prompt": msgs, "domain": "math", "answer": str(r["answer"]), "id": r["id"]}
        elif d == "code":
            msgs = P.code_messages(r["prompt"], r.get("starter_code") or "", system=cfg.data.lcb_system)
            rec = {"prompt": msgs, "domain": "code", "tests": json.dumps(r["tests"]), "id": r["id"]}
        elif d == "if":
            msgs = P.if_messages(r["prompt"])
            rec = {"prompt": msgs, "domain": "if", "ground_truth": r["ground_truth"], "id": r["id"]}
        else:
            raise ValueError(d)
        recs.append(rec)
    if tok is not None:
        budget = prompt_token_budget(cfg, tok)
        rendered = tok.apply_chat_template([r["prompt"] for r in recs], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=cfg.data.thinking)
        lens = [len(ids) for ids in tok(rendered, add_special_tokens=False)["input_ids"]]
        kept = [r for r, n in zip(recs, lens) if n <= budget]
        if is_main():
            print(f"[grpo] prompt-length filter: budget={budget} tokens, dropped {len(recs) - len(kept)}/{len(recs)}"
                  f" (max seen {max(lens)})", flush=True)
        recs = kept
    ds = Dataset.from_list(recs)
    return ds.shuffle(seed=cfg.data.seed)


class MopdGRPOTrainer(GRPOTrainer):
    def log(self, logs, *a, **k):
        for rf in self.reward_funcs:
            fl = getattr(rf, "flush_stats", None)
            if callable(fl):
                logs.update(fl())
        super().log(logs, *a, **k)


def is_main():
    return os.environ.get("RANK", "0") in ("0", "")


def main(argv=None):
    _patch_runtime()
    p = argparse.ArgumentParser()
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.override, base=DEFAULTS)
    if is_main():
        print("[grpo] config:\n" + dump(cfg), flush=True)

    tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.model.init, trust_remote_code=True))
    ds = build_dataset(cfg, tok)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    gen_bs = cfg.rl.prompts_per_step * cfg.rl.num_generations
    assert gen_bs % (cfg.rl.per_device_bs * world) == 0, \
        f"prompts_per_step*num_generations={gen_bs} must be divisible by per_device_bs*world={cfg.rl.per_device_bs * world}"
    grad_accum = gen_bs // (cfg.rl.per_device_bs * world)
    reward = DomainReward(log_dir=cfg.rl.output, code_workers=cfg.reward.code_workers, code_timeout=cfg.reward.code_timeout,
                          if_all_or_nothing=cfg.reward.if_all_or_nothing,
                          require_finished_thinking=cfg.reward.require_finished_thinking, thinking=cfg.data.thinking)

    targs = GRPOConfig(
        output_dir=cfg.rl.output,
        model_init_kwargs={"dtype": cfg.model.dtype, "attn_implementation": "sdpa"},  # fp32 master weights: bf16 Adam rounds lr=3e-6 to 0
        max_steps=cfg.rl.max_steps,
        per_device_train_batch_size=cfg.rl.per_device_bs,
        gradient_accumulation_steps=grad_accum,
        generation_batch_size=gen_bs,
        num_generations=cfg.rl.num_generations,
        max_completion_length=cfg.rl.max_completion_length,
        learning_rate=cfg.rl.lr, lr_scheduler_type=cfg.rl.scheduler, warmup_ratio=cfg.rl.warmup_ratio,
        max_grad_norm=cfg.rl.max_grad_norm, weight_decay=0.0,
        beta=cfg.rl.beta, epsilon=cfg.rl.epsilon, epsilon_high=cfg.rl.epsilon_high, loss_type=cfg.rl.loss_type,
        scale_rewards=cfg.rl.scale_rewards, mask_truncated_completions=cfg.rl.mask_truncated_completions,
        temperature=cfg.rl.temperature, top_p=cfg.rl.top_p,
        num_iterations=1,
        save_steps=cfg.rl.save_steps, save_strategy="steps", save_only_model=cfg.rl.save_only_model,
        save_total_limit=cfg.rl.save_total_limit, logging_steps=cfg.rl.logging_steps,
        report_to="tensorboard", logging_dir=os.path.join(cfg.rl.output, "tb"),
        bf16=True, tf32=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=cfg.rl.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=cfg.rl.vllm_tp,
        vllm_enable_sleep_mode=bool(cfg.rl.vllm_sleep),            # offload vLLM weights/KV during the train phase (fork study)
        vllm_max_model_length=cfg.rl.vllm_max_model_len,            # prompt + completion cap (KV budget)
        vllm_importance_sampling_mode=cfg.rl.vllm_is_mode, vllm_importance_sampling_cap=cfg.rl.vllm_is_cap,
        chat_template_kwargs={"enable_thinking": cfg.data.thinking},
        remove_unused_columns=False, ddp_timeout=cfg.rl.ddp_timeout, seed=cfg.data.seed,
        log_completions=True, num_completions_to_print=2,
    )
    trainer = MopdGRPOTrainer(model=cfg.model.init, args=targs, train_dataset=ds, reward_funcs=[reward], processing_class=tok)
    t0 = time.time()
    resume = find_resume_checkpoint(cfg.rl.output, cfg.rl.resume)
    if is_main():
        print(f"[grpo] resume_from_checkpoint = {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    final = os.path.join(cfg.rl.output, "final")
    trainer.save_model(final)
    if is_main():
        tok.save_pretrained(final)
        with open(os.path.join(final, "generation_config.json"), "w") as f:
            json.dump(chat.generation_config_dict(cfg.data.thinking), f, indent=2)
        with open(os.path.join(cfg.rl.output, "mopd_rl_done.json"), "w") as f:
            json.dump({"config": json.loads(json.dumps(cfg)), "grad_accum": grad_accum, "world": world,
                       "seconds": time.time() - t0}, f, indent=2)
        print("[grpo] DONE ->", final, flush=True)


if __name__ == "__main__":
    main()
