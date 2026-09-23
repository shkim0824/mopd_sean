"""Stage 3 — MOPD: distil the frozen per-domain RL teachers into the SFT student on the
student's own rollouts. Runs on the STUDENT node (8 ranks, accelerate + ZeRO-2, colocate
vLLM); the teachers are vLLM servers on the other node(s), reached through
``teacher_endpoints.json`` written by the sbatch from ``mopd.distill.placement``.

  accelerate launch --config_file <accel.yaml> --num_processes 8 -m mopd.distill.train_mopd \\
      --config configs/mopd.yaml --override model.student=outputs/sft/qwen3-8b \\
      --override teachers.endpoints_file=outputs/mopd/run1/teacher_endpoints.json

Paper hyper-parameters (Appendix A): BS 2048 sequences, N=1 rollout/prompt, no dynamic
sampling, domain ratio math:if:swe = 0.35:0.35:0.30 (here math:code:if), advantage clip
A_max=5 (PG form), k=64 (top-k form), max len 32,768, sampling T=1.0. LR is not stated;
default 1e-6 (verl / TRL MOPD examples).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import timedelta

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig

from mopd.common import chat
from mopd.common.config import add_config_args, dump, load_config
from mopd.common.io import read_jsonl
from mopd.common.resume import find_resume_checkpoint
from mopd.distill.teacher_client import TeacherPool, load_endpoints
from mopd.distill.trainer import MOPDTrainer
from mopd.eval import prompts as P
from mopd.rl.rewards import DomainReward

DEFAULTS = {
    "model": {"student": "outputs/sft/qwen3-8b", "dtype": "float32"},
    "teachers": {"endpoints_file": "", "top_k": 64, "max_workers": 64, "timeout": 900},
    "data": {"paths": {"math": "data/train/rl_math.jsonl", "code": "data/train/rl_code.jsonl", "if": "data/train/rl_if.jsonl"},
             "ratio": {"math": 0.35, "code": 0.30, "if": 0.35}, "max_rows_per_domain": None, "seed": 42,
             "math_style": "qwen", "lcb_system": False, "thinking": True},
    "distill": {"mode": "pg", "a_max": 5.0, "task_reward_coef": 0.0, "log_task_reward": False},
    "train": {"lr": 1e-6, "batch_size": 2048, "num_generations": 1, "max_completion_length": 32768, "max_steps": 60,
              "per_device_bs": 2, "temperature": 1.0, "top_p": 1.0, "beta": 0.0, "scheduler": "constant",
              "warmup_ratio": 0.0, "max_grad_norm": 1.0, "save_steps": 5, "save_total_limit": 2, "save_only_model": False,
              "resume": "auto", "logging_steps": 1,
              "vllm_gpu_memory_utilization": 0.3, "vllm_tp": 1, "ddp_timeout": 7200, "mask_truncated_completions": True,   # user order 2026-09-17: on for every run from now on
              "output": "outputs/mopd/qwen3-8b"},
}


def _patch_runtime():
    try:
        from vllm.entrypoints.llm import LLM as _LLM
        _orig = _LLM.__init__

        def _init(self, *a, **k):
            k["disable_custom_all_reduce"] = True
            _orig(self, *a, **k)
        _LLM.__init__ = _init
    except Exception:
        pass
    import torch.distributed as dist
    _o = dist.init_process_group

    def _pg(*a, **k):
        k.setdefault("timeout", timedelta(hours=2))
        return _o(*a, **k)
    dist.init_process_group = _pg


def build_mixture(cfg) -> Dataset:
    """Per-batch domain ratio is realised as a dataset-level ratio with a fixed interleave
    (row i of the shuffled dataset is math/code/if with prob = ratio) — every batch of 2048
    then matches the ratio in expectation with tiny variance."""
    rng = random.Random(cfg.data.seed)
    pools = {}
    for d, p in cfg.data.paths.items():
        if not p or not os.path.exists(p):
            continue
        rows = read_jsonl(p, limit=cfg.data.max_rows_per_domain)
        rng.shuffle(rows)
        pools[d] = rows
    ratio = {d: float(cfg.data.ratio[d]) for d in pools}
    z = sum(ratio.values())
    ratio = {d: r / z for d, r in ratio.items()}
    total = min(int(len(pools[d]) / ratio[d]) for d in pools)  # largest mixture the pools support
    recs = []
    counts = {d: int(round(total * ratio[d])) for d in pools}
    for d, n in counts.items():
        for r in pools[d][:n]:
            if d == "math":
                rec = {"prompt": P.math_messages(r["prompt"], cfg.data.math_style), "domain": "math",
                       "answer": str(r["answer"]), "tests": "", "ground_truth": "", "id": r["id"]}
            elif d == "code":
                rec = {"prompt": P.code_messages(r["prompt"], r.get("starter_code") or "", system=cfg.data.lcb_system),
                       "domain": "code", "answer": "", "tests": json.dumps(r["tests"]), "ground_truth": "", "id": r["id"]}
            else:
                rec = {"prompt": P.if_messages(r["prompt"]), "domain": "if", "answer": "", "tests": "",
                       "ground_truth": r["ground_truth"], "id": r["id"]}
            recs.append(rec)
    rng.shuffle(recs)
    return Dataset.from_list(recs), counts


def is_main():
    return os.environ.get("RANK", "0") in ("0", "")


def main(argv=None):
    _patch_runtime()
    p = argparse.ArgumentParser()
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.override, base=DEFAULTS)
    if is_main():
        print("[mopd] config:\n" + dump(cfg), flush=True)
    assert cfg.teachers.endpoints_file, "teachers.endpoints_file is required"
    endpoints, names = load_endpoints(cfg.teachers.endpoints_file)
    for d, pth in dict(cfg.data.paths).items():
        if not pth:  # empty path = domain disabled (build_mixture skips it too)
            continue
        assert d in endpoints, f"no teacher endpoint for domain {d}: {endpoints}"
    pool = TeacherPool(endpoints, names, top_k=(cfg.teachers.top_k if cfg.distill.mode == "topk" else 0),
                       max_workers=cfg.teachers.max_workers, timeout=cfg.teachers.timeout)
    if is_main():
        print("[mopd] teacher health:", pool.wait_ready(timeout=600), flush=True)

    tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.model.student, trust_remote_code=True))
    ds, counts = build_mixture(cfg)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    N = cfg.train.num_generations
    gen_bs = cfg.train.batch_size
    assert gen_bs % (cfg.train.per_device_bs * world) == 0
    grad_accum = gen_bs // (cfg.train.per_device_bs * world)
    task_reward = None
    if cfg.distill.task_reward_coef or cfg.distill.log_task_reward:
        task_reward = DomainReward(log_dir=cfg.train.output, thinking=cfg.data.thinking)

    targs = GRPOConfig(
        output_dir=cfg.train.output,
        model_init_kwargs={"dtype": cfg.model.dtype, "attn_implementation": "sdpa"},
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_bs,
        gradient_accumulation_steps=grad_accum,
        generation_batch_size=gen_bs,
        num_generations=max(N, 2),  # trl init check; overridden to N below
        max_completion_length=cfg.train.max_completion_length,
        learning_rate=cfg.train.lr, lr_scheduler_type=cfg.train.scheduler, warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm, weight_decay=0.0,
        beta=cfg.train.beta, loss_type="grpo", scale_rewards="none", epsilon=0.2, epsilon_high=0.2,
        mask_truncated_completions=cfg.train.mask_truncated_completions,
        temperature=cfg.train.temperature, top_p=cfg.train.top_p, num_iterations=1,
        save_steps=cfg.train.save_steps, save_strategy="steps", save_only_model=cfg.train.save_only_model,
        save_total_limit=cfg.train.save_total_limit, logging_steps=cfg.train.logging_steps,
        report_to="tensorboard", logging_dir=os.path.join(cfg.train.output, "tb"),
        bf16=True, tf32=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=cfg.train.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=cfg.train.vllm_tp, vllm_importance_sampling_correction=True,
        chat_template_kwargs={"enable_thinking": cfg.data.thinking},
        remove_unused_columns=False, ddp_timeout=cfg.train.ddp_timeout, seed=cfg.data.seed,
        log_completions=True, num_completions_to_print=2,
    )
    trainer = MOPDTrainer(model=cfg.model.student, args=targs, train_dataset=ds, processing_class=tok,
                          teacher_pool=pool, distill_mode=cfg.distill.mode, a_max=cfg.distill.a_max,
                          top_k=cfg.teachers.top_k, task_reward=task_reward, task_reward_coef=cfg.distill.task_reward_coef,
                          num_generations_override=(N if N != max(N, 2) else None))
    t0 = time.time()
    resume = find_resume_checkpoint(cfg.train.output, cfg.train.resume)
    if is_main():
        print(f"[mopd] resume_from_checkpoint = {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    final = os.path.join(cfg.train.output, "final")
    trainer.save_model(final)
    if is_main():
        tok.save_pretrained(final)
        with open(os.path.join(final, "generation_config.json"), "w") as f:
            json.dump(chat.generation_config_dict(cfg.data.thinking), f, indent=2)
        with open(os.path.join(cfg.train.output, "mopd_done.json"), "w") as f:
            json.dump({"config": json.loads(json.dumps(cfg)), "mixture_counts": counts, "grad_accum": grad_accum,
                       "world": world, "seconds": time.time() - t0, "endpoints": endpoints}, f, indent=2)
        print("[mopd] DONE ->", final, flush=True)


if __name__ == "__main__":
    main()
