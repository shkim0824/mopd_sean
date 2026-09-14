"""Domain on-policy distillation (OPD / MOPD) driver: one or several domain prompt pools, one teacher per domain.

Built on the paper implementation in mopd.distill (arXiv 2606.30406, §3.2):
  * mopd.distill.trainer.MOPDTrainer  -> trl 0.29 GRPOTrainer with the advantage/loss replaced by the
    teacher signal: mode "pg"   = Eq. 3-4 (Â_t = sg[log π_T(y_t) − log π_θ(y_t)] clipped to ±A_max)
                    mode "topk" = Eq. 5  (Σ_{v∈TopK_k(π_T)} π_θ(v) log(π_θ(v)/π_T(v)) − π_θ(v) + π_T(v),
                                          student FULL-softmax probs, no renormalisation)
  * mopd.distill.teacher_client.TeacherPool -> vLLM prefill servers (prompt_logprobs top-K, raw_logprobs)
Only the DATASET is different: this repo's RL prompt pools ({input, output, meta} jsonl) with arbitrary
domain names (law / fin), one teacher per domain, no math/code/if-specific prompt builders.
Launch: scripts/opd.sbatch (node0 = student, node1.. = teacher servers).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig

from mopd.common import chat
from mopd.common.config import add_config_args, dump, load_config
from mopd.common.io import read_jsonl
from mopd.common.resume import find_resume_checkpoint
from mopd.distill.teacher_client import TeacherPool, load_endpoints
from mopd.distill.trainer import MOPDTrainer
from mopd.distill.train_mopd import DEFAULTS as MOPD_DEFAULTS, _patch_runtime, is_main

DEFAULTS = copy.deepcopy(MOPD_DEFAULTS)
DEFAULTS["data"] = {"paths": {}, "max_rows_per_domain": None, "seed": 42, "thinking": True,
                    "max_prompt_chars": 24000,  # ~6k tokens: prompt + 16k completion must stay under the teacher/vLLM context
                    "per_domain_per_batch": None}  # MOPD: K prompts of EVERY domain in each generation batch (sequential, no shuffle)  # ~6k tokens: prompt + 16k completion must stay under the teacher/vLLM context
DEFAULTS["train"].update({"save_steps": 25, "save_total_limit": None, "warmup_ratio": 0.05,
                          "max_completion_length": 16384, "batch_size": 32, "per_device_bs": 1,
                          "vllm_gpu_memory_utilization": 0.25, "max_steps": 200,
                          "vllm_is_mode": "token_truncate"})


class StepTimer(TrainerCallback):
    """Wall-clock per optimizer step (one step = one rollout batch + teacher prefill + update)."""
    def __init__(self):
        self.t0 = None

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kw):
        if self.t0 is not None and is_main():
            print(f"[opd] step {state.global_step} wall {time.time() - self.t0:.1f}s", flush=True)


def build_dataset(cfg) -> tuple[Dataset, dict]:
    """Every row of every listed domain pool -> {prompt: [user turn], domain}. The prompt text is the
    same fully-formatted user message the RL/eval pipeline uses (question + options + answer-format line)."""
    rng = random.Random(cfg.data.seed)
    recs, counts = [], {}
    for d, p in dict(cfg.data.paths).items():
        if not p:
            continue
        rows = read_jsonl(p, limit=cfg.data.max_rows_per_domain)
        n_all = len(rows)
        rows = [r for r in rows if len(r["input"]) <= cfg.data.max_prompt_chars]
        if is_main():
            print(f"[opd] {d}: {n_all} rows, {n_all - len(rows)} dropped (> {cfg.data.max_prompt_chars} chars)", flush=True)
        for i, r in enumerate(rows):
            rid = r.get("id") or (r.get("meta") or {}).get("id") or f"{d}-{i}"
            recs.append({"prompt": [{"role": "user", "content": r["input"]}], "domain": d,
                         "answer": str(r.get("output", "")), "tests": "", "ground_truth": "", "id": str(rid)})
        counts[d] = len(rows)
    K = cfg.data.per_domain_per_batch
    if K:
        # balanced interleaving: block j = rows [j*B, (j+1)*B) holds exactly K prompts of every domain
        # (domain order = cfg.data.paths order); the trainer reads the dataset sequentially (shuffle_dataset=False),
        # so every generation batch of B = K * n_domains prompts is one block. Domains cycle (reshuffled) when exhausted.
        per = {}
        for r in recs:
            per.setdefault(r["domain"], []).append(r)
        doms = [d for d in dict(cfg.data.paths) if d in per]
        B = K * len(doms)
        assert cfg.train.batch_size == B, f"train.batch_size ({cfg.train.batch_size}) must equal per_domain_per_batch*n_domains ({B})"
        for d in doms:
            rng.shuffle(per[d])
        ptr = {d: 0 for d in doms}
        n_blocks = int(cfg.train.max_steps * 1.05) + 2
        out = []
        for _ in range(n_blocks):
            for d in doms:
                for _k in range(K):
                    if ptr[d] >= len(per[d]):
                        rng.shuffle(per[d]); ptr[d] = 0
                    out.append(per[d][ptr[d]]); ptr[d] += 1
        if is_main():
            print(f"[opd] balanced batches: {K} x {doms} per batch of {B}; {n_blocks} blocks = {len(out)} rows", flush=True)
        return Dataset.from_list(out), counts
    rng.shuffle(recs)
    return Dataset.from_list(recs), counts


def main(argv=None):
    _patch_runtime()
    p = argparse.ArgumentParser()
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.override, base=DEFAULTS)
    if is_main():
        print("[opd] config:\n" + dump(cfg), flush=True)
    assert cfg.teachers.endpoints_file, "teachers.endpoints_file is required (written by the sbatch)"
    endpoints, names = load_endpoints(cfg.teachers.endpoints_file)
    for d, pth in dict(cfg.data.paths).items():
        if pth:
            assert d in endpoints, f"no teacher endpoint for domain {d}: {endpoints}"
    pool = TeacherPool(endpoints, names, top_k=(cfg.teachers.top_k if cfg.distill.mode == "topk" else 0),
                       max_workers=cfg.teachers.max_workers, timeout=cfg.teachers.timeout)
    if is_main():
        print("[opd] teacher health:", pool.wait_ready(timeout=600), flush=True)
    tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.model.student, trust_remote_code=True))
    ds, counts = build_dataset(cfg)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    N = cfg.train.num_generations
    gen_bs = cfg.train.batch_size
    assert gen_bs % (cfg.train.per_device_bs * world) == 0, (gen_bs, cfg.train.per_device_bs, world)
    grad_accum = gen_bs // (cfg.train.per_device_bs * world)
    targs = GRPOConfig(
        output_dir=cfg.train.output,
        model_init_kwargs={"dtype": cfg.model.dtype, "attn_implementation": "sdpa"},
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_bs,
        gradient_accumulation_steps=grad_accum,
        generation_batch_size=gen_bs,
        num_generations=max(N, 2),  # trl init check; overridden to N in MOPDTrainer
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
        vllm_tensor_parallel_size=cfg.train.vllm_tp,
        # trl 0.29 traps (memory trl-grpo-silent-bugs): never sleep-mode (reloads the initial ckpt), and
        # per-token truncated IS instead of the default sequence_mask that zeroes long completions.
        vllm_importance_sampling_correction=True, vllm_importance_sampling_mode=cfg.train.vllm_is_mode,
        vllm_enable_sleep_mode=False,
        chat_template_kwargs={"enable_thinking": cfg.data.thinking},
        remove_unused_columns=False, ddp_timeout=cfg.train.ddp_timeout, seed=cfg.data.seed,
        shuffle_dataset=(not cfg.data.per_domain_per_batch),  # balanced MOPD batches rely on sequential order
        log_completions=True, num_completions_to_print=2,
    )
    trainer = MOPDTrainer(model=cfg.model.student, args=targs, train_dataset=ds, processing_class=tok,
                          teacher_pool=pool, distill_mode=cfg.distill.mode, a_max=cfg.distill.a_max,
                          top_k=cfg.teachers.top_k, task_reward=None, task_reward_coef=0.0,
                          num_generations_override=(N if N != max(N, 2) else None),
                          callbacks=[StepTimer()])
    t0 = time.time()
    resume = find_resume_checkpoint(cfg.train.output, cfg.train.resume)
    if is_main():
        print(f"[opd] dataset rows={len(ds)} counts={counts} world={world} grad_accum={grad_accum} resume={resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    final = os.path.join(cfg.train.output, "final")
    trainer.save_model(final)
    if is_main():
        tok.save_pretrained(final)
        with open(os.path.join(final, "generation_config.json"), "w") as f:
            json.dump(chat.generation_config_dict(cfg.data.thinking), f, indent=2)
        with open(os.path.join(cfg.train.output, "opd_done.json"), "w") as f:
            json.dump({"config": json.loads(json.dumps(cfg)), "counts": counts, "grad_accum": grad_accum,
                       "world": world, "seconds": time.time() - t0, "endpoints": endpoints}, f, indent=2)
        print("[opd] DONE ->", final, flush=True)


if __name__ == "__main__":
    main()
