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
                    "max_prompt_chars": 24000,  # legacy {input} pools: ~6k tokens; prompt + 16k completion must fit the 32k context
                    "max_prompt_tokens": None,  # pre-rendered pools (prompt_text + n_prompt_tokens): token cap, e.g. 15872
                    "per_domain_per_batch": None}  # MOPD: int K / dict per domain / "equal" (B split evenly, remainder by lottery)
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


def balanced_blocks(per: dict, doms: list, kdom: dict, n_blocks: int, rng: random.Random) -> list:
    """Sequential MOPD blocks: block j holds kdom[d] prompts of every domain d (in `doms` order). Each domain walks
    its own (pre-shuffled) pool and is reshuffled when exhausted, so no row is dropped or repeated within one pass
    over that domain's pool."""
    ptr = {d: 0 for d in doms}
    out = []
    for _ in range(n_blocks):
        for d in doms:
            for _k in range(kdom[d]):
                if ptr[d] >= len(per[d]):
                    rng.shuffle(per[d]); ptr[d] = 0
                out.append(per[d][ptr[d]]); ptr[d] += 1
    return out


def equal_blocks(per: dict, doms: list, B: int, n_blocks: int, rng: random.Random) -> list:
    """`per_domain_per_batch: equal` -- B prompts per block split as evenly as integers allow: every domain
    gets B // D, and the B mod D leftover slots go to a fresh random subset of domains each block, so over
    the run every domain averages exactly B / D (128 / 5 = 25.6 -> 25 each + a 3-of-5 lottery per block).
    Same pool walking as balanced_blocks."""
    base, rem = divmod(B, len(doms))
    assert base > 0, (B, doms)
    ptr = {d: 0 for d in doms}
    out = []
    for _ in range(n_blocks):
        extra = set(rng.sample(doms, rem)) if rem else set()
        for d in doms:
            for _k in range(base + (1 if d in extra else 0)):
                if ptr[d] >= len(per[d]):
                    rng.shuffle(per[d]); ptr[d] = 0
                out.append(per[d][ptr[d]]); ptr[d] += 1
    return out


def render_prompt(tok, messages: list, tools, thinking: bool) -> str:
    """The prompt text exactly as trl's conversational path renders it (checked at startup by
    check_render_matches_trl), so every domain enters the trainer as a plain string."""
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def check_render_matches_trl(tok, messages: list, thinking: bool) -> None:
    """Guard the pre-rendering: it must be byte-identical to what trl would have produced for the same
    conversational prompt (that is what every earlier OPD/MOPD run was trained on)."""
    from trl.data_utils import maybe_apply_chat_template
    ours = render_prompt(tok, messages, None, thinking)
    theirs = maybe_apply_chat_template({"prompt": messages}, tok, enable_thinking=thinking)["prompt"]
    assert ours == theirs, "pre-rendered prompt differs from trl's chat-template rendering:\n%r\n!=\n%r" % (
        ours[-300:], theirs[-300:])


def build_dataset(cfg, tok) -> tuple[Dataset, dict]:
    """Every row of every listed domain pool -> {prompt: <rendered string>, domain, id, ...}.

    Two pool formats:
      * legacy {input, output, meta}: one user turn, rendered here with the student template
        (add_generation_prompt, enable_thinking = data.thinking) -- byte-identical to trl's own path;
      * pre-rendered {prompt_text, n_prompt_tokens, ...} (mopd/data/prep_tau2_opd.py): a multi-message
        context with the domain's tool schemas already in the text; filtered by data.max_prompt_tokens.
    """
    rng = random.Random(cfg.data.seed)
    recs, counts = [], {}
    checked = False
    for d, p in dict(cfg.data.paths).items():
        if not p:
            continue
        rows = read_jsonl(p, limit=cfg.data.max_rows_per_domain)
        n_all, n_keep = len(rows), 0
        for i, r in enumerate(rows):
            if "prompt_text" in r:
                if cfg.data.max_prompt_tokens and int(r.get("n_prompt_tokens") or 0) > int(cfg.data.max_prompt_tokens):
                    continue
                text = r["prompt_text"]
            else:
                if len(r["input"]) > cfg.data.max_prompt_chars:
                    continue
                msgs = [{"role": "user", "content": r["input"]}]
                if not checked:
                    check_render_matches_trl(tok, msgs, cfg.data.thinking); checked = True
                text = render_prompt(tok, msgs, None, cfg.data.thinking)
            rid = r.get("id") or (r.get("meta") or {}).get("id") or f"{d}-{i}"
            recs.append({"prompt": text, "domain": d, "answer": str(r.get("output", "")), "tests": "",
                         "ground_truth": "", "id": str(rid)})
            n_keep += 1
        if is_main():
            print(f"[opd] {d}: {n_all} rows, {n_all - n_keep} dropped (prompt too long), {n_keep} kept", flush=True)
        counts[d] = n_keep
    K = cfg.data.per_domain_per_batch
    if K:
        # balanced interleaving: block j = rows [j*B, (j+1)*B) holds a fixed number of prompts of every domain
        # (int K = K prompts of EVERY domain; dict = per-domain counts, e.g. {law: 51, fin: 13, if: 13, med: 51};
        # "equal" = B // D each + the remainder by a per-block lottery; domain order = cfg.data.paths order);
        # the trainer reads the dataset sequentially (shuffle_dataset=False), so every generation batch of
        # B prompts is one block. Domains cycle (reshuffled) when exhausted.
        per = {}
        for r in recs:
            per.setdefault(r["domain"], []).append(r)
        doms = [d for d in dict(cfg.data.paths) if d in per]
        for d in doms:
            rng.shuffle(per[d])
        n_blocks = int(cfg.train.max_steps * 1.05) + 2
        B = int(cfg.train.batch_size)
        if isinstance(K, str):
            assert K == "equal", f"per_domain_per_batch: int, dict or 'equal' (got {K!r})"
            base, rem = divmod(B, len(doms))
            out = equal_blocks(per, doms, B, n_blocks, rng)
            desc = f"equal: {base} each of {doms} + {rem} extra slot(s) by lottery"
        else:
            if isinstance(K, dict):
                extra = [d for d in K if d not in doms]
                assert not extra, f"per_domain_per_batch names domains without a pool: {extra}"
                kdom = {d: int(K[d]) for d in doms}
            else:
                kdom = {d: int(K) for d in doms}
            assert all(v > 0 for v in kdom.values()), kdom
            assert B == sum(kdom.values()), f"train.batch_size ({B}) must equal the per-batch prompt count ({sum(kdom.values())} = {kdom})"
            out = balanced_blocks(per, doms, kdom, n_blocks, rng)
            desc = f"{kdom}"
        if is_main():
            print(f"[opd] balanced batches: {desc} per batch of {B}; {n_blocks} blocks = {len(out)} rows", flush=True)
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
    ds, counts = build_dataset(cfg, tok)
    if is_main():
        print(f"[opd] mask_truncated_completions={cfg.train.mask_truncated_completions} "
              f"(default True since 2026-09-17; v2 rescue only works while some rollouts survive)", flush=True)
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
