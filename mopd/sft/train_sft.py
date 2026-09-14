"""Stage 1 — general SFT of a Qwen3-*-Base on the math+code+if mixture (1 node, 8 GPU, ZeRO-2).

  accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \\
      -m mopd.sft.train_sft --config configs/sft.yaml --override model.base=models/Qwen3-8B-Base

Plain HF ``Trainer`` + manual assistant-only masking (no TRL SFTTrainer), sdpa
attention (no flash-attn in the venv), bf16, gradient checkpointing,
resumable ``checkpoint-N`` every ``save_steps`` with ``save_total_limit`` (k-sean jobs get
killed; ``train.resume=auto`` picks the latest complete one). The saved model gets the eos/generation contract of
``mopd.common.chat`` so it is directly usable by vLLM / RL / MOPD.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from mopd.common import chat
from mopd.common.config import add_config_args, dump, load_config
from mopd.common.resume import find_resume_checkpoint
from mopd.sft.dataset import SFTCollator, SFTDataset, build_mixture

DEFAULTS = {
    "model": {"base": "models/Qwen3-4B-Base", "attn": "auto"},   # auto: flash_attention_2 if importable else sdpa
    "data": {"sources": {"math": "data/train/sft_math.jsonl", "code": "data/train/sft_code.jsonl", "if": "data/train/sft_if.jsonl"},
             "cache_dir": None,  # pre-built memmap cache (mopd.data.prep_openthoughts3) -> sources ignored
             "weights": None, "max_rows": None, "max_len": 32768, "thinking": True, "packing": "flatten", "seed": 42},
    "train": {"output": "outputs/sft/qwen3-4b", "epochs": 1.0, "lr": 2e-5, "scheduler": "cosine", "warmup_ratio": 0.03,
              "weight_decay": 0.0, "max_grad_norm": 1.0, "per_device_bs": 2, "grad_accum": 1, "logging_steps": 5,
              "save_strategy": "steps", "save_steps": 100, "save_total_limit": None, "save_only_model": False,
              "max_steps": -1, "resume": "auto", "seed": 42, "dataloader_workers": 2, "liger": True, "grad_ckpt": True},
}


def is_main() -> bool:
    return os.environ.get("RANK", "0") in ("0", "") and os.environ.get("LOCAL_RANK", "0") in ("0", "")


def main(argv=None):
    p = argparse.ArgumentParser()
    add_config_args(p, default_config=None)
    p.add_argument("--inspect", type=int, default=0, help="print N masked examples and exit (CPU)")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.override, base=DEFAULTS)
    if is_main():
        print("[sft] config:\n" + dump(cfg), flush=True)

    tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.model.base, trust_remote_code=True))
    os.makedirs(cfg.train.output, exist_ok=True)
    t_ds = time.time()
    if cfg.data.get("cache_dir"):
        # pre-built memmap cache (built on the cpu-instance; page-cache shared across ranks)
        from mopd.sft.ot3_cache import MemmapSFTDataset
        ds = MemmapSFTDataset(cfg.data.cache_dir)
        assert ds.stats["max_len"] == cfg.data.max_len, \
            f"cache max_len {ds.stats['max_len']} != config {cfg.data.max_len}"
    else:
        rows = build_mixture({d: p_ for d, p_ in cfg.data.sources.items() if p_ and os.path.exists(p_)},
                             cfg.data.weights, cfg.data.max_rows, cfg.data.seed)
        import hashlib
        key = hashlib.sha1(json.dumps([sorted(cfg.data.sources.items()), cfg.data.weights, cfg.data.max_rows, cfg.data.max_len,
                                       cfg.data.thinking, cfg.data.packing, cfg.data.seed, cfg.model.base]).encode()).hexdigest()[:12]
        cache = None if args.inspect else os.path.join(cfg.train.output, f"dataset_cache_{key}.pt")
        ds = SFTDataset(rows, tok, max_len=cfg.data.max_len, thinking=cfg.data.thinking, packing=cfg.data.packing,
                        seed=cfg.data.seed, cache_path=cache)
    if is_main():
        print(f"[sft] dataset ready in {time.time() - t_ds:.0f}s", flush=True)
    if is_main():
        print(f"[sft] dataset stats: {json.dumps(ds.stats)}", flush=True)
    if args.inspect:
        for i in range(min(args.inspect, len(ds))):
            ex = ds[i]
            lab = [t for t in ex["labels"] if t != -100]
            print("=" * 80)
            print("PROMPT:", tok.decode(ex["input_ids"][: len(ex["input_ids"]) - len(lab)]))
            print("--- COMPLETION (loss):", tok.decode(lab)[:1500], "...")
        return

    attn = cfg.model.attn
    if attn == "auto":
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa"
    if ds.mode == "flatten" and attn != "flash_attention_2":
        raise SystemExit("packing=flatten needs flash_attention_2 (padding-free varlen); install flash-attn or use packing=none")
    if is_main():
        print(f"[sft] attn_implementation={attn} packing={ds.mode} liger={cfg.train.liger}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.base, dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation=attn)
    model.config.use_cache = False
    model.config.eos_token_id = tok.convert_tokens_to_ids(chat.IM_END)
    model.config.pad_token_id = tok.pad_token_id

    targs = TrainingArguments(
        output_dir=cfg.train.output,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_bs,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr, lr_scheduler_type=cfg.train.scheduler, warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay, max_grad_norm=cfg.train.max_grad_norm,
        seed=cfg.train.seed, bf16=True, tf32=True,
        gradient_checkpointing=bool(cfg.train.grad_ckpt), gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, logging_steps=cfg.train.logging_steps,
        max_steps=cfg.train.max_steps,
        # RESUMABLE checkpoints: weights + optimizer + scheduler + trainer_state (k-sean jobs get killed);
        # save_total_limit bounds disk (a 4B fp32-master ZeRO-2 ckpt is ~50 GB).
        save_strategy=cfg.train.save_strategy, save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit, save_only_model=cfg.train.save_only_model,
        report_to="tensorboard", logging_dir=os.path.join(cfg.train.output, "tb"),
        remove_unused_columns=False, dataloader_num_workers=cfg.train.dataloader_workers,
        group_by_length=False,
        # Liger: fused linear-CE never materialises the [T, 151936] fp32 logits (the pd>=2 OOM cause)
        # + fused RMSNorm / RoPE / SwiGLU triton kernels
        use_liger_kernel=bool(cfg.train.liger),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=SFTCollator(tok.pad_token_id), processing_class=tok)
    t0 = time.time()
    resume = find_resume_checkpoint(cfg.train.output, cfg.train.resume)
    if is_main():
        print(f"[sft] resume_from_checkpoint = {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(cfg.train.output)
    if is_main():
        chat.save_model_with_contract(trainer.model, tok, cfg.train.output, thinking=cfg.data.thinking) if not trainer.is_deepspeed_enabled else None
        # under DeepSpeed save_model already wrote the consolidated weights; still write the contract files
        tok.save_pretrained(cfg.train.output)
        with open(os.path.join(cfg.train.output, "generation_config.json"), "w") as f:
            json.dump(chat.generation_config_dict(cfg.data.thinking), f, indent=2)
        with open(os.path.join(cfg.train.output, "mopd_sft_done.json"), "w") as f:
            json.dump({"config": json.loads(json.dumps(cfg)), "dataset_stats": ds.stats, "seconds": time.time() - t0}, f, indent=2)
        print("[sft] DONE", flush=True)


if __name__ == "__main__":
    main()
