"""Offline vLLM generation for one GPU shard (tp=1 by default).

Prompts are passed as TOKEN IDS produced by the same ``render_prompt_ids`` used
for SFT targets / RL rollouts, so eval is byte-identical to training.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from mopd.common import chat


class VllmGenerator:
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, tp: int = 1,
                 max_model_len: int = 40960, gpu_mem_util: float = 0.90, seed: int = 0,
                 enforce_eager: bool = False):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = chat.prepare_tokenizer(AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True))
        # never ask vLLM for more context than the model config allows (Qwen3-*-Base: 32768)
        try:
            import json as _json
            with open(f"{model_path}/config.json") as _f:
                mpe = int(_json.load(_f).get("max_position_embeddings", max_model_len))
            if mpe < max_model_len:
                print(f"[gen] clamping max_model_len {max_model_len} -> {mpe} (max_position_embeddings)", flush=True)
                max_model_len = mpe
        except Exception:
            pass
        self.llm = LLM(model=model_path, tokenizer=tokenizer_path or model_path, tensor_parallel_size=tp,
                       disable_custom_all_reduce=(tp > 1),  # custom_all_reduce 'invalid argument' at tp>1 on this box
                       dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_util,
                       trust_remote_code=True, seed=seed, enforce_eager=(enforce_eager or tp > 2))
        self.stop_ids = chat.eos_token_ids(self.tok)
        self.max_model_len = max_model_len

    def prompt_ids(self, messages: Sequence[Dict[str, str]], mode: str = "chat", thinking: bool = True) -> List[int]:
        if mode == "chat":
            return chat.render_prompt_ids(self.tok, messages, thinking=thinking)
        if mode == "raw":
            from mopd.eval.prompts import raw_completion_prompt
            return self.tok(raw_completion_prompt(list(messages)), add_special_tokens=True)["input_ids"]
        raise ValueError(mode)

    def generate(self, messages_list: Sequence[Sequence[Dict[str, str]]], *, n: int = 1, max_tokens: int = 32768,
                 temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1, seed: int = 0,
                 mode: str = "chat", thinking: bool = True, presence_penalty: float = 0.0) -> List[List[Dict[str, Any]]]:
        """-> per prompt, a list of n dicts {text, n_tokens, finish_reason}."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts, budgets = [], []
        for m in messages_list:
            ids = self.prompt_ids(m, mode, thinking)
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            budgets.append(max(16, min(max_tokens, self.max_model_len - len(ids))))
        outs = []
        # group by budget so long prompts don't force a global cap
        sps = [SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=b,
                              seed=(seed if temperature > 0 else None), stop_token_ids=self.stop_ids,
                              presence_penalty=presence_penalty, skip_special_tokens=True) for b in budgets]
        results = self.llm.generate(prompts, sps, use_tqdm=True)
        for r in results:
            outs.append([{"text": c.text, "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason} for c in r.outputs])
        return outs
