"""SFT dataset: per-domain jsonl mixture -> tokenised (prompt masked) examples.

* prompt = ``render_prompt`` (chat template, add_generation_prompt, thinking flag)
* completion = assistant content + eos (``<|im_end|>``)
* labels = -100 on the prompt, completion ids elsewhere (assistant-only loss)
* overflow: LEFT-truncate the prompt, never the completion; if the completion
  alone exceeds max_len the example is DROPPED (a truncated reasoning trace
  teaches the model to stop mid-thought).

Packing modes (``packing``):
  "none"    one example per row (pad in the collator)
  "flatten" PADDING-FREE packing: examples are concatenated into bins of <= max_len
            tokens and the collator emits ONE row per bin with ``position_ids`` that
            restart at 0 for every example.  With ``attn_implementation="flash_attention_2"``
            transformers turns the position_ids resets into varlen (block-diagonal)
            attention, so examples never attend to each other and attention cost is
            sum(L_i^2), not (sum L_i)^2.  This is the HF "padding-free" recipe
            (DataCollatorWithFlattening); ~3-4x more tokens per micro-step than pd=1.
  "concat"  legacy packing WITHOUT position resets (cross-example attention; only for
            sdpa; not recommended)
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from mopd.common import chat
from mopd.common.io import read_jsonl

IGNORE_INDEX = -100


def build_mixture(sources: Dict[str, str], weights: Optional[Dict[str, float]], max_rows: Optional[int],
                  seed: int) -> List[Dict[str, Any]]:
    """sources: domain -> jsonl path. weights: domain -> share of the final mix
    (None = concatenate everything). Sampling without replacement, capped at pool size."""
    pools = {d: read_jsonl(p) for d, p in sources.items()}
    rng = random.Random(seed)
    for rows in pools.values():
        rng.shuffle(rows)
    if not weights:
        out = [r for rows in pools.values() for r in rows]
        rng.shuffle(out)
        return out[:max_rows] if max_rows else out
    total = max_rows or sum(len(v) for v in pools.values())
    wsum = sum(weights.values())
    out = []
    for d, w in weights.items():
        k = min(int(round(total * w / wsum)), len(pools[d]))
        out.extend(pools[d][:k])
    rng.shuffle(out)
    return out


_W = {}


def _init_worker(tok, max_len, thinking):
    _W["ds"] = SFTDataset.__new__(SFTDataset)
    _W["ds"].tok, _W["ds"].max_len, _W["ds"].thinking = tok, max_len, thinking
    _W["ds"].eos = tok.convert_tokens_to_ids(chat.IM_END)
    _W["ds"].stats = {"dropped_long_completion": 0, "prompt_truncated": 0, "tokens": 0}


def _encode_chunk(rows):
    ds = _W["ds"]
    ds.stats = {"dropped_long_completion": 0, "prompt_truncated": 0, "tokens": 0}  # fresh per chunk
    out = []
    for r in rows:
        ex = ds._encode(r["messages"])
        if ex is not None:
            out.append(ex)
            ds.stats["tokens"] += len(ex["input_ids"])
    return out, dict(ds.stats)


def _as_mode(packing) -> str:
    if packing is True:
        return "concat"
    if packing in (False, None, "", "none"):
        return "none"
    return str(packing)


class SFTDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tok, max_len: int, thinking: bool = True,
                 packing="none", seed: int = 0, cache_path: Optional[str] = None):
        """cache_path: if given, rank 0 tokenises+packs once and torch.saves {examples, stats};
        other ranks (and restarts) wait for the file and load it (tokenising 180k x 9k-token rows
        on every rank takes ~1 h)."""
        self.tok = tok
        if cache_path:
            rank = int(os.environ.get("RANK", "0") or 0)
            if rank != 0 or os.path.exists(cache_path):
                for _ in range(720):  # up to 2 h
                    if os.path.exists(cache_path):
                        break
                    time.sleep(10)
                d = torch.load(cache_path, weights_only=False)
                self.max_len, self.thinking, self.mode = max_len, thinking, _as_mode(packing)
                self.eos = tok.convert_tokens_to_ids(chat.IM_END)
                self.examples, self.stats = d["examples"], d["stats"]
                return
        self.max_len = max_len
        self.thinking = thinking
        self.mode = _as_mode(packing)
        self.eos = tok.convert_tokens_to_ids(chat.IM_END)
        self.examples: List[Dict[str, List[int]]] = []
        self.stats = {"n_in": len(rows), "dropped_long_completion": 0, "prompt_truncated": 0, "tokens": 0, "packing": self.mode}
        # tokenise in parallel (rank 0 only when cached): ~180k x 9k-token rows take ~1 h single-threaded
        nproc = int(os.environ.get("MOPD_TOKENIZE_PROCS", "0")) or max(1, min(32, (os.cpu_count() or 4) - 2))
        if len(rows) >= 2000 and nproc > 1:
            from concurrent.futures import ProcessPoolExecutor
            chunks = [list(rows[i::nproc]) for i in range(nproc)]
            with ProcessPoolExecutor(max_workers=nproc, initializer=_init_worker, initargs=(tok, max_len, thinking)) as ex:
                results = list(ex.map(_encode_chunk, chunks))
            for exs, st in results:
                self.examples.extend(exs)
                for k in ("dropped_long_completion", "prompt_truncated", "tokens"):
                    self.stats[k] += st[k]
        else:
            for r in rows:
                ex = self._encode(r["messages"])
                if ex is not None:
                    self.examples.append(ex)
                    self.stats["tokens"] += len(ex["input_ids"])
        if self.mode in ("concat", "flatten"):
            self.examples = self._pack(self.examples, seed, with_positions=(self.mode == "flatten"))
            self.stats["bins"] = len(self.examples)
            self.stats["fill"] = self.stats["tokens"] / max(1, len(self.examples) * max_len)
        self.stats["n_out"] = len(self.examples)
        if cache_path:
            tmp = cache_path + ".tmp"
            torch.save({"examples": self.examples, "stats": self.stats}, tmp)
            os.replace(tmp, cache_path)

    def _encode(self, messages: List[Dict[str, str]]):
        assert messages[-1]["role"] == "assistant"
        prompt_ids = chat.render_prompt_ids(self.tok, messages[:-1], thinking=self.thinking)
        comp = messages[-1]["content"]
        if not self.thinking:
            comp = chat.split_thinking(comp)["answer"]
        comp_ids = self.tok(comp, add_special_tokens=False)["input_ids"] + [self.eos]
        if len(comp_ids) >= self.max_len:
            self.stats["dropped_long_completion"] += 1
            return None
        overflow = len(prompt_ids) + len(comp_ids) - self.max_len
        if overflow > 0:
            prompt_ids = prompt_ids[overflow:]
            self.stats["prompt_truncated"] += 1
        return {"input_ids": np.asarray(prompt_ids + comp_ids, dtype=np.int32),
                "labels": np.asarray([IGNORE_INDEX] * len(prompt_ids) + comp_ids, dtype=np.int32)}

    def _pack(self, exs, seed, with_positions: bool):
        """Best-fit-decreasing bin packing into bins of <= max_len tokens, O(n log n):
        bins are kept sorted by remaining capacity; each example (largest first) goes into the
        bin with the SMALLEST remaining capacity that still fits (bisect), else opens a new bin.
        Bins hold lists of numpy chunks and are concatenated once at the end (compact int32)."""
        import bisect
        rng = random.Random(seed)
        exs = sorted(exs, key=lambda e: -len(e["input_ids"]))
        caps: List[int] = []          # remaining capacity, sorted ascending
        cap_bins: List[int] = []      # bin index aligned with caps
        parts: List[List[Dict[str, np.ndarray]]] = []
        for e in exs:
            n = len(e["input_ids"])
            k = bisect.bisect_left(caps, n)
            if k < len(caps):
                bi = cap_bins.pop(k)
                rem = caps.pop(k) - n
                parts[bi].append(e)
            else:
                parts.append([e])
                bi = len(parts) - 1
                rem = self.max_len - n
            j = bisect.bisect_left(caps, rem)
            caps.insert(j, rem)
            cap_bins.insert(j, bi)
        bins = []
        for ps in parts:
            b = {"input_ids": np.concatenate([e["input_ids"] for e in ps]),
                 "labels": np.concatenate([e["labels"] for e in ps])}
            if with_positions:
                b["position_ids"] = np.concatenate([np.arange(len(e["input_ids"]), dtype=np.int32) for e in ps])
            bins.append(b)
        rng.shuffle(bins)
        return bins

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


@dataclass
class SFTCollator:
    """'none'/'concat' rows: right-pad (input_ids / labels / attention_mask).
    'flatten' rows (carry position_ids): ALL rows of the micro-batch are concatenated into ONE
    padding-free row [1, sum(len)] with position_ids restarting per packed example; no
    attention_mask is emitted, so transformers' flash_attention_2 path derives cu_seqlens from
    the position resets (block-diagonal varlen attention). Works for any per_device_bs."""
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        if all("position_ids" in b for b in batch):
            ids = np.concatenate([np.asarray(b["input_ids"]) for b in batch])
            lab = np.concatenate([np.asarray(b["labels"]) for b in batch])
            pos = np.concatenate([np.asarray(b["position_ids"]) for b in batch])
            return {"input_ids": torch.from_numpy(ids.astype(np.int64))[None], "labels": torch.from_numpy(lab.astype(np.int64))[None],
                    "position_ids": torch.from_numpy(pos.astype(np.int64))[None]}
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), IGNORE_INDEX, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.as_tensor(np.asarray(b["input_ids"], dtype=np.int64))
            lab[i, :n] = torch.as_tensor(np.asarray(b["labels"], dtype=np.int64))
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
