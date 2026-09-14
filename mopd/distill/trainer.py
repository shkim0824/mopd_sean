"""MOPDTrainer = trl 0.29 GRPOTrainer with the reward/advantage stage replaced by
teacher prefill (Stage 3 of the paper).

Per optimisation step (paper §3.1):
 (1) sample a batch of prompts from the multi-domain mixture           -> dataset (domain column)
 (2) student rollouts + its per-token log-probs                       -> trl colocate vLLM + HF forward
 (3) dispatch every trajectory to ITS domain teacher for prefill      -> TeacherPool (HTTP, thread pool)
 (4) update the student on the per-token teacher signal:
       pg   : advantages_t = clip(log π_φd(y_t) − log π_θ(y_t), ±A_max), loss_type="grpo" (Eq. 4)
       topk : Eq. 5 over the teacher's top-k tokens (student full-softmax probs), computed in
              ``_compute_loss`` from the student logits

Rewards: trl demands >= 1 reward function; we register a no-op (zeros) — the verifiable
task reward can be logged/added with ``task_reward_coef`` (verl ``use_task_rewards``-style).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from trl import GRPOTrainer

from mopd.distill import loss as L
from mopd.distill.teacher_client import TeacherPool


def zero_reward(completions=None, **kw) -> List[float]:
    return [0.0] * len(completions)


zero_reward.__name__ = "zero"


class MOPDTrainer(GRPOTrainer):
    def __init__(self, *args, teacher_pool: TeacherPool, distill_mode: str = "pg", a_max: float = 5.0,
                 top_k: int = 64, task_reward: Optional[Any] = None, task_reward_coef: float = 0.0,
                 num_generations_override: Optional[int] = None, **kwargs):
        rfs = [zero_reward] if task_reward is None else [task_reward]
        super().__init__(*args, reward_funcs=rfs, **kwargs)
        self.teacher_pool = teacher_pool
        self.distill_mode = distill_mode
        self.a_max = a_max
        self.top_k = top_k if distill_mode == "topk" else 0
        self.task_reward_coef = task_reward_coef
        self._task_reward = task_reward
        if num_generations_override is not None:
            # trl refuses num_generations < 2 in __init__ (it needs groups for GRPO std);
            # MOPD uses N=1 (paper: "BS 2048, N=1") — the sampler honours this value, and the
            # GRPO group statistics are irrelevant because we overwrite the advantages.
            self.num_generations = num_generations_override
        self._mopd_metrics: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ rollouts -> teacher signal
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        device = out["completion_ids"].device
        prompt_ids, prompt_mask = out["prompt_ids"], out["prompt_mask"]
        completion_ids, completion_mask = out["completion_ids"], out["completion_mask"]
        B, T = completion_ids.shape
        # student per-token logps of the sampled tokens (computed by trl under no_grad when vLLM IS-correction is on)
        student_logps = out.get("old_per_token_logps")
        if student_logps is None:
            input_ids = torch.cat([prompt_ids, completion_ids], 1)
            attn = torch.cat([prompt_mask, completion_mask], 1)
            with torch.no_grad():
                student_logps, _ = self._get_per_token_logps_and_entropies(self.model, input_ids, attn, T,
                                                                           self.args.per_device_train_batch_size)
            out["old_per_token_logps"] = student_logps
        # domains: inputs are the local prompts repeated num_generations times (trl order) — the
        # extra columns are carried in ``inputs`` (list of dicts, one per completion)
        domains = [x["domain"] for x in inputs]
        assert len(domains) == B, (len(domains), B)
        # token id lists without padding
        seqs, starts = [], []
        for i in range(B):
            p = prompt_ids[i][prompt_mask[i].bool()].tolist()
            c_len = int(completion_mask[i].sum().item())
            # completion_mask may be zeroed by mask_truncated_completions -> still prefill the tokens
            c_full = completion_ids[i].tolist()
            c_len = c_len if c_len > 0 else len(c_full)
            c = c_full[:c_len]
            seqs.append(p + c)
            starts.append(len(p))
        res = self.teacher_pool.prefill_batch(domains, seqs, starts, top_k=self.top_k)
        teacher_logps = torch.zeros((B, T), dtype=torch.float32, device=device)
        for i, (lp, tk) in enumerate(res):
            n = min(len(lp), T)
            teacher_logps[i, :n] = torch.tensor(lp[:n], dtype=torch.float32, device=device)
        mask = completion_mask.float()
        if self.distill_mode == "pg":
            adv = L.pg_advantages(teacher_logps, student_logps.float(), mask, self.a_max)
            if self.task_reward_coef and self._task_reward is not None:
                # optional: add the GRPO group-normalised task advantage (verl use_task_rewards style)
                adv = adv + self.task_reward_coef * out["advantages"].unsqueeze(1) * mask
            out["advantages"] = adv
        else:
            K = self.top_k
            tk_ids = torch.full((B, T, K), -1, dtype=torch.long, device=device)
            tk_lp = torch.zeros((B, T, K), dtype=torch.float32, device=device)
            for i, (lp, tk) in enumerate(res):
                ids, lps = tk
                n = min(len(ids), T)
                if n:
                    tk_ids[i, :n] = torch.tensor(ids[:n], dtype=torch.long, device=device)
                    tk_lp[i, :n] = torch.tensor(lps[:n], dtype=torch.float32, device=device)
            out["teacher_topk_ids"] = tk_ids
            out["teacher_topk_logps"] = tk_lp
            out["advantages"] = torch.zeros((B,), device=device)  # unused
        out["teacher_logps"] = teacher_logps
        # metrics (Fig. 3): per-token reverse KL estimate, advantage stats, teacher stats
        kl = L.reverse_kl_estimate(teacher_logps, student_logps.float(), mask)
        self._mopd_metrics.setdefault("mopd/student_teacher_kl", []).append(self.accelerator.gather(kl.mean()).mean().item())
        if self.distill_mode == "pg":
            a = out["advantages"][mask.bool()]
            self._mopd_metrics.setdefault("mopd/adv_mean", []).append(a.mean().item() if a.numel() else 0.0)
            self._mopd_metrics.setdefault("mopd/adv_abs_mean", []).append(a.abs().mean().item() if a.numel() else 0.0)
            self._mopd_metrics.setdefault("mopd/adv_clipped_frac", []).append(((a.abs() >= self.a_max - 1e-6).float().mean().item()) if a.numel() else 0.0)
        for d in set(domains):
            self._mopd_metrics.setdefault(f"mopd/frac_{d}", []).append(sum(1 for x in domains if x == d) / B)
        return out

    # ------------------------------------------------------------------ top-k loss
    def _compute_loss(self, model, inputs):
        if self.distill_mode != "topk":
            return super()._compute_loss(model, inputs)
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], 1)
        attention_mask = torch.cat([prompt_mask, completion_mask], 1)
        T = completion_ids.size(1)
        # one forward for the whole micro-batch; logits for completion positions only
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=T + 1).logits
        logits = logits[:, :-1, :]  # position t predicts completion token t
        per_seq, _ = L.topk_loss_from_logits(logits, inputs["teacher_topk_ids"], inputs["teacher_topk_logps"],
                                             completion_mask.float(), temperature=self.temperature)
        loss = per_seq.mean() / self.current_gradient_accumulation_steps
        self._metrics["train"]["mopd/topk_loss"].append(self.accelerator.gather(per_seq.mean()).mean().item())
        return loss

    # ------------------------------------------------------------------ logging
    def log(self, logs, *a, **k):
        for key, vals in self._mopd_metrics.items():
            if vals:
                logs[key] = sum(vals) / len(vals)
        self._mopd_metrics = {}
        logs.update(self.teacher_pool.flush_stats())
        fl = getattr(self._task_reward, "flush_stats", None)
        if callable(fl):
            logs.update(fl())
        super().log(logs, *a, **k)
