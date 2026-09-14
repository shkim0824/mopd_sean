"""MOPD objectives (paper §3.2), pure tensor functions (unit-tested on CPU).

Policy-gradient form (Eq. 3-4)
    Â_t  = sg[ log π_φd(y_t) − log π_θ(y_t) ]  clipped to [−A_max, A_max]
    L_PG = −E[ 1/|y| Σ_t Â_t log π_θ(y_t) ]
  -> implemented as per-token advantages fed to the GRPO loss with loss_type="grpo"
     (sequence-mean over tokens, batch-mean), ratio π_θ/π_old == 1 on-policy.

Top-k form (Eq. 5)
    L_TopK = E[ 1/|y| Σ_t Σ_{v∈TopK_k(π_φd(·|s_t))} π_θ(v) log(π_θ(v)/π_φd(v)) − π_θ(v) + π_φd(v) ]
  where π_θ(v) is the student's FULL-softmax probability of token v (not renormalised on the
  top-k set) — the extra −π_θ(v)+π_φd(v) term makes the truncated objective minimised at
  π_θ = π_φd on the top-k support.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def pg_advantages(teacher_logps: torch.Tensor, student_logps: torch.Tensor, mask: torch.Tensor,
                  a_max: float = 5.0) -> torch.Tensor:
    """(B,T) clipped per-token advantages, zero outside the mask. Both inputs are detached."""
    adv = (teacher_logps.detach() - student_logps.detach()).clamp(-a_max, a_max)
    return adv * mask


def topk_loss_from_logits(logits: torch.Tensor, topk_ids: torch.Tensor, topk_logps: torch.Tensor,
                          mask: torch.Tensor, temperature: float = 1.0,
                          chunk: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """logits (B,T,V) student logits at completion positions; topk_ids/logps (B,T,K) from the
    teacher (ids < 0 = padding); mask (B,T). Returns (per_sequence_loss (B,), per_token_kl (B,T)).
    Computed in time-chunks in fp32 to bound memory."""
    B, T, _ = logits.shape
    per_token = logits.new_zeros((B, T), dtype=torch.float32)
    valid = (topk_ids >= 0) & (mask.unsqueeze(-1) > 0)
    ids = topk_ids.clamp(min=0)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        lg = logits[:, s:e].float() / temperature
        logp_full = F.log_softmax(lg, dim=-1)                         # (B,c,V)
        stu_logp = torch.gather(logp_full, -1, ids[:, s:e])           # (B,c,K)
        stu_p = stu_logp.exp()
        tea_logp = topk_logps[:, s:e].float()
        tea_p = tea_logp.exp()
        term = stu_p * (stu_logp - tea_logp) - stu_p + tea_p           # Eq. 5 summand
        term = term * valid[:, s:e]
        per_token[:, s:e] = term.sum(-1)
    m = mask.float()
    per_seq = (per_token * m).sum(-1) / m.sum(-1).clamp(min=1.0)
    return per_seq, per_token


def reverse_kl_estimate(teacher_logps: torch.Tensor, student_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Monte-Carlo estimate of per-token reverse KL(π_θ‖π_φ) along the student rollout =
    mean_t [log π_θ(y_t) − log π_φ(y_t)]  (Fig. 3 'student-teacher KL')."""
    m = mask.float()
    return ((student_logps - teacher_logps) * m).sum(-1) / m.sum(-1).clamp(min=1.0)
