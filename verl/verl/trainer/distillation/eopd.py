"""Pure tensor helpers for entropy-gated on-policy distillation."""

from __future__ import annotations

import math

import torch


def compute_entropy_gated_forward_kl(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_entropy: torch.Tensor,
    *,
    entropy_threshold: float,
) -> torch.Tensor:
    """Return EOPD's per-token normalized-Teacher Top-k forward KL.

    ``student_topk_log_probs`` must come from the Student's full-vocabulary
    log-softmax.  It is deliberately not normalized again inside Teacher's
    Top-k set.  Teacher probabilities, on the other hand, are normalized over
    that set exactly as required by EOPD.
    """

    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "EOPD Student and Teacher Top-k log-probabilities must have "
            "identical shapes."
        )
    if student_topk_log_probs.ndim < 2:
        raise ValueError("EOPD Top-k tensors must include a non-empty Top-k dimension.")
    if student_topk_log_probs.shape[-1] <= 0:
        raise ValueError("EOPD Top-k tensors must contain at least one token.")
    if teacher_entropy.shape != teacher_topk_log_probs.shape[:-1]:
        raise ValueError(
            "EOPD Teacher entropy must match the Top-k tensors' token dimensions."
        )
    entropy_threshold = float(entropy_threshold)
    if not math.isfinite(entropy_threshold) or entropy_threshold < 0:
        raise ValueError(
            "EOPD entropy_threshold must be finite and non-negative, got "
            f"{entropy_threshold}."
        )

    teacher_log_probs = teacher_topk_log_probs.float()
    normalized_teacher_log_probs = teacher_log_probs - torch.logsumexp(
        teacher_log_probs, dim=-1, keepdim=True
    )
    normalized_teacher_probs = normalized_teacher_log_probs.exp()
    forward_kl = (
        normalized_teacher_probs
        * (normalized_teacher_log_probs - student_topk_log_probs.float())
    ).sum(dim=-1)
    high_entropy = teacher_entropy.float() > entropy_threshold
    return torch.where(high_entropy, forward_kl, torch.zeros_like(forward_kl))
