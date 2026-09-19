"""Signed REINFORCE on-policy distillation (PG-OPD)."""

from __future__ import annotations

import torch

from utils.opd_runtime import (
    BasePGOPDTrainer,
    validate_pg_opd_runtime_config,
    validate_token_selection_config,
)


PG_OPD_VARIANT = "pg_opd"


def signed_pg_opd_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return detached ``log pi_teacher - log pi_student`` advantages."""

    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and Teacher log-probability shapes must match")
    advantage = (teacher_log_probs - student_log_probs).detach()
    if response_mask is not None:
        if response_mask.shape != advantage.shape:
            raise ValueError("response_mask must match the log-probability shape")
        advantage = advantage * response_mask.to(dtype=advantage.dtype)
    return advantage


def signed_pg_opd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference CPU kernel for the sequence-mean/token-mean PG-OPD loss."""

    advantage = signed_pg_opd_advantage(
        student_log_probs, teacher_log_probs, response_mask
    )
    mask = response_mask.to(dtype=student_log_probs.dtype)
    token_loss = -(advantage * student_log_probs) * mask
    per_sequence = token_loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return per_sequence.mean()


def validate_pg_opd_config(config) -> None:
    validate_pg_opd_runtime_config(config)
    if str(config.algorithm.name) != PG_OPD_VARIANT:
        raise ValueError(f"PG-OPD requires algorithm.name={PG_OPD_VARIANT}")
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "reverse_kl":
        raise ValueError("PG-OPD requires loss_mode=reverse_kl")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("PG-OPD requires policy_loss_mode=reinforce")
    validate_token_selection_config(loss)


class PGOPDTrainer(BasePGOPDTrainer):
    """Trainer for signed detached-advantage PG-OPD."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_pg_opd_config(config)
        super().__init__(*args, **kwargs)


__all__ = [
    "PGOPDTrainer",
    "PG_OPD_VARIANT",
    "signed_pg_opd_advantage",
    "signed_pg_opd_loss",
    "validate_pg_opd_config",
]
