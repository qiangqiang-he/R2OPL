"""OPDVR: On-policy Distillation with Verifiable Reward (arXiv:2608.24696).

The ReLU gate re-keys sampled-token OPD's implicit reward by trajectory
correctness so that verifier-correct trajectories only receive non-negative
advantages and verifier-incorrect ones only non-positive advantages, with
direction-conflicting tokens masked to zero.  ``grpd`` optionally scales the
gated reward by a Dr.GRPO-style group-relative advantage (GRPD variant).
"""

from __future__ import annotations

import torch

from utils.opd_runtime import (
    BasePGOPDTrainer,
    validate_pg_opd_runtime_config,
    validate_token_selection_config,
)


OPDVR_VARIANT = "opdvr"
OPDVR_LOSS_MODE = "opdvr"


def opdvr_gated_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    correct: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference CPU kernel: detached correctness-gated token advantages.

    ``correct`` is a boolean per-trajectory mask broadcast over tokens.
    """

    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and Teacher log-probability shapes must match")
    sampled_reverse_kl = (student_log_probs - teacher_log_probs).detach()
    correct_rows = correct.reshape(-1)
    if correct_rows.numel() != sampled_reverse_kl.shape[0]:
        raise ValueError("correct must carry one flag per trajectory")
    correct_mask = correct_rows.to(dtype=torch.bool).unsqueeze(-1).expand_as(
        sampled_reverse_kl
    )
    gated = torch.where(
        correct_mask,
        torch.clamp(-sampled_reverse_kl, min=0.0),
        -torch.clamp(sampled_reverse_kl, min=0.0),
    )
    if response_mask is not None:
        if response_mask.shape != gated.shape:
            raise ValueError("response_mask must match the log-probability shape")
        gated = gated * response_mask.to(dtype=gated.dtype)
    return gated


def opdvr_reinforce_loss(
    student_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Sequence-mean/token-mean REINFORCE loss on detached advantages."""

    if advantage.shape != student_log_probs.shape:
        raise ValueError("advantage must match the student log-probability shape")
    mask = response_mask.to(dtype=student_log_probs.dtype)
    token_loss = -(advantage * student_log_probs) * mask
    per_sequence = token_loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return per_sequence.mean()


def opdvr_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    correct: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference CPU kernel for the full OPDVR loss."""

    advantage = opdvr_gated_advantage(
        student_log_probs, teacher_log_probs, correct, response_mask
    )
    return opdvr_reinforce_loss(student_log_probs, advantage, response_mask)


def grpd_group_advantage(
    verifier_rewards: torch.Tensor, group_ids: list[str]
) -> torch.Tensor:
    """Dr.GRPO advantage: reward minus prompt-group mean, no std division."""

    rewards = verifier_rewards.reshape(-1).to(dtype=torch.float32)
    if len(group_ids) != rewards.numel():
        raise ValueError("group_ids must match the trajectory count")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for group_id, value in zip(group_ids, rewards.tolist(), strict=True):
        key = str(group_id)
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1
    return torch.tensor(
        [
            float(rewards[i]) - sums[str(group_ids[i])] / counts[str(group_ids[i])]
            for i in range(rewards.numel())
        ],
        dtype=verifier_rewards.dtype,
        device=verifier_rewards.device,
    )


def validate_opdvr_config(config) -> None:
    validate_pg_opd_runtime_config(config, expected_loss_mode=OPDVR_LOSS_MODE)
    if str(config.algorithm.name) != OPDVR_VARIANT:
        raise ValueError(f"OPDVR requires algorithm.name={OPDVR_VARIANT}")
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != OPDVR_LOSS_MODE:
        raise ValueError(f"OPDVR requires loss_mode={OPDVR_LOSS_MODE}")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("OPDVR requires policy_loss_mode=reinforce")
    validate_token_selection_config(loss)


class OPDVRTrainer(BasePGOPDTrainer):
    """Trainer for correctness-gated OPDVR (optional GRPD scaling)."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_opdvr_config(config)
        super().__init__(*args, **kwargs)

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        aliases = {
            "actor/distillation/opdvr_gated_reward_mean": (
                "opdvr/train/gated_reward_mean"
            ),
            "actor/distillation/opdvr_correct_reward_mean": (
                "opdvr/train/correct_reward_mean"
            ),
            "actor/distillation/opdvr_incorrect_reward_mean": (
                "opdvr/train/incorrect_reward_mean"
            ),
            "actor/distillation/opdvr_conflicting_token_ratio": (
                "opdvr/train/conflicting_token_ratio"
            ),
            "actor/distillation/loss": "opdvr/train/policy_loss",
            "actor/grad_norm": "opdvr/train/grad_norm",
        }
        for source, destination in aliases.items():
            if source in metrics:
                metrics[destination] = metrics[source]


__all__ = [
    "OPDVRTrainer",
    "OPDVR_VARIANT",
    "OPDVR_LOSS_MODE",
    "grpd_group_advantage",
    "opdvr_gated_advantage",
    "opdvr_loss",
    "opdvr_reinforce_loss",
    "validate_opdvr_config",
]
