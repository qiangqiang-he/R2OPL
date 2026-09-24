"""Correct-only R^2OPL self-reinforcement (Correct-R2OPL).

Correct-R2OPL retains R^2OPL-base's prompt-group difficulty controller but
removes its Teacher/OPD branch completely.  A verifier-correct trajectory
receives the detached token advantage ``difficulty * miu``; a verifier-wrong
trajectory receives an identically zero advantage.  The policy reducer still
sees every physical trajectory, so the loss is divided by the *complete*
rollout batch rather than by the count of correct trajectories.

Truncated trajectories retain R^2OPL-base's Student head/tail answer probe.
A passing probe reclassifies the trajectory as correct before both the
prompt-group success rate and the self-reinforcement advantage are computed.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch
from tensordict import TensorDict

from utils.opd_runtime import (
    BaseR2OPLTrainer,
    configure_r2opl_defaults,
    configure_student_batch,
    validate_r2opl_runtime_config,
)
from verl.trainer import main_ppo_sync as verl_sync
from verl.workers.utils.padding import response_to_nested


CORRECT_R2OPL_VARIANT = "correct_r2opl"
CORRECT_R2OPL_WANDB_GROUP = "Correct-R2OPL"
CORRECT_R2OPL_DEFAULT_MIU = 16.0


@dataclass(frozen=True)
class CorrectR2OPLBatchResult:
    """Complete-batch correctness, difficulty, and self-RL token mask."""

    correct_mask: torch.Tensor
    difficulty: torch.Tensor
    correctness: torch.Tensor
    truncated: torch.Tensor
    probe_rescued: torch.Tensor
    group_success: torch.Tensor
    metrics: dict[str, float]


def _ordered_groups(prompt_group_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(group_id) for group_id in prompt_group_ids))


def _validate_binary_rewards(sequence_rewards: torch.Tensor) -> None:
    if not bool(torch.isfinite(sequence_rewards).all()):
        raise ValueError("Correct-R2OPL verifier rewards must be finite.")
    binary = sequence_rewards.eq(0.0) | sequence_rewards.eq(1.0)
    if not bool(binary.all()):
        invalid = sequence_rewards[~binary].detach().cpu().tolist()
        raise ValueError(
            "Correct-R2OPL requires binary verifier rewards in {0, 1}; "
            f"got invalid values {invalid[:5]}"
        )


def compute_correct_r2opl_batch(
    response_mask: torch.Tensor,
    verifier_rewards: torch.Tensor,
    prompt_group_ids: Sequence[str],
    *,
    miu: float = CORRECT_R2OPL_DEFAULT_MIU,
    truncated_mask: torch.Tensor | None = None,
    probe_correct_mask: torch.Tensor | None = None,
    genuine_trajectory_mask: torch.Tensor | None = None,
) -> CorrectR2OPLBatchResult:
    """Compute Correct-R2OPL's complete-batch difficulty and correct mask.

    Incorrect trajectories are represented explicitly in the physical batch,
    but receive no optimization signal.  They must never be removed here:
    keeping their zero advantages is what preserves complete-batch
    normalization in the actor's ``seq-mean-token-mean`` REINFORCE reducer.
    """

    if response_mask.ndim != 2:
        raise ValueError("Correct-R2OPL response_mask must be two-dimensional.")
    response_mask = response_mask.bool()
    batch_size = response_mask.shape[0]
    if len(prompt_group_ids) != batch_size:
        raise ValueError(
            "Correct-R2OPL prompt_group_ids must have one entry per trajectory."
        )

    miu = float(miu)
    if not math.isfinite(miu) or miu <= 0.0:
        raise ValueError(f"Correct-R2OPL miu must be finite and positive; got {miu}.")

    if genuine_trajectory_mask is None:
        genuine_trajectory_mask = torch.ones(
            batch_size, dtype=torch.bool, device=response_mask.device
        )
    elif (
        genuine_trajectory_mask.ndim != 1
        or genuine_trajectory_mask.shape[0] != batch_size
    ):
        raise ValueError(
            "Correct-R2OPL genuine_trajectory_mask must have one entry per trajectory."
        )
    else:
        genuine_trajectory_mask = genuine_trajectory_mask.to(
            device=response_mask.device, dtype=torch.bool
        )

    def _mask(name: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=response_mask.device)
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(
                f"Correct-R2OPL {name} must have one entry per trajectory."
            )
        return value.to(device=response_mask.device, dtype=torch.bool)

    truncated_mask = _mask("truncated_mask", truncated_mask)
    probe_correct_mask = _mask("probe_correct_mask", probe_correct_mask)

    if verifier_rewards.ndim == 2:
        if verifier_rewards.shape[0] != batch_size:
            raise ValueError(
                "Correct-R2OPL verifier reward batch size must match trajectories."
            )
        sequence_rewards = verifier_rewards.float().sum(dim=-1)
    elif verifier_rewards.ndim == 1:
        sequence_rewards = verifier_rewards.float()
    else:
        raise ValueError("Correct-R2OPL verifier_rewards must be one- or two-dimensional.")
    if sequence_rewards.shape[0] != batch_size:
        raise ValueError(
            "Correct-R2OPL verifier reward batch size must match trajectories."
        )

    genuine_tokens = response_mask & genuine_trajectory_mask.unsqueeze(-1)
    token_counts = genuine_tokens.sum(dim=-1)
    if bool((genuine_trajectory_mask & token_counts.eq(0)).any()):
        rows = (genuine_trajectory_mask & token_counts.eq(0)).nonzero(
            as_tuple=False
        ).flatten().tolist()
        raise ValueError(
            "Correct-R2OPL cannot score a genuine trajectory with no response tokens; "
            f"empty rows: {rows[:5]}"
        )
    _validate_binary_rewards(sequence_rewards[genuine_trajectory_mask])

    verifier_correct = sequence_rewards.eq(1.0)
    probe_rescued = truncated_mask & probe_correct_mask & ~verifier_correct
    correctness = (verifier_correct | probe_rescued) & genuine_trajectory_mask

    difficulty = torch.zeros(
        batch_size, dtype=torch.float32, device=response_mask.device
    )
    group_success = []
    for group_id in _ordered_groups(prompt_group_ids):
        indices = [
            index
            for index, row_group_id in enumerate(prompt_group_ids)
            if str(row_group_id) == group_id and bool(genuine_trajectory_mask[index])
        ]
        if not indices:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=response_mask.device)
        success = correctness[index_tensor].float().mean()
        difficulty[index_tensor] = 1.0 - success
        group_success.append(success)
    if not group_success:
        raise ValueError("Correct-R2OPL requires at least one genuine prompt group.")

    correct_mask = response_mask & correctness.unsqueeze(-1)
    group_success_tensor = torch.stack(group_success)
    genuine_rewards = correctness[genuine_trajectory_mask].float()
    correct_advantages = difficulty[correctness] * miu
    zero = difficulty.new_zeros(())
    correct_advantage_mean = (
        correct_advantages.mean() if correct_advantages.numel() else zero
    )
    truncated_count = int((genuine_trajectory_mask & truncated_mask).sum().item())
    metrics = {
        "correct_r2opl/train/mean_score": float(genuine_rewards.mean().item()),
        "correct_r2opl/train/group_success_rate": float(group_success_tensor.mean().item()),
        "correct_r2opl/train/prompt_difficulty_mean": float(
            (1.0 - group_success_tensor).mean().item()
        ),
        "correct_r2opl/train/correct_trajectory_count": float(correctness.sum().item()),
        "correct_r2opl/train/error_trajectory_count": float(
            (genuine_trajectory_mask & ~correctness).sum().item()
        ),
        "correct_r2opl/train/correct_trajectory_ratio": float(genuine_rewards.mean().item()),
        "correct_r2opl/train/error_trajectory_ratio": float(
            (~correctness[genuine_trajectory_mask]).float().mean().item()
        ),
        "correct_r2opl/train/truncated_trajectory_count": float(truncated_count),
        "correct_r2opl/train/response_truncated_ratio": float(
            truncated_count / int(genuine_trajectory_mask.sum().item())
        ),
        "correct_r2opl/train/probe_rescued_trajectory_count": float(probe_rescued.sum().item()),
        "correct_r2opl/train/probe_rescued_ratio": float(
            probe_rescued[genuine_trajectory_mask].float().mean().item()
        ),
        "correct_r2opl/train/correct_advantage_mean": float(correct_advantage_mean.item()),
        "correct_r2opl/train/correct_raw_advantage_mean": (
            miu if bool(correct_mask.any()) else 0.0
        ),
        "correct_r2opl/train/correct_token_count": float(correct_mask.sum().item()),
        "correct_r2opl/train/error_token_count": float(
            (genuine_tokens & ~correctness.unsqueeze(-1)).sum().item()
        ),
        "correct_r2opl/train/miu": miu,
    }
    return CorrectR2OPLBatchResult(
        correct_mask=correct_mask,
        difficulty=difficulty,
        correctness=correctness,
        truncated=truncated_mask & genuine_trajectory_mask,
        probe_rescued=probe_rescued,
        group_success=group_success_tensor,
        metrics=metrics,
    )


def correct_r2opl_token_advantage(
    correct_mask: torch.Tensor,
    difficulty: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    miu: float = CORRECT_R2OPL_DEFAULT_MIU,
) -> torch.Tensor:
    """Return ``difficulty * miu`` on correct tokens and exactly zero elsewhere."""

    if not (
        correct_mask.shape == difficulty.shape == response_mask.shape
        and correct_mask.ndim == 2
    ):
        raise ValueError(
            "Correct-R2OPL masks and token difficulty must be matching two-dimensional tensors."
        )
    advantage = difficulty.float() * float(miu)
    return advantage * correct_mask.to(advantage.dtype) * response_mask.to(advantage.dtype)


def correct_r2opl_reinforce_loss(
    student_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Full-batch sequence-mean/token-mean REINFORCE objective.

    The response mask is deliberately not reduced to correct trajectories:
    zero-advantage wrong rows remain in the final mean denominator.
    """

    if not (
        student_log_probs.shape == advantage.shape == response_mask.shape
        and student_log_probs.ndim == 2
    ):
        raise ValueError("Correct-R2OPL loss tensors must have matching two-dimensional shapes.")
    mask = response_mask.to(dtype=student_log_probs.dtype)
    token_loss = -(advantage.detach().to(mask.dtype) * student_log_probs) * mask
    per_sequence = token_loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return per_sequence.mean()


def correct_r2opl_hyperparameter(config) -> float:
    """Read and validate Correct-R2OPL's self-reinforcement multiplier."""

    settings = config.algorithm.get(CORRECT_R2OPL_VARIANT, {})
    miu = float(settings.get("miu", CORRECT_R2OPL_DEFAULT_MIU))
    if not math.isfinite(miu) or miu <= 0.0:
        raise ValueError(f"Correct-R2OPL miu must be finite and positive; got {miu}.")
    return miu


def configure_correct_r2opl_defaults(config) -> str:
    """Install Student-only dataset and prompt defaults."""

    student_family = configure_r2opl_defaults(config)
    correct_r2opl_hyperparameter(config)
    return student_family


def configure_correct_r2opl_batch(config) -> int:
    """Round the question batch to the complete Student data-parallel world."""

    return configure_student_batch(config, algorithm_name="Correct-R2OPL")


def validate_correct_r2opl_config(config) -> None:
    """Fail closed unless this is the Student-only complete-batch objective."""

    validate_r2opl_runtime_config(config)
    if str(config.algorithm.name) != CORRECT_R2OPL_VARIANT:
        raise ValueError(
            f"Correct-R2OPL requires algorithm.name={CORRECT_R2OPL_VARIANT}"
        )
    if str(config.get("group_name", "")).strip() != CORRECT_R2OPL_WANDB_GROUP:
        raise ValueError(
            f"Correct-R2OPL requires group_name={CORRECT_R2OPL_WANDB_GROUP!r}"
        )
    correct_r2opl_hyperparameter(config)
    if bool(config.distillation.enabled):
        raise ValueError("Correct-R2OPL is Student-only; distillation.enabled must be false.")
    teacher_models = config.distillation.get("teacher_models")
    if teacher_models is not None:
        teacher_paths = [
            teacher.get("model_path")
            for teacher in teacher_models.values()
            if teacher is not None and teacher.get("model_path") is not None
        ]
        if teacher_paths:
            raise ValueError("Correct-R2OPL must not configure a Teacher model path.")
    if config.data.get("native_teacher") is not None:
        raise ValueError("Correct-R2OPL must not configure data.native_teacher.")
    if bool(config.algorithm.use_kl_in_reward):
        raise ValueError("Correct-R2OPL must not add an in-reward KL penalty.")

    actor = config.actor_rollout_ref.actor
    if str(actor.policy_loss.loss_mode) != "reinforce":
        raise ValueError("Correct-R2OPL requires actor.policy_loss.loss_mode=reinforce")
    if bool(actor.use_kl_loss) or float(actor.kl_loss_coef) != 0.0:
        raise ValueError("Correct-R2OPL must not add an actor KL loss.")
    if str(actor.loss_agg_mode) != "seq-mean-token-mean":
        raise ValueError(
            "Correct-R2OPL requires loss_agg_mode=seq-mean-token-mean so the "
            "complete physical rollout batch is the sequence denominator."
        )
    if int(config.actor_rollout_ref.rollout.n) < 2:
        raise ValueError("Correct-R2OPL requires at least two rollouts per prompt.")
    if int(config.data.train_batch_size) <= 0:
        raise ValueError("Correct-R2OPL requires a positive training question batch size.")


class CorrectR2OPLTrainer(BaseR2OPLTrainer):
    """Student-only R^2OPL controller with zeroed incorrect advantages."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_correct_r2opl_config(config)
        super().__init__(*args, **kwargs)

    def _validate_complete_rollout_batch(
        self, batch, prompt_group_ids: Sequence[str]
    ) -> None:
        expected_questions = int(self.config.data.train_batch_size)
        expected_per_group = int(self.config.actor_rollout_ref.rollout.n)
        expected_rollouts = expected_questions * expected_per_group
        genuine_rollouts = sum(
            not bool(tag.get("is_padding", False)) for tag in batch.tags
        )
        if len(batch) != expected_rollouts or genuine_rollouts != expected_rollouts:
            raise RuntimeError(
                "Correct-R2OPL requires one genuine rollout per sampled trajectory; got "
                f"batch_size={len(batch)}, genuine_rollouts={genuine_rollouts}, "
                f"expected={expected_rollouts}."
            )
        counts = Counter(str(group_id) for group_id in prompt_group_ids)
        bad_counts = {
            group_id: count
            for group_id, count in counts.items()
            if count != expected_per_group
        }
        if len(counts) != expected_questions or bad_counts:
            raise RuntimeError(
                f"Correct-R2OPL requires {expected_questions} prompt groups with "
                f"{expected_per_group} rollouts each; group_count={len(counts)}, "
                f"mismatched_counts={bad_counts}."
            )

    @staticmethod
    def _scalar_field(data: TensorDict, name: str, batch_size: int) -> torch.Tensor:
        """Read one required per-trajectory Student-probe field."""

        if name not in data.keys():
            raise RuntimeError(
                f"Correct-R2OPL controller input is missing the {name!r} field."
            )
        value = data[name]
        if value.is_nested:
            value = value.to_padded_tensor(0.0)
        value = value.reshape(-1).float()
        if value.numel() != batch_size:
            raise RuntimeError(
                f"Correct-R2OPL {name} must have one entry per trajectory; "
                f"got {value.numel()} values for batch size {batch_size}."
            )
        return value

    def _compute_advantage(self, batch, metrics):
        """Materialize detached, zero-on-error advantages before actor updates."""

        fields = [
            "uid",
            "response_mask",
            "rm_scores",
            "r2opl_v2_truncated",
            "r2opl_v2_probe_correct",
            "r2opl_v2_probe_attempted",
        ]
        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        prompt_group_ids = [str(value) for value in data["uid"]]
        self._validate_complete_rollout_batch(batch, prompt_group_ids)

        response_mask_nested = data["response_mask"]
        if not response_mask_nested.is_nested:
            raise RuntimeError(
                "Correct-R2OPL controller expects jagged response masks from TransferQueue."
            )
        response_mask = response_mask_nested.to_padded_tensor(False).bool()
        verifier_rewards = data["rm_scores"]
        if verifier_rewards.is_nested:
            verifier_rewards = verifier_rewards.to_padded_tensor(0.0)
        genuine_mask = torch.tensor(
            [not bool(tag.get("is_padding", False)) for tag in batch.tags],
            dtype=torch.bool,
            device=response_mask.device,
        )
        truncated_mask = self._scalar_field(
            data, "r2opl_v2_truncated", len(batch)
        ).gt(0.5).to(device=response_mask.device)
        probe_correct_mask = self._scalar_field(
            data, "r2opl_v2_probe_correct", len(batch)
        ).gt(0.5).to(device=response_mask.device)
        miu = correct_r2opl_hyperparameter(self.config)
        result = compute_correct_r2opl_batch(
            response_mask=response_mask,
            verifier_rewards=verifier_rewards,
            prompt_group_ids=prompt_group_ids,
            miu=miu,
            truncated_mask=truncated_mask,
            probe_correct_mask=probe_correct_mask,
            genuine_trajectory_mask=genuine_mask,
        )
        difficulty_tokens = result.difficulty.unsqueeze(-1).expand_as(response_mask).clone()
        difficulty_tokens *= response_mask.to(difficulty_tokens.dtype)
        advantages = correct_r2opl_token_advantage(
            result.correct_mask,
            difficulty_tokens,
            response_mask,
            miu=miu,
        )
        output = TensorDict(
            {
                "advantages": response_to_nested(advantages, response_mask_nested),
                # No critic is used, but retaining returns keeps the standard PPO
                # batch contract intact for diagnostics and future extensions.
                "returns": response_to_nested(advantages, response_mask_nested),
            },
            batch_size=len(batch),
        )
        batch = verl_sync.tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=output,
        )
        metrics.update(result.metrics)
        metrics["correct_r2opl/train/probe_attempted_trajectory_count"] = float(
            self._scalar_field(data, "r2opl_v2_probe_attempted", len(batch))[genuine_mask]
            .sum()
            .item()
        )
        metrics["correct_r2opl/train/verifier_correct_trajectory_count"] = float(
            verifier_rewards.sum(dim=-1).eq(1.0)[genuine_mask].sum().item()
            if verifier_rewards.ndim == 2
            else verifier_rewards.eq(1.0)[genuine_mask].sum().item()
        )
        return batch

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        aliases = {
            "actor/entropy": "correct_r2opl/train/entropy",
            "actor/grad_norm": "correct_r2opl/train/grad_norm",
            "actor/pg_loss": "correct_r2opl/train/policy_loss",
            "actor/ppo_kl": "correct_r2opl/train/old_policy_kl",
            "response_length/mean": "correct_r2opl/train/response_length_mean",
            "response_length/max": "correct_r2opl/train/response_length_max",
            "perf/throughput": "correct_r2opl/perf/tokens_per_second_per_gpu",
            "perf/time_per_step": "correct_r2opl/perf/time_per_step",
        }
        for source, destination in aliases.items():
            if source in metrics:
                metrics[destination] = metrics[source]


__all__ = [
    "CORRECT_R2OPL_DEFAULT_MIU",
    "CORRECT_R2OPL_VARIANT",
    "CORRECT_R2OPL_WANDB_GROUP",
    "CorrectR2OPLBatchResult",
    "CorrectR2OPLTrainer",
    "compute_correct_r2opl_batch",
    "configure_correct_r2opl_batch",
    "configure_correct_r2opl_defaults",
    "correct_r2opl_hyperparameter",
    "correct_r2opl_reinforce_loss",
    "correct_r2opl_token_advantage",
    "validate_correct_r2opl_config",
]
