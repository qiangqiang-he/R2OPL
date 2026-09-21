"""Difficulty-aware hybrid self-reinforcement and OPD (R^2OPL-base).

Each training prompt samples a group of Student rollouts that the verifier
grades binarily.  The per-prompt difficulty ``1 - group_mean`` scales two
branches:

* correct trajectories reinforce themselves with the constant advantage
  ``difficulty * miu`` (the ``miu = 1`` case is classic R^2OPL-base);
* incorrect trajectories follow the Teacher through the sampled-token OPD
  advantage ``difficulty * lambda * (log pi_T - log pi_S)``.

Truncated rollouts are checked by the OPD_Forge v2 head/tail answer probe.
If the Student's gold-answer probability improves by more than 0.3 after
the truncated reasoning prefix, they are reclassified as correct with
reward 1 and join the self-reinforcement branch. Probe metadata is required
from the agent loop; missing fields must never silently disable rescue.

Everything algorithm-specific lives in this module: the complete-batch group
controller kernel, the runtime configuration/validation, and the trainer.
Only VERL built-ins (the registered ``r2opl_base`` distillation estimator,
the TransferQueue controller hooks, and the ERSR evaluation fabric VERL
itself consumes) are imported.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from utils.ersr import (
    STUDENT_ACTION,
    TEACHER_REPLACE,
    aggregate_ersr_advantages,
    build_ersr_step_cases,
    parse_ersr_settings,
    validate_ersr_dataset_subset,
)
from utils.prompts import PROMPT_TEMPLATES, normalize_model_family
from verl.trainer import main_ppo_sync as verl_sync
from verl.workers.utils.padding import no_padding_2_padding, response_to_nested


R2OPL_BASE_VARIANT = "r2opl_base"
R2OPL_BASE_LOSS_MODE = "r2opl_base"
R2OPL_BASE_WANDB_GROUP = "R2OPL-Base"
R2OPL_BASE_DEFAULT_LAMBDA = 0.05
R2OPL_BASE_DEFAULT_MIU = 16.0
R2OPL_BASE_PROBE_RESCUE_MARGIN = 0.3


@dataclass(frozen=True)
class R2OPLBaseBatchResult:
    """Controller-materialized R^2OPL group statistics and token masks."""

    correct_mask: torch.Tensor
    error_mask: torch.Tensor
    difficulty: torch.Tensor
    correctness: torch.Tensor
    truncated: torch.Tensor
    probe_rescued: torch.Tensor
    group_success: torch.Tensor
    metrics: dict[str, float]


def _safe_mean(values: torch.Tensor, *, zero: torch.Tensor) -> torch.Tensor:
    """Mean over a possibly empty tensor without creating NaNs."""

    return values.mean() if values.numel() else zero


def _trajectory_means(
    values: torch.Tensor,
    token_mask: torch.Tensor,
    trajectory_mask: torch.Tensor,
    *,
    zero: torch.Tensor,
) -> torch.Tensor:
    """Return one token-mean value for each selected trajectory."""

    token_mask = token_mask.bool()
    counts = token_mask.sum(dim=-1)
    selected = trajectory_mask.bool()
    if not bool(selected.any()):
        return zero.reshape(1)[:0]
    if bool((selected & counts.eq(0)).any()):
        raise ValueError(
            "R^2OPL-base cannot reduce a selected trajectory with no response tokens."
        )
    sequence_sums = (values * token_mask.to(values.dtype)).sum(dim=-1)
    sequence_means = sequence_sums / counts.clamp_min(1).to(values.dtype)
    return sequence_means[selected]


def _validate_binary_rewards(sequence_rewards: torch.Tensor) -> None:
    if not bool(torch.isfinite(sequence_rewards).all()):
        raise ValueError("R^2OPL-base verifier rewards must be finite.")
    binary = sequence_rewards.eq(0.0) | sequence_rewards.eq(1.0)
    if not bool(binary.all()):
        invalid = sequence_rewards[~binary].detach().cpu().tolist()
        raise ValueError(
            "R^2OPL-base requires binary verifier rewards in {0, 1}; "
            f"got invalid values {invalid[:5]}"
        )


def _ordered_groups(prompt_group_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(group_id) for group_id in prompt_group_ids))


def compute_r2opl_base_batch(
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    verifier_rewards: torch.Tensor,
    prompt_group_ids: Sequence[str],
    *,
    lambda_: float = R2OPL_BASE_DEFAULT_LAMBDA,
    miu: float = R2OPL_BASE_DEFAULT_MIU,
    truncated_mask: torch.Tensor | None = None,
    probe_correct_mask: torch.Tensor | None = None,
    genuine_trajectory_mask: torch.Tensor | None = None,
) -> R2OPLBaseBatchResult:
    """Compute complete-batch R^2OPL group difficulty and branch masks.

    Truncated trajectories whose answer probe passes are reclassified as
    correct for both the branch assignment and the reward statistic.  The
    incorrect-trajectory OPD signal is recomputed from current Student
    log-probabilities inside VERL's registered estimator; this controller
    kernel reports it from the pre-update ``old_log_probs`` for metrics.
    """

    tensors = {
        "old_log_probs": old_log_probs,
        "teacher_log_probs": teacher_log_probs,
        "response_mask": response_mask,
    }
    if any(value.ndim != 2 for value in tensors.values()):
        raise ValueError("R^2OPL-base token tensors must all be two-dimensional.")
    shapes = {tuple(value.shape) for value in tensors.values()}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={tuple(value.shape)}" for name, value in tensors.items())
        raise ValueError(
            "R^2OPL-base Student/Teacher log-probs and response mask must have "
            f"identical shapes; got {details}."
        )

    lambda_ = float(lambda_)
    if not math.isfinite(lambda_) or lambda_ < 0.0:
        raise ValueError(f"R^2OPL-base lambda must be finite and non-negative; got {lambda_}.")
    miu = float(miu)
    if not math.isfinite(miu) or miu <= 0.0:
        raise ValueError(f"R^2OPL-base miu must be finite and positive; got {miu}.")

    response_mask = response_mask.bool()
    batch_size = response_mask.shape[0]
    if len(prompt_group_ids) != batch_size:
        raise ValueError(
            "R^2OPL-base prompt_group_ids must have one entry per trajectory."
        )
    if genuine_trajectory_mask is None:
        genuine_trajectory_mask = torch.ones(
            batch_size, dtype=torch.bool, device=response_mask.device
        )
    elif genuine_trajectory_mask.ndim != 1 or genuine_trajectory_mask.shape[0] != batch_size:
        raise ValueError(
            "R^2OPL-base genuine_trajectory_mask must have one entry per trajectory."
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
                f"R^2OPL-base {name} must have one entry per trajectory."
            )
        return value.to(device=response_mask.device, dtype=torch.bool)

    truncated_mask = _mask("truncated_mask", truncated_mask)
    probe_correct_mask = _mask("probe_correct_mask", probe_correct_mask)

    if verifier_rewards.ndim == 2:
        if verifier_rewards.shape[0] != batch_size:
            raise ValueError("R^2OPL-base verifier reward batch size must match trajectories.")
        sequence_rewards = verifier_rewards.float().sum(dim=-1)
    elif verifier_rewards.ndim == 1:
        sequence_rewards = verifier_rewards.float()
    else:
        raise ValueError("R^2OPL-base verifier_rewards must be one- or two-dimensional.")
    if sequence_rewards.shape[0] != batch_size:
        raise ValueError("R^2OPL-base verifier reward batch size must match trajectories.")

    genuine_tokens = response_mask & genuine_trajectory_mask.unsqueeze(-1)
    token_counts = genuine_tokens.sum(dim=-1)
    if bool((genuine_trajectory_mask & token_counts.eq(0)).any()):
        rows = (genuine_trajectory_mask & token_counts.eq(0)).nonzero(
            as_tuple=False
        ).flatten().tolist()
        raise ValueError(
            "R^2OPL-base cannot score a genuine trajectory with no response tokens; "
            f"empty rows: {rows[:5]}"
        )
    _validate_binary_rewards(sequence_rewards[genuine_trajectory_mask])
    if not bool(torch.isfinite(old_log_probs[genuine_tokens]).all()):
        raise ValueError("R^2OPL-base Student log-probabilities must be finite.")
    if not bool(torch.isfinite(teacher_log_probs[genuine_tokens]).all()):
        raise ValueError("R^2OPL-base Teacher log-probabilities must be finite.")

    # A probe can only rescue a truncated trajectory; the verifier verdict
    # takes precedence everywhere else.
    verifier_correct = sequence_rewards.eq(1.0)
    probe_rescued = truncated_mask & probe_correct_mask & ~verifier_correct
    correctness = (verifier_correct | probe_rescued) & genuine_trajectory_mask

    difficulty = torch.zeros(
        batch_size, dtype=torch.float32, device=response_mask.device
    )
    group_success = []
    groups = _ordered_groups(prompt_group_ids)
    for group_id in groups:
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
        raise ValueError("R^2OPL-base requires at least one genuine prompt group.")

    correct_mask = response_mask & correctness.unsqueeze(-1)
    error_mask = response_mask & genuine_trajectory_mask.unsqueeze(-1) & ~correctness.unsqueeze(-1)
    teacher_minus_student = teacher_log_probs.float() - old_log_probs.float()
    zero = teacher_minus_student.new_zeros(())
    correct_scaled = difficulty.unsqueeze(-1).expand_as(teacher_minus_student) * miu
    error_scaled = difficulty.unsqueeze(-1).expand_as(teacher_minus_student) * lambda_ * teacher_minus_student

    valid_correct = correct_mask
    valid_error = error_mask
    correct_mean = _safe_mean(
        _trajectory_means(correct_scaled, valid_correct, correctness, zero=zero),
        zero=zero,
    )
    error_trajectory_mask = genuine_trajectory_mask & ~correctness
    error_mean = _safe_mean(
        _trajectory_means(error_scaled, valid_error, error_trajectory_mask, zero=zero),
        zero=zero,
    )
    error_abs_mean = _safe_mean(
        _trajectory_means(error_scaled.abs(), valid_error, error_trajectory_mask, zero=zero),
        zero=zero,
    )
    raw_error_mean = _safe_mean(
        _trajectory_means(teacher_minus_student, valid_error, error_trajectory_mask, zero=zero),
        zero=zero,
    )
    raw_error_abs_mean = _safe_mean(
        _trajectory_means(teacher_minus_student.abs(), valid_error, error_trajectory_mask, zero=zero),
        zero=zero,
    )
    group_success_tensor = torch.stack(group_success)
    genuine_rewards = correctness[genuine_trajectory_mask].float()
    truncated_count = int((genuine_trajectory_mask & truncated_mask).sum().item())
    metrics = {
        "r2opl_base/train/mean_score": float(genuine_rewards.mean().item()),
        "r2opl_base/train/group_success_rate": float(group_success_tensor.mean().item()),
        "r2opl_base/train/prompt_difficulty_mean": float(
            (1.0 - group_success_tensor).mean().item()
        ),
        "r2opl_base/train/correct_trajectory_count": float(correctness.sum().item()),
        "r2opl_base/train/error_trajectory_count": float(
            (genuine_trajectory_mask & ~correctness).sum().item()
        ),
        "r2opl_base/train/correct_trajectory_ratio": float(
            correctness[genuine_trajectory_mask].float().mean().item()
        ),
        "r2opl_base/train/error_trajectory_ratio": float(
            (~correctness[genuine_trajectory_mask]).float().mean().item()
        ),
        "r2opl_base/train/truncated_trajectory_count": float(truncated_count),
        "r2opl_base/train/response_truncated_ratio": float(
            truncated_count / int(genuine_trajectory_mask.sum().item())
        ),
        "r2opl_base/train/probe_rescued_trajectory_count": float(probe_rescued.sum().item()),
        "r2opl_base/train/probe_rescued_ratio": float(
            probe_rescued[genuine_trajectory_mask].float().mean().item()
        ),
        "r2opl_base/train/correct_advantage_mean": float(correct_mean.item()),
        "r2opl_base/train/correct_raw_advantage_mean": miu if bool(valid_correct.any()) else 0.0,
        "r2opl_base/train/error_opd_advantage_mean": float(error_mean.item()),
        "r2opl_base/train/error_opd_abs_advantage_mean": float(error_abs_mean.item()),
        "r2opl_base/train/error_raw_opd_mean": float(raw_error_mean.item()),
        "r2opl_base/train/error_raw_opd_abs_mean": float(raw_error_abs_mean.item()),
        "r2opl_base/train/correct_token_count": float(valid_correct.sum().item()),
        "r2opl_base/train/error_token_count": float(valid_error.sum().item()),
        "r2opl_base/train/lambda": lambda_,
        "r2opl_base/train/miu": miu,
    }
    return R2OPLBaseBatchResult(
        correct_mask=correct_mask,
        error_mask=error_mask,
        difficulty=difficulty,
        correctness=correctness,
        truncated=truncated_mask & genuine_trajectory_mask,
        probe_rescued=probe_rescued,
        group_success=group_success_tensor,
        metrics=metrics,
    )


def r2opl_base_token_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    correct_mask: torch.Tensor,
    error_mask: torch.Tensor,
    difficulty: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    lambda_: float = R2OPL_BASE_DEFAULT_LAMBDA,
    miu: float = R2OPL_BASE_DEFAULT_MIU,
) -> torch.Tensor:
    """Reference CPU kernel for VERL's ``total``-branch token advantage.

    ``A = difficulty * (correct ? miu : lambda * (log pi_T - log pi_S))``
    computed with the supplied (current) Student log-probabilities, exactly
    as the registered estimator does inside the actor forward.
    """

    advantage = torch.where(
        correct_mask.bool(),
        difficulty.float() * float(miu),
        difficulty.float() * float(lambda_) * (teacher_log_probs.float() - student_log_probs.float()),
    )
    return advantage * response_mask.to(advantage.dtype)


def r2opl_base_reinforce_loss(
    student_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Sequence-mean/token-mean REINFORCE on the branch advantage."""

    mask = response_mask.to(dtype=student_log_probs.dtype)
    token_loss = -(advantage.detach().to(mask.dtype) * student_log_probs) * mask
    per_sequence = token_loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return per_sequence.mean()


# ---------------------------------------------------------------------------
# Runtime configuration and validation (self-contained).
# ---------------------------------------------------------------------------


def _validation_namespace(data_source: str) -> str:
    name = Path(str(data_source)).stem.lower()
    name = re.sub(r"20(\d{2})$", r"\1", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    if not name:
        raise ValueError(f"Cannot derive a validation name from {data_source!r}")
    return f"val-{name}"


def compute_r2opl_base_pass_avg_metrics(
    data_sources, sample_uids, accuracies, *, expected_rollouts: int
) -> dict[str, float]:
    """Local Pass@1 / Pass@k / Avg@k validation metrics for R^2OPL-base."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for source, uid, accuracy in zip(data_sources, sample_uids, accuracies, strict=True):
        grouped[str(source)][str(uid)].append(float(accuracy))

    metrics: dict[str, float] = {}
    for source, uid_to_accuracy in grouped.items():
        bad = {
            uid: len(values)
            for uid, values in uid_to_accuracy.items()
            if len(values) != expected_rollouts
        }
        if bad:
            raise RuntimeError(
                f"Expected {expected_rollouts} rollouts per question; "
                f"mismatches: {list(bad.items())[:5]}"
            )
        values = list(uid_to_accuracy.values())
        prefix = f"{_validation_namespace(source)}-core"
        metrics[f"{prefix}/Pass@1"] = float(np.mean([row[0] for row in values]))
        metrics[f"{prefix}/Pass@{expected_rollouts}"] = float(
            np.mean([max(row) for row in values])
        )
        metrics[f"{prefix}/Avg@{expected_rollouts}"] = float(
            np.mean([np.mean(row) for row in values])
        )
    return metrics


def selected_avg_metrics(
    metrics: dict[str, float],
    dataset_names,
    *,
    expected_rollouts: int,
) -> dict[str, float]:
    """Extract one ordered Avg@k value for every configured benchmark."""

    selected: dict[str, float] = {}
    missing = []
    for dataset_name in dataset_names:
        key = f"{_validation_namespace(dataset_name)}-core/Avg@{expected_rollouts}"
        if key not in metrics:
            missing.append(dataset_name)
            continue
        selected[str(dataset_name)] = float(metrics[key])
    if missing:
        raise RuntimeError(
            "Validation did not produce Avg@"
            f"{expected_rollouts} for configured datasets: {', '.join(missing)}"
        )
    return selected


def _configure_run_metadata(config) -> tuple[str, str]:
    run_name = str(config.get("run_name", "") or "").strip()
    if not run_name:
        prefix = str(config.get("run_name_prefix", "r2opl") or "r2opl").strip()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{prefix}_{timestamp}_p{os.getpid()}"
    group_name = str(
        config.get("group_name", R2OPL_BASE_WANDB_GROUP) or R2OPL_BASE_WANDB_GROUP
    ).strip()
    OmegaConf.update(config, "run_name", run_name, force_add=True)
    OmegaConf.update(config, "group_name", group_name, force_add=True)
    OmegaConf.update(config, "trainer.experiment_name", run_name, force_add=True)
    return run_name, group_name


def r2opl_base_hyperparameters(config) -> tuple[float, float]:
    """Read the ``(lambda, miu)`` pair from the algorithm settings."""

    settings = config.algorithm.get(R2OPL_BASE_VARIANT, {})
    lambda_ = float(settings.get("lambda", R2OPL_BASE_DEFAULT_LAMBDA))
    miu = float(settings.get("miu", R2OPL_BASE_DEFAULT_MIU))
    if not math.isfinite(lambda_) or lambda_ < 0.0:
        raise ValueError(
            f"R^2OPL-base lambda must be finite and non-negative; got {lambda_}."
        )
    if not math.isfinite(miu) or miu <= 0.0:
        raise ValueError(f"R^2OPL-base miu must be finite and positive; got {miu}.")
    return lambda_, miu


def _configure_native_teacher_prompt(config, teacher_family: str) -> None:
    """Give the shared dataset the one Teacher's own tokenizer contract."""
    teachers = config.distillation.teacher_models
    if set(teachers) != {"teacher_model"}:
        raise ValueError("R^2OPL-base native prompts currently require exactly one Teacher")
    teacher = teachers.teacher_model
    OmegaConf.update(
        config,
        "data.native_teacher",
        {
            "model_path": str(teacher.model_path),
            "model_family": str(teacher_family),
            "max_prompt_length": int(teacher.inference.prompt_length),
            "prompt_name": str(config.prompt_template),
        },
        force_add=True,
    )


def configure_r2opl_base_defaults(config) -> str:
    """Install R^2OPL-base's dataset/prompt defaults before VERL starts."""

    project_root = Path(__file__).resolve().parents[1]
    prompt_name = str(config.get("prompt_template", "")).strip()
    if prompt_name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt_template={prompt_name!r}; expected one of: {available}"
        )
    student_family = normalize_model_family(config.student_model_family)
    teacher_family = normalize_model_family(config.teacher_model_family)
    if student_family != teacher_family:
        raise ValueError(
            "R^2OPL-base requires Student and Teacher to share a tokenizer family "
            "so the sampled token IDs have identical meaning"
        )
    r2opl_base_hyperparameters(config)
    OmegaConf.update(
        config,
        "data.custom_cls",
        {
            "path": str(project_root / "utils" / "custom_dataset.py"),
            "name": "CustomDataset",
        },
        force_add=True,
    )
    OmegaConf.update(config, "data.model_family", student_family, force_add=True)
    OmegaConf.update(config, "student_prompt", prompt_name, force_add=True)
    OmegaConf.update(config, "teacher_prompt", prompt_name, force_add=True)
    _configure_native_teacher_prompt(config, teacher_family)
    _configure_run_metadata(config)
    return student_family


def configure_r2opl_base_batch(config) -> int:
    """Round R^2OPL-base's question batch to the Student data-parallel world."""

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.tensor_model_parallel_size) != 1:
        raise ValueError("R^2OPL-base Student rollout tensor parallelism must be 1")
    if int(rollout.data_parallel_size) != student_gpus:
        raise ValueError(
            "Student rollout data_parallel_size must equal the Student GPU count"
        )
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("R^2OPL-base Ulysses sequence parallelism must be 1")
    configured = int(config.data.train_batch_size)
    adjusted = math.ceil(configured / student_gpus) * student_gpus
    OmegaConf.update(config, "data.train_batch_size", adjusted)
    OmegaConf.update(config, "actor_rollout_ref.actor.ppo_mini_batch_size", adjusted)
    return adjusted


def _validate_verl_r2opl_base_support() -> None:
    """Fail early when the active VERL lacks the R^2OPL-base kernels."""

    from verl.trainer.distillation import losses

    if not hasattr(losses, "compute_r2opl_base_sampled_token_loss"):
        raise RuntimeError(
            "The active VERL build does not provide R^2OPL-base support; "
            "missing: compute_r2opl_base_sampled_token_loss"
        )


def validate_r2opl_base_config(config) -> None:
    """Fail closed unless the configuration is standard R^2OPL-base."""

    _validate_verl_r2opl_base_support()

    prompt_name = str(config.get("prompt_template", "")).strip()
    if prompt_name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt_template={prompt_name!r}; expected one of: {available}"
        )
    student_family = normalize_model_family(config.student_model_family)
    teacher_family = normalize_model_family(config.teacher_model_family)
    if student_family != teacher_family:
        raise ValueError("R^2OPL-base Student and Teacher tokenizer families must match")
    if str(config.trainer.project_name) != "R^2OPL":
        raise ValueError("All R2OPL runs must use trainer.project_name='R^2OPL'")
    if str(config.get("group_name", "")).strip() != R2OPL_BASE_WANDB_GROUP:
        raise ValueError(f"R^2OPL-base requires group_name={R2OPL_BASE_WANDB_GROUP!r}")
    if str(config.algorithm.name) != R2OPL_BASE_VARIANT:
        raise ValueError(f"R^2OPL-base requires algorithm.name={R2OPL_BASE_VARIANT}")
    if not bool(config.distillation.enabled):
        raise ValueError("R^2OPL-base requires distillation.enabled=true")
    r2opl_base_hyperparameters(config)

    val_datasets = config.data.get("val_datasets")
    if val_datasets is None or isinstance(val_datasets, str):
        raise ValueError(
            "data.val_datasets must be a non-empty list of evaluation dataset names"
        )
    val_datasets = [str(name).strip() for name in val_datasets]
    if not val_datasets or any(not name for name in val_datasets):
        raise ValueError(
            "data.val_datasets must be a non-empty list of evaluation dataset names"
        )
    if len(val_datasets) != len(set(val_datasets)):
        raise ValueError("data.val_datasets must not contain duplicate names")

    ersr_config = config.get("ersr")
    if ersr_config is not None and bool(ersr_config.get("enabled", False)):
        ersr_settings = parse_ersr_settings(
            ersr_config,
            max_response_tokens=int(config.rlvr_generation.val_max_new_tokens),
        )
        validate_ersr_dataset_subset(ersr_settings.datasets, val_datasets)

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != R2OPL_BASE_LOSS_MODE:
        raise ValueError(f"R^2OPL-base requires loss_mode={R2OPL_BASE_LOSS_MODE}")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("R^2OPL-base requires policy_loss_mode=reinforce")
    if not bool(loss.use_policy_gradient) or bool(loss.use_task_rewards):
        raise ValueError(
            "R^2OPL-base requires use_policy_gradient=true and use_task_rewards=false."
        )
    if loss.topk is not None:
        raise ValueError("R^2OPL-base uses sampled-token OPD and requires topk=null.")
    if loss.loss_max_clamp is not None:
        clamp_value = float(loss.loss_max_clamp)
        if not math.isfinite(clamp_value) or clamp_value < 0.0:
            raise ValueError(
                "R^2OPL-base loss_max_clamp must be finite and non-negative when set."
            )
    if float(loss.selection_ratio) != 1.0:
        raise ValueError(
            "R^2OPL-base uses every valid response token; selection_ratio must be 1.0."
        )
    if str(config.actor_rollout_ref.actor.loss_agg_mode) != "seq-mean-token-mean":
        raise ValueError(
            "R^2OPL-base requires loss_agg_mode=seq-mean-token-mean so each "
            "trajectory is averaged by its own response-token count."
        )

    if int(config.actor_rollout_ref.rollout.n) < 2:
        raise ValueError("R^2OPL-base requires at least two rollouts per prompt.")
    if int(config.data.train_batch_size) <= 0:
        raise ValueError(
            "R^2OPL-base requires a positive number of training questions per step."
        )

    max_prompt = int(config.data.max_prompt_length)
    train_response = int(config.rlvr_generation.train_max_new_tokens)
    val_response = int(config.rlvr_generation.val_max_new_tokens)
    if min(max_prompt, train_response, val_response) <= 0:
        raise ValueError("Prompt and response limits must be positive")
    if int(config.data.max_response_length) < max(train_response, val_response):
        raise ValueError("data.max_response_length does not cover generation limits")

    rollout = config.actor_rollout_ref.rollout
    if int(rollout.max_model_len) < max_prompt + val_response:
        raise ValueError("Student max_model_len does not cover validation context")
    required_train_tokens = max_prompt + train_response + 1
    limits = (
        config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
        rollout.log_prob_max_token_len_per_gpu,
        config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu,
    )
    if any(int(limit) < required_train_tokens for limit in limits):
        raise ValueError("Student log-prob token limits do not cover training context")

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    teacher_gpus = int(config.distillation.n_gpus_per_node) * int(
        config.distillation.nnodes
    )
    if student_gpus <= 0 or teacher_gpus <= 0:
        raise ValueError("Student and Teacher pools must each contain GPUs")
    if int(rollout.tensor_model_parallel_size) != 1:
        raise ValueError("R^2OPL-base currently requires Student rollout tensor parallelism 1")
    if int(rollout.data_parallel_size) != student_gpus:
        raise ValueError(
            "Student rollout data_parallel_size must equal the Student GPU count"
        )
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("R^2OPL-base currently requires Ulysses sequence parallelism 1")

    inference = config.distillation.teacher_models.teacher_model.inference
    if str(inference.name) != "vllm":
        raise ValueError("R^2OPL-base sampled-token Teacher scoring requires vLLM.")
    if int(inference.prompt_length) < max_prompt:
        raise ValueError("Teacher prompt_length is too small")
    if int(inference.response_length) < train_response:
        raise ValueError("Teacher response_length is too small")
    if int(inference.max_model_len) < int(inference.prompt_length) + train_response + 1:
        raise ValueError("Teacher max_model_len does not cover the scored sequence")
    if (
        ersr_config is not None
        and bool(ersr_config.get("enabled", False))
        and int(inference.max_model_len) < max_prompt + val_response
    ):
        raise ValueError(
            "Teacher max_model_len must cover the validation prompt and response "
            "when Teacher-replacement ERSR is enabled"
        )
    vllm_kwargs = inference.engine_kwargs.get("vllm", {})
    max_logprobs = vllm_kwargs.get("max_logprobs")
    if max_logprobs is None or int(max_logprobs) < 1:
        raise ValueError(
            "R^2OPL-base sampled-token Teacher scoring requires vLLM max_logprobs >= 1."
        )

    teacher_tp = int(inference.tensor_model_parallel_size)
    teacher_dp = int(inference.data_parallel_size)
    teacher_pp = int(inference.pipeline_model_parallel_size)
    if teacher_dp != 1:
        raise ValueError("Teacher per-replica data_parallel_size must be 1")
    replica_size = teacher_tp * teacher_dp * teacher_pp
    if replica_size <= 0 or teacher_gpus % replica_size:
        raise ValueError("Teacher replicas do not fit evenly in the Teacher pool")


# ---------------------------------------------------------------------------
# Trainer.
# ---------------------------------------------------------------------------


class R2OPLBaseTrainer(verl_sync.PPOTrainer):
    """Controller-side group difficulty plus detached hybrid OPD policy loss.

    R^2OPL-base is RL+OPD, so its ERSR evaluates BOTH actions: the Student
    action on correct validation trajectories and the Teacher replacement on
    incorrect ones.
    """

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_r2opl_base_config(config)
        super().__init__(*args, **kwargs)

    def _log_wandb_avg_chart(self, values: dict[str, float], *, expected_rollouts: int) -> None:
        tracking = getattr(self, "logger", None)
        backends = getattr(tracking, "logger", {})
        wandb = backends.get("wandb")
        if wandb is None:
            return
        step = int(self.global_steps)
        skip_until = os.environ.get("RLVR_WANDB_SKIP_UNTIL_STEP")
        if skip_until is not None and step <= int(skip_until):
            return
        histories = getattr(self, "_r2opl_avg_histories", None)
        if histories is None:
            histories = {}
            self._r2opl_avg_histories = histories
        history = histories.get(int(expected_rollouts))
        if history is None:
            history = {name: [] for name in values}
            histories[int(expected_rollouts)] = history
        if list(history) != list(values):
            raise RuntimeError(
                "Configured validation datasets changed during the run; cannot "
                f"build a stable Avg@{expected_rollouts} chart"
            )
        for name, value in values.items():
            points = history[name]
            if points and points[-1][0] == step:
                points[-1] = (step, float(value))
            else:
                points.append((step, float(value)))
        chart = wandb.plot.line_series(
            xs=[[point[0] for point in history[name]] for name in values],
            ys=[[point[1] for point in history[name]] for name in values],
            keys=list(values),
            title=f"Validation Avg@{expected_rollouts} - All Datasets",
            xname="Training Step",
        )
        wandb.log(
            {f"val-summary/Avg@{expected_rollouts}-all-datasets": chart},
            step=step,
            commit=False,
        )

    def _log_wandb_ersr_chart(self, values: dict[str, float | None]) -> None:
        tracking = getattr(self, "logger", None)
        backends = getattr(tracking, "logger", {})
        wandb = backends.get("wandb")
        if wandb is None:
            return
        step = int(self.global_steps)
        skip_until = os.environ.get("RLVR_WANDB_SKIP_UNTIL_STEP")
        if skip_until is not None and step <= int(skip_until):
            return
        history = getattr(self, "_r2opl_ersr_history", None)
        if history is None:
            history = {name: [] for name in values}
            self._r2opl_ersr_history = history
        if list(history) != list(values):
            raise RuntimeError(
                "Configured ERSR datasets/actions changed during the run; cannot "
                "build a stable combined chart"
            )
        for name, value in values.items():
            if value is None:
                continue
            points = history[name]
            if points and points[-1][0] == step:
                points[-1] = (step, float(value))
            else:
                points.append((step, float(value)))
        if not any(history.values()):
            return
        chart = wandb.plot.line_series(
            xs=[[point[0] for point in history[name]] for name in values],
            ys=[[point[1] for point in history[name]] for name in values],
            keys=list(values),
            title="Validation ERSR - All Datasets",
            xname="Training Step",
        )
        wandb.log({"val-summary/ERSR-all-datasets": chart}, step=step, commit=False)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        metrics = super()._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns
        )
        accuracies = reward_extra_infos_dict.get("acc")
        if accuracies is None:
            raise RuntimeError("R^2OPL-base validation reward must return acc")
        expected_rollouts = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        custom_metrics = compute_r2opl_base_pass_avg_metrics(
            data_sources,
            sample_uids,
            accuracies,
            expected_rollouts=expected_rollouts,
        )
        metrics.update(custom_metrics)
        avg_values = selected_avg_metrics(
            custom_metrics,
            [str(name) for name in self.config.data.val_datasets],
            expected_rollouts=expected_rollouts,
        )
        self._log_wandb_avg_chart(avg_values, expected_rollouts=expected_rollouts)
        return metrics

    def _additional_validation_metrics(self, validation_rollout_records: list[dict]) -> dict[str, float]:
        ersr_config = self.config.get("ersr")
        if ersr_config is None or not bool(ersr_config.get("enabled", False)):
            return {}
        settings = parse_ersr_settings(
            ersr_config,
            max_response_tokens=int(self.config.rlvr_generation.val_max_new_tokens),
        )
        # RL+OPD hybrid: correct trajectories evaluate the Student action and
        # incorrect trajectories evaluate the Teacher replacement.
        cases, _selection = build_ersr_step_cases(
            validation_rollout_records,
            self.tokenizer,
            settings,
            include_student_action=True,
            include_teacher_replace=True,
        )
        results = self.async_rollout_manager.evaluate_ersr_cases(
            cases,
            asdict(settings),
        )
        means, skipped = aggregate_ersr_advantages(
            results,
            settings.datasets,
            [STUDENT_ACTION, TEACHER_REPLACE],
        )
        self._log_wandb_ersr_chart(means)

        valid_results = [result for result in results if result.get("valid")]
        return {
            "val-ersr/selected_steps": float(len(cases)),
            "val-ersr/evaluated_steps": float(len(valid_results)),
            "val-ersr/skipped_steps": float(sum(skipped.values())),
            "val-ersr/empty_student_continuations": float(
                sum(
                    int(result.get("baseline_empty_continuations", 0))
                    + int(result.get("action_empty_continuations", 0))
                    for result in valid_results
                )
            ),
        }

    def _include_additional_validation_record(self, dataset: str, correct: bool) -> bool:
        ersr_config = self.config.get("ersr")
        if ersr_config is None or not bool(ersr_config.get("enabled", False)):
            return False
        # Keep both branches: correct trajectories feed Student-action cases
        # and incorrect trajectories feed Teacher-replacement cases.
        return str(dataset) in {
            str(name) for name in ersr_config.datasets
        }

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
                "R^2OPL-base requires one genuine rollout per sampled trajectory; got "
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
                f"R^2OPL-base requires {expected_questions} prompt groups with "
                f"{expected_per_group} rollouts each; group_count={len(counts)}, "
                f"mismatched_counts={bad_counts}."
            )

    @staticmethod
    def _scalar_field(
        data: TensorDict, name: str, batch_size: int
    ) -> torch.Tensor:
        """Read a required per-trajectory probe field produced by the agent loop."""

        if name not in data.keys():
            raise RuntimeError(f"R2OPL-base controller input is missing the {name!r} field.")
        value = data[name]
        if value.is_nested:
            value = value.to_padded_tensor(0.0)
        value = value.reshape(-1).float()
        if value.numel() != batch_size:
            raise RuntimeError(
                f"R^2OPL-base {name} must have one entry per trajectory; "
                f"got {value.numel()} values for batch size {batch_size}."
            )
        return value

    def _compute_old_log_prob(self, batch, metrics):
        batch = super()._compute_old_log_prob(batch, metrics)
        fields = [
            "uid",
            "prompts",
            "responses",
            "response_mask",
            "rm_scores",
            "teacher_logprobs",
            "old_log_probs",
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
                "R^2OPL-base controller expects jagged response masks from TransferQueue."
            )
        response_mask = response_mask_nested.to_padded_tensor(False).bool()
        old_log_probs = data["old_log_probs"]
        if old_log_probs.is_nested:
            old_log_probs = old_log_probs.to_padded_tensor(0.0)
        teacher_log_probs = no_padding_2_padding(
            data["teacher_logprobs"], data
        ).squeeze(-1)
        verifier_rewards = data["rm_scores"]
        if verifier_rewards.is_nested:
            verifier_rewards = verifier_rewards.to_padded_tensor(0.0)

        lambda_, miu = r2opl_base_hyperparameters(self.config)
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

        result = compute_r2opl_base_batch(
            old_log_probs=old_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            verifier_rewards=verifier_rewards,
            prompt_group_ids=prompt_group_ids,
            lambda_=lambda_,
            miu=miu,
            truncated_mask=truncated_mask,
            probe_correct_mask=probe_correct_mask,
            genuine_trajectory_mask=genuine_mask,
        )

        difficulty_tokens = result.difficulty.unsqueeze(-1).expand_as(response_mask).clone()
        difficulty_tokens *= response_mask.to(difficulty_tokens.dtype)
        output = TensorDict(
            {
                "r2opl_base_correct_mask": response_to_nested(
                    result.correct_mask, response_mask_nested
                ),
                "r2opl_base_error_mask": response_to_nested(
                    result.error_mask, response_mask_nested
                ),
                "r2opl_base_difficulty": response_to_nested(
                    difficulty_tokens, response_mask_nested
                ),
                "r2opl_base_lambda": torch.full(
                    (len(batch),),
                    lambda_,
                    dtype=torch.float32,
                    device=response_mask.device,
                ),
                "r2opl_base_miu": torch.full(
                    (len(batch),),
                    miu,
                    dtype=torch.float32,
                    device=response_mask.device,
                ),
            },
            batch_size=len(batch),
        )
        batch = verl_sync.tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=output,
        )
        metrics.update(result.metrics)
        metrics["r2opl_base/train/probe_attempted_trajectory_count"] = float(
            self._scalar_field(data, "r2opl_v2_probe_attempted", len(batch))[genuine_mask].sum().item()
        )
        metrics["r2opl_base/train/verifier_correct_trajectory_count"] = float(
            verifier_rewards.sum(dim=-1).eq(1.0)[genuine_mask].sum().item()
            if verifier_rewards.ndim == 2 else verifier_rewards.eq(1.0)[genuine_mask].sum().item()
        )
        return batch

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        aliases = {
            "actor/entropy": "r2opl_base/train/entropy",
            "actor/grad_norm": "r2opl_base/train/total_grad_norm",
            "response_length/mean": "r2opl_base/train/response_length_mean",
            "response_length/max": "r2opl_base/train/response_length_max",
            "perf/throughput": "r2opl_base/perf/tokens_per_second_per_gpu",
            "perf/time_per_step": "r2opl_base/perf/time_per_step",
            "actor/r2opl_base/correct_grad_norm": (
                "r2opl_base/train/correct_grad_norm"
            ),
            "actor/r2opl_base/error_grad_norm": (
                "r2opl_base/train/error_grad_norm"
            ),
            "actor/r2opl_base/total_grad_norm": (
                "r2opl_base/train/total_grad_norm"
            ),
            "actor/distillation/reverse_kl_estimate": (
                "r2opl_base/train/error_raw_opd_mean"
            ),
            "actor/distillation/loss": "r2opl_base/train/policy_loss",
        }
        for source, destination in aliases.items():
            if source in metrics:
                metrics[destination] = metrics[source]


__all__ = [
    "R2OPL_BASE_DEFAULT_LAMBDA",
    "R2OPL_BASE_DEFAULT_MIU",
    "R2OPL_BASE_LOSS_MODE",
    "R2OPL_BASE_PROBE_RESCUE_MARGIN",
    "R2OPL_BASE_VARIANT",
    "R2OPL_BASE_WANDB_GROUP",
    "R2OPLBaseBatchResult",
    "R2OPLBaseTrainer",
    "compute_r2opl_base_batch",
    "compute_r2opl_base_pass_avg_metrics",
    "configure_r2opl_base_batch",
    "configure_r2opl_base_defaults",
    "r2opl_base_hyperparameters",
    "r2opl_base_reinforce_loss",
    "r2opl_base_token_advantage",
    "selected_avg_metrics",
    "validate_r2opl_base_config",
]
