"""Entropy-Aware On-Policy Distillation (EOPD).

EOPD keeps PG-OPD's signed sampled-token reverse-KL policy gradient and adds
an entropy-gated Top-k forward KL: wherever the Teacher's full-vocabulary
entropy exceeds ``eopd_entropy_threshold``, the Student is pulled toward the
Teacher's Top-k distribution renormalized over that set.  The two terms are
combined as ``pg_loss + eopd_alpha * forward_kl_loss``.

Everything EOPD-specific lives in this module: the reference tensor kernels,
the runtime configuration/validation, and the trainer.  Only VERL built-ins
(the registered ``eopd`` distillation loss, the vLLM entropy-gather patch,
and the ERSR evaluation fabric that VERL itself consumes) are imported.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from algorithms.pg_opd import signed_pg_opd_loss
from utils.ersr import (
    TEACHER_REPLACE,
    aggregate_ersr_advantages,
    build_ersr_step_cases,
    parse_ersr_settings,
    validate_ersr_dataset_subset,
)
from utils.prompts import PROMPT_TEMPLATES, normalize_model_family
from verl.trainer import main_ppo_sync as verl_sync


EOPD_VARIANT = "eopd"
EOPD_LOSS_MODE = "eopd"
EOPD_WANDB_GROUP = "EOPD"
EOPD_TOPK = 16
EOPD_ENTROPY_THRESHOLD = 0.8
EOPD_ALPHA = 1.0
_TOKEN_SELECTION_METHODS = {"random", "topgap", "bottomgap"}


# ---------------------------------------------------------------------------
# Reference tensor kernels (CPU-testable; mirror VERL's registered eopd path).
# ---------------------------------------------------------------------------


def compute_entropy_gated_forward_kl(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_entropy: torch.Tensor,
    *,
    entropy_threshold: float,
) -> torch.Tensor:
    """Return EOPD's per-token normalized-Teacher Top-k forward KL.

    ``student_topk_log_probs`` must come from the Student's full-vocabulary
    log-softmax; it is deliberately not renormalized inside the Teacher's
    Top-k set.  Teacher probabilities are normalized over that set exactly as
    EOPD requires, and the gate uses a strict ``>`` on the Teacher's
    full-vocabulary entropy.
    """

    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "EOPD Student and Teacher Top-k log-probabilities must have "
            "identical shapes."
        )
    if student_topk_log_probs.ndim < 2 or student_topk_log_probs.shape[-1] <= 0:
        raise ValueError("EOPD Top-k tensors must include a non-empty Top-k dimension.")
    if teacher_entropy.shape != teacher_topk_log_probs.shape[:-1]:
        raise ValueError(
            "EOPD Teacher entropy must match the Top-k tensors' token dimensions."
        )
    entropy_threshold = float(entropy_threshold)
    if not math.isfinite(entropy_threshold) or entropy_threshold < 0:
        raise ValueError(
            f"EOPD entropy_threshold must be finite and non-negative, got "
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


def entropy_gated_topk_forward_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_entropy: torch.Tensor,
    *,
    entropy_threshold: float,
    chunk_size: int = 512,
):
    """Entropy-gated Top-k FKL from full-vocabulary Student logits.

    ``student_logits`` has shape ``(tokens, vocab)``; the Teacher tensors are
    token-aligned ``(tokens,)`` / ``(tokens, topk)``.  Response positions are
    processed in bounded chunks so a full-vocabulary projection is never
    materialized for the whole sequence at once.  The returned tensor carries
    gradients with respect to ``student_logits``.
    """

    tokens = int(student_logits.shape[0])
    if teacher_topk_ids.shape != teacher_topk_log_probs.shape:
        raise ValueError("EOPD Teacher Top-k ids and log-probs must align.")
    if teacher_topk_ids.shape[0] != tokens or teacher_entropy.shape[0] != tokens:
        raise ValueError("EOPD Teacher tensors must be token-aligned with the logits.")
    chunk_size = max(1, int(chunk_size))
    chunks = []
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        log_probs = torch.nn.functional.log_softmax(
            student_logits[start:end].float(), dim=-1
        )
        student_topk_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=teacher_topk_ids[start:end].to(device=log_probs.device),
        )
        chunks.append(
            compute_entropy_gated_forward_kl(
                student_topk_log_probs,
                teacher_topk_log_probs[start:end].to(
                    device=student_topk_log_probs.device
                ),
                teacher_entropy[start:end].to(
                    device=student_topk_log_probs.device
                ),
                entropy_threshold=entropy_threshold,
            )
        )
    return torch.cat(chunks, dim=0) if chunks else torch.zeros((0,))


def eopd_total_loss(
    student_log_probs: torch.Tensor,
    teacher_sampled_log_probs: torch.Tensor,
    forward_kl_losses: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Reference CPU kernel for ``pg + alpha * entropy-gated FKL``.

    Both terms use the sequence-mean/token-mean aggregation of the production
    actor (``loss_agg_mode=seq-mean-token-mean``).
    """

    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"EOPD alpha must be finite and non-negative, got {alpha}.")
    pg_term = signed_pg_opd_loss(
        student_log_probs, teacher_sampled_log_probs, response_mask
    )
    mask = response_mask.to(dtype=forward_kl_losses.dtype)
    per_sequence = (forward_kl_losses * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(
        1.0
    )
    return pg_term + alpha * per_sequence.mean()


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


def compute_eopd_pass_avg_metrics(
    data_sources, sample_uids, accuracies, *, expected_rollouts: int
) -> dict[str, float]:
    """Local Pass@1 / Pass@k / Avg@k validation metrics for EOPD."""

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
    group_name = str(config.get("group_name", EOPD_WANDB_GROUP) or EOPD_WANDB_GROUP).strip()
    OmegaConf.update(config, "run_name", run_name, force_add=True)
    OmegaConf.update(config, "group_name", group_name, force_add=True)
    OmegaConf.update(config, "trainer.experiment_name", run_name, force_add=True)
    return run_name, group_name


def configure_eopd_defaults(config) -> str:
    """Install EOPD's dataset/prompt defaults before VERL starts."""

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
            "EOPD requires Student and Teacher to share a tokenizer family so "
            "the sampled token IDs have identical meaning"
        )
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
    _configure_run_metadata(config)
    return student_family


def configure_eopd_batch(config) -> int:
    """Round EOPD's question batch to the Student data-parallel world."""

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.tensor_model_parallel_size) != 1:
        raise ValueError("EOPD Student rollout tensor parallelism must be 1")
    if int(rollout.data_parallel_size) != student_gpus:
        raise ValueError("Student rollout data_parallel_size must equal the Student GPU count")
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("EOPD Ulysses sequence parallelism must be 1")
    configured = int(config.data.train_batch_size)
    adjusted = math.ceil(configured / student_gpus) * student_gpus
    OmegaConf.update(config, "data.train_batch_size", adjusted)
    OmegaConf.update(config, "actor_rollout_ref.actor.ppo_mini_batch_size", adjusted)
    return adjusted


def _validate_verl_eopd_support() -> None:
    """Fail early when the active VERL lacks the EOPD kernels."""

    from verl.trainer.distillation import losses
    from verl.workers.rollout.vllm_rollout import utils as rollout_utils

    missing = [
        name
        for name in (
            "compute_eopd_sampled_token_reverse_kl",
            "distillation_loss",
        )
        if not hasattr(losses, name)
    ] + [
        name
        for name in ("enable_eopd_entropy_gather", "extract_prompt_logprobs")
        if not hasattr(rollout_utils, name)
    ]
    if missing:
        raise RuntimeError(
            "The active VERL build does not provide EOPD support; "
            f"missing: {', '.join(missing)}"
        )


def validate_eopd_config(config) -> None:
    """Fail closed unless the configuration is standard EOPD."""

    _validate_verl_eopd_support()

    prompt_name = str(config.get("prompt_template", "")).strip()
    if prompt_name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt_template={prompt_name!r}; expected one of: {available}"
        )
    student_family = normalize_model_family(config.student_model_family)
    teacher_family = normalize_model_family(config.teacher_model_family)
    if student_family != teacher_family:
        raise ValueError("EOPD Student and Teacher tokenizer families must match")
    if str(config.trainer.project_name) != "R^2OPL":
        raise ValueError("All R2OPL runs must use trainer.project_name='R^2OPL'")
    if str(config.get("group_name", "")).strip() != EOPD_WANDB_GROUP:
        raise ValueError(f"EOPD requires group_name={EOPD_WANDB_GROUP!r}")
    if str(config.algorithm.name) != EOPD_VARIANT:
        raise ValueError(f"EOPD requires algorithm.name={EOPD_VARIANT}")
    if not bool(config.distillation.enabled):
        raise ValueError("EOPD requires distillation.enabled=true")

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
    if str(loss.loss_mode) != EOPD_LOSS_MODE:
        raise ValueError(f"EOPD requires loss_mode={EOPD_LOSS_MODE}")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("EOPD requires policy_loss_mode=reinforce for standard OPD")
    if int(loss.diagnostic_topk) != 16:
        raise ValueError("EOPD diagnostic_topk must be 16")
    if not bool(loss.use_policy_gradient):
        raise ValueError("EOPD requires use_policy_gradient=true")
    if bool(loss.use_task_rewards):
        raise ValueError("EOPD is distillation-only; use_task_rewards must be false")
    topk = int(loss.topk)
    if topk <= 0:
        raise ValueError(f"EOPD topk must be positive, got {topk}.")
    entropy_threshold = float(loss.eopd_entropy_threshold)
    if not math.isfinite(entropy_threshold) or entropy_threshold < 0:
        raise ValueError(
            "EOPD entropy threshold must be finite and non-negative, got "
            f"{entropy_threshold}."
        )
    alpha = float(loss.eopd_alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"EOPD alpha must be finite and non-negative, got {alpha}.")
    selection_ratio = float(loss.selection_ratio)
    if not 0.0 <= selection_ratio <= 1.0:
        raise ValueError(f"selection_ratio must lie in [0, 1], got {selection_ratio}")
    if selection_ratio < 1.0 and str(loss.selection_method) not in _TOKEN_SELECTION_METHODS:
        expected = ", ".join(sorted(_TOKEN_SELECTION_METHODS))
        raise ValueError(f"selection_method must be one of {expected}")

    actor = config.actor_rollout_ref.actor
    if str(actor.strategy) != "fsdp":
        raise ValueError("EOPD's combined OPD + FKL path currently requires FSDP.")
    if float(actor.entropy_coeff) != 0.0:
        raise ValueError(
            "EOPD's chunked Student projection does not compute Student entropy; "
            "set actor.entropy_coeff=0."
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
        actor.ppo_max_token_len_per_gpu,
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
        raise ValueError("EOPD currently requires Student rollout tensor parallelism 1")
    if int(rollout.data_parallel_size) != student_gpus:
        raise ValueError("Student rollout data_parallel_size must equal the Student GPU count")
    if int(actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("EOPD currently requires Ulysses sequence parallelism 1")

    teacher = config.distillation.teacher_models.teacher_model
    inference = teacher.inference
    if str(inference.name) != "vllm":
        raise ValueError("EOPD full-vocabulary Teacher entropy currently requires vLLM.")
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
            "when ERSR Teacher replacement is enabled"
        )
    vllm_kwargs = inference.engine_kwargs.get("vllm", {})
    max_logprobs = vllm_kwargs.get("max_logprobs")
    if max_logprobs is None or int(max_logprobs) < topk + 1:
        raise ValueError(
            "EOPD requires vLLM max_logprobs >= topk + 1 for its bounded "
            "Top-k plus entropy output."
        )
    eopd_entropy_topk = vllm_kwargs.get("eopd_entropy_topk")
    if eopd_entropy_topk is None or int(eopd_entropy_topk) != topk:
        raise ValueError("EOPD requires vLLM eopd_entropy_topk to match distillation topk.")

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


class EOPDTrainer(verl_sync.PPOTrainer):
    """VERL trainer with EOPD's validation metrics and Teacher-replace ERSR."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_eopd_config(config)
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
        histories = getattr(self, "_eopd_avg_histories", None)
        if histories is None:
            histories = {}
            self._eopd_avg_histories = histories
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
        history = getattr(self, "_eopd_ersr_history", None)
        if history is None:
            history = {name: [] for name in values}
            self._eopd_ersr_history = history
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
            raise RuntimeError("EOPD validation reward must return acc")
        expected_rollouts = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        custom_metrics = compute_eopd_pass_avg_metrics(
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
        cases, _selection = build_ersr_step_cases(
            validation_rollout_records,
            self.tokenizer,
            settings,
            include_student_action=False,
            include_teacher_replace=True,
        )
        results = self.async_rollout_manager.evaluate_ersr_cases(
            cases,
            asdict(settings),
        )
        means, skipped = aggregate_ersr_advantages(
            results,
            settings.datasets,
            [TEACHER_REPLACE],
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
        return not correct and str(dataset) in {
            str(name) for name in ersr_config.datasets
        }

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        prefix = "eopd/train"
        aliases = {
            "actor/distillation/reverse_kl_estimate": f"{prefix}/reverse_kl_estimate",
            "actor/distillation/eopd_teacher_entropy": f"{prefix}/teacher_entropy",
            "actor/distillation/eopd_high_entropy_token_ratio": (
                f"{prefix}/high_entropy_token_ratio"
            ),
            "actor/distillation/eopd_forward_kl_per_token": f"{prefix}/forward_kl_per_token",
            "actor/distillation/eopd_forward_kl_loss": f"{prefix}/forward_kl_loss",
            "actor/distillation/eopd_scaled_forward_kl_loss": (
                f"{prefix}/scaled_forward_kl_loss"
            ),
            "actor/distillation/loss": f"{prefix}/total_loss",
            "actor/grad_norm": f"{prefix}/grad_norm",
        }
        for source, destination in aliases.items():
            if source in metrics:
                metrics[destination] = metrics[source]


__all__ = [
    "EOPD_ALPHA",
    "EOPD_ENTROPY_THRESHOLD",
    "EOPD_LOSS_MODE",
    "EOPD_TOPK",
    "EOPD_VARIANT",
    "EOPD_WANDB_GROUP",
    "EOPDTrainer",
    "compute_entropy_gated_forward_kl",
    "compute_eopd_pass_avg_metrics",
    "configure_eopd_batch",
    "configure_eopd_defaults",
    "entropy_gated_topk_forward_kl_from_logits",
    "eopd_total_loss",
    "selected_avg_metrics",
    "validate_eopd_config",
]
