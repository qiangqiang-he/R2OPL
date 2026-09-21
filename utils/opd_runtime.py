"""Minimal runtime shared by the single retained algorithm: PG-OPD."""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from utils.ersr import (
    TEACHER_REPLACE,
    aggregate_ersr_advantages,
    build_ersr_step_cases,
    parse_ersr_settings,
    validate_ersr_dataset_subset,
)
from utils.prompts import PROMPT_NAME, PROMPT_TEMPLATES, normalize_model_family
from verl.trainer import main_ppo_sync as verl_sync


TOKEN_SELECTION_METHODS = {"random", "topgap", "bottomgap"}


def configure_native_teacher_prompt(config, teacher_family: str) -> None:
    """Pass the Teacher's independent native-prompt contract to the dataset.

    A response sampled by the Student can be scored by a larger Teacher only
    when the two tokenizers have identical token-to-ID mappings.  That does
    *not* make their chat templates interchangeable: the dataset must render
    each template itself and preserve the Teacher's encoded prefix.
    """
    teachers = config.distillation.teacher_models
    if set(teachers) != {"teacher_model"}:
        raise ValueError("Native Teacher prompts currently require exactly one Teacher")
    teacher = teachers.teacher_model
    spec = {
        "model_path": str(teacher.model_path),
        "model_family": str(teacher_family),
        "max_prompt_length": int(teacher.inference.prompt_length),
        "prompt_name": str(config.prompt_template),
    }
    OmegaConf.update(config, "data.native_teacher", spec, force_add=True)
    # Retain the provisional PG-OPD key for already composed configurations
    # and local test tooling while all consumers move to ``native_teacher``.
    OmegaConf.update(config, "data.pg_opd_teacher", spec, force_add=True)


def validate_verl_pg_opd_support() -> None:
    """Fail early when the active VERL lacks the required PG-OPD kernel."""

    from verl.trainer.distillation import losses

    required = ("compute_sampled_token_reverse_kl", "distillation_loss")
    missing = [name for name in required if not hasattr(losses, name)]
    if missing:
        raise RuntimeError(
            "The active VERL build does not provide R2OPL PG-OPD support; "
            f"missing: {', '.join(missing)}"
        )


def validation_namespace(data_source: str) -> str:
    name = Path(str(data_source)).stem.lower()
    name = re.sub(r"20(\d{2})$", r"\1", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    if not name:
        raise ValueError(f"Cannot derive a validation name from {data_source!r}")
    return f"val-{name}"


def compute_pass_avg_metrics(
    data_sources: list[str],
    sample_uids: list[str],
    accuracies: list[float],
    *,
    expected_rollouts: int,
) -> dict[str, float]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for source, uid, accuracy in zip(
        data_sources, sample_uids, accuracies, strict=True
    ):
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
        prefix = f"{validation_namespace(source)}-core"
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
    dataset_names: list[str],
    *,
    expected_rollouts: int,
) -> dict[str, float]:
    """Extract one ordered Avg@k value for every configured benchmark."""

    selected: dict[str, float] = {}
    missing: list[str] = []
    for dataset_name in dataset_names:
        key = (
            f"{validation_namespace(dataset_name)}-core/"
            f"Avg@{expected_rollouts}"
        )
        if key not in metrics:
            missing.append(dataset_name)
            continue
        selected[dataset_name] = float(metrics[key])
    if missing:
        raise RuntimeError(
            "Validation did not produce Avg@"
            f"{expected_rollouts} for configured datasets: {', '.join(missing)}"
        )
    return selected


def configure_run_metadata(config) -> tuple[str, str]:
    run_name = str(config.get("run_name", "") or "").strip()
    if not run_name:
        prefix = str(config.get("run_name_prefix", "r2opl") or "r2opl").strip()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{prefix}_{timestamp}_p{os.getpid()}"
    group_name = str(config.get("group_name", "R2OPL") or "R2OPL").strip()
    OmegaConf.update(config, "run_name", run_name, force_add=True)
    OmegaConf.update(config, "group_name", group_name, force_add=True)
    OmegaConf.update(config, "trainer.experiment_name", run_name, force_add=True)
    return run_name, group_name


def configure_r2opl_defaults(config) -> str:
    """Install the shared dataset and prompt defaults before VERL starts."""

    project_root = Path(__file__).resolve().parents[1]
    prompt_name = str(config.get("prompt_template", "")).strip()
    if prompt_name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt_template={prompt_name!r}; expected one of: {available}"
        )
    student_family = normalize_model_family(config.student_model_family)
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
    # Retain this conventional field for prompt-aware VERL paths and logs.
    OmegaConf.update(config, "student_prompt", prompt_name, force_add=True)
    configure_run_metadata(config)
    return student_family


def configure_pg_opd_defaults(config) -> str:
    """Install PG-OPD defaults and enforce shared Student/Teacher token IDs."""

    student_family = configure_r2opl_defaults(config)
    teacher_family = normalize_model_family(config.teacher_model_family)
    if student_family != teacher_family:
        raise ValueError(
            "PG-OPD requires Student and Teacher to share a tokenizer family so "
            "the sampled token IDs have identical meaning"
        )
    OmegaConf.update(
        config,
        "teacher_prompt",
        str(config.prompt_template),
        force_add=True,
    )
    configure_native_teacher_prompt(config, teacher_family)
    return student_family


def configure_student_batch(config, *, algorithm_name: str) -> int:
    """Round question batches to the Student data-parallel world."""

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.tensor_model_parallel_size) != 1:
        raise ValueError(f"{algorithm_name} Student rollout tensor parallelism must be 1")
    if int(rollout.data_parallel_size) != student_gpus:
        raise ValueError(
            "Student rollout data_parallel_size must equal the Student GPU count"
        )
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError(f"{algorithm_name} Ulysses sequence parallelism must be 1")

    configured = int(config.data.train_batch_size)
    adjusted = math.ceil(configured / student_gpus) * student_gpus
    OmegaConf.update(config, "data.train_batch_size", adjusted)
    OmegaConf.update(config, "actor_rollout_ref.actor.ppo_mini_batch_size", adjusted)
    return adjusted


def configure_pg_opd_batch(config) -> int:
    """Configure the PG-OPD question batch for its Student pool."""

    return configure_student_batch(config, algorithm_name="PG-OPD")


def validate_token_selection_config(loss) -> None:
    ratio = float(loss.selection_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"selection_ratio must lie in [0, 1], got {ratio}")
    if ratio < 1.0 and str(loss.selection_method) not in TOKEN_SELECTION_METHODS:
        expected = ", ".join(sorted(TOKEN_SELECTION_METHODS))
        raise ValueError(f"selection_method must be one of {expected}")


def validate_r2opl_runtime_config(config) -> None:
    """Validate contracts shared by Student-only and distillation algorithms."""

    prompt_name = str(config.get("prompt_template", "")).strip()
    if prompt_name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt_template={prompt_name!r}; expected one of: {available}"
        )
    normalize_model_family(config.student_model_family)
    if str(config.trainer.project_name) != "R^2OPL":
        raise ValueError("All R2OPL runs must use trainer.project_name='R^2OPL'")
    if not str(config.get("group_name", "")).strip():
        raise ValueError("Every algorithm must define a non-empty W&B group_name")

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
    )
    if any(int(limit) < required_train_tokens for limit in limits):
        raise ValueError("Student log-prob token limits do not cover training context")

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    if student_gpus <= 0:
        raise ValueError("The Student pool must contain at least one GPU")
    if int(rollout.tensor_model_parallel_size) != 1:
        raise ValueError("R2OPL currently requires Student rollout tensor parallelism 1")
    if int(rollout.data_parallel_size) != student_gpus:
        raise ValueError(
            "Student rollout data_parallel_size must equal the Student GPU count"
        )
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("R2OPL currently requires Ulysses sequence parallelism 1")


def validate_pg_opd_runtime_config(config, expected_loss_mode: str = "reverse_kl") -> None:
    """Validate the PG-OPD-style runtime and 4+4 resource contract.

    ``expected_loss_mode`` lets OPD-style variants (e.g. OPDVR's ``opdvr``)
    reuse every shared contract while requiring their own loss kernel.
    """

    validate_verl_pg_opd_support()
    validate_r2opl_runtime_config(config)
    student_family = normalize_model_family(config.student_model_family)
    teacher_family = normalize_model_family(config.teacher_model_family)
    if student_family != teacher_family:
        raise ValueError("Student and Teacher tokenizer families must match")
    if not bool(config.distillation.enabled):
        raise ValueError("PG-OPD requires distillation.enabled=true")

    ersr_config = config.get("ersr")
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != expected_loss_mode:
        raise ValueError(
            f"Expected loss_mode={expected_loss_mode}, got {loss.loss_mode!r}"
        )
    if loss.topk is not None:
        raise ValueError("Sampled-token reverse KL requires topk=null")
    if int(loss.diagnostic_topk) != 16:
        raise ValueError("PG-OPD diagnostic_topk must be 16")
    if not bool(loss.use_policy_gradient):
        raise ValueError("PG-OPD requires use_policy_gradient=true")
    if bool(loss.use_task_rewards):
        raise ValueError("PG-OPD is distillation-only; use_task_rewards must be false")
    validate_token_selection_config(loss)

    max_prompt = int(config.data.max_prompt_length)
    train_response = int(config.rlvr_generation.train_max_new_tokens)
    val_response = int(config.rlvr_generation.val_max_new_tokens)
    required_train_tokens = max_prompt + train_response + 1
    if int(config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu) < required_train_tokens:
        raise ValueError("Student reference log-prob limit does not cover training context")

    teacher = config.distillation.teacher_models.teacher_model.inference
    if int(teacher.prompt_length) < max_prompt:
        raise ValueError("Teacher prompt_length is too small")
    if int(teacher.response_length) < train_response:
        raise ValueError("Teacher response_length is too small")
    if int(teacher.max_model_len) < int(teacher.prompt_length) + train_response + 1:
        raise ValueError("Teacher max_model_len does not cover the scored sequence")
    if (
        ersr_config is not None
        and bool(ersr_config.get("enabled", False))
        and int(teacher.max_model_len) < max_prompt + val_response
    ):
        raise ValueError(
            "Teacher max_model_len must cover the validation prompt and response "
            "when ERSR Teacher replacement is enabled"
        )

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    teacher_gpus = int(config.distillation.n_gpus_per_node) * int(
        config.distillation.nnodes
    )
    if student_gpus <= 0 or teacher_gpus <= 0:
        raise ValueError("Student and Teacher pools must each contain GPUs")
    teacher_tp = int(teacher.tensor_model_parallel_size)
    teacher_dp = int(teacher.data_parallel_size)
    teacher_pp = int(teacher.pipeline_model_parallel_size)
    if teacher_dp != 1:
        raise ValueError("Teacher per-replica data_parallel_size must be 1")
    replica_size = teacher_tp * teacher_dp * teacher_pp
    if replica_size <= 0 or teacher_gpus % replica_size:
        raise ValueError("Teacher replicas do not fit evenly in the Teacher pool")


class BaseR2OPLTrainer(verl_sync.PPOTrainer):
    """VERL trainer with R2OPL's shared validation metrics."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_r2opl_runtime_config(config)
        super().__init__(*args, **kwargs)

    def _log_wandb_avg_chart(
        self, values: dict[str, float], *, expected_rollouts: int
    ) -> None:
        """Log all benchmark Avg@K histories as one W&B multi-line chart."""

        tracking = getattr(self, "logger", None)
        backends = getattr(tracking, "logger", {})
        wandb = backends.get("wandb")
        if wandb is None:
            return

        step = int(self.global_steps)
        skip_until = os.environ.get("RLVR_WANDB_SKIP_UNTIL_STEP")
        if skip_until is not None and step <= int(skip_until):
            return

        history_key = int(expected_rollouts)
        histories = getattr(self, "_validation_avg_histories", None)
        if histories is None:
            histories = {}
            self._validation_avg_histories = histories
        history = histories.get(history_key)
        if history is None:
            history = {name: [] for name in values}
            histories[history_key] = history
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
        # The ordinary scalar validation metrics are logged immediately after
        # _validate returns. Leave this W&B call uncommitted so both the custom
        # chart and the unchanged per-dataset metrics share the same step.
        wandb.log(
            {f"val-summary/Avg@{expected_rollouts}-all-datasets": chart},
            step=step,
            commit=False,
        )

    def _log_wandb_ersr_chart(
        self, values: dict[str, float | None]
    ) -> None:
        """Log all dataset/action ERSR histories in one W&B chart."""

        tracking = getattr(self, "logger", None)
        backends = getattr(tracking, "logger", {})
        wandb = backends.get("wandb")
        if wandb is None:
            return

        step = int(self.global_steps)
        skip_until = os.environ.get("RLVR_WANDB_SKIP_UNTIL_STEP")
        if skip_until is not None and step <= int(skip_until):
            return

        history = getattr(self, "_validation_ersr_history", None)
        if history is None:
            history = {name: [] for name in values}
            self._validation_ersr_history = history
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
        wandb.log(
            {"val-summary/ERSR-all-datasets": chart},
            step=step,
            commit=False,
        )

    def _additional_validation_metrics(
        self, validation_rollout_records: list[dict]
    ) -> dict[str, float]:
        del validation_rollout_records
        return {}

    def _include_additional_validation_record(
        self, dataset: str, correct: bool
    ) -> bool:
        del dataset, correct
        return False

    def _val_metrics_update(
        self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns
    ):
        metrics = super()._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns
        )
        accuracies = reward_extra_infos_dict.get("acc")
        if accuracies is None:
            raise RuntimeError("R2OPL validation reward must return acc")
        expected_rollouts = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        custom_metrics = compute_pass_avg_metrics(
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
        self._log_wandb_avg_chart(
            avg_values,
            expected_rollouts=expected_rollouts,
        )
        return metrics

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)


class BasePGOPDTrainer(BaseR2OPLTrainer):
    """Shared validation plus PG-OPD Teacher-replacement ERSR."""

    expected_loss_mode = "reverse_kl"

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_pg_opd_runtime_config(config, expected_loss_mode=self.expected_loss_mode)
        super().__init__(*args, **kwargs)

    def _additional_validation_metrics(
        self, validation_rollout_records: list[dict]
    ) -> dict[str, float]:
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

    def _include_additional_validation_record(
        self, dataset: str, correct: bool
    ) -> bool:
        ersr_config = self.config.get("ersr")
        if ersr_config is None or not bool(ersr_config.get("enabled", False)):
            return False
        return not correct and str(dataset) in {
            str(name) for name in ersr_config.datasets
        }

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        aliases = {
            "actor/distillation/reverse_kl_estimate": (
                "pg-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/selected_token_ratio": (
                "pg-opd/train/selected_token_ratio"
            ),
            "actor/distillation/selection_gap_mean": "pg-opd/train/gap_mean",
            "actor/distillation/loss": "pg-opd/train/policy_loss",
            "actor/grad_norm": "pg-opd/train/grad_norm",
        }
        for source, destination in aliases.items():
            if source in metrics:
                metrics[destination] = metrics[source]


__all__ = [
    "BasePGOPDTrainer",
    "BaseR2OPLTrainer",
    "TOKEN_SELECTION_METHODS",
    "compute_pass_avg_metrics",
    "configure_pg_opd_batch",
    "configure_pg_opd_defaults",
    "configure_r2opl_defaults",
    "configure_student_batch",
    "selected_avg_metrics",
    "validate_pg_opd_runtime_config",
    "validate_r2opl_runtime_config",
    "validate_token_selection_config",
    "validate_verl_pg_opd_support",
    "validation_namespace",
]
