"""Group Relative Policy Optimization (GRPO)."""

from __future__ import annotations

from dataclasses import asdict

from utils.ersr import (
    STUDENT_ACTION,
    aggregate_ersr_advantages,
    build_ersr_step_cases,
    parse_ersr_settings,
)
from utils.opd_runtime import (
    BaseR2OPLTrainer,
    configure_r2opl_defaults,
    configure_student_batch,
    validate_r2opl_runtime_config,
)


GRPO_VARIANT = "grpo"


def configure_grpo_defaults(config) -> str:
    """Install the common Student-only defaults for GRPO."""

    return configure_r2opl_defaults(config)


def configure_grpo_batch(config) -> int:
    """Configure GRPO's question batch for the Student data-parallel pool."""

    return configure_student_batch(config, algorithm_name="GRPO")


def validate_grpo_config(config) -> None:
    """Fail closed unless the configuration is standard, KL-free GRPO."""

    validate_r2opl_runtime_config(config)
    if str(config.algorithm.name) != GRPO_VARIANT:
        raise ValueError(f"GRPO requires algorithm.name={GRPO_VARIANT}")
    if str(config.algorithm.adv_estimator) != GRPO_VARIANT:
        raise ValueError("GRPO requires algorithm.adv_estimator=grpo")
    if not bool(config.algorithm.norm_adv_by_std_in_grpo):
        raise ValueError("GRPO requires group-wise standard-deviation normalization")
    if bool(config.algorithm.use_kl_in_reward):
        raise ValueError("R2OPL GRPO must not use an in-reward KL penalty")

    actor = config.actor_rollout_ref.actor
    if bool(actor.use_kl_loss):
        raise ValueError("R2OPL GRPO must not use an actor KL loss")
    if float(actor.kl_loss_coef) != 0.0:
        raise ValueError("R2OPL GRPO requires actor.kl_loss_coef=0")
    if str(actor.loss_agg_mode) != "seq-mean-token-mean":
        raise ValueError("GRPO requires loss_agg_mode=seq-mean-token-mean")
    for field in ("clip_ratio", "clip_ratio_low", "clip_ratio_high"):
        if float(actor[field]) != 0.2:
            raise ValueError(f"GRPO requires actor.{field}=0.2")

    distillation = config.get("distillation")
    if distillation is not None and bool(distillation.get("enabled", False)):
        raise ValueError("GRPO is Student-only; distillation must be disabled")
    teacher_models = None if distillation is None else distillation.get("teacher_models")
    if teacher_models is not None:
        configured_teachers = [
            str(teacher.get("model_path"))
            for teacher in teacher_models.values()
            if teacher is not None and teacher.get("model_path") is not None
        ]
        if configured_teachers:
            raise ValueError("GRPO must not configure any Teacher model path")

    if int(config.actor_rollout_ref.rollout.n) < 2:
        raise ValueError("GRPO requires at least two responses per question")


class GRPOTrainer(BaseR2OPLTrainer):
    """KL-free GRPO with correct-trajectory Student-action ERSR."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_grpo_config(config)
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
            include_student_action=True,
            include_teacher_replace=False,
        )
        results = self.async_rollout_manager.evaluate_ersr_cases(
            cases,
            asdict(settings),
        )
        means, skipped = aggregate_ersr_advantages(
            results,
            settings.datasets,
            [STUDENT_ACTION],
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
        return correct and str(dataset) in {
            str(name) for name in ersr_config.datasets
        }


__all__ = [
    "GRPOTrainer",
    "GRPO_VARIANT",
    "configure_grpo_batch",
    "configure_grpo_defaults",
    "validate_grpo_config",
]
