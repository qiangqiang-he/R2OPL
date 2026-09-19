"""Group-Sequence Policy Optimization (GSPO).

See https://arxiv.org/abs/2507.18071.  GSPO keeps GRPO's group-relative
outcome advantage and KL-free setup, but replaces the token-level importance
ratio with the sequence-level ratio
``s_i = exp(|y_i|^-1 * sum_t log(pi/pi_old))`` clipped at
``[1 - eps_low, 1 + eps_high]`` (standard ``eps = 0.2``).  The loss kernel is
VERL's registered ``gspo`` policy loss, selected via
``actor.policy_loss.loss_mode: gspo``.
"""

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


GSPO_VARIANT = "gspo"


def configure_gspo_defaults(config) -> str:
    """Install the common Student-only defaults for GSPO."""

    return configure_r2opl_defaults(config)


def configure_gspo_batch(config) -> int:
    """Configure GSPO's question batch for the Student data-parallel pool."""

    return configure_student_batch(config, algorithm_name="GSPO")


def validate_gspo_config(config) -> None:
    """Fail closed unless the configuration is standard, KL-free GSPO."""

    validate_r2opl_runtime_config(config)
    if str(config.algorithm.name) != GSPO_VARIANT:
        raise ValueError(f"GSPO requires algorithm.name={GSPO_VARIANT}")
    # GSPO inherits GRPO's group-relative outcome advantage unchanged; only
    # the policy-loss ratio moves to the sequence level.
    if str(config.algorithm.adv_estimator) != "grpo":
        raise ValueError("GSPO requires algorithm.adv_estimator=grpo")
    if not bool(config.algorithm.norm_adv_by_std_in_grpo):
        raise ValueError("GSPO requires group-wise standard-deviation normalization")
    if bool(config.algorithm.use_kl_in_reward):
        raise ValueError("R2OPL GSPO must not use an in-reward KL penalty")

    actor = config.actor_rollout_ref.actor
    if str(actor.policy_loss.loss_mode) != GSPO_VARIANT:
        raise ValueError("GSPO requires actor.policy_loss.loss_mode=gspo")
    if bool(actor.use_kl_loss):
        raise ValueError("R2OPL GSPO must not use an actor KL loss")
    if float(actor.kl_loss_coef) != 0.0:
        raise ValueError("R2OPL GSPO requires actor.kl_loss_coef=0")
    if str(actor.loss_agg_mode) != "seq-mean-token-mean":
        raise ValueError("GSPO requires loss_agg_mode=seq-mean-token-mean")
    # Standard GSPO clips the sequence-level ratio at [1 - eps, 1 + eps] with
    # eps = 0.2 on both sides, exactly as in the paper's main configuration.
    for field in ("clip_ratio", "clip_ratio_low", "clip_ratio_high"):
        if float(actor[field]) != 0.2:
            raise ValueError(f"GSPO requires actor.{field}=0.2")

    distillation = config.get("distillation")
    if distillation is not None and bool(distillation.get("enabled", False)):
        raise ValueError("GSPO is Student-only; distillation must be disabled")
    teacher_models = None if distillation is None else distillation.get("teacher_models")
    if teacher_models is not None:
        configured_teachers = [
            str(teacher.get("model_path"))
            for teacher in teacher_models.values()
            if teacher is not None and teacher.get("model_path") is not None
        ]
        if configured_teachers:
            raise ValueError("GSPO must not configure any Teacher model path")

    if int(config.actor_rollout_ref.rollout.n) < 2:
        raise ValueError("GSPO requires at least two responses per question")


class GSPOTrainer(BaseR2OPLTrainer):
    """KL-free GSPO with correct-trajectory Student-action ERSR."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_gspo_config(config)
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
    "GSPOTrainer",
    "GSPO_VARIANT",
    "configure_gspo_batch",
    "configure_gspo_defaults",
    "validate_gspo_config",
]
