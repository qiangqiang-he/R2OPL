"""Paper-faithful Return-Referenced On-Policy Learning (R²OPL).

This module intentionally lives beside :mod:`algorithms.r2opl_base`.  The
base implementation is a useful, simpler RL/OPD baseline; it does *not*
implement the paper's semantic-step answer probes.  R²OPL routes successful
rollouts to reward reinforcement and failed rollouts to sampled-token OPD,
then lets the actor-side packed Student forward apply the answer-probe gate.

The controller owns only outcome routing and prompt difficulty.  Probe
probabilities are deliberately computed by the optimisation forward itself in
``FSDPEngineWithLMHead`` so that there is no stale or separately-forwarded
Student score in the objective.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from algorithms.r2opl_base import R2OPLBaseTrainer
from utils.prompts import PROMPT_TEMPLATES, normalize_model_family
from verl.trainer import main_ppo_sync as verl_sync
from verl.workers.utils.padding import response_to_nested


R2OPL_VARIANT = "r2opl"
R2OPL_LOSS_MODE = "r2opl"
R2OPL_WANDB_GROUP = "R2OPL"
R2OPL_DEFAULT_LAMBDA = 0.1
R2OPL_DEFAULT_MIU = 16.0
R2OPL_DEFAULT_ALPHA_R = 0.25
R2OPL_DEFAULT_ALPHA_D = 0.5
R2OPL_DEFAULT_EPSILON = 0.8


@dataclass(frozen=True)
class R2OPLHyperparameters:
    """The five scalar coefficients in the R²OPL paper objective."""

    lambda_: float
    miu: float
    alpha_r: float
    alpha_d: float
    epsilon: float


@dataclass(frozen=True)
class R2OPLControllerBatchResult:
    """Per-rollout routing state materialised by the trainer controller."""

    correctness: torch.Tensor
    difficulty: torch.Tensor
    group_success: torch.Tensor
    metrics: dict[str, float]


def _ordered_groups(prompt_group_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(group_id) for group_id in prompt_group_ids))


def _validate_binary_rewards(sequence_rewards: torch.Tensor) -> None:
    if not bool(torch.isfinite(sequence_rewards).all()):
        raise ValueError("R²OPL verifier rewards must be finite.")
    binary = sequence_rewards.eq(0.0) | sequence_rewards.eq(1.0)
    if not bool(binary.all()):
        invalid = sequence_rewards[~binary].detach().cpu().tolist()
        raise ValueError(
            "R²OPL requires binary verifier rewards in {0, 1}; "
            f"got invalid values {invalid[:5]}."
        )


def compute_r2opl_controller_batch(
    response_mask: torch.Tensor,
    verifier_rewards: torch.Tensor,
    prompt_group_ids: Sequence[str],
    *,
    genuine_trajectory_mask: torch.Tensor | None = None,
) -> R2OPLControllerBatchResult:
    """Compute Eq. (routing/scaling)'s outcome and prompt-difficulty terms.

    ``difficulty`` is exactly ``1 - mean_group_reward``.  The step gates do
    not belong here: they depend on probabilities emitted by the current actor
    optimisation forward, not the old-policy forward used to build PPO state.
    """

    if response_mask.ndim != 2:
        raise ValueError("R²OPL response_mask must be two-dimensional.")
    response_mask = response_mask.bool()
    batch_size = response_mask.shape[0]
    if len(prompt_group_ids) != batch_size:
        raise ValueError("R²OPL prompt_group_ids must have one entry per rollout.")

    if genuine_trajectory_mask is None:
        genuine_trajectory_mask = torch.ones(
            batch_size, dtype=torch.bool, device=response_mask.device
        )
    elif genuine_trajectory_mask.ndim != 1 or genuine_trajectory_mask.numel() != batch_size:
        raise ValueError(
            "R²OPL genuine_trajectory_mask must have one entry per rollout."
        )
    else:
        genuine_trajectory_mask = genuine_trajectory_mask.to(
            device=response_mask.device, dtype=torch.bool
        )

    if verifier_rewards.ndim == 2:
        if verifier_rewards.shape[0] != batch_size:
            raise ValueError("R²OPL verifier rewards must match rollout batch size.")
        sequence_rewards = verifier_rewards.float().sum(dim=-1)
    elif verifier_rewards.ndim == 1:
        sequence_rewards = verifier_rewards.float()
    else:
        raise ValueError("R²OPL verifier_rewards must be one- or two-dimensional.")
    if sequence_rewards.numel() != batch_size:
        raise ValueError("R²OPL verifier rewards must match rollout batch size.")

    genuine_tokens = response_mask & genuine_trajectory_mask.unsqueeze(-1)
    if bool((genuine_trajectory_mask & genuine_tokens.sum(dim=-1).eq(0)).any()):
        rows = (genuine_trajectory_mask & genuine_tokens.sum(dim=-1).eq(0)).nonzero(
            as_tuple=False
        ).flatten().tolist()
        raise ValueError(
            "R²OPL cannot train a genuine rollout with no response tokens; "
            f"empty rows: {rows[:5]}."
        )
    _validate_binary_rewards(sequence_rewards[genuine_trajectory_mask])

    correctness = sequence_rewards.eq(1.0) & genuine_trajectory_mask
    difficulty = torch.zeros(batch_size, dtype=torch.float32, device=response_mask.device)
    group_success: list[torch.Tensor] = []
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
        raise ValueError("R²OPL requires at least one genuine prompt group.")

    group_success_tensor = torch.stack(group_success)
    correct_tokens = response_mask & correctness.unsqueeze(-1)
    error_tokens = (
        response_mask
        & genuine_trajectory_mask.unsqueeze(-1)
        & ~correctness.unsqueeze(-1)
    )
    genuine_count = int(genuine_trajectory_mask.sum().item())
    metrics = {
        "r2opl/train/mean_score": float(correctness[genuine_trajectory_mask].float().mean().item()),
        "r2opl/train/group_success_rate": float(group_success_tensor.mean().item()),
        "r2opl/train/prompt_difficulty_mean": float((1.0 - group_success_tensor).mean().item()),
        "r2opl/train/correct_trajectory_count": float(correctness.sum().item()),
        "r2opl/train/error_trajectory_count": float(
            (genuine_trajectory_mask & ~correctness).sum().item()
        ),
        "r2opl/train/correct_trajectory_ratio": float(
            correctness[genuine_trajectory_mask].float().mean().item()
        ),
        "r2opl/train/error_trajectory_ratio": float(
            (~correctness[genuine_trajectory_mask]).float().mean().item()
        ),
        "r2opl/train/correct_token_count": float(correct_tokens.sum().item()),
        "r2opl/train/error_token_count": float(error_tokens.sum().item()),
        "r2opl/train/genuine_trajectory_count": float(genuine_count),
    }
    return R2OPLControllerBatchResult(
        correctness=correctness,
        difficulty=difficulty,
        group_success=group_success_tensor,
        metrics=metrics,
    )


def r2opl_hyperparameters(config) -> R2OPLHyperparameters:
    """Read and validate the paper defaults/overrides from ``algorithm.r2opl``."""

    settings = config.algorithm.get(R2OPL_VARIANT, {})
    values = R2OPLHyperparameters(
        lambda_=float(settings.get("lambda", R2OPL_DEFAULT_LAMBDA)),
        miu=float(settings.get("miu", R2OPL_DEFAULT_MIU)),
        alpha_r=float(settings.get("alpha_R", R2OPL_DEFAULT_ALPHA_R)),
        alpha_d=float(settings.get("alpha_D", R2OPL_DEFAULT_ALPHA_D)),
        epsilon=float(settings.get("epsilon", R2OPL_DEFAULT_EPSILON)),
    )
    if not math.isfinite(values.lambda_) or values.lambda_ < 0.0:
        raise ValueError(f"R²OPL lambda must be finite and non-negative; got {values.lambda_}.")
    if not math.isfinite(values.miu) or values.miu <= 0.0:
        raise ValueError(f"R²OPL miu must be finite and positive; got {values.miu}.")
    if not math.isfinite(values.alpha_r) or values.alpha_r < 0.0:
        raise ValueError(f"R²OPL alpha_R must be finite and non-negative; got {values.alpha_r}.")
    if not math.isfinite(values.alpha_d) or values.alpha_d < 0.0:
        raise ValueError(f"R²OPL alpha_D must be finite and non-negative; got {values.alpha_d}.")
    if not math.isfinite(values.epsilon) or values.epsilon <= 0.0:
        raise ValueError(f"R²OPL epsilon must be finite and positive; got {values.epsilon}.")
    return values


def _configure_native_teacher_prompt(config, teacher_family: str) -> None:
    teachers = config.distillation.teacher_models
    if set(teachers) != {"teacher_model"}:
        raise ValueError("R²OPL native prompts currently require exactly one Teacher.")
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


def _configure_run_metadata(config) -> tuple[str, str]:
    run_name = str(config.get("run_name", "") or "").strip()
    if not run_name:
        prefix = str(config.get("run_name_prefix", R2OPL_VARIANT) or R2OPL_VARIANT).strip()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{prefix}_{timestamp}_p{os.getpid()}"
    group_name = str(config.get("group_name", R2OPL_WANDB_GROUP) or R2OPL_WANDB_GROUP).strip()
    OmegaConf.update(config, "run_name", run_name, force_add=True)
    OmegaConf.update(config, "group_name", group_name, force_add=True)
    OmegaConf.update(config, "trainer.experiment_name", run_name, force_add=True)
    return run_name, group_name


def configure_r2opl_defaults(config) -> str:
    """Install the native-tokenizer and packed-probe defaults before VERL starts."""

    project_root = Path(__file__).resolve().parents[1]
    prompt_name = str(config.get("prompt_template", "")).strip()
    if prompt_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt_template={prompt_name!r}.")
    student_family = normalize_model_family(config.student_model_family)
    teacher_family = normalize_model_family(config.teacher_model_family)
    if student_family != teacher_family:
        raise ValueError(
            "R²OPL requires Student and Teacher to share a tokenizer family so "
            "sampled response token IDs retain their meaning."
        )
    r2opl_hyperparameters(config)
    OmegaConf.update(
        config,
        "data.custom_cls",
        {"path": str(project_root / "utils" / "custom_dataset.py"), "name": "CustomDataset"},
        force_add=True,
    )
    OmegaConf.update(config, "data.model_family", student_family, force_add=True)
    OmegaConf.update(config, "student_prompt", prompt_name, force_add=True)
    OmegaConf.update(config, "teacher_prompt", prompt_name, force_add=True)
    _configure_native_teacher_prompt(config, teacher_family)
    _configure_run_metadata(config)
    return student_family


def configure_r2opl_batch(config) -> int:
    """Keep a complete group on each training update and disallow SP packing."""

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.tensor_model_parallel_size) != 1:
        raise ValueError("R²OPL Student rollout tensor parallelism must be 1.")
    if int(rollout.data_parallel_size) != student_gpus:
        raise ValueError("R²OPL rollout data_parallel_size must equal Student GPU count.")
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("R²OPL packed probes currently require Ulysses sequence parallelism 1.")
    adjusted = math.ceil(int(config.data.train_batch_size) / student_gpus) * student_gpus
    OmegaConf.update(config, "data.train_batch_size", adjusted)
    OmegaConf.update(config, "actor_rollout_ref.actor.ppo_mini_batch_size", adjusted)
    return adjusted


def _validate_verl_r2opl_support() -> None:
    from verl.trainer.distillation import losses

    if not hasattr(losses, "compute_r2opl_sampled_token_loss"):
        raise RuntimeError(
            "The active VERL build does not provide R²OPL support; missing "
            "compute_r2opl_sampled_token_loss."
        )


def validate_r2opl_config(config) -> None:
    """Fail early when a configuration would make packed probes incorrect."""

    _validate_verl_r2opl_support()
    if str(config.algorithm.name) != R2OPL_VARIANT:
        raise ValueError(f"R²OPL requires algorithm.name={R2OPL_VARIANT!r}.")
    if str(config.trainer.project_name) != "R^2OPL":
        raise ValueError("All R²OPL runs must use trainer.project_name='R^2OPL'.")
    if str(config.get("group_name", "")).strip() != R2OPL_WANDB_GROUP:
        raise ValueError(f"R²OPL requires group_name={R2OPL_WANDB_GROUP!r}.")
    if not bool(config.distillation.enabled):
        raise ValueError("R²OPL requires distillation.enabled=true for failed rollouts.")
    prompt_name = str(config.get("prompt_template", "")).strip()
    if prompt_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt_template={prompt_name!r}.")
    if normalize_model_family(config.student_model_family) != normalize_model_family(config.teacher_model_family):
        raise ValueError("R²OPL Student and Teacher tokenizer families must match.")
    r2opl_hyperparameters(config)

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != R2OPL_LOSS_MODE:
        raise ValueError(f"R²OPL requires loss_mode={R2OPL_LOSS_MODE!r}.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("R²OPL requires policy_loss_mode=reinforce.")
    if not bool(loss.use_policy_gradient) or bool(loss.use_task_rewards):
        raise ValueError("R²OPL requires use_policy_gradient=true and use_task_rewards=false.")
    if loss.topk is not None:
        raise ValueError("R²OPL uses sampled-token OPD and requires topk=null.")
    if float(loss.selection_ratio) != 1.0:
        raise ValueError("R²OPL must score every valid response token.")
    if str(config.actor_rollout_ref.actor.loss_agg_mode) != "seq-mean-token-sum":
        raise ValueError("R²OPL requires loss_agg_mode=seq-mean-token-sum.")
    if loss.loss_max_clamp is not None:
        raise ValueError("R²OPL requires loss_max_clamp=null to preserve its paper objective.")

    model = config.actor_rollout_ref.model
    if bool(model.use_remove_padding):
        raise ValueError("R²OPL packed probes require model.use_remove_padding=false.")
    if bool(model.use_fused_kernels):
        raise ValueError("R²OPL packed probes require model.use_fused_kernels=false.")
    override = model.get("override_config", {})
    if str(override.get("attn_implementation", "")) != "sdpa":
        raise ValueError("R²OPL packed probes require override_config.attn_implementation=sdpa.")
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("R²OPL packed probes require ulysses_sequence_parallel_size=1.")
    # The packed answer-probe representation needs a dense [L, L] visibility
    # mask.  A micro-batch of more than one rollout would multiply that
    # quadratic allocation after padding, and dynamic batching can recreate
    # such a micro-batch even when the configured fixed size is one.  Keep the
    # allocation contract explicit rather than treating the ordinary PPO token
    # budget as a reliable dense-mask memory guard.
    if int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu) != 1:
        raise ValueError(
            "R²OPL packed probes require actor.ppo_micro_batch_size_per_gpu=1."
        )
    if bool(config.actor_rollout_ref.actor.use_dynamic_bsz):
        raise ValueError(
            "R²OPL packed probes require actor.use_dynamic_bsz=false so each "
            "dense-mask forward contains exactly one rollout."
        )
    if int(config.actor_rollout_ref.rollout.n) < 2:
        raise ValueError("R²OPL requires at least two rollouts per prompt.")
    if int(config.data.train_batch_size) <= 0:
        raise ValueError("R²OPL requires a positive number of training questions per update.")
    if min(
        int(config.data.max_prompt_length),
        int(config.rlvr_generation.train_max_new_tokens),
        int(config.rlvr_generation.val_max_new_tokens),
    ) <= 0:
        raise ValueError("R²OPL prompt and response limits must be positive.")
    if int(config.data.max_response_length) < max(
        int(config.rlvr_generation.train_max_new_tokens),
        int(config.rlvr_generation.val_max_new_tokens),
    ):
        raise ValueError("data.max_response_length does not cover generation limits.")

    max_prompt = int(config.data.max_prompt_length)
    train_response = int(config.rlvr_generation.train_max_new_tokens)
    val_response = int(config.rlvr_generation.val_max_new_tokens)
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.max_model_len) < max_prompt + val_response:
        raise ValueError("R²OPL Student max_model_len does not cover validation context.")
    # Teacher sampled-token scoring needs the causal predictor for every
    # generated response token, including a full-length rollout's final token.
    required_train_tokens = max_prompt + train_response + 1
    token_limits = (
        config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
        rollout.log_prob_max_token_len_per_gpu,
        config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu,
    )
    if any(int(limit) < required_train_tokens for limit in token_limits):
        raise ValueError("R²OPL Student log-prob token limits do not cover training context.")

    inference = config.distillation.teacher_models.teacher_model.inference
    if str(inference.name) != "vllm":
        raise ValueError("R²OPL sampled-token Teacher scoring requires vLLM.")
    if int(inference.prompt_length) < int(config.data.max_prompt_length):
        raise ValueError("R²OPL Teacher prompt_length is too small.")
    if int(inference.response_length) < int(config.rlvr_generation.train_max_new_tokens):
        raise ValueError("R²OPL Teacher response_length is too small.")
    if int(inference.max_model_len) < required_train_tokens:
        raise ValueError(
            "R²OPL Teacher max_model_len must cover the Student prompt, the "
            "full sampled response, and one causal predictor token."
        )
    max_logprobs = inference.engine_kwargs.get("vllm", {}).get("max_logprobs")
    if max_logprobs is None or int(max_logprobs) < 1:
        raise ValueError("R²OPL sampled-token Teacher scoring requires vLLM max_logprobs >= 1.")


class R2OPLTrainer(R2OPLBaseTrainer):
    """R²OPL trainer: group routing in the controller, one actor backward."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config") if kwargs else None
        if config is None and args:
            config = args[0]
        validate_r2opl_config(config)
        # Do not call R2OPLBaseTrainer.__init__: it intentionally validates the
        # different baseline algorithm.  Its non-constructor helper methods
        # (validation charts and complete-group checking) remain reusable.
        verl_sync.PPOTrainer.__init__(self, *args, **kwargs)

    def _compute_old_log_prob(self, batch, metrics):
        batch = verl_sync.PPOTrainer._compute_old_log_prob(self, batch, metrics)
        fields = ["uid", "response_mask", "rm_scores"]
        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        prompt_group_ids = [str(value) for value in data["uid"]]
        self._validate_complete_rollout_batch(batch, prompt_group_ids)
        response_mask_nested = data["response_mask"]
        if not response_mask_nested.is_nested:
            raise RuntimeError("R²OPL controller expects jagged response masks from TransferQueue.")
        response_mask = response_mask_nested.to_padded_tensor(False).bool()
        verifier_rewards = data["rm_scores"]
        if verifier_rewards.is_nested:
            verifier_rewards = verifier_rewards.to_padded_tensor(0.0)
        genuine_mask = torch.tensor(
            [not bool(tag.get("is_padding", False)) for tag in batch.tags],
            dtype=torch.bool,
            device=response_mask.device,
        )
        result = compute_r2opl_controller_batch(
            response_mask=response_mask,
            verifier_rewards=verifier_rewards,
            prompt_group_ids=prompt_group_ids,
            genuine_trajectory_mask=genuine_mask,
        )
        hparams = r2opl_hyperparameters(self.config)
        count = len(batch)
        scalar = lambda value: torch.full(
            (count,), float(value), dtype=torch.float32, device=response_mask.device
        )
        output = TensorDict(
            {
                "r2opl_correctness": result.correctness.to(torch.float32),
                "r2opl_difficulty": result.difficulty,
                "r2opl_lambda": scalar(hparams.lambda_),
                "r2opl_miu": scalar(hparams.miu),
                "r2opl_alpha_r": scalar(hparams.alpha_r),
                "r2opl_alpha_d": scalar(hparams.alpha_d),
                "r2opl_epsilon": scalar(hparams.epsilon),
            },
            batch_size=count,
        )
        batch = verl_sync.tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=output
        )
        metrics.update(result.metrics)
        metrics.update(
            {
                "r2opl/train/lambda": hparams.lambda_,
                "r2opl/train/miu": hparams.miu,
                "r2opl/train/alpha_r": hparams.alpha_r,
                "r2opl/train/alpha_d": hparams.alpha_d,
                "r2opl/train/epsilon": hparams.epsilon,
            }
        )
        return batch

    def _update_actor(self, batch, metrics):
        # This flag selects the custom 4-D-mask forward only during the actual
        # optimisation call.  Old-policy logprob computation stays standard.
        batch.extra_info.update(
            {
                "r2opl_enable_probe": True,
                "use_remove_padding": False,
                "use_fused_kernels": False,
            }
        )
        return super()._update_actor(batch, metrics)

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        verl_sync.PPOTrainer._compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch)
        aliases = {
            "actor/entropy": "r2opl/train/entropy",
            "actor/grad_norm": "r2opl/train/total_grad_norm",
            "response_length/mean": "r2opl/train/response_length_mean",
            "response_length/max": "r2opl/train/response_length_max",
            "perf/throughput": "r2opl/perf/tokens_per_second_per_gpu",
            "perf/time_per_step": "r2opl/perf/time_per_step",
            "actor/distillation/loss": "r2opl/train/policy_loss",
        }
        for source, destination in aliases.items():
            if source in metrics:
                metrics[destination] = metrics[source]
        # Actor loss metrics are returned under actor/r2opl/... by the normal
        # worker reducer; expose stable controller-level names as well.
        for source, value in list(metrics.items()):
            if source.startswith("actor/r2opl/"):
                metrics[source[len("actor/") :]] = value

        # The loss emits outcome-specific sufficient statistics as SUM metrics.
        # Materialize means only after the actor reducer has combined every
        # micro-batch and DP rank.  This makes, for example, a ten-step error
        # trajectory contribute ten delta values rather than one rollout-level
        # average, and represents a missing outcome honestly as NaN instead of
        # a misleading zero.
        for sum_name, total in list(metrics.items()):
            if not sum_name.startswith("r2opl/train/") or not sum_name.endswith("_sum"):
                continue
            stem = sum_name[: -len("_sum")]
            count_name = f"{stem}_count"
            if count_name not in metrics:
                continue
            try:
                denominator = float(metrics[count_name])
                numerator = float(total)
            except (TypeError, ValueError):
                continue
            metrics[f"{stem}_mean"] = (
                numerator / denominator if denominator > 0.0 else math.nan
            )


__all__ = [
    "R2OPLControllerBatchResult",
    "R2OPLHyperparameters",
    "R2OPLTrainer",
    "R2OPL_DEFAULT_ALPHA_D",
    "R2OPL_DEFAULT_ALPHA_R",
    "R2OPL_DEFAULT_EPSILON",
    "R2OPL_DEFAULT_LAMBDA",
    "R2OPL_DEFAULT_MIU",
    "R2OPL_LOSS_MODE",
    "R2OPL_VARIANT",
    "R2OPL_WANDB_GROUP",
    "compute_r2opl_controller_batch",
    "configure_r2opl_batch",
    "configure_r2opl_defaults",
    "r2opl_hyperparameters",
    "validate_r2opl_config",
]
