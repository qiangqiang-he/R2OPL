# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False
    use_full_vocab_teacher_entropy: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni" | "fsdp2":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            if distillation_config.distillation_loss.loss_mode == "eopd":
                distillation_loss_fn = fsdp_losses.compute_eopd_forward_kl_topk
            else:
                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    kwargs = {
        "student_logits": student_logits,
        "teacher_topk_log_probs": data["teacher_logprobs"],
        "teacher_topk_ids": data["teacher_ids"],
        "config": distillation_config,
        "data_format": data_format,
    }
    if distillation_config.distillation_loss.loss_mode == "eopd":
        kwargs["teacher_entropy"] = data["teacher_entropy"]
    outputs = distillation_loss_fn(**kwargs)

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if not distillation_loss_config.use_task_rewards and not distillation_loss_config.use_policy_gradient:
        # no need to compute policy loss
        policy_loss = 0.0
        policy_metrics = {}
    else:
        policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
        if not distillation_loss_config.use_task_rewards:
            policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode
    # The direct OPD path does not call ``ppo_loss`` (task rewards are
    # disabled), so the actor-level global reducer metadata is not populated
    # by that helper.  Copy the same metadata from the engine input here so
    # sampled-token policy losses, including R²OPL-base, preserve the global
    # batch/sequence-mean normalization under FSDP.
    for key in ("dp_size", "batch_num_tokens", "global_batch_size"):
        value = tu.get_non_tensor_data(data, key, default=None)
        if value is not None:
            config.global_batch_info[key] = value
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    eopd_forward_kl_losses = None
    if loss_config.loss_mode == "eopd":
        eopd_forward_kl_losses = no_padding_2_padding(
            model_output["distillation_losses"], data
        )

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.loss_mode == "error_reverse_kl":
        loss_normalization = data.get("error_opd_loss_normalization", None)
        if loss_normalization is None:
            raise RuntimeError(
                "Error-OPD actor input is missing error_opd_loss_normalization."
            )
        loss_normalization = tu.unwrap_non_tensor_data(loss_normalization)
        if not torch.is_tensor(loss_normalization):
            loss_normalization = torch.tensor(
                float(loss_normalization),
                dtype=distillation_losses.dtype,
                device=distillation_losses.device,
            )
        loss_normalization = loss_normalization.to(
            dtype=distillation_losses.dtype,
            device=distillation_losses.device,
        ).reshape(-1)
        if loss_normalization.numel() == 1:
            loss_normalization = loss_normalization.expand(
                distillation_losses.shape[0]
            )
        elif loss_normalization.numel() != distillation_losses.shape[0]:
            raise ValueError(
                "Error-OPD loss normalization must be scalar or have one value "
                f"per trajectory; got {loss_normalization.numel()} values for "
                f"{distillation_losses.shape[0]} trajectories."
            )
        if (
            not bool(torch.isfinite(loss_normalization).all())
            or bool(loss_normalization.lt(0.0).any())
        ):
            raise ValueError(
                "Error-OPD loss normalization values must be finite and non-negative."
            )
        # The clamp above remains identical to PG-OPD.  Scaling afterwards
        # changes only the reducer denominator and not the per-token signal.
        distillation_losses = (
            distillation_losses * loss_normalization.unsqueeze(-1)
        )

    if loss_config.loss_mode == "correct_reverse_kl":
        loss_normalization = data.get("correct_opd_loss_normalization", None)
        if loss_normalization is None:
            raise RuntimeError(
                "Correct-OPD actor input is missing correct_opd_loss_normalization."
            )
        loss_normalization = tu.unwrap_non_tensor_data(loss_normalization)
        if not torch.is_tensor(loss_normalization):
            loss_normalization = torch.tensor(
                float(loss_normalization),
                dtype=distillation_losses.dtype,
                device=distillation_losses.device,
            )
        loss_normalization = loss_normalization.to(
            dtype=distillation_losses.dtype,
            device=distillation_losses.device,
        ).reshape(-1)
        if loss_normalization.numel() == 1:
            loss_normalization = loss_normalization.expand(
                distillation_losses.shape[0]
            )
        elif loss_normalization.numel() != distillation_losses.shape[0]:
            raise ValueError(
                "Correct-OPD loss normalization must be scalar or have one value "
                f"per trajectory; got {loss_normalization.numel()} values for "
                f"{distillation_losses.shape[0]} trajectories."
            )
        if (
            not bool(torch.isfinite(loss_normalization).all())
            or bool(loss_normalization.lt(0.0).any())
        ):
            raise ValueError(
                "Correct-OPD loss normalization values must be finite and non-negative."
            )
        # Scaling after the same clamp as Error-OPD changes only the reducer
        # denominator, making the physical mixed batch equivalent to deleting
        # verifier-incorrect trajectories before the loss.
        distillation_losses = (
            distillation_losses * loss_normalization.unsqueeze(-1)
        )

    if loss_config.loss_mode == "fire_opd":
        loss_normalization = data.get("fire_opd_loss_normalization", None)
        if loss_normalization is None:
            raise RuntimeError(
                "FiRe-OPD actor input is missing fire_opd_loss_normalization."
            )
        loss_normalization = tu.unwrap_non_tensor_data(loss_normalization)
        if not torch.is_tensor(loss_normalization):
            loss_normalization = torch.tensor(
                float(loss_normalization),
                dtype=distillation_losses.dtype,
                device=distillation_losses.device,
            )
        loss_normalization = loss_normalization.to(
            dtype=distillation_losses.dtype,
            device=distillation_losses.device,
        ).reshape(-1)
        if loss_normalization.numel() == 1:
            loss_normalization = loss_normalization.expand(
                distillation_losses.shape[0]
            )
        elif loss_normalization.numel() != distillation_losses.shape[0]:
            raise ValueError(
                "FiRe-OPD loss normalization must be scalar or have one value "
                f"per trajectory; got {loss_normalization.numel()} values for "
                f"{distillation_losses.shape[0]} trajectories."
            )
        if (
            not bool(torch.isfinite(loss_normalization).all())
            or bool(loss_normalization.lt(1.0).any())
        ):
            raise ValueError(
                "FiRe-OPD loss normalization values must be finite and at "
                "least one."
            )
        # Filtered trajectories stay in the physical actor batch with zero
        # advantage so a size-one micro-batch is never all-masked.  Correct
        # the shared reducer's unchanged global denominator to make this
        # exactly equivalent to deleting those trajectories before the loss.
        distillation_losses = (
            distillation_losses * loss_normalization.unsqueeze(-1)
        )
    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    if eopd_forward_kl_losses is not None:
        forward_kl_loss = agg_loss(
            loss_mat=eopd_forward_kl_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        distillation_loss = distillation_loss + loss_config.eopd_alpha * forward_kl_loss
        distillation_metrics["distillation/eopd_forward_kl_loss"] = Metric(
            AggregationType.SUM, forward_kl_loss
        )
        distillation_metrics["distillation/eopd_scaled_forward_kl_loss"] = Metric(
            AggregationType.SUM, loss_config.eopd_alpha * forward_kl_loss
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["uni_opd"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_uni_opd_trajectory_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``-tilde_G`` for Uni-OPD's detached REINFORCE path.

    The complete-batch trajectory reduction, correctness balancing, group
    margin shift, and token broadcast are performed by ``UniOPDTrainer`` before
    the actor batch is split across workers.  Recomputing any of them here
    would incorrectly use only the current micro-batch.
    """

    del config, distillation_config, model_output
    if "uni_opd_advantages" not in data:
        raise RuntimeError(
            "Uni-OPD actor input is missing controller-computed "
            "uni_opd_advantages."
        )

    advantage = data["uni_opd_advantages"]
    if advantage.is_nested:
        advantage = advantage.to_padded_tensor(0.0)
    advantage = advantage.detach()
    if data["response_mask"].is_nested:
        response_mask = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask = data["response_mask"].bool()
    if advantage.shape != response_mask.shape:
        raise ValueError(
            "Uni-OPD broadcast advantage and response mask must have identical "
            f"shapes; got {advantage.shape} and {response_mask.shape}."
        )
    loss_normalization = data.get("uni_opd_loss_normalization", None)
    if loss_normalization is None:
        raise RuntimeError(
            "Uni-OPD actor input is missing uni_opd_loss_normalization."
        )
    loss_normalization = tu.unwrap_non_tensor_data(loss_normalization)
    if not torch.is_tensor(loss_normalization):
        loss_normalization = torch.tensor(
            float(loss_normalization),
            dtype=advantage.dtype,
            device=advantage.device,
        )
    loss_normalization = loss_normalization.to(
        dtype=advantage.dtype, device=advantage.device
    ).reshape(-1)
    if loss_normalization.numel() == 1:
        loss_normalization = loss_normalization.expand(advantage.shape[0])
    elif loss_normalization.numel() != advantage.shape[0]:
        raise ValueError(
            "Uni-OPD loss normalization must be scalar or have one value per "
            f"trajectory; got {loss_normalization.numel()} values for "
            f"{advantage.shape[0]} trajectories."
        )
    if (
        not bool(torch.isfinite(loss_normalization).all())
        or bool(loss_normalization.lt(1.0).any())
    ):
        raise ValueError(
            "Uni-OPD loss normalization values must be finite and at least one."
        )
    # distillation_loss() negates and detaches this tensor before REINFORCE,
    # recovering the controller-computed calibrated return.  The global scale
    # changes only the denominator convention: averaging over all 256 physical
    # rows becomes averaging over the correctness-balanced subset.
    return -(advantage * loss_normalization.unsqueeze(-1)), {}


@register_distillation_loss(
    DistillationLossSettings(
        names=["fire_opd"],
        use_estimator=True,
        use_full_vocab_teacher_entropy=True,
    )
)  # type: ignore[arg-type]
def compute_fire_opd_trajectory_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the controller-computed ``-A_FiRe`` to the shared PG path."""

    del config, distillation_config, model_output
    if "fire_opd_advantages" not in data:
        raise RuntimeError(
            "FiRe-OPD actor input is missing controller-computed "
            "fire_opd_advantages."
        )
    advantage = data["fire_opd_advantages"]
    if advantage.is_nested:
        advantage = advantage.to_padded_tensor(0.0)
    advantage = advantage.detach()
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.bool().to_padded_tensor(False)
    else:
        response_mask = response_mask.bool()
    if advantage.shape != response_mask.shape:
        raise ValueError(
            "FiRe-OPD advantage and response mask must have identical shapes; "
            f"got {advantage.shape} and {response_mask.shape}."
        )
    # distillation_loss() negates and detaches this tensor before invoking the
    # configured OPD policy-loss implementation, recovering A_FiRe.
    return -advantage, {}


@register_distillation_loss(
    DistillationLossSettings(names=["r2opl_base", "r2opl_base_v2"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_r2opl_base_sampled_token_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the detached hybrid R²OPL-base policy-gradient signal.

    The controller materializes ``r2opl_base_correct_mask`` and the
    prompt-level ``r2opl_base_difficulty`` for the complete rollout batch.
    The Student log-probability is intentionally taken from this actor
    forward, rather than from the controller's pre-update pass, so the OPD
    branch remains aligned with the current policy during a PPO epoch.

    The shared distillation path negates the returned tensor before invoking
    the ``reinforce`` policy loss; returning ``-advantage`` therefore recovers
    the objective ``-A * log pi``.
    """

    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(
        data["teacher_logprobs"], data
    ).squeeze(-1)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.bool().to_padded_tensor(False)
    else:
        response_mask = response_mask.bool()

    correct_mask = data.get("r2opl_base_correct_mask", None)
    error_mask = data.get("r2opl_base_error_mask", None)
    difficulty = data.get("r2opl_base_difficulty", None)
    if correct_mask is None or error_mask is None or difficulty is None:
        raise RuntimeError(
            "R²OPL-base actor input is missing controller-computed masks or difficulty."
        )
    if correct_mask.is_nested:
        correct_mask = correct_mask.bool().to_padded_tensor(False)
    else:
        correct_mask = correct_mask.bool()
    if error_mask.is_nested:
        error_mask = error_mask.bool().to_padded_tensor(False)
    else:
        error_mask = error_mask.bool()
    if difficulty.is_nested:
        difficulty = difficulty.to_padded_tensor(0.0)
    else:
        difficulty = difficulty.float()

    expected_shape = response_mask.shape
    if not (
        student_log_probs.shape
        == teacher_log_probs.shape
        == correct_mask.shape
        == error_mask.shape
        == difficulty.shape
        == expected_shape
    ):
        raise ValueError(
            "R²OPL-base actor token tensors must have identical shapes; got "
            f"student={tuple(student_log_probs.shape)}, "
            f"teacher={tuple(teacher_log_probs.shape)}, "
            f"response_mask={tuple(response_mask.shape)}, "
            f"correct_mask={tuple(correct_mask.shape)}, "
            f"error_mask={tuple(error_mask.shape)}, "
            f"difficulty={tuple(difficulty.shape)}."
        )
    if bool((correct_mask & error_mask).any()):
        raise ValueError("R²OPL-base correct and error token masks overlap.")
    if bool(((correct_mask | error_mask) & ~response_mask).any()):
        raise ValueError("R²OPL-base branch masks must be subsets of response_mask.")
    if not bool(torch.isfinite(difficulty[response_mask]).all()):
        raise ValueError("R²OPL-base prompt difficulties must be finite.")

    opd_advantage = teacher_log_probs.float() - student_log_probs.float()
    loss_config = distillation_config.distillation_loss
    lambda_field = data.get("r2opl_base_lambda", None)
    if lambda_field is None:
        lambda_value = float(getattr(loss_config, "r2opl_lambda", 1.0 / 20.0))
        if not torch.isfinite(torch.tensor(lambda_value)) or lambda_value < 0.0:
            raise ValueError(
                f"R²OPL-base lambda must be finite and non-negative; got {lambda_value}."
            )
        lambda_tensor = torch.tensor(
            lambda_value, dtype=opd_advantage.dtype, device=opd_advantage.device
        )
    else:
        # TransferQueue normally preserves this as a tensor, but unwrap the
        # metadata form too so scalar test fixtures and alternate dispatch
        # paths cannot change the mathematical signal silently.
        lambda_field = tu.unwrap_non_tensor_data(lambda_field)
        if not torch.is_tensor(lambda_field):
            lambda_field = torch.tensor(
                float(lambda_field),
                dtype=opd_advantage.dtype,
                device=opd_advantage.device,
            )
        if lambda_field.is_nested:
            lambda_field = lambda_field.to_padded_tensor(0.0)
        lambda_tensor = lambda_field.to(
            dtype=opd_advantage.dtype, device=opd_advantage.device
        ).reshape(-1)
        if lambda_tensor.numel() == 1:
            lambda_tensor = lambda_tensor.reshape(1, 1)
        elif lambda_tensor.numel() == opd_advantage.shape[0]:
            lambda_tensor = lambda_tensor.reshape(-1, 1)
        else:
            raise ValueError(
                "R²OPL-base lambda must be scalar or have one value per trajectory; "
                f"got {lambda_tensor.numel()} values for batch size {opd_advantage.shape[0]}."
            )
        if not bool(torch.isfinite(lambda_tensor).all()) or bool(lambda_tensor.lt(0.0).any()):
            raise ValueError("R²OPL-base lambda values must be finite and non-negative.")

    miu_field = data.get("r2opl_base_miu", None)
    if miu_field is None:
        # R²OPL-base v1 never transports a miu field; its correct-branch
        # advantage is exactly the fixed ``1``.
        miu_tensor = torch.tensor(
            1.0, dtype=opd_advantage.dtype, device=opd_advantage.device
        )
    else:
        miu_field = tu.unwrap_non_tensor_data(miu_field)
        if not torch.is_tensor(miu_field):
            miu_field = torch.tensor(
                float(miu_field),
                dtype=opd_advantage.dtype,
                device=opd_advantage.device,
            )
        if miu_field.is_nested:
            miu_field = miu_field.to_padded_tensor(1.0)
        miu_tensor = miu_field.to(
            dtype=opd_advantage.dtype, device=opd_advantage.device
        ).reshape(-1)
        if miu_tensor.numel() == 1:
            miu_tensor = miu_tensor.reshape(1, 1)
        elif miu_tensor.numel() == opd_advantage.shape[0]:
            miu_tensor = miu_tensor.reshape(-1, 1)
        else:
            raise ValueError(
                "R²OPL-base miu must be scalar or have one value per trajectory; "
                f"got {miu_tensor.numel()} values for batch size {opd_advantage.shape[0]}."
            )
        if not bool(torch.isfinite(miu_tensor).all()) or bool(miu_tensor.le(0.0).any()):
            raise ValueError("R²OPL-base miu values must be finite and positive.")

    correct_advantage = difficulty.float() * miu_tensor
    error_advantage = difficulty.float() * lambda_tensor * opd_advantage
    branch = str(
        tu.get_non_tensor_data(data, "r2opl_base_gradient_branch", default="total")
    )
    if branch == "correct":
        advantage = correct_advantage * correct_mask.to(correct_advantage.dtype)
        metric_mask = correct_mask
    elif branch == "error":
        advantage = error_advantage * error_mask.to(error_advantage.dtype)
        metric_mask = error_mask
    elif branch == "total":
        advantage = torch.where(correct_mask, correct_advantage, error_advantage)
        advantage = advantage * response_mask.to(advantage.dtype)
        metric_mask = response_mask
    else:
        raise ValueError(
            "R²OPL-base gradient branch must be one of 'correct', 'error', or "
            f"'total'; got {branch!r}."
        )
    advantage = advantage * response_mask.to(advantage.dtype)
    distillation_losses = -advantage

    error_valid = error_mask & response_mask
    zero = opd_advantage.float().sum() * 0.0

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask].float()
        return selected.mean() if selected.numel() else zero

    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN, masked_mean(opd_advantage, error_valid)
        ),
        "distillation/opd_advantage_mean": Metric(
            AggregationType.MEAN, masked_mean(opd_advantage, error_valid)
        ),
        "distillation/opd_advantage_abs_mean": Metric(
            AggregationType.MEAN, masked_mean(opd_advantage.abs(), error_valid)
        ),
        "distillation/applied_advantage_mean": Metric(
            AggregationType.MEAN, masked_mean(advantage, metric_mask)
        ),
    }
    return distillation_losses, metrics


def compute_opd_outcome_statistics(
    advantage: torch.Tensor,
    response_mask: torch.Tensor,
    rm_scores: torch.Tensor,
    statistics_threshold: float = 1.0e-4,
) -> dict[str, Metric]:
    """Return sufficient statistics for standard OPD signal by outcome.

    Ratios are intentionally deferred until after micro-batch and data-parallel
    aggregation. This keeps variable-length trajectories weighted per token.
    """

    if advantage.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("OPD outcome statistics expect 2-D advantage and response mask tensors.")
    if advantage.shape != response_mask.shape:
        raise ValueError("OPD outcome advantage and response mask must have identical shapes.")
    if statistics_threshold < 0:
        raise ValueError(f"statistics_threshold must be non-negative, got {statistics_threshold}.")

    reward_tensor = rm_scores.to_padded_tensor(0.0) if rm_scores.is_nested else rm_scores
    if reward_tensor.ndim != 2 or reward_tensor.shape[0] != advantage.shape[0]:
        raise ValueError("OPD outcome rm_scores must be a 2-D tensor with matching batch size.")

    values = advantage.float()
    valid = response_mask.bool()
    rollout_correct = reward_tensor.float().sum(dim=-1) > 0.5
    correct = valid & rollout_correct.unsqueeze(-1)
    wrong = valid & ~rollout_correct.unsqueeze(-1)
    positive = values > float(statistics_threshold)
    negative = values < -float(statistics_threshold)
    positive_mass = values.clamp_min(0.0)
    negative_mass = (-values).clamp_min(0.0)

    prefix = "distillation/opd_outcome_stats/"

    def sum_metric(value: torch.Tensor) -> Metric:
        return Metric(AggregationType.SUM, value)

    return {
        f"{prefix}correct_token_count": sum_metric(correct.float().sum()),
        f"{prefix}wrong_token_count": sum_metric(wrong.float().sum()),
        f"{prefix}correct_advantage_sum": sum_metric(values[correct].sum()),
        f"{prefix}wrong_advantage_sum": sum_metric(values[wrong].sum()),
        f"{prefix}correct_abs_advantage_sum": sum_metric(values[correct].abs().sum()),
        f"{prefix}wrong_abs_advantage_sum": sum_metric(values[wrong].abs().sum()),
        f"{prefix}correct_positive_token_count": sum_metric((correct & positive).float().sum()),
        f"{prefix}wrong_positive_token_count": sum_metric((wrong & positive).float().sum()),
        f"{prefix}correct_negative_token_count": sum_metric((correct & negative).float().sum()),
        f"{prefix}wrong_negative_token_count": sum_metric((wrong & negative).float().sum()),
        f"{prefix}correct_positive_mass_sum": sum_metric(positive_mass[correct].sum()),
        f"{prefix}wrong_positive_mass_sum": sum_metric(positive_mass[wrong].sum()),
        f"{prefix}correct_negative_mass_sum": sum_metric(negative_mass[correct].sum()),
        f"{prefix}wrong_negative_mass_sum": sum_metric(negative_mass[wrong].sum()),
    }


def select_opd_reverse_kl_tokens(
    reverse_kl: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    selection_ratio: float,
    selection_method: str,
) -> torch.Tensor:
    """Return the valid tokens selected for a reverse-KL OPD update."""

    ratio = float(selection_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"selection_ratio must lie in [0, 1], got {ratio}.")

    valid = response_mask.bool()
    if ratio == 1.0:
        # Do not inspect the method: full-token training has no selection step.
        return valid.clone()

    method = str(selection_method)
    if method not in {"random", "topgap", "bottomgap"}:
        raise ValueError(
            "selection_method must be one of random, topgap, bottomgap; "
            f"got {method!r}."
        )
    if ratio == 0.0:
        return torch.zeros_like(valid)
    if method == "random":
        return (torch.rand_like(reverse_kl, dtype=torch.float32) < ratio) & valid

    selected = torch.zeros_like(valid)
    largest = method == "topgap"
    gap = reverse_kl.abs()
    for row in range(valid.shape[0]):
        valid_indices = valid[row].nonzero(as_tuple=False).squeeze(-1)
        valid_count = int(valid_indices.numel())
        if valid_count == 0:
            continue
        selected_count = min(valid_count, max(1, math.ceil(valid_count * ratio)))
        ranked_indices = torch.topk(
            gap[row, valid_indices],
            k=selected_count,
            largest=largest,
            sorted=False,
        ).indices
        selected[row, valid_indices[ranked_indices]] = True
    return selected


@register_distillation_loss(DistillationLossSettings(names=["reverse_kl"], use_estimator=True))  # type: ignore[arg-type]
def compute_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate KL(student || teacher) on tokens sampled by the student.

    The returned per-token value is ``log p_student(a) - log p_teacher(a)``.
    GKD-OPD and PG-OPD consume it as a detached policy-gradient advantage.
    ``selection_ratio`` and ``selection_method`` optionally retain a token
    subset without changing the sign of the selected reverse-KL signals. No
    teacher or student top-k distribution is materialized.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config = distillation_config.distillation_loss
    sampled_reverse_kl = student_log_probs - teacher_log_probs
    selected_mask = select_opd_reverse_kl_tokens(
        sampled_reverse_kl,
        response_mask_bool,
        selection_ratio=float(loss_config.selection_ratio),
        selection_method=str(loss_config.selection_method),
    )
    selected_reverse_kl = sampled_reverse_kl * selected_mask.to(
        sampled_reverse_kl.dtype
    )
    valid_student_log_probs = student_log_probs[response_mask_bool].float()
    valid_teacher_log_probs = teacher_log_probs[response_mask_bool].float()
    valid_gap = sampled_reverse_kl[response_mask_bool].float().abs()
    selected_gap = sampled_reverse_kl[selected_mask].float().abs()
    base_signal = sampled_reverse_kl
    if loss_config.loss_max_clamp is not None:
        base_signal = base_signal.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp,
        )
    valid_base_signal = base_signal[response_mask_bool].float()
    valid_selected_signal = (
        base_signal * selected_mask.to(base_signal.dtype)
    )[response_mask_bool].float()
    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN, sampled_reverse_kl[response_mask_bool].mean()
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, valid_student_log_probs.exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, valid_teacher_log_probs.exp().mean()
        ),
        "distillation/selected_token_ratio": Metric(
            AggregationType.MEAN,
            selected_mask[response_mask_bool].float().mean(),
        ),
        "distillation/selection_gap_mean": Metric(
            AggregationType.MEAN, valid_gap.mean()
        ),
        "distillation/selected_gap_mean": Metric(
            AggregationType.MEAN,
            selected_gap.mean() if selected_gap.numel() else valid_gap.new_zeros(()),
        ),
        "distillation/selection_gradient_signal_relative_change": Metric(
            AggregationType.MEAN,
            torch.linalg.vector_norm(valid_selected_signal - valid_base_signal)
            / torch.linalg.vector_norm(valid_base_signal).clamp_min(1e-12),
        ),
    }
    track_outcome_metrics = (
        bool(tu.get_non_tensor_data(data, "track_opd_outcome_metrics", default=False))
        if isinstance(data, TensorDict)
        else bool(data.get("track_opd_outcome_metrics", False))
    )
    if track_outcome_metrics and "rm_scores" in data:
        metrics.update(
            compute_opd_outcome_statistics(
                advantage=-selected_reverse_kl,
                response_mask=selected_mask,
                rm_scores=data["rm_scores"],
                statistics_threshold=float(
                    getattr(
                        distillation_config.distillation_loss,
                        "opd_statistics_threshold",
                        1.0e-4,
                    )
                ),
            )
        )
    return selected_reverse_kl, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["error_reverse_kl"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_error_only_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return PG-OPD reverse-KL only for verifier-incorrect rollouts."""

    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(
        data["teacher_logprobs"], data
    ).squeeze(-1)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.bool().to_padded_tensor(False)
    else:
        response_mask = response_mask.bool()
    error_token_mask = data.get("error_opd_token_mask", None)
    if error_token_mask is None:
        raise RuntimeError(
            "Error-OPD actor input is missing controller-computed "
            "error_opd_token_mask."
        )
    if error_token_mask.is_nested:
        error_token_mask = error_token_mask.bool().to_padded_tensor(False)
    else:
        error_token_mask = error_token_mask.bool()
    if not (
        teacher_log_probs.shape
        == student_log_probs.shape
        == response_mask.shape
        == error_token_mask.shape
    ):
        raise ValueError("Error-OPD token tensors must have identical response shapes.")
    if bool((error_token_mask & ~response_mask).any()):
        raise ValueError("Error-OPD token mask must be a subset of response_mask.")

    loss_config = distillation_config.distillation_loss
    sampled_reverse_kl = student_log_probs - teacher_log_probs
    selected_mask = select_opd_reverse_kl_tokens(
        sampled_reverse_kl,
        error_token_mask,
        selection_ratio=float(loss_config.selection_ratio),
        selection_method=str(loss_config.selection_method),
    )
    selected_reverse_kl = sampled_reverse_kl * selected_mask.to(
        sampled_reverse_kl.dtype
    )

    zero = sampled_reverse_kl.float().sum() * 0.0

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask].float()
        return selected.mean() if selected.numel() else zero

    valid_gap = sampled_reverse_kl.float().abs()
    base_signal = sampled_reverse_kl
    if loss_config.loss_max_clamp is not None:
        base_signal = base_signal.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp,
        )
    selected_signal = base_signal * selected_mask.to(base_signal.dtype)
    if bool(error_token_mask.any()):
        relative_change = (
            torch.linalg.vector_norm(
                selected_signal[error_token_mask].float()
                - base_signal[error_token_mask].float()
            )
            / torch.linalg.vector_norm(
                base_signal[error_token_mask].float()
            ).clamp_min(1e-12)
        )
        selected_ratio = selected_mask[error_token_mask].float().mean()
    else:
        relative_change = zero
        selected_ratio = zero

    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN,
            masked_mean(sampled_reverse_kl, error_token_mask),
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN,
            masked_mean(student_log_probs.float().exp(), error_token_mask),
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN,
            masked_mean(teacher_log_probs.float().exp(), error_token_mask),
        ),
        "distillation/selected_token_ratio": Metric(
            AggregationType.MEAN, selected_ratio
        ),
        "distillation/selection_gap_mean": Metric(
            AggregationType.MEAN, masked_mean(valid_gap, error_token_mask)
        ),
        "distillation/selected_gap_mean": Metric(
            AggregationType.MEAN, masked_mean(valid_gap, selected_mask)
        ),
        "distillation/selection_gradient_signal_relative_change": Metric(
            AggregationType.MEAN, relative_change
        ),
    }
    track_outcome_metrics = (
        bool(
            tu.get_non_tensor_data(
                data, "track_opd_outcome_metrics", default=False
            )
        )
        if isinstance(data, TensorDict)
        else bool(data.get("track_opd_outcome_metrics", False))
    )
    if track_outcome_metrics and "rm_scores" in data:
        metrics.update(
            compute_opd_outcome_statistics(
                advantage=-selected_reverse_kl,
                response_mask=selected_mask,
                rm_scores=data["rm_scores"],
                statistics_threshold=float(
                    getattr(loss_config, "opd_statistics_threshold", 1.0e-4)
                ),
            )
        )
    return selected_reverse_kl, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["correct_reverse_kl"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_correct_only_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return PG-OPD reverse-KL only for verifier-correct rollouts."""

    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(
        data["teacher_logprobs"], data
    ).squeeze(-1)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.bool().to_padded_tensor(False)
    else:
        response_mask = response_mask.bool()
    correct_token_mask = data.get("correct_opd_token_mask", None)
    if correct_token_mask is None:
        raise RuntimeError(
            "Correct-OPD actor input is missing controller-computed "
            "correct_opd_token_mask."
        )
    if correct_token_mask.is_nested:
        correct_token_mask = correct_token_mask.bool().to_padded_tensor(False)
    else:
        correct_token_mask = correct_token_mask.bool()
    if not (
        teacher_log_probs.shape
        == student_log_probs.shape
        == response_mask.shape
        == correct_token_mask.shape
    ):
        raise ValueError("Correct-OPD token tensors must have identical response shapes.")
    if bool((correct_token_mask & ~response_mask).any()):
        raise ValueError("Correct-OPD token mask must be a subset of response_mask.")

    loss_config = distillation_config.distillation_loss
    sampled_reverse_kl = student_log_probs - teacher_log_probs
    selected_mask = select_opd_reverse_kl_tokens(
        sampled_reverse_kl,
        correct_token_mask,
        selection_ratio=float(loss_config.selection_ratio),
        selection_method=str(loss_config.selection_method),
    )
    selected_reverse_kl = sampled_reverse_kl * selected_mask.to(
        sampled_reverse_kl.dtype
    )

    zero = sampled_reverse_kl.float().sum() * 0.0

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask].float()
        return selected.mean() if selected.numel() else zero

    valid_gap = sampled_reverse_kl.float().abs()
    base_signal = sampled_reverse_kl
    if loss_config.loss_max_clamp is not None:
        base_signal = base_signal.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp,
        )
    selected_signal = base_signal * selected_mask.to(base_signal.dtype)
    if bool(correct_token_mask.any()):
        relative_change = (
            torch.linalg.vector_norm(
                selected_signal[correct_token_mask].float()
                - base_signal[correct_token_mask].float()
            )
            / torch.linalg.vector_norm(
                base_signal[correct_token_mask].float()
            ).clamp_min(1e-12)
        )
        selected_ratio = selected_mask[correct_token_mask].float().mean()
    else:
        relative_change = zero
        selected_ratio = zero

    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN,
            masked_mean(sampled_reverse_kl, correct_token_mask),
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN,
            masked_mean(student_log_probs.float().exp(), correct_token_mask),
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN,
            masked_mean(teacher_log_probs.float().exp(), correct_token_mask),
        ),
        "distillation/selected_token_ratio": Metric(
            AggregationType.MEAN, selected_ratio
        ),
        "distillation/selection_gap_mean": Metric(
            AggregationType.MEAN, masked_mean(valid_gap, correct_token_mask)
        ),
        "distillation/selected_gap_mean": Metric(
            AggregationType.MEAN, masked_mean(valid_gap, selected_mask)
        ),
        "distillation/selection_gradient_signal_relative_change": Metric(
            AggregationType.MEAN, relative_change
        ),
    }
    track_outcome_metrics = (
        bool(
            tu.get_non_tensor_data(
                data, "track_opd_outcome_metrics", default=False
            )
        )
        if isinstance(data, TensorDict)
        else bool(data.get("track_opd_outcome_metrics", False))
    )
    if track_outcome_metrics and "rm_scores" in data:
        metrics.update(
            compute_opd_outcome_statistics(
                advantage=-selected_reverse_kl,
                response_mask=selected_mask,
                rm_scores=data["rm_scores"],
                statistics_threshold=float(
                    getattr(loss_config, "opd_statistics_threshold", 1.0e-4)
                ),
            )
        )
    return selected_reverse_kl, metrics


def compute_sol_opd_comparison_statistics(
    opd_advantage: torch.Tensor,
    sol_advantage: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> dict[str, Metric]:
    """Return aggregatable raw-advantage statistics for Sol-OPD.

    Token-pooled ratios are represented by separate sums/counts so they can be
    combined exactly across micro-batches and data-parallel ranks. Rollout
    statistics are reduced within each trajectory first and materialized as a
    sum plus a rollout count for the same reason. Headline magnitude rates use
    strict comparisons over every valid token. ``same_direction_*`` rates are
    conditional on both advantages having magnitude greater than ``epsilon``
    and having the same sign. The exhaustive ``category_*`` rates remain
    unconditional over all valid tokens.
    """

    if (
        opd_advantage.ndim != 2
        or sol_advantage.ndim != 2
        or response_mask.ndim != 2
    ):
        raise ValueError("Sol-OPD comparison tensors must all be two-dimensional.")
    if not (
        opd_advantage.shape == sol_advantage.shape == response_mask.shape
    ):
        raise ValueError("Sol-OPD comparison tensors must have identical shapes.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError(f"Sol-OPD epsilon must be finite and positive, got {epsilon}.")

    values_opd = opd_advantage.float()
    values_sol = sol_advantage.float()
    valid = response_mask.bool()
    if not bool(torch.isfinite(values_opd[valid]).all()) or not bool(
        torch.isfinite(values_sol[valid]).all()
    ):
        raise ValueError("Sol-OPD advantages must be finite on valid response tokens.")

    eps = float(epsilon)
    abs_opd = values_opd.abs()
    abs_sol = values_sol.abs()
    magnitude_delta = abs_sol - abs_opd

    # Keep the three headline magnitude rates faithful to their literal
    # definitions.  In particular, ``equal`` means exact equality here; the
    # epsilon dead zone belongs to sign reliability and to the separate
    # approximately-unchanged category below.
    amplified = valid & abs_sol.gt(abs_opd)
    reduced = valid & abs_sol.lt(abs_opd)
    equal = valid & abs_sol.eq(abs_opd)

    both_active = valid & abs_opd.gt(eps) & abs_sol.gt(eps)
    sign_flipped = both_active & values_opd.mul(values_sol).lt(0.0)
    same_direction_eligible = both_active & ~sign_flipped
    same_direction_amplified = same_direction_eligible & amplified
    same_direction_reduced = same_direction_eligible & reduced

    # These four masks are mutually exclusive and exhaustive. Near-zero sign
    # transitions are intentionally treated as non-flips. Tiny magnitude
    # changes (including those involving a near-zero advantage) go into the
    # approximately-unchanged bucket rather than the strict headline rates.
    category_sign_flipped = sign_flipped
    category_amplified = (
        valid & ~category_sign_flipped & magnitude_delta.gt(eps)
    )
    category_reduced = (
        valid & ~category_sign_flipped & magnitude_delta.lt(-eps)
    )
    category_unchanged = valid & ~(
        category_sign_flipped | category_amplified | category_reduced
    )

    token_counts = valid.sum(dim=-1)
    valid_rollouts = token_counts.gt(0)
    token_denominators = token_counts.clamp_min(1).float()
    same_direction_counts = same_direction_eligible.sum(dim=-1)
    same_direction_rollouts = same_direction_counts.gt(0)
    same_direction_denominators = same_direction_counts.clamp_min(1).float()

    def rollout_rate_sum(
        selection: torch.Tensor,
        *,
        denominators: torch.Tensor = token_denominators,
        included_rollouts: torch.Tensor = valid_rollouts,
    ) -> torch.Tensor:
        rates = selection.sum(dim=-1).float() / denominators
        return rates[included_rollouts].sum()

    opd_abs_by_rollout = (abs_opd * valid).sum(dim=-1)
    sol_abs_by_rollout = (abs_sol * valid).sum(dim=-1)
    deviation_abs_by_rollout = ((values_sol - values_opd).abs() * valid).sum(
        dim=-1
    )
    ratio_rollouts = valid_rollouts & opd_abs_by_rollout.gt(eps)

    def sum_metric(value: torch.Tensor) -> Metric:
        return Metric(AggregationType.SUM, value)

    stats_prefix = "distillation/sol_stats/"
    return {
        f"{stats_prefix}token_count": sum_metric(valid.float().sum()),
        f"{stats_prefix}rollout_count": sum_metric(valid_rollouts.float().sum()),
        f"{stats_prefix}ratio_rollout_count": sum_metric(
            ratio_rollouts.float().sum()
        ),
        f"{stats_prefix}same_direction_token_count": sum_metric(
            same_direction_eligible.float().sum()
        ),
        f"{stats_prefix}same_direction_rollout_count": sum_metric(
            same_direction_rollouts.float().sum()
        ),
        f"{stats_prefix}opd_advantage_sum": sum_metric(values_opd[valid].sum()),
        f"{stats_prefix}sol_advantage_sum": sum_metric(values_sol[valid].sum()),
        f"{stats_prefix}opd_abs_sum": sum_metric(abs_opd[valid].sum()),
        f"{stats_prefix}sol_abs_sum": sum_metric(abs_sol[valid].sum()),
        f"{stats_prefix}deviation_abs_sum": sum_metric(
            (values_sol - values_opd)[valid].abs().sum()
        ),
        f"{stats_prefix}amplified_count": sum_metric(amplified.float().sum()),
        f"{stats_prefix}reduced_count": sum_metric(reduced.float().sum()),
        f"{stats_prefix}equal_count": sum_metric(equal.float().sum()),
        f"{stats_prefix}same_direction_amplified_count": sum_metric(
            same_direction_amplified.float().sum()
        ),
        f"{stats_prefix}same_direction_reduced_count": sum_metric(
            same_direction_reduced.float().sum()
        ),
        f"{stats_prefix}sign_flipped_count": sum_metric(
            sign_flipped.float().sum()
        ),
        f"{stats_prefix}category_amplified_count": sum_metric(
            category_amplified.float().sum()
        ),
        f"{stats_prefix}category_reduced_count": sum_metric(
            category_reduced.float().sum()
        ),
        f"{stats_prefix}category_sign_flipped_count": sum_metric(
            category_sign_flipped.float().sum()
        ),
        f"{stats_prefix}category_unchanged_count": sum_metric(
            category_unchanged.float().sum()
        ),
        f"{stats_prefix}rollout_magnitude_ratio_sum": sum_metric(
            (sol_abs_by_rollout[ratio_rollouts] / opd_abs_by_rollout[ratio_rollouts]).sum()
        ),
        f"{stats_prefix}rollout_deviation_ratio_sum": sum_metric(
            (
                deviation_abs_by_rollout[ratio_rollouts]
                / opd_abs_by_rollout[ratio_rollouts]
            ).sum()
        ),
        f"{stats_prefix}rollout_amplification_rate_sum": sum_metric(
            rollout_rate_sum(amplified)
        ),
        f"{stats_prefix}rollout_reduction_rate_sum": sum_metric(
            rollout_rate_sum(reduced)
        ),
        f"{stats_prefix}rollout_equal_rate_sum": sum_metric(
            rollout_rate_sum(equal)
        ),
        f"{stats_prefix}rollout_same_direction_amplification_rate_sum": sum_metric(
            rollout_rate_sum(
                same_direction_amplified,
                denominators=same_direction_denominators,
                included_rollouts=same_direction_rollouts,
            )
        ),
        f"{stats_prefix}rollout_same_direction_reduction_rate_sum": sum_metric(
            rollout_rate_sum(
                same_direction_reduced,
                denominators=same_direction_denominators,
                included_rollouts=same_direction_rollouts,
            )
        ),
        f"{stats_prefix}rollout_sign_flip_rate_sum": sum_metric(
            rollout_rate_sum(sign_flipped)
        ),
        f"{stats_prefix}rollout_category_amplified_rate_sum": sum_metric(
            rollout_rate_sum(category_amplified)
        ),
        f"{stats_prefix}rollout_category_reduced_rate_sum": sum_metric(
            rollout_rate_sum(category_reduced)
        ),
        f"{stats_prefix}rollout_category_sign_flipped_rate_sum": sum_metric(
            rollout_rate_sum(category_sign_flipped)
        ),
        f"{stats_prefix}rollout_category_unchanged_rate_sum": sum_metric(
            rollout_rate_sum(category_unchanged)
        ),
    }


@register_distillation_loss(
    DistillationLossSettings(names=["sol_reverse_kl"], use_estimator=True)  # type: ignore[arg-type]
)
def compute_solution_privileged_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``log p_student - log p_solution_teacher`` for Sol-OPD."""

    del config
    if "sol_unprivileged_teacher_logprobs" not in data:
        raise RuntimeError(
            "Sol-OPD requires the statistics-only unprivileged Teacher forward; "
            "actor input is missing sol_unprivileged_teacher_logprobs."
        )
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    solution_teacher_log_probs = no_padding_2_padding(
        data["teacher_logprobs"], data
    ).squeeze(-1)
    unprivileged_teacher_log_probs = no_padding_2_padding(
        data["sol_unprivileged_teacher_logprobs"], data
    ).squeeze(-1)
    if data["response_mask"].is_nested:
        valid = data["response_mask"].bool().to_padded_tensor(False)
    else:
        valid = data["response_mask"].bool()
    if not (
        student_log_probs.shape
        == solution_teacher_log_probs.shape
        == unprivileged_teacher_log_probs.shape
        == valid.shape
    ):
        raise ValueError(
            "Sol-OPD Student, solution-Teacher, unprivileged-Teacher, and mask "
            "tensors must have identical shapes."
        )

    loss_config = distillation_config.distillation_loss
    solution_reverse_kl = student_log_probs - solution_teacher_log_probs
    selected_mask = select_opd_reverse_kl_tokens(
        solution_reverse_kl,
        valid,
        selection_ratio=float(loss_config.selection_ratio),
        selection_method=str(loss_config.selection_method),
    )
    selected_reverse_kl = solution_reverse_kl * selected_mask.to(
        solution_reverse_kl.dtype
    )

    with torch.no_grad():
        opd_advantage = unprivileged_teacher_log_probs - student_log_probs
        sol_advantage = solution_teacher_log_probs - student_log_probs
        metrics = compute_sol_opd_comparison_statistics(
            opd_advantage,
            sol_advantage,
            valid,
            epsilon=float(loss_config.sol_opd_epsilon),
        )
        valid_student = student_log_probs[valid].float()
        valid_solution_teacher = solution_teacher_log_probs[valid].float()
        valid_unprivileged_teacher = unprivileged_teacher_log_probs[valid].float()
        valid_gap = solution_reverse_kl[valid].float().abs()
        selected_gap = solution_reverse_kl[selected_mask].float().abs()
        base_signal = solution_reverse_kl
        if loss_config.loss_max_clamp is not None:
            base_signal = base_signal.clamp(
                min=-loss_config.loss_max_clamp,
                max=loss_config.loss_max_clamp,
            )
        valid_base_signal = base_signal[valid].float()
        valid_selected_signal = (
            base_signal * selected_mask.to(base_signal.dtype)
        )[valid].float()
        metrics.update(
            {
                "distillation/reverse_kl_estimate": Metric(
                    AggregationType.MEAN, solution_reverse_kl[valid].mean()
                ),
                "distillation/student_sampled_token_prob": Metric(
                    AggregationType.MEAN, valid_student.exp().mean()
                ),
                "distillation/teacher_sampled_token_prob": Metric(
                    AggregationType.MEAN, valid_solution_teacher.exp().mean()
                ),
                "distillation/sol_unprivileged_teacher_sampled_token_prob": Metric(
                    AggregationType.MEAN, valid_unprivileged_teacher.exp().mean()
                ),
                "distillation/selected_token_ratio": Metric(
                    AggregationType.MEAN, selected_mask[valid].float().mean()
                ),
                "distillation/selection_gap_mean": Metric(
                    AggregationType.MEAN, valid_gap.mean()
                ),
                "distillation/selected_gap_mean": Metric(
                    AggregationType.MEAN,
                    selected_gap.mean()
                    if selected_gap.numel()
                    else valid_gap.new_zeros(()),
                ),
                "distillation/selection_gradient_signal_relative_change": Metric(
                    AggregationType.MEAN,
                    torch.linalg.vector_norm(
                        valid_selected_signal - valid_base_signal
                    )
                    / torch.linalg.vector_norm(valid_base_signal).clamp_min(1e-12),
                ),
            }
        )
    return selected_reverse_kl, metrics


@register_distillation_loss(DistillationLossSettings(names=["oa_opd"], use_estimator=True))  # type: ignore[arg-type]
def compute_outcome_aware_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply semantic-step outcome weights to the sampled-token OPD signal.

    This kernel returns ``w * (logS - logT)``.  The shared policy-gradient
    path negates and detaches it, yielding the requested advantage
    ``w * (logT - logS)`` while retaining the full response mask as the loss
    denominator.
    """

    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(
        data["teacher_logprobs"], data
    ).squeeze(-1)
    response_mask = data["response_mask"]
    response_mask_bool = (
        response_mask.bool().to_padded_tensor(False)
        if response_mask.is_nested
        else response_mask.bool()
    )
    weights = data.get("oa_opd_weights", None)
    if weights is None:
        raise RuntimeError("OA-OPD actor input is missing oa_opd_weights.")
    if weights.is_nested:
        weights = weights.to_padded_tensor(0.0)
    # Keep intervention weights in float32.  Casting a valid saturated weight
    # to bf16/fp16 can round it up to exactly 1.0, violating OA-OPD's [0, 1)
    # invariant and needlessly rejecting a valid batch.
    weights = weights.to(dtype=torch.float32, device=student_log_probs.device)
    if not (
        student_log_probs.shape
        == teacher_log_probs.shape
        == response_mask_bool.shape
        == weights.shape
    ):
        raise ValueError(
            "OA-OPD Student/Teacher logprobs, response mask, and weights must "
            f"have identical shapes; got {student_log_probs.shape}, "
            f"{teacher_log_probs.shape}, {response_mask_bool.shape}, and {weights.shape}."
        )
    valid_weights = weights[response_mask_bool]
    if not bool(torch.isfinite(valid_weights).all()):
        raise ValueError("OA-OPD intervention weights must be finite.")
    if bool((valid_weights < 0.0).any()) or bool((valid_weights >= 1.0).any()):
        raise ValueError("OA-OPD intervention weights must lie in [0, 1).")
    if bool((weights[~response_mask_bool] != 0.0).any()):
        raise ValueError("OA-OPD padding positions must have zero intervention weight.")

    sampled_reverse_kl = student_log_probs - teacher_log_probs
    weighted_reverse_kl = sampled_reverse_kl * weights
    valid_student = student_log_probs[response_mask_bool].float()
    valid_teacher = teacher_log_probs[response_mask_bool].float()
    metrics: dict[str, Any] = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN,
            sampled_reverse_kl[response_mask_bool].mean(),
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, valid_student.exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, valid_teacher.exp().mean()
        ),
    }

    if "rm_scores" not in data:
        raise RuntimeError("OA-OPD requires rm_scores for correct/wrong metrics.")
    reward_tensor = data["rm_scores"]
    if reward_tensor.is_nested:
        reward_tensor = reward_tensor.to_padded_tensor(0.0)
    rollout_correct = reward_tensor.float().sum(dim=-1) > 0.5
    if rollout_correct.shape != (student_log_probs.shape[0],):
        raise ValueError("OA-OPD rm_scores batch dimension is misaligned.")

    def sequence_field(name: str) -> torch.Tensor:
        value = data.get(name, None)
        if value is None:
            raise RuntimeError(f"OA-OPD actor input is missing {name}.")
        if value.is_nested:
            value = value.to_padded_tensor(0.0)
        value = value.to(device=student_log_probs.device, dtype=torch.float32).reshape(-1)
        if value.numel() != student_log_probs.shape[0]:
            raise ValueError(
                f"OA-OPD {name} must contain one value per trajectory; got "
                f"{value.numel()} for batch size {student_log_probs.shape[0]}."
            )
        return value

    step_count = sequence_field("oa_opd_num_steps")
    active_step_count = sequence_field("oa_opd_active_steps")
    probe_failures = sequence_field("oa_opd_probe_failures")
    if bool((step_count < 0).any()) or bool((active_step_count < 0).any()):
        raise ValueError("OA-OPD step counts must be non-negative.")
    if bool((active_step_count > step_count).any()):
        raise ValueError("OA-OPD active step count cannot exceed total step count.")

    correct_tokens = response_mask_bool & rollout_correct.unsqueeze(-1)
    wrong_tokens = response_mask_bool & ~rollout_correct.unsqueeze(-1)
    active_tokens = response_mask_bool & (weights > 0.0)
    absolute_advantage = sampled_reverse_kl.float().abs()
    retained_advantage = weighted_reverse_kl.float().abs()
    stats_prefix = "distillation/oa_opd_stats/"

    def sum_metric(value: torch.Tensor) -> Metric:
        return Metric(AggregationType.SUM, value)

    raw_stats = {
        "step_count": step_count.sum(),
        "active_step_count": active_step_count.sum(),
        "token_count": response_mask_bool.float().sum(),
        "active_token_count": active_tokens.float().sum(),
        "weight_sum": weights[response_mask_bool].float().sum(),
        "opd_adv_mass": absolute_advantage[response_mask_bool].sum(),
        "oa_adv_mass": retained_advantage[response_mask_bool].sum(),
        "probe_failure_count": probe_failures.sum(),
    }
    for outcome_name, sequence_mask, token_mask in (
        ("correct", rollout_correct, correct_tokens),
        ("wrong", ~rollout_correct, wrong_tokens),
    ):
        raw_stats.update(
            {
                f"{outcome_name}_step_count": step_count[sequence_mask].sum(),
                f"{outcome_name}_active_step_count": active_step_count[
                    sequence_mask
                ].sum(),
                f"{outcome_name}_token_count": token_mask.float().sum(),
                f"{outcome_name}_active_token_count": (
                    active_tokens & token_mask
                ).float().sum(),
                f"{outcome_name}_weight_sum": weights[token_mask].float().sum(),
                f"{outcome_name}_opd_adv_mass": absolute_advantage[token_mask].sum(),
                f"{outcome_name}_oa_adv_mass": retained_advantage[token_mask].sum(),
            }
        )
    metrics.update(
        {
            f"{stats_prefix}{name}": sum_metric(value)
            for name, value in raw_stats.items()
        }
    )

    track_outcome_metrics = bool(
        tu.get_non_tensor_data(
            data, "track_opd_outcome_metrics", default=False
        )
    )
    if track_outcome_metrics:
        metrics.update(
            compute_opd_outcome_statistics(
                advantage=-weighted_reverse_kl,
                response_mask=response_mask_bool,
                rm_scores=data["rm_scores"],
                statistics_threshold=float(
                    getattr(
                        distillation_config.distillation_loss,
                        "opd_statistics_threshold",
                        1.0e-4,
                    )
                ),
            )
        )
    return weighted_reverse_kl, metrics


@register_distillation_loss(
    DistillationLossSettings(
        names=["eopd"],
        use_topk=True,
        use_full_vocab_teacher_entropy=True,
    )
)  # type: ignore[arg-type]
def compute_eopd_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return standard sampled-token OPD plus EOPD-specific diagnostics.

    The returned tensor is exactly the existing OPD reverse-KL estimator.  The
    differentiable Forward KL is combined separately in ``distillation_loss``
    so its gradient is not detached or routed through REINFORCE.
    """

    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(
        data["teacher_sampled_logprobs"], data
    ).squeeze(-1)
    teacher_entropy = no_padding_2_padding(data["teacher_entropy"], data).squeeze(
        -1
    )
    forward_kl = no_padding_2_padding(
        model_output["distillation_losses"], data
    )
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    if not (
        student_log_probs.shape
        == teacher_log_probs.shape
        == teacher_entropy.shape
        == forward_kl.shape
        == response_mask_bool.shape
    ):
        raise ValueError("EOPD token tensors must have identical response shapes.")

    sampled_reverse_kl = student_log_probs - teacher_log_probs
    valid = response_mask_bool
    high_entropy = (
        teacher_entropy > distillation_config.distillation_loss.eopd_entropy_threshold
    ) & valid
    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN, sampled_reverse_kl[valid].mean()
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, student_log_probs[valid].float().exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, teacher_log_probs[valid].float().exp().mean()
        ),
        "distillation/eopd_teacher_entropy": Metric(
            AggregationType.MEAN, teacher_entropy[valid].float().mean()
        ),
        "distillation/eopd_high_entropy_token_ratio": Metric(
            AggregationType.MEAN, high_entropy[valid].float().mean()
        ),
        "distillation/eopd_forward_kl_per_token": Metric(
            AggregationType.MEAN, forward_kl[valid].float().mean()
        ),
    }
    return sampled_reverse_kl, metrics


def compute_exopd_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    *,
    exopd_lambda: float = 1.25,
) -> torch.Tensor:
    """Return the detached ExOPD token-level advantage.

    ``A = lambda * (log pi_T - log pi_ref) - (log pi_S - log pi_ref)``.
    The initial student reference and teacher are supplied by forward-only
    workers; the explicit no-grad region also prevents the Student term inside
    the advantage from creating an autograd path.
    """

    if not (
        student_log_probs.shape == teacher_log_probs.shape == reference_log_probs.shape
    ):
        raise ValueError("ExOPD log-probability tensors must have identical shapes.")
    exopd_lambda = float(exopd_lambda)
    if not math.isfinite(exopd_lambda) or exopd_lambda < 0:
        raise ValueError(
            "ExOPD lambda must be finite and non-negative, got "
            f"{exopd_lambda}."
        )

    with torch.no_grad():
        advantage = exopd_lambda * (teacher_log_probs - reference_log_probs) - (
            student_log_probs - reference_log_probs
        )
    return advantage


@register_distillation_loss(
    DistillationLossSettings(names=["exopd_reverse_kl"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_exopd_sampled_token_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``-A_ExOPD`` for the shared detached REINFORCE loss path."""

    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    reference_log_probs = data["ref_log_prob"]
    if reference_log_probs.is_nested:
        reference_log_probs = reference_log_probs.to_padded_tensor(0.0)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert (
        student_log_probs.shape
        == teacher_log_probs.shape
        == reference_log_probs.shape
        == response_mask_bool.shape
    )

    advantage = compute_exopd_advantage(
        student_log_probs,
        teacher_log_probs,
        reference_log_probs,
        exopd_lambda=distillation_config.distillation_loss.exopd_lambda,
    )
    valid = response_mask_bool
    valid_advantage = advantage[valid].float()
    metrics = {
        "distillation/exopd_advantage_mean": Metric(
            AggregationType.MEAN, valid_advantage.mean()
        ),
        "distillation/exopd_advantage_abs_mean": Metric(
            AggregationType.MEAN, valid_advantage.abs().mean()
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, student_log_probs[valid].float().exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, teacher_log_probs[valid].float().exp().mean()
        ),
        "distillation/exopd_reference_sampled_token_prob": Metric(
            AggregationType.MEAN, reference_log_probs[valid].float().exp().mean()
        ),
    }
    # distillation_loss() passes -loss.detach() to REINFORCE, recovering A.
    return -advantage, metrics


def compute_calibrated_opd_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    positive_teacher_log_probs: torch.Tensor,
    negative_teacher_log_probs: torch.Tensor,
    cal_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lambda-scaled Cal-OPD advantage and teacher self-deviation.

    Both feedback prompts are treated as interventions rather than assumed
    directions: their maximum positive and minimum negative likelihood shifts
    are computed first.  Only the extreme shift toward the student is removed
    from the original signed gap.  Both signs subtract ``cal_lambda`` times
    the self-deviation from their magnitude, and the matching threshold
    prevents either branch from reversing sign.
    """

    shapes = {
        student_log_probs.shape,
        teacher_log_probs.shape,
        positive_teacher_log_probs.shape,
        negative_teacher_log_probs.shape,
    }
    if len(shapes) != 1:
        raise ValueError("All Cal-OPD log-probability tensors must have identical shapes.")
    cal_lambda = float(cal_lambda)
    if not math.isfinite(cal_lambda) or cal_lambda < 0:
        raise ValueError(
            f"Cal-OPD cal_lambda must be finite and non-negative, got {cal_lambda}."
        )

    base_advantage = teacher_log_probs - student_log_probs
    positive_shift = positive_teacher_log_probs - teacher_log_probs
    negative_shift = negative_teacher_log_probs - teacher_log_probs
    zero = torch.zeros_like(base_advantage)
    upward_deviation = torch.maximum(zero, torch.maximum(positive_shift, negative_shift))
    downward_deviation = torch.minimum(zero, torch.minimum(positive_shift, negative_shift))
    teacher_self_deviation = torch.where(
        base_advantage > 0,
        -downward_deviation,
        torch.where(base_advantage < 0, upward_deviation, zero),
    )
    calibrated_magnitude = torch.relu(
        base_advantage.abs() - cal_lambda * teacher_self_deviation
    )
    calibrated_advantage = base_advantage.sign() * calibrated_magnitude
    return calibrated_advantage, teacher_self_deviation


@register_distillation_loss(DistillationLossSettings(names=["cal_reverse_kl"], use_estimator=True))  # type: ignore[arg-type]
def compute_calibrated_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``-A_cal`` for the shared detached REINFORCE loss path."""

    del config
    loss_config = getattr(distillation_config, "distillation_loss", None)
    cal_lambda = float(getattr(loss_config, "cal_lambda", 1.0))
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    positive_teacher_log_probs = no_padding_2_padding(
        data["cal_positive_teacher_logprobs"], data
    ).squeeze(-1)
    negative_teacher_log_probs = no_padding_2_padding(
        data["cal_negative_teacher_logprobs"], data
    ).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert (
        student_log_probs.shape
        == teacher_log_probs.shape
        == positive_teacher_log_probs.shape
        == negative_teacher_log_probs.shape
        == response_mask_bool.shape
    )

    # The calibrated advantage is a policy-gradient weight and must remain
    # stop-gradient.
    with torch.no_grad():
        calibrated_advantage, teacher_self_deviation = compute_calibrated_opd_advantage(
            student_log_probs,
            teacher_log_probs,
            positive_teacher_log_probs,
            negative_teacher_log_probs,
            cal_lambda=cal_lambda,
        )
        base_advantage = teacher_log_probs - student_log_probs
        valid = response_mask_bool
        valid_base = base_advantage[valid].float()
        valid_calibrated = calibrated_advantage[valid].float()
        valid_deviation = teacher_self_deviation[valid].float()

        positive = valid_base > 0
        negative = valid_base < 0

        def retained_magnitude_ratio(selection: torch.Tensor) -> torch.Tensor:
            base = valid_base[selection]
            calibrated = valid_calibrated[selection]
            # L1 measures the total absolute policy-gradient weight that
            # survives calibration.  Unlike an L2 norm, it does not let a few
            # large advantages hide the removal of many smaller advantages.
            return calibrated.abs().sum() / base.abs().sum().clamp_min(1e-12)

        metrics = {
            "distillation/reverse_kl_estimate": Metric(
                AggregationType.MEAN, -valid_base.mean()
            ),
            "distillation/student_sampled_token_prob": Metric(
                AggregationType.MEAN, student_log_probs[valid].float().exp().mean()
            ),
            "distillation/teacher_sampled_token_prob": Metric(
                AggregationType.MEAN, teacher_log_probs[valid].float().exp().mean()
            ),
            "distillation/cal_advantage_mean": Metric(
                AggregationType.MEAN, valid_calibrated.mean()
            ),
            "distillation/cal_advantage_abs_mean": Metric(
                AggregationType.MEAN, valid_calibrated.abs().mean()
            ),
            "distillation/cal_zero_token_ratio": Metric(
                AggregationType.MEAN, valid_calibrated.eq(0).float().mean()
            ),
            "distillation/cal_teacher_self_deviation_mean": Metric(
                AggregationType.MEAN, valid_deviation.mean()
            ),
            "distillation/cal_retained_magnitude_ratio": Metric(
                AggregationType.MEAN,
                retained_magnitude_ratio(torch.ones_like(positive, dtype=torch.bool)),
            ),
            "distillation/cal_positive_retained_magnitude_ratio": Metric(
                AggregationType.MEAN, retained_magnitude_ratio(positive)
            ),
            "distillation/cal_negative_retained_magnitude_ratio": Metric(
                AggregationType.MEAN, retained_magnitude_ratio(negative)
            ),
        }
    # The shared PG path consumes advantages=-distillation_loss.detach().
    return -calibrated_advantage, metrics


@register_distillation_loss(DistillationLossSettings(names=["ps_reverse_kl"], use_estimator=True))  # type: ignore[arg-type]
def compute_privilege_sensitive_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Downweight sampled reverse-KL tokens that are sensitive to false feedback."""
    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    privileged_log_probs = no_padding_2_padding(data["privileged_teacher_logprobs"], data).squeeze(-1)
    teacher_max_log_probs = None
    if data.get("teacher_max_logprobs", None) is not None:
        teacher_max_log_probs = no_padding_2_padding(data["teacher_max_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert (
        teacher_log_probs.shape
        == privileged_log_probs.shape
        == student_log_probs.shape
        == response_mask_bool.shape
    )

    loss_config = distillation_config.distillation_loss
    base_reverse_kl = student_log_probs - teacher_log_probs
    # Weight the same stabilized signal that standard OPD would consume.
    if loss_config.loss_max_clamp is not None:
        base_reverse_kl = base_reverse_kl.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp,
        )
    sensitivity = (privileged_log_probs - teacher_log_probs).abs()

    stats_dir = loss_config.sensitivity_stats_dir
    if stats_dir:
        if teacher_max_log_probs is None:
            raise RuntimeError("Sensitivity diagnostics require teacher_max_logprobs.")
        stats_path = Path(str(stats_dir))
        stats_path.mkdir(parents=True, exist_ok=True)
        records = []
        for row in range(response_mask_bool.shape[0]):
            row_valid = response_mask_bool[row]
            records.append(
                {
                    "student_logprob": student_log_probs[row][row_valid].detach().float().cpu(),
                    "teacher_logprob": teacher_log_probs[row][row_valid].detach().float().cpu(),
                    "teacher_max_logprob": teacher_max_log_probs[row][row_valid].detach().float().cpu(),
                    "sensitivity": sensitivity[row][row_valid].detach().float().cpu(),
                }
            )
        torch.save(
            records,
            stats_path / f"tokens_pid{os.getpid()}_{uuid4().hex}.pt",
        )
    sensitive_mask = sensitivity > float(loss_config.sensitivity_threshold)
    weights = torch.where(
        sensitive_mask,
        torch.full_like(base_reverse_kl, float(loss_config.w_sens)),
        torch.full_like(base_reverse_kl, float(loss_config.w_stable)),
    )
    weighted_reverse_kl = base_reverse_kl * weights

    valid = response_mask_bool
    base_signal = base_reverse_kl[valid].float()
    weighted_signal = weighted_reverse_kl[valid].float()
    signal_relative_change = torch.linalg.vector_norm(weighted_signal - base_signal) / torch.linalg.vector_norm(
        base_signal
    ).clamp_min(1e-12)
    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN, base_signal.mean()
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, student_log_probs[valid].float().exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, teacher_log_probs[valid].float().exp().mean()
        ),
        "distillation/ps_sensitive_token_ratio": Metric(
            AggregationType.MEAN, sensitive_mask[valid].float().mean()
        ),
        "distillation/ps_sensitivity_mean": Metric(
            AggregationType.MEAN, sensitivity[valid].float().mean()
        ),
        "distillation/ps_gradient_signal_relative_change": Metric(
            AggregationType.MEAN, signal_relative_change
        ),
    }
    return weighted_reverse_kl, metrics
