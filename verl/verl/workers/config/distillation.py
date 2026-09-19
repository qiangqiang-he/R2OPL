# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "reverse_kl"
    topk: Optional[int] = None
    diagnostic_topk: int = 16
    # EOPD entropy gate threshold (tau).
    eopd_entropy_threshold: float = 0.8
    # Weight applied to EOPD's gated Forward KL term (alpha).
    eopd_alpha: float = 1.0
    sensitivity_threshold: float = 0.05
    w_sens: float = 0.25
    w_stable: float = 1.0
    selection_ratio: float = 1.0
    selection_method: str = "random"
    # OA-OPD step intervention threshold, strength, and maximum number of
    # boundary probes submitted together for one rollout.
    oa_opd_tau: float = 0.0
    oa_opd_beta: float = 1.0
    oa_opd_probe_batch_size: int = 8
    # Multiplier applied to Teacher self-deviation on both sides of Cal-OPD.
    cal_lambda: float = 1.0
    # Teacher coefficient in A_ExOPD = lambda * (logT - logRef) - (logS - logRef).
    exopd_lambda: float = 1.25
    # Statistics-only dead zone for classifying standard OPD advantages as
    # positive or negative. It never changes the training signal.
    opd_statistics_threshold: float = 1.0e-4
    # Numerical dead zone used only by Sol-OPD comparison diagnostics.
    sol_opd_epsilon: float = 1.0e-6
    sensitivity_stats_dir: Optional[str] = None
    use_task_rewards: bool = False
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)
        if self.diagnostic_topk <= 0:
            raise ValueError(f"diagnostic_topk must be positive, got {self.diagnostic_topk}.")
        if not math.isfinite(self.eopd_entropy_threshold) or self.eopd_entropy_threshold < 0:
            raise ValueError(
                "eopd_entropy_threshold must be finite and non-negative, got "
                f"{self.eopd_entropy_threshold}."
            )
        if not math.isfinite(self.eopd_alpha) or self.eopd_alpha < 0:
            raise ValueError(
                f"eopd_alpha must be finite and non-negative, got {self.eopd_alpha}."
            )
        if self.sensitivity_threshold < 0:
            raise ValueError(
                f"sensitivity_threshold must be non-negative, got {self.sensitivity_threshold}."
            )
        if self.w_sens < 0 or self.w_stable < 0:
            raise ValueError(
                "PS-OPD token weights must be non-negative, got "
                f"w_sens={self.w_sens}, w_stable={self.w_stable}."
            )
        if not 0.0 <= self.selection_ratio <= 1.0:
            raise ValueError(
                "selection_ratio must lie in [0, 1], got "
                f"{self.selection_ratio}."
            )
        if self.selection_ratio < 1.0 and self.selection_method not in {
            "random",
            "topgap",
            "bottomgap",
        }:
            raise ValueError(
                "selection_method must be one of random, topgap, bottomgap; got "
                f"{self.selection_method!r}."
            )
        if not math.isfinite(self.oa_opd_tau):
            raise ValueError(f"oa_opd_tau must be finite, got {self.oa_opd_tau}.")
        if not math.isfinite(self.oa_opd_beta) or self.oa_opd_beta <= 0:
            raise ValueError(
                f"oa_opd_beta must be finite and positive, got {self.oa_opd_beta}."
            )
        if not 1 <= self.oa_opd_probe_batch_size <= 8:
            raise ValueError(
                "oa_opd_probe_batch_size must lie in [1, 8], got "
                f"{self.oa_opd_probe_batch_size}."
            )
        if not math.isfinite(self.cal_lambda) or self.cal_lambda < 0:
            raise ValueError(
                "cal_lambda must be finite and non-negative, got "
                f"{self.cal_lambda}."
            )
        if not math.isfinite(self.exopd_lambda) or self.exopd_lambda < 0:
            raise ValueError(
                "exopd_lambda must be finite and non-negative, got "
                f"{self.exopd_lambda}."
            )
        if self.opd_statistics_threshold < 0:
            raise ValueError(
                "opd_statistics_threshold must be non-negative, got "
                f"{self.opd_statistics_threshold}."
            )
        if not math.isfinite(self.sol_opd_epsilon) or self.sol_opd_epsilon <= 0:
            raise ValueError(
                "sol_opd_epsilon must be finite and positive, got "
                f"{self.sol_opd_epsilon}."
            )
        if self.policy_loss_mode not in {"vanilla", "reinforce"}:
            raise NotImplementedError(
                f"Only vanilla and reinforce policy losses are supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if not self.use_policy_gradient and self.loss_mode in {
            "k1",
            "reverse_kl",
            "error_reverse_kl",
            "correct_reverse_kl",
            "sol_reverse_kl",
            "cal_reverse_kl",
            "eopd",
            "exopd_reverse_kl",
            "ps_reverse_kl",
            "uni_opd",
            "fire_opd",
            "oa_opd",
            "r2opl_base",
        }:
            raise ValueError(
                f"Directly backpropagating {self.loss_mode} is incorrect since its sampled-token loss "
                "must be consumed as a policy-gradient advantage."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(
        self,
        use_topk: bool,
        topk: Optional[int],
        use_full_vocab_teacher_entropy: bool = False,
    ) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(
            use_topk=use_topk,
            topk=topk,
            use_full_vocab_teacher_entropy=use_full_vocab_teacher_entropy,
        )

    def _validate_topk_logprobs(
        self,
        use_topk: bool,
        topk: Optional[int],
        use_full_vocab_teacher_entropy: bool = False,
    ) -> None:
        if not use_topk and not use_full_vocab_teacher_entropy:
            return
        if topk is None:
            raise ValueError(
                "topk must be specified for top-k distillation or exact "
                "Teacher-entropy transfer."
            )

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if use_full_vocab_teacher_entropy:
                    required_logprobs = int(topk) + 1
                    if max_logprobs is None:
                        vllm_engine_kwargs["max_logprobs"] = required_logprobs
                        max_logprobs = required_logprobs
                    if max_logprobs < required_logprobs:
                        raise ValueError(
                            "Full-vocabulary Teacher entropy requires vLLM "
                            "max_logprobs >= topk + 1; got "
                            f"{max_logprobs} < {required_logprobs}."
                        )
                    entropy_topk = vllm_engine_kwargs.get(
                        "full_vocab_entropy_topk",
                        vllm_engine_kwargs.get("eopd_entropy_topk", topk),
                    )
                    if int(entropy_topk) != int(topk):
                        raise ValueError(
                            "vLLM full-vocabulary entropy topk must match "
                            "distillation topk."
                        )
                    if not any(
                        key in vllm_engine_kwargs
                        for key in ("full_vocab_entropy_topk", "eopd_entropy_topk")
                    ):
                        vllm_engine_kwargs["full_vocab_entropy_topk"] = int(topk)
                    engine_kwargs["vllm"] = vllm_engine_kwargs
                    return
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                if use_full_vocab_teacher_entropy:
                    raise NotImplementedError(
                        "Full-vocabulary Teacher entropy currently requires vLLM."
                    )
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    # Student: number of final hidden-state tokens projected through the LM
    # head at once during the differentiable OPD loss.
    student_chunk_size: int = 1024
    # Teacher: maximum number of prompt tokens for which vLLM computes logits
    # in one scheduler iteration. With chunked prefill enabled this bounds the
    # teacher's transient [tokens, vocab] projection.
    teacher_chunk_size: int = 1024
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        if not self.enabled:
            return

        if self.student_chunk_size <= 0 or self.teacher_chunk_size <= 0:
            raise ValueError(
                "distillation student_chunk_size and teacher_chunk_size must both be positive, "
                f"got {self.student_chunk_size=} and {self.teacher_chunk_size=}."
            )

        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            if not teacher_model.inference.enable_chunked_prefill:
                raise ValueError("teacher_chunk_size requires teacher inference.enable_chunked_prefill=true")
            teacher_model.inference.max_num_batched_tokens = self.teacher_chunk_size
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
                use_full_vocab_teacher_entropy=(
                    self.distillation_loss.loss_settings.use_full_vocab_teacher_entropy
                ),
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
