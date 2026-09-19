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
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    if getattr(
        distillation_loss_config.loss_settings,
        "use_full_vocab_teacher_entropy",
        False,
    ):
        num_logprobs = int(distillation_loss_config.topk) + 1
    else:
        num_logprobs = (
            distillation_loss_config.topk
            if distillation_loss_config.loss_settings.use_topk
            else distillation_loss_config.diagnostic_topk
        )
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


def _slice_response_prediction_outputs(
    sequence_outputs: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Select causal output rows that predict the response tokens.

    Prompt-logprob extractors store the distribution for sequence token ``i``
    at row ``i - 1`` and append one dummy row for the final sequence token.
    Therefore a response of length ``R`` occupies ``[-R - 1 : -1]``, not
    ``[-R:]``.
    """
    if sequence_outputs.ndim == 0:
        raise ValueError("Sequence outputs must have a sequence dimension.")
    sequence_length = sequence_outputs.shape[0]
    if response_length <= 0 or response_length >= sequence_length:
        raise ValueError(
            f"Invalid OPD response length {response_length} for causal sequence output "
            f"length {sequence_length}; at least one prompt token is required."
        )
    return sequence_outputs[sequence_length - response_length - 1 : sequence_length - 1]


def _build_causal_response_layout(
    response_outputs: torch.Tensor,
    student_prompt_length: int,
) -> torch.Tensor:
    """Place response predictions in a student's full causal-output layout."""
    if response_outputs.ndim == 0 or response_outputs.shape[0] == 0:
        raise ValueError("Response outputs must contain at least one token row.")
    if student_prompt_length <= 0:
        raise ValueError(
            f"Invalid student prompt length {student_prompt_length}; at least one prompt token is required."
        )

    leading_padding = response_outputs.new_zeros((student_prompt_length - 1, *response_outputs.shape[1:]))
    trailing_padding = response_outputs.new_zeros((1, *response_outputs.shape[1:]))
    return torch.cat((leading_padding, response_outputs, trailing_padding), dim=0)


def _align_teacher_response_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    teacher_sequence_length: int,
    student_prompt_length: int,
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align teacher response predictions to an independently tokenized student prompt."""
    if teacher_ids.shape != teacher_logprobs.shape or teacher_ids.shape[0] != teacher_sequence_length:
        raise ValueError("Teacher ids/logprobs must cover the complete teacher prompt and response sequence.")
    response_ids = _slice_response_prediction_outputs(teacher_ids, response_length)
    response_logprobs = _slice_response_prediction_outputs(teacher_logprobs, response_length)
    return (
        _build_causal_response_layout(response_ids, student_prompt_length),
        _build_causal_response_layout(response_logprobs, student_prompt_length),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        # ``DistillationTeacherModelConfig.validate_and_prepare_for_distillation``
        # rewrites inference.prompt_length to prompt_length + response_length so
        # the Teacher server can consume the complete sequence as a prefill.  Keep
        # the user-configured prompt capacities before that conversion: Sol-OPD
        # needs the routed Teacher's *prompt-only* limit when it constructs the
        # longer solution-privileged prompt.
        raw_teacher_prompt_configs: dict[str, tuple[Optional[str], int]] = {}
        for entry_name, teacher_config in config.distillation.teacher_models.items():
            raw_routing_key = teacher_config.get("key")
            raw_teacher_prompt_configs[str(entry_name)] = (
                str(raw_routing_key) if raw_routing_key is not None else None,
                int(teacher_config.inference.prompt_length),
            )

        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

        if len(self.teacher_model_configs) == 1:
            resolved_key = next(iter(self.teacher_model_configs))
            try:
                prompt_length = raw_teacher_prompt_configs["teacher_model"][1]
            except KeyError as exc:
                raise ValueError(
                    "Single-teacher distillation requires the teacher_model configuration entry."
                ) from exc
            self._teacher_prompt_lengths = {resolved_key: prompt_length}
        else:
            self._teacher_prompt_lengths = {
                routing_key: prompt_length
                for entry_name, (routing_key, prompt_length) in raw_teacher_prompt_configs.items()
                if entry_name != "teacher_model" and routing_key is not None
            }
        if set(self._teacher_prompt_lengths) != expected:
            raise ValueError(
                "Could not associate every routed Teacher with its configured prompt length: "
                f"length keys={sorted(self._teacher_prompt_lengths)}, "
                f"teacher keys={sorted(expected)}."
            )
        invalid_prompt_lengths = {
            key: value for key, value in self._teacher_prompt_lengths.items() if value <= 0
        }
        if invalid_prompt_lengths:
            raise ValueError(
                "Teacher prompt lengths must be positive, got "
                f"{invalid_prompt_lengths}."
            )

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    def get_teacher_prompt_length(self, routing_key: Optional[str] = None) -> int:
        """Return the routed Teacher's configured prompt-only capacity."""

        teacher_key = self._resolve_teacher_key(routing_key)
        return self._teacher_prompt_lengths[teacher_key]

    async def generate_replacement_step_tokens(
        self,
        *,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        routing_key: Optional[str] = None,
    ) -> list[int]:
        """Sample tokens for one ERSR Teacher replacement proposal.

        Distillation normally uses Teacher servers only for prompt log-probability
        scoring.  ERSR reuses the same already-loaded servers for generation so
        validation never creates a second copy of either model.
        """

        if not prompt_ids:
            raise ValueError("ERSR Teacher replacement requires a non-empty prompt")
        if max_tokens <= 0:
            return []
        teacher_key = self._resolve_teacher_key(routing_key)
        client = self.teacher_client[teacher_key]
        output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=[int(value) for value in prompt_ids],
            sampling_params={
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "seed": int(seed),
            },
        )
        # An empty proposal is returned to the caller as data.  The ERSR
        # evaluator can retry or mark that individual case unavailable without
        # turning a recoverable generation outcome into a training failure.
        return [int(value) for value in output.token_ids]

    async def compute_answer_probe_mean_logprob_single(
        self,
        sequence_ids: list[int],
        answer_token_positions: list[int],
        routing_key: Optional[str] = None,
    ) -> float:
        """Score selected answer tokens in one pre-tokenized OA-OPD probe.

        ``sequence_ids`` is already the exact original rollout prefix followed
        by a separately tokenized complete probe.  ``answer_token_positions``
        therefore refers directly to this sequence; no text slicing or prefix
        re-tokenization occurs here.
        """

        if not sequence_ids:
            raise ValueError("OA-OPD answer probe sequence must be non-empty.")
        if not answer_token_positions:
            raise ValueError("OA-OPD answer probe must select at least one answer token.")
        positions = [int(position) for position in answer_token_positions]
        if any(position <= 0 or position >= len(sequence_ids) for position in positions):
            raise ValueError(
                "OA-OPD answer-token positions must have a causal predecessor "
                f"and lie inside the sequence; got {positions} for length {len(sequence_ids)}."
            )

        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params={
                "max_tokens": 1,
                "temperature": teacher_model_config.inference.temperature,
                # Zero requests only the observed-token log probability.  It
                # avoids transferring an unused Top-k distribution for every
                # boundary probe.
                "prompt_logprobs": 0,
            },
        )
        sampled_ids = torch.tensor(
            teacher_output.extra_fields["prompt_sampled_ids"], dtype=torch.int64
        ).reshape(len(sequence_ids), -1)
        sampled_logprobs = torch.tensor(
            teacher_output.extra_fields["prompt_sampled_logprobs"],
            dtype=torch.float32,
        ).reshape(len(sequence_ids), -1)

        # The server stores the distribution predicting sequence token p in
        # row p-1, followed by one dummy final row.
        causal_rows = torch.tensor([position - 1 for position in positions], dtype=torch.long)
        scored_ids = sampled_ids[causal_rows, 0]
        expected_ids = torch.tensor(
            [sequence_ids[position] for position in positions], dtype=scored_ids.dtype
        )
        if not torch.equal(scored_ids.cpu(), expected_ids.cpu()):
            raise RuntimeError(
                "OA-OPD answer-probe causal alignment failed: returned sampled IDs "
                "do not equal the selected probe answer IDs."
            )
        scores = sampled_logprobs[causal_rows, 0]
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("OA-OPD answer probe returned a non-finite log probability.")
        return float(scores.mean().item())

    async def compute_masked_answer_probe_mean_logprobs_single(
        self,
        *,
        sequence_ids: list[int],
        position_ids: list[int],
        original_sequence_length: int,
        branches: list[dict[str, Any]],
        routing_key: Optional[str] = None,
    ) -> list[float]:
        """Score every appended Fast OA-OPD probe in one masked forward."""

        if not sequence_ids or len(sequence_ids) != len(position_ids):
            raise ValueError(
                "Fast OA-OPD requires aligned, non-empty sequence and position IDs."
            )
        if not branches:
            raise ValueError("Fast OA-OPD requires at least one probe branch.")
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        if str(teacher_model_config.inference.name) != "vllm":
            raise ValueError("Fast OA-OPD masked probes require the vLLM Teacher backend.")
        client = self.teacher_client[teacher_key]
        values = await client.compute_fast_oa_opd_probe_logprobs(
            request_id=uuid4().hex,
            sequence_ids=[int(value) for value in sequence_ids],
            position_ids=[int(value) for value in position_ids],
            original_sequence_length=int(original_sequence_length),
            branches=branches,
        )
        if len(values) != len(branches):
            raise RuntimeError(
                "Fast OA-OPD Teacher returned a boundary-value count mismatch: "
                f"{len(values)} != {len(branches)}."
            )
        value_tensor = torch.tensor(values, dtype=torch.float64)
        if not bool(torch.isfinite(value_tensor).all()):
            raise RuntimeError("Fast OA-OPD Teacher returned a non-finite boundary value.")
        return [float(value) for value in values]

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        student_prompt_length: int,
        response_length: int,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        sampled_only: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        dict[str, float],
    ]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        if response_length <= 0 or len(sequence_ids) <= response_length:
            raise ValueError(
                "Teacher scoring requires at least one prompt token and one response token; "
                f"got sequence_length={len(sequence_ids)}, response_length={response_length}."
            )
        # A few focused alignment tests construct the manager via ``__new__``
        # to exercise extraction without a full runtime config.  Production
        # construction always initializes this map and therefore always enforces
        # the routed prompt capacity.
        if hasattr(self, "_teacher_prompt_lengths"):
            teacher_prompt_length = len(sequence_ids) - response_length
            max_teacher_prompt_length = self.get_teacher_prompt_length(teacher_key)
            if teacher_prompt_length > max_teacher_prompt_length:
                raise ValueError(
                    "Teacher prompt exceeds its configured prompt-only capacity: "
                    f"{teacher_prompt_length} > {max_teacher_prompt_length} "
                    f"for routing key {teacher_key!r}."
                )
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        loss_settings = self.distillation_loss_config.loss_settings
        if sampled_only and (
            not getattr(loss_settings, "use_estimator", False)
            or getattr(loss_settings, "use_topk", False)
            or getattr(loss_settings, "use_full_vocab_teacher_entropy", False)
        ):
            raise ValueError(
                "sampled_only Teacher scoring is valid only for estimator-based "
                "distillation losses without Top-k or full-vocabulary entropy."
            )
        extra_generate_kwargs = {}
        if not sampled_only and getattr(
            loss_settings,
            "use_full_vocab_teacher_entropy",
            False,
        ):
            extra_generate_kwargs["prompt_logprobs_topk"] = int(
                self.distillation_loss_config.topk
            )
        sampling_params = _get_teacher_sampling_params(
            teacher_model_config, self.distillation_loss_config
        )
        if sampled_only:
            # Ask the server only for the observed token's log probability at
            # each causal position.  Sol-OPD's second Teacher forward is purely
            # an A_opd baseline and does not consume Top-k diagnostics.
            sampling_params["prompt_logprobs"] = 0
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=sampling_params,
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
            **extra_generate_kwargs,
        )
        # Shapes: [S, 1 or K], where S is the complete teacher prompt-plus-response
        # sequence length. Each row predicts the following token, and the last row is dummy.
        diagnostic_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        diagnostic_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        if sampled_only:
            sampled_ids = diagnostic_ids
            sampled_logprobs = diagnostic_logprobs
        else:
            sampled_ids = torch.tensor(
                teacher_output.extra_fields["prompt_sampled_ids"], dtype=torch.int32
            )
            sampled_logprobs = torch.tensor(
                teacher_output.extra_fields["prompt_sampled_logprobs"]
            )
        teacher_entropy = None
        if not sampled_only and getattr(
            loss_settings,
            "use_full_vocab_teacher_entropy",
            False,
        ):
            teacher_entropy = torch.tensor(teacher_output.extra_fields["prompt_entropies"])
        if loss_settings.use_topk:
            teacher_ids = diagnostic_ids
            teacher_logprobs = diagnostic_logprobs
        else:
            teacher_ids = sampled_ids
            teacher_logprobs = sampled_logprobs
        # Teacher and student prompts may differ when their thinking modes are
        # configured independently.  Only response-token distributions enter
        # the OPD loss, so align the teacher's final response positions to the
        # student prompt width and leave ignored prompt positions as padding.
        teacher_ids, teacher_logprobs = _align_teacher_response_outputs(
            teacher_ids,
            teacher_logprobs,
            teacher_sequence_length=len(sequence_ids),
            student_prompt_length=student_prompt_length,
            response_length=response_length,
        )
        diagnostic_ids, diagnostic_logprobs = _align_teacher_response_outputs(
            diagnostic_ids,
            diagnostic_logprobs,
            teacher_sequence_length=len(sequence_ids),
            student_prompt_length=student_prompt_length,
            response_length=response_length,
        )
        sampled_ids, sampled_logprobs = _align_teacher_response_outputs(
            sampled_ids,
            sampled_logprobs,
            teacher_sequence_length=len(sequence_ids),
            student_prompt_length=student_prompt_length,
            response_length=response_length,
        )
        if teacher_entropy is not None:
            _, teacher_entropy = _align_teacher_response_outputs(
                torch.zeros_like(teacher_entropy, dtype=torch.int32),
                teacher_entropy,
                teacher_sequence_length=len(sequence_ids),
                student_prompt_length=student_prompt_length,
                response_length=response_length,
            )
        aligned_response_ids = _slice_response_prediction_outputs(
            sampled_ids, response_length
        ).reshape(-1)
        expected_response_ids = torch.as_tensor(
            sequence_ids[-response_length:],
            dtype=aligned_response_ids.dtype,
            device=aligned_response_ids.device,
        )
        if not torch.equal(aligned_response_ids, expected_response_ids):
            raise RuntimeError(
                "OPD teacher sampled-token alignment invariant failed: teacher ids at "
                "the loss positions do not equal the student response ids."
            )
        timing = {
            "teacher_engine_s": float(teacher_output.extra_fields.get("engine_generate_s", 0.0)),
            "teacher_logprob_extract_s": float(
                teacher_output.extra_fields.get("prompt_logprobs_extract_s", 0.0)
            ),
        }
        return (
            teacher_ids,
            teacher_logprobs,
            diagnostic_ids,
            diagnostic_logprobs,
            sampled_logprobs,
            teacher_entropy,
            timing,
        )
