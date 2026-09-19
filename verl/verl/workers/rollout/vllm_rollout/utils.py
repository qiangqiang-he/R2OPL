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
import ctypes
import json
import logging
import os
import platform
import signal
import threading
from types import MethodType
from typing import Any, Literal, Optional, get_args

import torch
from vllm.outputs import RequestOutput

from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        return current_platform.get_device_uuid(device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


def enable_eopd_entropy_gather(sampler, topk: int) -> None:
    """Add full-vocabulary entropy to vLLM's bounded prompt-logprob output.

    EOPD requests ``topk + 1`` prompt logprobs.  The patched gather computes
    entropy from the already-materialized full-vocabulary log-softmax, keeps
    only the requested Top-k, and uses the final bounded output slot as an
    entropy carrier.  Thus no full-vocabulary tensor or Python object crosses
    the vLLM worker boundary.
    """

    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"EOPD Top-k must be positive, got {topk}.")
    if getattr(sampler, "_verl_eopd_entropy_topk", None) == topk:
        return
    original_gather_logprobs = sampler.gather_logprobs
    requested_logprobs = topk + 1

    def gather_logprobs(logprobs, num_logprobs, token_ids):
        if int(num_logprobs) != requested_logprobs:
            return original_gather_logprobs(logprobs, num_logprobs, token_ids)

        # Ask the original gather for two candidates outside the retained
        # Top-k. At least one cannot equal the sampled token and is therefore a
        # collision-free, tokenizer-valid carrier id.
        gathered = original_gather_logprobs(
            logprobs, requested_logprobs + 1, token_ids
        )
        sampled_ids = gathered.logprob_token_ids[:, 0]
        first_carrier = gathered.logprob_token_ids[:, topk + 1]
        second_carrier = gathered.logprob_token_ids[:, topk + 2]
        carrier_ids = torch.where(
            first_carrier == sampled_ids, second_carrier, first_carrier
        ).unsqueeze(-1)

        probabilities = logprobs.exp()
        entropy = torch.special.entr(probabilities).sum(dim=-1, keepdim=True)
        retained_ids = torch.cat(
            (gathered.logprob_token_ids[:, : topk + 1], carrier_ids), dim=-1
        )
        retained_logprobs = torch.cat(
            (gathered.logprobs[:, : topk + 1], entropy), dim=-1
        )
        # The engine assigns ranks 1..topk+1 to the retained non-sampled
        # slots. Keep the sampled slot outside that range so only the final
        # slot is interpreted as the entropy carrier.
        sampled_ranks = torch.full_like(
            gathered.selected_token_ranks, requested_logprobs + 1
        )
        return gathered._replace(
            logprob_token_ids=retained_ids,
            logprobs=retained_logprobs,
            selected_token_ranks=sampled_ranks,
        )

    sampler.gather_logprobs = gather_logprobs
    sampler._verl_eopd_entropy_topk = topk


def _validate_fast_oa_opd_worker_payload(
    *,
    sequence_ids: list[int],
    position_ids: list[int],
    original_sequence_length: int,
    branches: list[dict[str, Any]],
) -> None:
    """Validate primitive Fast OA-OPD layout data inside the vLLM worker."""

    sequence_length = len(sequence_ids)
    if sequence_length != len(position_ids):
        raise ValueError("Fast OA-OPD sequence IDs and position IDs must align.")
    if not 1 <= int(original_sequence_length) <= sequence_length:
        raise ValueError("Fast OA-OPD original sequence length is invalid.")
    if [int(value) for value in position_ids[:original_sequence_length]] != list(
        range(original_sequence_length)
    ):
        raise ValueError("Fast OA-OPD changed original rollout position IDs.")
    if not branches:
        raise ValueError("Fast OA-OPD requires at least one appended probe branch.")

    previous_end = int(original_sequence_length)
    for expected_index, branch in enumerate(branches):
        index = int(branch["index"])
        visible_end = int(branch["visible_prefix_end"])
        probe_start = int(branch["probe_start"])
        probe_end = int(branch["probe_end"])
        answer_positions = [int(value) for value in branch["answer_token_positions"]]
        if index != expected_index:
            raise ValueError("Fast OA-OPD branch indices must be consecutive.")
        if probe_start != previous_end:
            raise ValueError("Fast OA-OPD probe branches must be contiguous.")
        if not (
            1
            <= visible_end
            <= original_sequence_length
            <= probe_start
            < probe_end
            <= sequence_length
        ):
            raise ValueError(f"Invalid Fast OA-OPD branch {index}.")
        expected_positions = list(
            range(visible_end, visible_end + probe_end - probe_start)
        )
        if [int(value) for value in position_ids[probe_start:probe_end]] != expected_positions:
            raise ValueError(
                f"Fast OA-OPD branch {index} has incorrect logical position IDs."
            )
        if not answer_positions or any(
            value <= probe_start or value >= probe_end for value in answer_positions
        ):
            raise ValueError(
                f"Fast OA-OPD branch {index} has invalid answer-token positions."
            )
        previous_end = probe_end
    if previous_end != sequence_length:
        raise ValueError("Fast OA-OPD packed sequence has unassigned trailing tokens.")


def _fast_oa_opd_flash_attention(
    *,
    self_attn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    original_sequence_length: int,
    branches: list[dict[str, Any]],
) -> torch.Tensor:
    """Evaluate the branch mask with vLLM's own ragged FlashAttention.

    Query rows remain in their physical packed order.  The KV view contains
    one ordinary causal rollout plus one logical ``[visible prefix, probe]``
    sequence per branch.  FlashAttention's bottom-right causal alignment then
    gives every probe exactly its allowed original prefix and its own causal
    probe prefix, without materializing a dense quadratic mask.
    """

    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    impl = getattr(self_attn.attn, "impl", None)
    if impl is None or type(impl).__name__ != "FlashAttentionImpl":
        raise NotImplementedError(
            "Fast OA-OPD requires the vLLM FlashAttention Teacher backend, "
            f"got {type(impl).__module__}.{type(impl).__name__}."
        )
    if impl.vllm_flash_attn_version is None:
        raise RuntimeError("Fast OA-OPD could not resolve a FlashAttention version.")
    if not bool(impl.batch_invariant_enabled):
        raise RuntimeError(
            "Fast OA-OPD requires vLLM batch-invariant kernels for strict "
            "independent-probe equivalence."
        )

    query_lengths = [int(original_sequence_length)]
    key_lengths = [int(original_sequence_length)]
    key_parts = [key[:original_sequence_length]]
    value_parts = [value[:original_sequence_length]]
    for branch in branches:
        visible_end = int(branch["visible_prefix_end"])
        probe_start = int(branch["probe_start"])
        probe_end = int(branch["probe_end"])
        probe_length = probe_end - probe_start
        query_lengths.append(probe_length)
        key_lengths.append(visible_end + probe_length)
        key_parts.extend((key[:visible_end], key[probe_start:probe_end]))
        value_parts.extend((value[:visible_end], value[probe_start:probe_end]))

    if sum(query_lengths) != query.shape[0]:
        raise RuntimeError("Fast OA-OPD query rows do not cover the packed sequence.")

    def cumulative_lengths(lengths: list[int]) -> torch.Tensor:
        values = [0]
        for length in lengths:
            values.append(values[-1] + length)
        return torch.tensor(values, dtype=torch.int32, device=query.device)

    packed_key = torch.cat(key_parts, dim=0).contiguous()
    packed_value = torch.cat(value_parts, dim=0).contiguous()
    descale_shape = (len(query_lengths), self_attn.num_kv_heads)
    output = flash_attn_varlen_func(
        q=query.contiguous(),
        k=packed_key,
        v=packed_value,
        cu_seqlens_q=cumulative_lengths(query_lengths),
        cu_seqlens_k=cumulative_lengths(key_lengths),
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(key_lengths),
        softmax_scale=float(self_attn.scaling),
        causal=True,
        window_size=list(impl.sliding_window),
        softcap=float(impl.logits_soft_cap),
        alibi_slopes=impl.alibi_slopes,
        fa_version=int(impl.vllm_flash_attn_version),
        q_descale=self_attn.attn._q_scale.expand(descale_shape),
        k_descale=self_attn.attn._k_scale.expand(descale_shape),
        v_descale=self_attn.attn._v_scale.expand(descale_shape),
        num_splits=1 if impl.batch_invariant_enabled else 0,
        s_aux=impl.sinks,
    )
    return output


def _fast_oa_opd_qwen3_forward(
    *,
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    original_sequence_length: int,
    branches: list[dict[str, Any]],
) -> torch.Tensor:
    """Run a Qwen2/Qwen3 decoder with a branch-causal mask.

    vLLM's ordinary decoder attention obtains its causal structure from the
    scheduler and paged-KV metadata.  Fast OA-OPD needs a branch-causal mask,
    represented here as one ragged causal rollout plus independent ragged probe
    branches.  Embeddings, projections, optional Q/K norms, RoPE, attention,
    MLPs and RMS norms all reuse the already-loaded vLLM implementation and
    weights.  Qwen3 always has Q/K norms, while the Qwen2 family only exposes
    them when ``qk_norm`` is enabled; the conditional below keeps the same
    path compatible with both model families.
    """

    decoder = getattr(model, "model", None)
    decoder_name = type(decoder).__name__ if decoder is not None else "None"
    if decoder is None or decoder_name not in {"Qwen2Model", "Qwen3Model"}:
        raise NotImplementedError(
            "Fast OA-OPD masked forward currently supports vLLM Qwen2/Qwen3 "
            "ForCausalLM, "
            f"got {type(model).__module__}.{type(model).__name__}."
        )
    if not hasattr(decoder, "layers"):
        raise RuntimeError(
            "Fast OA-OPD cannot locate the vLLM Qwen2/Qwen3 decoder layers."
        )

    hidden_states = decoder.embed_input_ids(input_ids)
    residual = None
    sequence_length = input_ids.numel()

    for layer in decoder.layers:
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states, residual
            )

        self_attn = layer.self_attn
        qkv, _ = self_attn.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            [self_attn.q_size, self_attn.kv_size, self_attn.kv_size], dim=-1
        )
        q_norm = getattr(self_attn, "q_norm", None)
        k_norm = getattr(self_attn, "k_norm", None)
        if q_norm is not None and k_norm is not None:
            query = q_norm(
                query.view(
                    sequence_length,
                    self_attn.num_heads,
                    self_attn.head_dim,
                )
            ).view(sequence_length, self_attn.q_size)
            key = k_norm(
                key.view(
                    sequence_length,
                    self_attn.num_kv_heads,
                    self_attn.head_dim,
                )
            ).view(sequence_length, self_attn.kv_size)
        query, key = self_attn.rotary_emb(position_ids, query, key)

        query = query.view(
            sequence_length, self_attn.num_heads, self_attn.head_dim
        )
        key = key.view(
            sequence_length, self_attn.num_kv_heads, self_attn.head_dim
        )
        value = value.view(
            sequence_length, self_attn.num_kv_heads, self_attn.head_dim
        )
        attention_output = _fast_oa_opd_flash_attention(
            self_attn=self_attn,
            query=query,
            key=key,
            value=value,
            original_sequence_length=int(original_sequence_length),
            branches=branches,
        )
        attention_output = attention_output.view(
            sequence_length, self_attn.q_size
        )
        hidden_states, _ = self_attn.o_proj(attention_output)
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = layer.mlp(hidden_states)

    hidden_states, _ = decoder.norm(hidden_states, residual)
    return hidden_states


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMHijack.hijack()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            apply_modelopt_nvfp4_patches()
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def _get_drafter_model(self):
        """Return the drafter's model object, or None if unavailable."""
        drafter = getattr(self.model_runner, "drafter", None)
        return drafter.model if drafter is not None and hasattr(drafter, "model") else None

    def _get_draft_model_config(self):
        """Return the draft model config from speculative_config, or None."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec.draft_model_config if spec is not None and spec.draft_model_config is not None else None

    def _use_mtp_drafter_weight_sync(self):
        """Return whether the vLLM MTP drafter should receive actor weights."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec is not None and spec.method == "mtp" and self._get_drafter_model() is not None

    def _iter_all_models(self):
        """Yield models that need weight updates.

        Only vLLM MTP drafter sync is supported for now. Independent non-MTP
        draft models are not compatible with actor weight loading through this path.
        """
        yield self.model_runner.model
        if self._use_mtp_drafter_weight_sync():
            yield self._get_drafter_model()

    def _iter_all_models_with_config(self):
        """Yield (model, model_config) for models that need post-processing."""
        yield self.model_runner.model, self.model_runner.vllm_config.model_config
        if self._use_mtp_drafter_weight_sync():
            draft_cfg = self._get_draft_model_config()
            if draft_cfg is not None:
                yield self._get_drafter_model(), draft_cfg

    def monkey_patch_model(self, vocab_size: int):
        for model in self._iter_all_models():
            # patch compute_logits to avoid sampling OOV token
            monkey_patch_compute_logits(model, vocab_size)
            # patch weight loader to support MoE model
            patch_vllm_moe_model_weight_loader(model)

    def enable_eopd_teacher_entropy(self, topk: int):
        """Enable bounded exact-entropy prompt-logprob output on this worker."""

        enable_eopd_entropy_gather(self.model_runner.sampler, topk)

    def compute_fast_oa_opd_probe_logprobs(
        self,
        *,
        sequence_ids: list[int],
        position_ids: list[int],
        original_sequence_length: int,
        branches: list[dict[str, Any]],
    ) -> list[float]:
        """Score all appended OA-OPD branches in one masked Qwen2/Qwen3 forward."""

        _validate_fast_oa_opd_worker_payload(
            sequence_ids=sequence_ids,
            position_ids=position_ids,
            original_sequence_length=int(original_sequence_length),
            branches=branches,
        )
        parallel_config = self.model_runner.vllm_config.parallel_config
        if (
            int(parallel_config.tensor_parallel_size) != 1
            or int(parallel_config.pipeline_parallel_size) != 1
        ):
            raise NotImplementedError(
                "Fast OA-OPD masked probes currently require Teacher TP=1 and PP=1."
            )

        device = self.device
        input_ids = torch.tensor(sequence_ids, dtype=torch.long, device=device)
        positions = torch.tensor(position_ids, dtype=torch.long, device=device)
        predictor_rows: list[int] = []
        target_ids: list[int] = []
        branch_widths: list[int] = []
        for branch in branches:
            answer_positions = [
                int(value) for value in branch["answer_token_positions"]
            ]
            branch_widths.append(len(answer_positions))
            predictor_rows.extend(position - 1 for position in answer_positions)
            target_ids.extend(int(sequence_ids[position]) for position in answer_positions)

        with torch.inference_mode():
            hidden_states = _fast_oa_opd_qwen3_forward(
                model=self.model_runner.model,
                input_ids=input_ids,
                position_ids=positions,
                original_sequence_length=int(original_sequence_length),
                branches=branches,
            )
            row_indices = torch.tensor(
                predictor_rows, dtype=torch.long, device=device
            )
            selected_hidden = hidden_states.index_select(0, row_indices)
            logits = self.model_runner.model.compute_logits(selected_hidden)
            if logits is None:
                raise RuntimeError("Fast OA-OPD Teacher produced no logits.")
            targets = torch.tensor(target_ids, dtype=torch.long, device=device)
            token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)
            if not bool(torch.isfinite(token_logprobs).all()):
                raise RuntimeError(
                    "Fast OA-OPD Teacher returned non-finite answer log probabilities."
                )

            values: list[float] = []
            offset = 0
            for width in branch_widths:
                values.append(
                    float(token_logprobs[offset : offset + width].mean().item())
                )
                offset += width
        return values

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            for model in self._iter_all_models():
                prepare_qat_for_load_weights(model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            for model in self._iter_all_models():
                patch_vllm_moe_model_weight_loader(model)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            for model in self._iter_all_models():
                manual_process_weights_after_loading(model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            for model, model_config in self._iter_all_models_with_config():
                process_weights_after_loading(model, model_config, self.device)

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
                # Keep the draft model in sync when present.
                if self._use_mtp_drafter_weight_sync():
                    load_quanted_weights(weights, self.model_runner, is_drafter=True)
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                for model in self._iter_all_models():
                    model.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.
        Uses Ray job id + replica_rank + local_rank to form the handle so it
        matches the sender side regardless of CUDA_VISIBLE_DEVICES differences,
        avoids collisions when multiple replicas share the same node, and is
        unique per Ray job to avoid cross-job collisions on shared hosts. The
        job id is forwarded by the vLLMHttpServer actor as VERL_RAY_JOB_ID and
        inherited by this vLLM worker subprocess.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def extract_prompt_logprobs(
    output: RequestOutput,
    num_prompt_logprobs: Optional[int],
    result_dict: dict[str, list],
    prompt_logprobs_topk: Optional[int] = None,
):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return
    if prompt_logprobs_topk is not None:
        if prompt_logprobs_topk <= 0:
            raise ValueError("EOPD prompt-logprob Top-k must be positive.")
        if num_prompt_logprobs != prompt_logprobs_topk + 1:
            raise ValueError(
                "EOPD prompt logprobs must reserve exactly one entropy slot."
            )

    prompt_logprobs_ls, prompt_ids_ls = [], []
    sampled_logprobs_ls, sampled_ids_ls = [], []
    prompt_entropies_ls = []
    # NOTE: logprob of first prompt token is None.
    for position, logprobs_dict in enumerate(output.prompt_logprobs[1:], start=1):
        sampled_id = int(output.prompt_token_ids[position])
        sampled_value = logprobs_dict.get(sampled_id, logprobs_dict.get(str(sampled_id)))
        if sampled_value is None:
            raise KeyError(f"Sampled prompt token {sampled_id} is missing from vLLM prompt_logprobs.")
        sampled_ids_ls.append([sampled_id])
        sampled_logprobs_ls.append([sampled_value.logprob])
        if prompt_logprobs_topk is not None:
            prompt_ids = [None] * prompt_logprobs_topk
            prompt_logprobs = [None] * prompt_logprobs_topk
            entropy = None
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = int(token_logprob.rank)
                if rank == num_prompt_logprobs:
                    entropy = float(token_logprob.logprob)
                    continue
                if rank <= prompt_logprobs_topk:
                    prompt_ids[rank - 1] = int(token_id_str)
                    prompt_logprobs[rank - 1] = float(token_logprob.logprob)
            if any(value is None for value in prompt_ids + prompt_logprobs):
                raise RuntimeError(
                    "vLLM EOPD prompt logprobs did not contain the "
                    f"requested Top-{prompt_logprobs_topk}."
                )
            if entropy is None:
                raise RuntimeError("vLLM EOPD prompt logprobs omitted Teacher entropy.")
            prompt_ids_ls.append(prompt_ids)
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_entropies_ls.append([entropy])
        elif num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    output_width = prompt_logprobs_topk or max(num_prompt_logprobs, 1)
    prompt_logprobs_ls.append([0.0] * output_width)
    prompt_ids_ls.append([0] * output_width)
    sampled_logprobs_ls.append([0.0])
    sampled_ids_ls.append([0])
    if prompt_logprobs_topk is not None:
        prompt_entropies_ls.append([0.0])

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls
    result_dict["prompt_sampled_ids"] = sampled_ids_ls
    result_dict["prompt_sampled_logprobs"] = sampled_logprobs_ls
    if prompt_logprobs_topk is not None:
        result_dict["prompt_entropies"] = prompt_entropies_ls
