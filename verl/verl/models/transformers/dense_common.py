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

from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    distillation_losses: Optional[torch.FloatTensor] = None
    student_mass: Optional[torch.FloatTensor] = None
    teacher_mass: Optional[torch.FloatTensor] = None
    overlap_count: Optional[torch.FloatTensor] = None
    overlap_token_advantage: Optional[torch.FloatTensor] = None


def _chunked_forward_kl_topk(
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    temperature: float,
    chunk_size: int,
    log_prob_min_clamp: Optional[float],
    shift_labels: Optional[torch.Tensor] = None,
    teacher_entropy: Optional[torch.Tensor] = None,
    eopd_entropy_threshold: float = 0.8,
) -> tuple[torch.Tensor, ...]:
    """Compute direct top-k OPD without retaining sequence x vocabulary logits.

    Each token chunk is activation-checkpointed.  Forward therefore retains
    only the final hidden states and small token x top-k outputs; backward
    recomputes and releases one chunk-sized vocabulary projection at a time.
    """
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    if chunk_size <= 0:
        raise ValueError(f"distillation chunk_size must be positive, got {chunk_size}")
    if hidden_states.shape[:2] != teacher_topk_ids.shape[:2]:
        raise ValueError(
            "student hidden states and teacher top-k tensors must have matching token dimensions: "
            f"{hidden_states.shape[:2]} != {teacher_topk_ids.shape[:2]}"
        )
    is_eopd = teacher_entropy is not None
    if is_eopd:
        if shift_labels is None:
            raise ValueError("EOPD chunked FKL requires shifted Student token labels.")
        if teacher_entropy.shape != hidden_states.shape[:2]:
            raise ValueError(
                "EOPD Teacher entropy and Student hidden states must have matching "
                "token dimensions."
            )
        if shift_labels.shape != hidden_states.shape[:2]:
            raise ValueError(
                "EOPD shifted labels and Student hidden states must have matching "
                "token dimensions."
            )

    def compute_chunk(
        hidden_chunk,
        teacher_ids_chunk,
        teacher_logps_chunk,
        labels_chunk=None,
        teacher_entropy_chunk=None,
    ):
        logits = F.linear(hidden_chunk, vocab_weights)
        logits = logits / max(float(temperature), 1e-8)
        student_logps = F.log_softmax(logits, dim=-1)
        student_topk_ids = torch.topk(student_logps, k=teacher_ids_chunk.shape[-1], dim=-1).indices
        student_teacher_logps = torch.gather(student_logps, dim=-1, index=teacher_ids_chunk)

        student_mass = student_teacher_logps.exp().sum(dim=-1)
        teacher_mass = teacher_logps_chunk.exp().sum(dim=-1)
        if teacher_entropy_chunk is not None:
            from verl.trainer.distillation.eopd import (
                compute_entropy_gated_forward_kl,
            )

            student_loss_logps = student_teacher_logps
            teacher_loss_logps = teacher_logps_chunk.float()
            losses = compute_entropy_gated_forward_kl(
                student_loss_logps,
                teacher_loss_logps,
                teacher_entropy_chunk,
                entropy_threshold=eopd_entropy_threshold,
            )
            normalized_teacher_logps = teacher_loss_logps - torch.logsumexp(
                teacher_loss_logps, dim=-1, keepdim=True
            )
            teacher_probs = normalized_teacher_logps.exp()
            token_kl = teacher_probs * (
                normalized_teacher_logps - student_loss_logps.float()
            )
        else:
            student_loss_logps = student_teacher_logps
            teacher_loss_logps = teacher_logps_chunk.float()
            if log_prob_min_clamp is not None:
                student_loss_logps = student_loss_logps.clamp_min(log_prob_min_clamp)
                teacher_loss_logps = teacher_loss_logps.clamp_min(log_prob_min_clamp)
            teacher_probs = teacher_loss_logps.exp()
            losses = (
                teacher_probs * (teacher_loss_logps - student_loss_logps.float())
            ).sum(dim=-1)
            token_kl = teacher_probs * (
                teacher_loss_logps - student_loss_logps.float()
            )

        overlap_mask = (teacher_ids_chunk.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)
        overlap_advantage = (-token_kl * overlap_mask).sum(dim=-1) / overlap_count.clamp_min(1)
        overlap_advantage = torch.where(
            overlap_count > 0, overlap_advantage, torch.zeros_like(overlap_advantage)
        )
        outputs = (losses, student_mass, teacher_mass, overlap_count, overlap_advantage)
        if labels_chunk is not None:
            sampled_log_probs = torch.gather(
                student_logps, dim=-1, index=labels_chunk.unsqueeze(-1)
            ).squeeze(-1)
            outputs = (*outputs, sampled_log_probs)
        return outputs

    chunk_outputs = [[] for _ in range(6 if is_eopd else 5)]
    for start in range(0, hidden_states.shape[1], chunk_size):
        stop = min(start + chunk_size, hidden_states.shape[1])
        checkpoint_args = [
            hidden_states[:, start:stop],
            teacher_topk_ids[:, start:stop],
            teacher_topk_log_probs[:, start:stop],
        ]
        if is_eopd:
            checkpoint_args.extend(
                [
                    shift_labels[:, start:stop],
                    teacher_entropy[:, start:stop],
                ]
            )
        outputs = checkpoint(compute_chunk, *checkpoint_args, use_reentrant=False)
        for destination, value in zip(chunk_outputs, outputs, strict=True):
            destination.append(value)
    return tuple(torch.cat(values, dim=1) for values in chunk_outputs)


def forward_base_model(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> CausalLMOutputWithPast:
    r"""
    Copy paste LLaMa's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/llama.py

    This function should be generic enough for all pure text models.
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    return outputs


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    teacher_topk_ids: Optional[torch.LongTensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_entropy: Optional[torch.Tensor] = None,
    eopd_entropy_threshold: float = 0.8,
    distillation_chunk_size: int = 1024,
    distillation_log_prob_min_clamp: Optional[float] = None,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    if teacher_topk_ids is not None:
        if teacher_topk_log_probs is None:
            raise ValueError("teacher_topk_log_probs is required with teacher_topk_ids")
        chunked_outputs = _chunked_forward_kl_topk(
            hidden_states=hidden_states,
            vocab_weights=self.lm_head.weight,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_log_probs=teacher_topk_log_probs,
            temperature=temperature,
            chunk_size=distillation_chunk_size,
            log_prob_min_clamp=distillation_log_prob_min_clamp,
            shift_labels=shift_labels,
            teacher_entropy=teacher_entropy,
            eopd_entropy_threshold=eopd_entropy_threshold,
        )
        return CausalLMOutputForPPO(
            log_probs=chunked_outputs[5] if teacher_entropy is not None else None,
            distillation_losses=chunked_outputs[0],
            student_mass=chunked_outputs[1],
            teacher_mass=chunked_outputs[2],
            overlap_count=chunked_outputs[3],
            overlap_token_advantage=chunked_outputs[4],
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Loss calculations.
    # When the engine has already prepared globally-rolled labels (e.g. the FSDP
    # path under Ulysses SP, see issue #6068), it passes them as `shift_labels`
    # so we don't redo `torch.roll` on a sequence-parallel-sliced shard.
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    # Loss calculations. See `forward_with_torch_backend` for why `shift_labels`
    # takes precedence over local `torch.roll` (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
