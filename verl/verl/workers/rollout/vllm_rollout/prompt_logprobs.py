# Copyright 2026 R2OPL contributors
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
"""Bound V1 prompt-score memory independently of the prefill token budget.

R2OPL asks the Student for prompt top-k scores after generation. V1 vLLM
otherwise projects an entire scheduled prompt into the vocabulary at once;
training-side fused/chunked logits do not protect this separate process.
The request lifecycle follows vLLM's GPUModelRunner._get_prompt_logprobs_dict.
"""

import logging
from types import MethodType

import torch
from vllm.v1.outputs import LogprobsTensors

logger = logging.getLogger(__name__)
PROMPT_LOGPROBS_CHUNK_SIZE = 128


def enable_chunked_prompt_logprobs(runner, chunk_size=PROMPT_LOGPROBS_CHUNK_SIZE):
    """Patch a V1 runner instance; V2 already has native chunked projection."""
    if chunk_size < 1:
        raise ValueError("prompt logprob chunk size must be positive")
    original = getattr(runner, "_get_prompt_logprobs_dict", None)
    if original is None:
        # V2's PromptLogprobsWorker uses compute_prompt_logprobs_with_chunking.
        return False
    if hasattr(runner, "_verl_prompt_logprobs_chunk_size"):
        return True
    # vLLM 0.19 always normalized prompt scores; newer V1 versions also
    # support raw/processed logits. Preserve the installed method's semantics.
    runner._verl_prompt_scores_support_logits = "logprobs_mode" in original.__func__.__code__.co_names
    runner._verl_prompt_logprobs_chunk_size = int(chunk_size)
    runner._get_prompt_logprobs_dict = MethodType(_get_chunked_prompt_logprobs, runner)
    logger.info("Enabled vLLM V1 prompt-logprob projection chunks of %s tokens", chunk_size)
    return True


def _get_chunked_prompt_logprobs(self, hidden_states, num_scheduled_tokens):
    if not self.num_prompt_logprobs:
        return {}

    # vLLM 0.19 stores partial CPU scores on InputBatch; 0.29 on each request.
    legacy_cache = getattr(self.input_batch, "in_progress_prompt_logprobs_cpu", None)
    completed = {}
    for req_id, topk in self.num_prompt_logprobs.items():
        num_tokens = num_scheduled_tokens.get(req_id)
        if num_tokens is None:  # e.g. preempted during prefill
            continue
        request = self.requests[req_id]
        if request.prompt_token_ids is None:
            continue
        prompt_len = len(request.prompt_token_ids)
        result = (
            legacy_cache.get(req_id)
            if legacy_cache is not None
            else request.in_progress_prompt_logprobs_cpu
        )
        if result is None:
            result = LogprobsTensors.empty_cpu(prompt_len - 1, topk + 1)
            if legacy_cache is not None:
                legacy_cache[req_id] = result
            else:
                request.in_progress_prompt_logprobs_cpu = result

        start_idx = request.num_computed_tokens
        start_tok = start_idx + 1
        remaining = prompt_len - start_tok
        num_logits = min(num_tokens, remaining)
        # Equality deliberately defers returning scores until the next step,
        # when vLLM also has a generated token to return to the caller.
        if num_tokens > remaining:
            completed[req_id] = result
        if num_logits <= 0:
            continue

        target_ids = torch.tensor(request.prompt_token_ids, dtype=torch.long).to(self.device, non_blocking=True)
        req_idx = self.input_batch.req_id_to_index[req_id]
        offset = self.query_start_loc.np[req_idx].item()
        logits_mode = self._verl_prompt_scores_support_logits and self.model_config.logprobs_mode in (
            "raw_logits",
            "processed_logits",
        )
        for begin in range(0, num_logits, self._verl_prompt_logprobs_chunk_size):
            end = min(begin + self._verl_prompt_logprobs_chunk_size, num_logits)
            logits = self.model.compute_logits(hidden_states[offset + begin : offset + end])
            scores = logits.float() if logits_mode else self.sampler.compute_logprobs(logits)
            # Keep the sampler hook (including EOPD's entropy carrier), sampled
            # token rank, and top-k ordering identical to the upstream path.
            token_ids, logprobs, ranks, *_ = self.sampler.gather_logprobs(
                scores, topk, target_ids[start_tok + begin : start_tok + end]
            )
            destination = slice(start_idx + begin, start_idx + end)
            result.logprob_token_ids[destination].copy_(token_ids, non_blocking=True)
            result.logprobs[destination].copy_(logprobs, non_blocking=True)
            result.selected_token_ranks[destination].copy_(ranks, non_blocking=True)
            # Do not keep the previous dense vocabulary tensors alive during
            # the next projection. Only the compact CPU top-k output persists.
            del logits, scores, token_ids, logprobs, ranks

    for req_id in completed:
        del self.num_prompt_logprobs[req_id]
        if legacy_cache is not None:
            del legacy_cache[req_id]
        else:
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None
    if completed:
        self._sync_device()
    return completed
