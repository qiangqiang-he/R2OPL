"""R²OPL-v2 truncated-trajectory answer-probe reclassification.

The probe is a lightweight head/tail confidence check used *only* on truncated
training trajectories.  A truncated rollout cannot be graded by the verifier;
instead we ask the Student how confident it is in the gold answer (a) given the
prompt alone and (b) after the truncated reasoning prefix.  A large
improvement of the tail confidence over the head prior marks the rollout as
on-track and moves it to the self-reinforcement branch.

Migrated from OPD_Forge/utils/r2opl_v2.py. Non-truncated trajectories are
never probed. RPC or non-finite failures stop training so a broken probe
cannot silently disable the correct branch.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from utils.answer_probe import AnswerProbe, build_answer_probe


R2OPL_V2_DELTA_THRESHOLD = 0.3


@dataclass(frozen=True)
class R2OPLV2ProbeResult:
    """Per-rollout head/tail answer-probe outcome."""

    truncated: bool
    probe_correct: bool
    head_mean_logprob: Optional[float]
    tail_mean_logprob: Optional[float]


def _decode(
    tokenizer: Any,
    token_ids: list[int],
    *,
    skip_special_tokens: bool,
) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def find_tail_split_token_end(
    tokenizer: Any,
    response_ids: list[int],
) -> Optional[int]:
    """Return the original-token end of the last sentence delimiter before the end.

    Only sentence-like delimiters that end a clause are accepted (period,
    newline, and the related punctuation set used by the offline cutoff
    analysis).  The returned value is a token count derived by decoding
    original-token prefixes, never by re-tokenizing the sliced text.
    """

    response_ids = [int(value) for value in response_ids]
    if not response_ids:
        return None
    response_text = _decode(tokenizer, response_ids, skip_special_tokens=True)
    if not response_text:
        return None
    matches = list(re.finditer(r"(?:\n+|[.!?。！？；;])(?=\s|$)", response_text))
    if not matches:
        return None
    char_end = matches[-1].end()

    cache: dict[int, str] = {0: ""}

    def decode_prefix(end: int) -> str:
        if end not in cache:
            cache[end] = _decode(
                tokenizer,
                response_ids[:end],
                skip_special_tokens=True,
            )
        return cache[end]

    # Length gives a very accurate binary-search approximation for BPE/SentencePiece
    # tokenizers; the exact prefix check below removes any byte-fallback artifacts.
    low, high = 0, len(response_ids)
    while low < high:
        middle = (low + high) // 2
        if len(decode_prefix(middle)) < char_end:
            low = middle + 1
        else:
            high = middle

    def valid_token_end(token_end: int) -> bool:
        prefix = decode_prefix(token_end)
        return len(prefix) >= char_end and response_text.startswith(prefix)

    for token_end in range(max(0, low - 8), min(len(response_ids), low + 8) + 1):
        if valid_token_end(token_end):
            return token_end
    for token_end in range(0, len(response_ids) + 1):
        if valid_token_end(token_end):
            return token_end
    return None


async def compute_r2opl_v2_probe(
    *,
    tokenizer: Any,
    student_probe: Callable[..., Awaitable[float]],
    prompt_ids: list[int],
    response_ids: list[int],
    answer: str,
    max_new_tokens: int,
    routing_key: Optional[str] = None,
) -> R2OPLV2ProbeResult:
    """Detect truncation and, when truncated, run the two-shot answer probe.

    ``student_probe`` scores answers using the same Student that generated the
    trajectory, with its current rollout weights. The head and tail probes are
    issued as two independent forwards so no masked multi-branch machinery is
    required.
    """

    prompt_ids = [int(value) for value in prompt_ids]
    response_ids = [int(value) for value in response_ids]
    if len(response_ids) < int(max_new_tokens):
        return R2OPLV2ProbeResult(
            truncated=False,
            probe_correct=False,
            head_mean_logprob=None,
            tail_mean_logprob=None,
        )

    answer = str(answer)
    if not answer.strip():
        raise ValueError("R2OPL answer probe requires a non-empty dataset answer.")
    tail_token_end = find_tail_split_token_end(tokenizer, response_ids)
    if tail_token_end is None or tail_token_end <= 0:
        return R2OPLV2ProbeResult(
            truncated=True,
            probe_correct=False,
            head_mean_logprob=None,
            tail_mean_logprob=None,
        )

    probe: AnswerProbe = build_answer_probe(tokenizer, answer)
    prompt_length = len(prompt_ids)
    head_sequence = [*prompt_ids, *probe.token_ids]
    head_positions = [
        prompt_length + position for position in probe.answer_token_positions
    ]
    tail_prefix_length = prompt_length + tail_token_end
    tail_sequence = [
        *prompt_ids,
        *response_ids[:tail_token_end],
        *probe.token_ids,
    ]
    tail_positions = [
        tail_prefix_length + position for position in probe.answer_token_positions
    ]

    head_logprob, tail_logprob = await asyncio.gather(
        student_probe(
            sequence_ids=head_sequence,
            answer_token_positions=head_positions,
            routing_key=routing_key,
        ),
        student_probe(
            sequence_ids=tail_sequence,
            answer_token_positions=tail_positions,
            routing_key=routing_key,
        ),
    )

    head_logprob = float(head_logprob)
    tail_logprob = float(tail_logprob)
    if not math.isfinite(head_logprob) or not math.isfinite(tail_logprob):
        raise RuntimeError("R2OPL answer probe returned a non-finite log probability.")

    head_probability = math.exp(head_logprob)
    tail_probability = math.exp(tail_logprob)
    probe_correct = (tail_probability - head_probability) > R2OPL_V2_DELTA_THRESHOLD
    return R2OPLV2ProbeResult(
        truncated=True,
        probe_correct=probe_correct,
        head_mean_logprob=head_logprob,
        tail_mean_logprob=tail_logprob,
    )


__all__ = [
    "R2OPL_V2_DELTA_THRESHOLD",
    "R2OPLV2ProbeResult",
    "compute_r2opl_v2_probe",
    "find_tail_split_token_end",
]
