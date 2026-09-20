"""Answer-probe token alignment migrated from OPD_Forge/utils/oa_opd.py."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from uuid import uuid4

PROBE_PREFIX = "\n\nTherefore, the answer is \\boxed{"
PROBE_SUFFIX = "}"

@dataclass(frozen=True)
class AnswerProbe:
    """A separately tokenized probe and its scored token positions."""

    text: str
    token_ids: list[int]
    answer_token_positions: list[int]
    answer_char_start: int
    answer_char_end: int


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


def _probe_offsets_fallback(
    tokenizer: Any,
    probe_ids: list[int],
    probe_text: str,
) -> list[tuple[int, int]]:
    """Recover probe-token character extents from the already encoded IDs."""

    boundaries = [0]
    for token_end in range(1, len(probe_ids) + 1):
        decoded = _decode(
            tokenizer,
            probe_ids[:token_end],
            skip_special_tokens=False,
        )
        # Byte-fallback prefixes can temporarily contain a replacement
        # character.  Holding the preceding valid boundary assigns the full
        # recovered character span to the token that completes the bytes.
        boundaries.append(
            len(decoded) if probe_text.startswith(decoded) else boundaries[-1]
        )
    if boundaries[-1] != len(probe_text):
        raise ValueError("Cannot recover probe offsets from its original token IDs.")
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def build_answer_probe(tokenizer: Any, answer: str) -> AnswerProbe:
    """Tokenize the complete appended probe once and mark answer-overlap tokens."""

    answer = str(answer)
    if not answer:
        raise ValueError("OA-OPD cannot construct an answer probe for an empty dataset answer.")
    probe_text = f"{PROBE_PREFIX}{answer}{PROBE_SUFFIX}"
    answer_char_start = len(PROBE_PREFIX)
    answer_char_end = answer_char_start + len(answer)

    encoded = tokenizer(
        probe_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    probe_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        offsets = _probe_offsets_fallback(tokenizer, probe_ids, probe_text)
    offsets = [(int(start), int(end)) for start, end in offsets]
    if len(offsets) != len(probe_ids):
        raise ValueError("Probe offset mapping and token IDs have different lengths.")
    if _decode(tokenizer, probe_ids, skip_special_tokens=False) != probe_text:
        raise ValueError("The complete OA-OPD answer probe does not round-trip through the tokenizer.")

    # Overlap, rather than containment, is intentional.  A token such as
    # ``{4`` overlaps answer character ``4`` and must be scored, whereas a
    # standalone closing brace begins at answer_char_end and is excluded.
    answer_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > answer_char_start and start < answer_char_end
    ]
    if not answer_positions:
        raise ValueError("OA-OPD probe tokenization produced no token overlapping the answer text.")
    return AnswerProbe(
        text=probe_text,
        token_ids=probe_ids,
        answer_token_positions=answer_positions,
        answer_char_start=answer_char_start,
        answer_char_end=answer_char_end,
    )


async def score_answer_probe(*, client, sequence_ids, answer_token_positions, max_model_len):
    """Score gold-answer tokens under the supplied model's current weights."""
    if not sequence_ids or not answer_token_positions:
        raise ValueError("Answer probe requires a non-empty sequence and answer positions.")
    positions = [int(position) for position in answer_token_positions]
    if any(position <= 0 or position >= len(sequence_ids) for position in positions):
        raise ValueError("Answer-probe token positions must have a causal predecessor.")
    if len(sequence_ids) + 1 > int(max_model_len):
        raise ValueError(
            f"Student answer probe requires {len(sequence_ids) + 1} context tokens, "
            f"but max_model_len={max_model_len}; reserve space for the appended answer probe."
        )
    output = await client.generate(
        request_id=uuid4().hex,
        prompt_ids=[int(token) for token in sequence_ids],
        sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
    )
    ids = output.extra_fields["prompt_sampled_ids"]
    logprobs = output.extra_fields["prompt_sampled_logprobs"]
    if len(ids) != len(sequence_ids) or len(logprobs) != len(sequence_ids):
        raise RuntimeError("Student answer probe returned a mismatched prompt-score length.")
    scores = []
    for position in positions:
        if int(ids[position - 1][0]) != int(sequence_ids[position]):
            raise RuntimeError("Student answer-probe causal alignment failed.")
        score = float(logprobs[position - 1][0])
        if not math.isfinite(score):
            raise RuntimeError("Student answer probe returned a non-finite log probability.")
        scores.append(score)
    return sum(scores) / len(scores)
