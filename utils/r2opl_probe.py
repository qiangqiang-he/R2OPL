"""Source-preserving probe metadata for R²OPL.

R²OPL probes the gold answer after the original prompt and after every
*non-terminal* semantic reasoning step.  This module deliberately performs
only deterministic token/layout construction; model scoring belongs to the
rollout/actor paths.  Keeping the layout separate makes the important
invariant explicit: the final semantic step is never a probe boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from utils.answer_probe import build_answer_probe
from utils.conservative_step_splitter import split_rollout_steps


# This is deliberately separate from OA-OPD's legacy probe template.  The
# paper fixes c_probe exactly (including no leading newlines and the final
# full stop), so changing it would change each measured P_k and hence ΔP_k.
R2OPL_PROBE_PREFIX = "Therefore, the answer is \\boxed{"
R2OPL_PROBE_SUFFIX = "}."


@dataclass(frozen=True)
class R2OPLProbeMetadata:
    """Token metadata consumed by the packed R²OPL student probe forward.

    ``semantic_step_ends`` indexes the original, unpadded response IDs.  The
    probe boundaries are derived by the consumer as
    ``[0] + semantic_step_ends[:-1]``.  Thus every non-terminal semantic step
    is measured, while the terminal step (and any invisible trailing tokens)
    retains the ordinary RL/OPD signal.

    ``probe_answer_positions`` are local indices within ``probe_token_ids``;
    the packed-forward caller offsets them for each branch.
    """

    probe_token_ids: list[int]
    probe_answer_positions: list[int]
    semantic_step_ends: list[int]

    @property
    def probe_boundaries(self) -> list[int]:
        """Original-response token ends at which an answer probe is appended."""
        if not self.probe_token_ids or not self.semantic_step_ends:
            return []
        return [0, *self.semantic_step_ends[:-1]]


def _validate_semantic_step_ends(
    semantic_step_ends: Sequence[int], *, response_length: int
) -> list[int]:
    """Validate the splitter's source-token spans before exposing metadata."""
    ends = [int(value) for value in semantic_step_ends]
    previous = 0
    for end in ends:
        if end <= previous or end > response_length:
            raise RuntimeError(
                "R²OPL semantic-step token ends must be strictly increasing and "
                f"within the response; got {ends} for response length {response_length}."
            )
        previous = end
    return ends


def build_r2opl_answer_probe(*, tokenizer: Any, answer: str) -> Any:
    """Build the paper's exact fixed answer probe without affecting OA-OPD.

    Keeping this construction in the R²OPL module makes it difficult for a
    caller to accidentally inherit a different method's prompt formatting.
    """

    return build_answer_probe(
        tokenizer,
        answer,
        prefix=R2OPL_PROBE_PREFIX,
        suffix=R2OPL_PROBE_SUFFIX,
        probe_name="R²OPL",
    )


def build_r2opl_probe_metadata(
    *,
    tokenizer: Any,
    response_ids: Sequence[int],
    answer: str | None,
) -> R2OPLProbeMetadata:
    """Build R²OPL's reusable answer probe and semantic-step endpoints.

    A response with zero or one semantic step has no non-terminal step, hence
    it intentionally carries empty probe tensors and does not require an
    answer.  For two or more semantic steps, a missing/empty answer is a hard
    error: silently disabling probes would change the requested algorithm.
    """
    original_response_ids = [int(token_id) for token_id in response_ids]
    try:
        _, steps = split_rollout_steps(original_response_ids, tokenizer)
    except Exception as exc:
        raise RuntimeError("R²OPL could not construct semantic reasoning steps.") from exc

    semantic_step_ends = _validate_semantic_step_ends(
        [step.token_end for step in steps], response_length=len(original_response_ids)
    )
    if len(semantic_step_ends) < 2:
        return R2OPLProbeMetadata(
            probe_token_ids=[],
            probe_answer_positions=[],
            semantic_step_ends=semantic_step_ends,
        )

    answer_text = "" if answer is None else str(answer)
    if not answer_text.strip():
        raise ValueError(
            "R²OPL requires a non-empty dataset answer when a rollout has "
            "non-terminal semantic steps to probe."
        )
    try:
        answer_probe = build_r2opl_answer_probe(tokenizer=tokenizer, answer=answer_text)
    except Exception as exc:
        raise RuntimeError("R²OPL could not construct its gold-answer probe.") from exc

    probe_token_ids = [int(token_id) for token_id in answer_probe.token_ids]
    probe_answer_positions = [int(position) for position in answer_probe.answer_token_positions]
    if not probe_token_ids or not probe_answer_positions:
        raise RuntimeError("R²OPL answer probe must contain scored answer tokens.")
    if any(position <= 0 or position >= len(probe_token_ids) for position in probe_answer_positions):
        raise RuntimeError(
            "R²OPL answer-probe positions must have causal predecessors within "
            "the probe suffix."
        )

    return R2OPLProbeMetadata(
        probe_token_ids=probe_token_ids,
        probe_answer_positions=probe_answer_positions,
        semantic_step_ends=semantic_step_ends,
    )


__all__ = [
    "R2OPL_PROBE_PREFIX",
    "R2OPL_PROBE_SUFFIX",
    "R2OPLProbeMetadata",
    "build_r2opl_answer_probe",
    "build_r2opl_probe_metadata",
]
