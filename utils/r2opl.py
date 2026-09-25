"""Core packed-probe and objective helpers for R²OPL.

The training implementation uses a *Student* answer probe at every semantic
boundary except the final semantic step.  The probes are packed after the
ordinary prompt/response sequence, but each appended probe is an independent
causal-attention branch.  This module keeps that representation and the
objective arithmetic independent from VERL so it can be unit-tested directly.

Two invariants are deliberately encoded here rather than left to callers:

* The original prompt/response tokens and their position IDs are untouched.
* If a response has ``K`` semantic steps, probes are ``P_0, ..., P_{K-1}``:
  ``P_0`` sees the prompt and ``P_j`` sees the prompt plus steps ``0..j-1``.
  Therefore only steps ``0..K-2`` receive a probe-derived gain.  The final
  step (and any invisible trailing response tokens) always use multiplier
  one.

The probability measured by a probe is the arithmetic mean of the Student's
gold-answer *token probabilities*, not a mean log-probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from utils.answer_probe import AnswerProbe
from utils.conservative_step_splitter import TokenStep


R2OPL_DEFAULT_MIU = 16.0
R2OPL_DEFAULT_LAMBDA = 0.1
R2OPL_DEFAULT_ALPHA_R = 0.25
R2OPL_DEFAULT_ALPHA_D = 0.5
R2OPL_DEFAULT_EPSILON = 0.8


@dataclass(frozen=True)
class R2OPLProbeBranch:
    """Physical and logical placement of one appended answer-probe branch.

    ``after_step_index`` is ``None`` for the prompt-only prior ``P_0`` and is
    otherwise the response-local semantic step after which the branch is
    evaluated.  It intentionally never names the final semantic step.
    """

    index: int
    after_step_index: int | None
    visible_prefix_end: int
    probe_start: int
    probe_end: int
    answer_token_positions: tuple[int, ...]


@dataclass(frozen=True)
class R2OPLProbeLayout:
    """One packed sequence plus all metadata needed to score its probes."""

    sequence_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    original_sequence_length: int
    prompt_length: int
    response_length: int
    semantic_step_ranges: tuple[tuple[int, int], ...]
    probe_token_ids: tuple[int, ...]
    branches: tuple[R2OPLProbeBranch, ...]

    @property
    def num_semantic_steps(self) -> int:
        return len(self.semantic_step_ranges)

    @property
    def num_probed_steps(self) -> int:
        """Number of semantic steps that can receive a non-unit multiplier."""

        return max(0, self.num_semantic_steps - 1)

    @property
    def last_step_range(self) -> tuple[int, int] | None:
        return self.semantic_step_ranges[-1] if self.semantic_step_ranges else None


@dataclass(frozen=True)
class R2OPLProbeBatch:
    """Padded packed-probe layouts for one model forward.

    ``attention_mask`` is a boolean ``[batch, length, length]`` visibility
    mask where ``True`` means a query may attend to a key.  A model that
    expects a custom additive mask can use
    :func:`r2opl_bool_mask_to_additive`.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    sequence_mask: torch.Tensor
    original_sequence_lengths: torch.Tensor
    response_lengths: torch.Tensor
    semantic_step_ranges: torch.Tensor
    semantic_step_mask: torch.Tensor
    probe_branch_mask: torch.Tensor
    layouts: tuple[R2OPLProbeLayout, ...]


@dataclass(frozen=True)
class R2OPLTokenModulation:
    """Probe deltas, gates, and per-token multipliers for one batch.

    The two multiplier tensors are intentionally both returned.  This makes
    it possible to log correct and incorrect trajectories independently while
    selecting the mathematically appropriate one only when the advantage is
    constructed.
    """

    probe_probabilities: torch.Tensor
    probe_branch_mask: torch.Tensor
    semantic_step_mask: torch.Tensor
    probed_step_mask: torch.Tensor
    probe_delta_prob: torch.Tensor
    correct_step_gate: torch.Tensor
    error_step_gate: torch.Tensor
    correct_token_multiplier: torch.Tensor
    error_token_multiplier: torch.Tensor
    probed_token_mask: torch.Tensor
    last_step_token_mask: torch.Tensor


@dataclass(frozen=True)
class R2OPLBatchResult:
    """Prompt-group difficulty and outcome masks independent of model logits."""

    correct_mask: torch.Tensor
    error_mask: torch.Tensor
    difficulty: torch.Tensor
    correctness: torch.Tensor
    group_success: torch.Tensor
    genuine_trajectory_mask: torch.Tensor
    metrics: dict[str, float]


def _coerce_token_ids(name: str, values: Sequence[int]) -> list[int]:
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"R2OPL {name} must be a sequence of integer token IDs.") from exc
    return result


def _step_range(step: TokenStep | Sequence[int], index: int) -> tuple[int, int]:
    """Accept a splitter ``TokenStep`` or a small ``(start, end)`` tuple."""

    if isinstance(step, TokenStep):
        if int(step.index) != index:
            raise ValueError(
                "R2OPL semantic steps must retain consecutive source indices; "
                f"expected {index}, got {step.index}."
            )
        return int(step.token_start), int(step.token_end)
    if hasattr(step, "token_start") and hasattr(step, "token_end"):
        source_index = getattr(step, "index", index)
        if int(source_index) != index:
            raise ValueError(
                "R2OPL semantic steps must retain consecutive source indices; "
                f"expected {index}, got {source_index}."
            )
        return int(step.token_start), int(step.token_end)
    try:
        start, end = step  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Each R2OPL semantic step must be a TokenStep or a (token_start, token_end) pair."
        ) from exc
    return int(start), int(end)


def _validate_semantic_ranges(
    ranges: Sequence[tuple[int, int]], *, response_length: int
) -> tuple[tuple[int, int], ...]:
    """Validate the source-preserving, response-local semantic partition.

    The conservative splitter can leave invisible trailing special tokens
    outside the visible semantic partition.  Those tokens are valid and use
    the standard signal, so the last range need not end at ``response_length``.
    """

    previous_end = 0
    normalized: list[tuple[int, int]] = []
    for index, raw_range in enumerate(ranges):
        start, end = int(raw_range[0]), int(raw_range[1])
        if not 0 <= start < end <= response_length:
            raise ValueError(
                "Invalid R2OPL semantic token interval "
                f"at step {index}: ({start}, {end}) for response length {response_length}."
            )
        if start != previous_end:
            raise ValueError(
                "R2OPL semantic steps must be contiguous from response token zero; "
                f"step {index} starts at {start}, expected {previous_end}."
            )
        normalized.append((start, end))
        previous_end = end
    return tuple(normalized)


def semantic_step_ranges_from_ends(
    semantic_step_ends: Sequence[int] | torch.Tensor,
    *,
    response_length: int,
    semantic_step_count: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Recover response-local ``(start, end)`` ranges from cumulative ends.

    Agent-loop and actor metadata use cumulative token ends because they are
    compact and easy to pad.  With no explicit ``semantic_step_count``, a
    zero-padded tail is accepted, but a non-zero value after the first zero is
    rejected rather than silently changing the semantic partition.
    """

    if isinstance(semantic_step_ends, torch.Tensor):
        if semantic_step_ends.ndim != 1:
            raise ValueError("R2OPL semantic_step_ends must be one-dimensional.")
        values = [int(value) for value in semantic_step_ends.detach().cpu().tolist()]
    else:
        values = [int(value) for value in semantic_step_ends]
    if semantic_step_count is not None:
        count = int(semantic_step_count)
        if not 0 <= count <= len(values):
            raise ValueError(
                "R2OPL semantic_step_count must lie between zero and the number of supplied ends."
            )
        values = values[:count]
    elif 0 in values:
        first_padding = values.index(0)
        if any(value != 0 for value in values[first_padding:]):
            raise ValueError(
                "R2OPL zero-padded semantic_step_ends cannot contain a non-zero value later."
            )
        values = values[:first_padding]
    if any(value <= 0 for value in values):
        raise ValueError("R2OPL semantic step ends must be positive before padding.")
    starts = [0, *values[:-1]]
    return _validate_semantic_ranges(
        list(zip(starts, values, strict=True)), response_length=response_length
    )


def build_r2opl_probe_layout(
    *,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    semantic_steps: Sequence[TokenStep | Sequence[int]],
    probe: AnswerProbe,
) -> R2OPLProbeLayout:
    """Build one branch-isolated packed Student answer-probe layout.

    Let the response contain ``K`` semantic steps.  For ``K >= 2`` the result
    appends exactly ``K`` copies of ``probe`` after the unmodified original
    sequence: a prompt-only prior ``P_0`` and one branch after each of steps
    ``0 .. K-2``.  The final semantic step receives no probe.  For zero or one
    semantic step no probe is appended because no delta can be used.
    """

    prompt = _coerce_token_ids("prompt_ids", prompt_ids)
    response = _coerce_token_ids("response_ids", response_ids)
    if not prompt:
        raise ValueError("R2OPL packed probes require at least one prompt token.")
    ranges = _validate_semantic_ranges(
        [_step_range(step, index) for index, step in enumerate(semantic_steps)],
        response_length=len(response),
    )
    # A zero/one-step response has no usable delta, so metadata producers are
    # intentionally allowed to transmit empty probe fields.  Requiring a gold
    # answer in this case would both waste a forward and violate the terminal
    # step's standard-signal rule.
    requires_probe = len(ranges) >= 2
    if requires_probe and not probe.token_ids:
        raise ValueError("R2OPL packed probes require a non-empty answer probe.")
    if requires_probe and not probe.answer_token_positions:
        raise ValueError("R2OPL packed probes require at least one answer token position.")

    probe_ids = _coerce_token_ids("probe.token_ids", probe.token_ids)
    local_answer_positions = tuple(int(position) for position in probe.answer_token_positions)
    if requires_probe and any(
        position <= 0 or position >= len(probe_ids) for position in local_answer_positions
    ):
        raise ValueError(
            "Every R2OPL answer token must have its causal predictor inside the "
            "appended answer-probe branch."
        )
    original_ids = [*prompt, *response]
    sequence_ids = list(original_ids)
    position_ids = list(range(len(original_ids)))
    branches: list[R2OPLProbeBranch] = []

    # P_0 plus the values after every *non-final* semantic step.  No branch is
    # useful when there is no non-final step, hence the K <= 1 special case.
    if requires_probe:
        visible_response_ends = [0, *[end for _start, end in ranges[:-1]]]
        for index, response_end in enumerate(visible_response_ends):
            visible_prefix_end = len(prompt) + response_end
            probe_start = len(sequence_ids)
            probe_end = probe_start + len(probe_ids)
            sequence_ids.extend(probe_ids)
            position_ids.extend(range(visible_prefix_end, visible_prefix_end + len(probe_ids)))
            branches.append(
                R2OPLProbeBranch(
                    index=index,
                    after_step_index=None if index == 0 else index - 1,
                    visible_prefix_end=visible_prefix_end,
                    probe_start=probe_start,
                    probe_end=probe_end,
                    answer_token_positions=tuple(
                        probe_start + position for position in local_answer_positions
                    ),
                )
            )

    layout = R2OPLProbeLayout(
        sequence_ids=tuple(sequence_ids),
        position_ids=tuple(position_ids),
        original_sequence_length=len(original_ids),
        prompt_length=len(prompt),
        response_length=len(response),
        semantic_step_ranges=ranges,
        probe_token_ids=tuple(probe_ids),
        branches=tuple(branches),
    )
    validate_r2opl_probe_layout(layout)
    return layout


def build_r2opl_probe_layout_from_step_ends(
    *,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    semantic_step_ends: Sequence[int] | torch.Tensor,
    probe_token_ids: Sequence[int],
    probe_answer_positions: Sequence[int],
    semantic_step_count: int | None = None,
) -> R2OPLProbeLayout:
    """Build a layout directly from actor-friendly step/probe metadata.

    This avoids constructing a :class:`TokenStep` or tokenizer-dependent
    :class:`AnswerProbe` inside FSDP.  ``probe_answer_positions`` are local
    offsets in ``probe_token_ids`` (not absolute packed-sequence positions).
    """

    response = _coerce_token_ids("response_ids", response_ids)
    ranges = semantic_step_ranges_from_ends(
        semantic_step_ends,
        response_length=len(response),
        semantic_step_count=semantic_step_count,
    )
    # ``build_r2opl_probe_layout`` only requires these three AnswerProbe
    # fields.  The text/character fields are deliberately irrelevant once the
    # agent loop has already tokenized and aligned the answer probe.
    metadata_probe = AnswerProbe(
        text="",
        token_ids=_coerce_token_ids("probe_token_ids", probe_token_ids),
        answer_token_positions=[int(position) for position in probe_answer_positions],
        answer_char_start=0,
        answer_char_end=0,
    )
    return build_r2opl_probe_layout(
        prompt_ids=prompt_ids,
        response_ids=response,
        semantic_steps=ranges,
        probe=metadata_probe,
    )


def validate_r2opl_probe_layout(layout: R2OPLProbeLayout) -> None:
    """Fail fast if a packed layout can leak suffixes or change positions."""

    sequence_length = len(layout.sequence_ids)
    if sequence_length != len(layout.position_ids):
        raise ValueError("R2OPL sequence IDs and position IDs must have identical length.")
    if layout.prompt_length <= 0:
        raise ValueError("R2OPL packed layouts require a non-empty prompt.")
    if layout.response_length < 0:
        raise ValueError("R2OPL response length cannot be negative.")
    if layout.original_sequence_length != layout.prompt_length + layout.response_length:
        raise ValueError("R2OPL original sequence length must equal prompt plus response length.")
    if not 1 <= layout.original_sequence_length <= sequence_length:
        raise ValueError("R2OPL original sequence length is invalid.")
    if tuple(layout.position_ids[: layout.original_sequence_length]) != tuple(
        range(layout.original_sequence_length)
    ):
        raise ValueError("R2OPL must preserve original prompt/response position IDs exactly.")

    ranges = _validate_semantic_ranges(
        layout.semantic_step_ranges,
        response_length=layout.response_length,
    )
    if tuple(layout.semantic_step_ranges) != ranges:
        raise ValueError("R2OPL semantic step ranges must be integer contiguous ranges.")
    probe_ids = tuple(int(token_id) for token_id in layout.probe_token_ids)
    if layout.branches and not probe_ids:
        raise ValueError("R2OPL branches require non-empty probe token IDs.")

    expected_count = len(ranges) if len(ranges) >= 2 else 0
    if len(layout.branches) != expected_count:
        raise ValueError(
            "R2OPL must include P_0 and one probe after every non-final step; "
            f"got {len(layout.branches)} branches for {len(ranges)} semantic steps."
        )

    previous_probe_end = layout.original_sequence_length
    for expected_index, branch in enumerate(layout.branches):
        expected_after_step = None if expected_index == 0 else expected_index - 1
        expected_response_end = 0 if expected_index == 0 else ranges[expected_index - 1][1]
        expected_visible_prefix_end = layout.prompt_length + expected_response_end
        if branch.index != expected_index:
            raise ValueError("R2OPL probe branch indices must be consecutive from zero.")
        if branch.after_step_index != expected_after_step:
            raise ValueError(
                f"R2OPL branch {branch.index} has after_step_index="
                f"{branch.after_step_index}, expected {expected_after_step}."
            )
        if branch.visible_prefix_end != expected_visible_prefix_end:
            raise ValueError(
                f"R2OPL branch {branch.index} has incorrect visible prefix length."
            )
        if branch.probe_start != previous_probe_end:
            raise ValueError("R2OPL probe branches must be physically contiguous.")
        if not (
            1
            <= branch.visible_prefix_end
            <= layout.original_sequence_length
            <= branch.probe_start
            < branch.probe_end
            <= sequence_length
        ):
            raise ValueError(f"Invalid R2OPL probe branch: {branch}.")
        if branch.probe_end - branch.probe_start != len(probe_ids):
            raise ValueError(f"R2OPL branch {branch.index} has the wrong probe length.")
        if tuple(layout.sequence_ids[branch.probe_start : branch.probe_end]) != probe_ids:
            raise ValueError(f"R2OPL branch {branch.index} does not contain the shared probe IDs.")
        expected_positions = tuple(
            range(branch.visible_prefix_end, branch.visible_prefix_end + len(probe_ids))
        )
        if tuple(layout.position_ids[branch.probe_start : branch.probe_end]) != expected_positions:
            raise ValueError(f"R2OPL branch {branch.index} has incorrect logical position IDs.")
        if not branch.answer_token_positions:
            raise ValueError(f"R2OPL branch {branch.index} has no answer-token positions.")
        local_positions = tuple(position - branch.probe_start for position in branch.answer_token_positions)
        if any(position <= 0 or position >= len(probe_ids) for position in local_positions):
            raise ValueError(
                f"R2OPL branch {branch.index} has answer tokens without an in-branch predictor."
            )
        previous_probe_end = branch.probe_end
    if previous_probe_end != sequence_length:
        raise ValueError("R2OPL packed layout has unassigned trailing tokens.")


def build_dense_attention_mask(
    layout: R2OPLProbeLayout,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the exact ``[query, key]`` branch-causal visibility mask.

    A ``True`` entry means the query can attend to the key, matching PyTorch
    SDPA's boolean-mask convention.  Original tokens remain ordinary causal
    attention.  A probe branch sees only its logical rollout prefix and its
    own causal probe history, never a response suffix or another branch.
    """

    validate_r2opl_probe_layout(layout)
    sequence_length = len(layout.sequence_ids)
    mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool, device=device)
    original_length = layout.original_sequence_length
    mask[:original_length, :original_length] = torch.ones(
        (original_length, original_length), dtype=torch.bool, device=device
    ).tril()
    for branch in layout.branches:
        mask[branch.probe_start : branch.probe_end, : branch.visible_prefix_end] = True
        branch_length = branch.probe_end - branch.probe_start
        mask[
            branch.probe_start : branch.probe_end,
            branch.probe_start : branch.probe_end,
        ] = torch.ones((branch_length, branch_length), dtype=torch.bool, device=device).tril()
    return mask


def r2opl_bool_mask_to_additive(
    attention_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    add_head_dimension: bool = True,
) -> torch.Tensor:
    """Convert an R2OPL boolean visibility mask to a Transformer 4-D mask.

    The returned values are zero for allowed locations and ``finfo.min`` for
    blocked locations.  ``attention_mask`` may be ``[L, L]`` or ``[B, L, L]``.
    """

    if attention_mask.dtype != torch.bool or attention_mask.ndim not in (2, 3):
        raise ValueError("R2OPL attention_mask must be a boolean [L,L] or [B,L,L] tensor.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("R2OPL additive attention masks require a floating-point dtype.")
    additive = torch.full(
        attention_mask.shape,
        torch.finfo(dtype).min,
        dtype=dtype,
        device=attention_mask.device,
    )
    additive.masked_fill_(attention_mask, 0.0)
    if not add_head_dimension:
        return additive
    if attention_mask.ndim == 2:
        return additive.unsqueeze(0).unsqueeze(0)
    return additive.unsqueeze(1)


def build_r2opl_additive_attention_mask(
    layout: R2OPLProbeLayout,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
    add_head_dimension: bool = True,
) -> torch.Tensor:
    """Build a model-ready additive attention mask for one packed layout."""

    return r2opl_bool_mask_to_additive(
        build_dense_attention_mask(layout, device=device),
        dtype=dtype,
        add_head_dimension=add_head_dimension,
    )


def build_r2opl_probe_batch(
    layouts: Sequence[R2OPLProbeLayout],
    *,
    pad_token_id: int = 0,
    device: torch.device | str | None = None,
) -> R2OPLProbeBatch:
    """Pad layouts and their branch masks for one Student packed forward."""

    if not layouts:
        raise ValueError("R2OPL probe batching requires at least one layout.")
    layouts = tuple(layouts)
    for layout in layouts:
        validate_r2opl_probe_layout(layout)

    batch_size = len(layouts)
    max_length = max(len(layout.sequence_ids) for layout in layouts)
    max_steps = max((layout.num_semantic_steps for layout in layouts), default=0)
    max_branches = max((len(layout.branches) for layout in layouts), default=0)
    input_ids = torch.full(
        (batch_size, max_length), int(pad_token_id), dtype=torch.long, device=device
    )
    position_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    attention_mask = torch.zeros(
        (batch_size, max_length, max_length), dtype=torch.bool, device=device
    )
    sequence_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    original_lengths = torch.empty(batch_size, dtype=torch.long, device=device)
    response_lengths = torch.empty(batch_size, dtype=torch.long, device=device)
    step_ranges = torch.zeros((batch_size, max_steps, 2), dtype=torch.long, device=device)
    step_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool, device=device)
    branch_mask = torch.zeros((batch_size, max_branches), dtype=torch.bool, device=device)

    for row, layout in enumerate(layouts):
        length = len(layout.sequence_ids)
        input_ids[row, :length] = torch.tensor(layout.sequence_ids, dtype=torch.long, device=device)
        position_ids[row, :length] = torch.tensor(layout.position_ids, dtype=torch.long, device=device)
        attention_mask[row, :length, :length] = build_dense_attention_mask(layout, device=device)
        sequence_mask[row, :length] = True
        original_lengths[row] = layout.original_sequence_length
        response_lengths[row] = layout.response_length
        if layout.semantic_step_ranges:
            count = len(layout.semantic_step_ranges)
            step_ranges[row, :count] = torch.tensor(
                layout.semantic_step_ranges, dtype=torch.long, device=device
            )
            step_mask[row, :count] = True
        if layout.branches:
            branch_mask[row, : len(layout.branches)] = True

    return R2OPLProbeBatch(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        sequence_mask=sequence_mask,
        original_sequence_lengths=original_lengths,
        response_lengths=response_lengths,
        semantic_step_ranges=step_ranges,
        semantic_step_mask=step_mask,
        probe_branch_mask=branch_mask,
        layouts=layouts,
    )


def compute_r2opl_probe_probabilities(
    logits: torch.Tensor,
    layout: R2OPLProbeLayout,
) -> torch.Tensor:
    """Return one mean gold-token probability for every probe branch.

    ``logits[p - 1]`` predicts ``sequence_ids[p]`` under the standard causal
    LM convention.  Computation is promoted to float32 for stable probabilities
    when the model forward uses bf16/fp16.
    """

    validate_r2opl_probe_layout(layout)
    if logits.ndim != 2:
        raise ValueError("R2OPL probe logits must have shape [sequence_length, vocabulary].")
    if logits.shape[0] < len(layout.sequence_ids):
        raise ValueError(
            "R2OPL probe logits are shorter than the packed sequence: "
            f"{logits.shape[0]} < {len(layout.sequence_ids)}."
        )
    if not logits.is_floating_point():
        raise ValueError("R2OPL probe logits must be floating point.")
    if not layout.branches:
        return logits.new_empty((0,), dtype=torch.float32)

    scores: list[torch.Tensor] = []
    for branch in layout.branches:
        answer_positions = torch.tensor(
            branch.answer_token_positions, dtype=torch.long, device=logits.device
        )
        target_ids = torch.tensor(
            [layout.sequence_ids[position] for position in branch.answer_token_positions],
            dtype=torch.long,
            device=logits.device,
        )
        predictor_logits = logits.index_select(0, answer_positions - 1).float()
        probabilities = torch.softmax(predictor_logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        scores.append(probabilities.mean())
    return torch.stack(scores)


def compute_r2opl_packed_probe_probabilities(
    packed_logits: torch.Tensor,
    layouts: Sequence[R2OPLProbeLayout],
    *,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a padded batch of packed forwards.

    Returns ``(probabilities, branch_mask)`` with shape ``[batch, max_branches]``.
    Padding values are never interpreted as probe probabilities; use the mask.
    """

    if packed_logits.ndim != 3:
        raise ValueError("R2OPL packed logits must have shape [batch, sequence_length, vocabulary].")
    if packed_logits.shape[0] != len(layouts):
        raise ValueError("R2OPL packed logits and layouts must have the same batch size.")
    if not packed_logits.is_floating_point():
        raise ValueError("R2OPL packed logits must be floating point.")
    layouts = tuple(layouts)
    rows = [compute_r2opl_probe_probabilities(packed_logits[index], layout) for index, layout in enumerate(layouts)]
    width = max((row.numel() for row in rows), default=0)
    if width == 0:
        probabilities = packed_logits.new_empty((len(rows), 0), dtype=torch.float32)
        branch_mask = torch.zeros((len(rows), 0), dtype=torch.bool, device=packed_logits.device)
        return probabilities, branch_mask
    probabilities = torch.stack(
        [F.pad(row, (0, width - row.numel()), value=float(pad_value)) for row in rows]
    )
    branch_mask = torch.zeros((len(rows), width), dtype=torch.bool, device=packed_logits.device)
    for row_index, row in enumerate(rows):
        branch_mask[row_index, : row.numel()] = True
    return probabilities, branch_mask


def layout_rpc_payload(layout: R2OPLProbeLayout) -> dict[str, Any]:
    """Convert a layout to primitive, Ray-serializable metadata."""

    validate_r2opl_probe_layout(layout)
    return {
        "sequence_ids": list(layout.sequence_ids),
        "position_ids": list(layout.position_ids),
        "original_sequence_length": int(layout.original_sequence_length),
        "prompt_length": int(layout.prompt_length),
        "response_length": int(layout.response_length),
        "semantic_step_ranges": [list(item) for item in layout.semantic_step_ranges],
        "probe_token_ids": list(layout.probe_token_ids),
        "branches": [
            {
                "index": int(branch.index),
                "after_step_index": (
                    None if branch.after_step_index is None else int(branch.after_step_index)
                ),
                "visible_prefix_end": int(branch.visible_prefix_end),
                "probe_start": int(branch.probe_start),
                "probe_end": int(branch.probe_end),
                "answer_token_positions": list(branch.answer_token_positions),
            }
            for branch in layout.branches
        ],
    }


def _validate_hyperparameters(
    *,
    lambda_: float | None = None,
    miu: float | None = None,
    alpha_r: float | None = None,
    alpha_d: float | None = None,
    epsilon: float | None = None,
) -> None:
    if lambda_ is not None and (not math.isfinite(float(lambda_)) or float(lambda_) < 0.0):
        raise ValueError(f"R2OPL lambda must be finite and non-negative, got {lambda_}.")
    if miu is not None and (not math.isfinite(float(miu)) or float(miu) <= 0.0):
        raise ValueError(f"R2OPL miu must be finite and positive, got {miu}.")
    if alpha_r is not None and (not math.isfinite(float(alpha_r)) or float(alpha_r) < 0.0):
        raise ValueError(f"R2OPL alpha_R must be finite and non-negative, got {alpha_r}.")
    if alpha_d is not None and (not math.isfinite(float(alpha_d)) or float(alpha_d) < 0.0):
        raise ValueError(f"R2OPL alpha_D must be finite and non-negative, got {alpha_d}.")
    if epsilon is not None and (
        not math.isfinite(float(epsilon)) or not 0.0 <= float(epsilon) <= 1.0
    ):
        raise ValueError(f"R2OPL epsilon must lie in [0, 1], got {epsilon}.")


def _prefix_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    """Validate a left-aligned boolean padded mask and return row counts."""

    if mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError(f"R2OPL {name} must be a two-dimensional boolean tensor.")
    counts = mask.sum(dim=-1)
    expected = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < counts.unsqueeze(1)
    if not bool(torch.equal(mask, expected)):
        raise ValueError(f"R2OPL {name} must contain a left-aligned true prefix per row.")
    return counts


def compute_r2opl_token_modulation(
    probe_probabilities: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    semantic_step_ranges: torch.Tensor,
    semantic_step_mask: torch.Tensor | None = None,
    probe_branch_mask: torch.Tensor | None = None,
    alpha_r: float = R2OPL_DEFAULT_ALPHA_R,
    alpha_d: float = R2OPL_DEFAULT_ALPHA_D,
    epsilon: float = R2OPL_DEFAULT_EPSILON,
) -> R2OPLTokenModulation:
    """Map packed probe probabilities to correct/error token multipliers.

    For a probed step ``k`` the gates are

    ``max(epsilon, 1 + alpha_R * (P[k+1] - P[k]))`` and
    ``max(0, 1 - alpha_D * (P[k+1] - P[k]))``.

    The final semantic step and all response tokens not owned by a visible
    semantic step retain exact multiplier ``1``.  This is also true for a
    zero/one-step response, for which the packed-probe branch mask is empty.
    """

    _validate_hyperparameters(alpha_r=alpha_r, alpha_d=alpha_d, epsilon=epsilon)
    if not probe_probabilities.is_floating_point() or probe_probabilities.ndim != 2:
        raise ValueError("R2OPL probe_probabilities must be a floating [batch, branches] tensor.")
    if response_mask.ndim != 2:
        raise ValueError("R2OPL response_mask must have shape [batch, response_length].")
    if semantic_step_ranges.ndim != 3 or semantic_step_ranges.shape[-1] != 2:
        raise ValueError("R2OPL semantic_step_ranges must have shape [batch, steps, 2].")
    batch_size, response_width = response_mask.shape
    if (
        probe_probabilities.shape[0] != batch_size
        or semantic_step_ranges.shape[0] != batch_size
    ):
        raise ValueError("R2OPL probe, response, and semantic-step tensors must share a batch size.")
    if probe_probabilities.device != response_mask.device or semantic_step_ranges.device != response_mask.device:
        raise ValueError("R2OPL probe, response, and semantic-step tensors must share a device.")
    response_mask = response_mask.bool()
    step_count_width = semantic_step_ranges.shape[1]
    if semantic_step_mask is None:
        # Ranges are produced by ``build_r2opl_probe_batch`` with zero padding.
        semantic_step_mask = semantic_step_ranges[..., 1].gt(semantic_step_ranges[..., 0])
    if semantic_step_mask.shape != semantic_step_ranges.shape[:2]:
        raise ValueError("R2OPL semantic_step_mask must have shape [batch, steps].")
    semantic_step_mask = semantic_step_mask.to(device=response_mask.device, dtype=torch.bool)
    step_counts = _prefix_mask(semantic_step_mask, name="semantic_step_mask")

    expected_branch_counts = torch.where(step_counts.ge(2), step_counts, torch.zeros_like(step_counts))
    branch_width = probe_probabilities.shape[1]
    expected_branch_mask = (
        torch.arange(branch_width, device=response_mask.device).unsqueeze(0)
        < expected_branch_counts.unsqueeze(1)
    )
    if probe_branch_mask is None:
        probe_branch_mask = expected_branch_mask
    if probe_branch_mask.shape != probe_probabilities.shape:
        raise ValueError("R2OPL probe_branch_mask must have shape [batch, branches].")
    probe_branch_mask = probe_branch_mask.to(device=response_mask.device, dtype=torch.bool)
    if not bool(torch.equal(probe_branch_mask, expected_branch_mask)):
        raise ValueError(
            "R2OPL probe branches must be a left-aligned P_0..P_(K-1) prefix "
            "for every response with K >= 2 semantic steps."
        )
    if bool((expected_branch_counts > branch_width).any()):
        raise ValueError("R2OPL probe_probabilities do not include every required branch.")
    if not bool(torch.isfinite(probe_probabilities[probe_branch_mask]).all()):
        raise ValueError("R2OPL active probe probabilities must be finite.")
    if bool(
        ((probe_probabilities[probe_branch_mask] < 0.0) | (probe_probabilities[probe_branch_mask] > 1.0)).any()
    ):
        raise ValueError("R2OPL active probe probabilities must lie in [0, 1].")

    # Enforce source-preserving response-local semantic ranges.  The loop is
    # intentionally tiny (semantic reasoning steps, not generated tokens) and
    # gives clear errors for malformed rollout metadata.
    probed_token_mask = torch.zeros_like(response_mask)
    last_step_token_mask = torch.zeros_like(response_mask)
    for row in range(batch_size):
        previous_end = 0
        count = int(step_counts[row].item())
        for step_index in range(count):
            start = int(semantic_step_ranges[row, step_index, 0].item())
            end = int(semantic_step_ranges[row, step_index, 1].item())
            if not 0 <= start < end <= response_width:
                raise ValueError(
                    f"Invalid R2OPL semantic range ({start}, {end}) in batch row {row}."
                )
            if start != previous_end:
                raise ValueError(
                    "R2OPL semantic ranges must be contiguous from response token zero; "
                    f"row {row}, step {step_index} starts at {start}, expected {previous_end}."
                )
            if not bool(response_mask[row, start:end].all()):
                raise ValueError(
                    f"R2OPL semantic range ({start}, {end}) includes a masked response token."
                )
            if step_index == count - 1:
                last_step_token_mask[row, start:end] = True
            else:
                probed_token_mask[row, start:end] = True
            previous_end = end

    step_positions = torch.arange(step_count_width, device=response_mask.device).unsqueeze(0)
    probed_step_mask = step_positions < (step_counts - 1).clamp_min(0).unsqueeze(1)
    safe_probabilities = torch.where(
        probe_branch_mask, probe_probabilities, torch.zeros_like(probe_probabilities)
    )
    probe_delta_prob = torch.zeros(
        (batch_size, step_count_width),
        dtype=probe_probabilities.dtype,
        device=response_mask.device,
    )
    if step_count_width and branch_width >= 2:
        usable_width = min(step_count_width, branch_width - 1)
        raw_deltas = safe_probabilities[:, 1 : usable_width + 1] - safe_probabilities[:, :usable_width]
        probe_delta_prob[:, :usable_width] = torch.where(
            probed_step_mask[:, :usable_width], raw_deltas, torch.zeros_like(raw_deltas)
        )

    correct_step_gate = torch.ones_like(probe_delta_prob)
    error_step_gate = torch.ones_like(probe_delta_prob)
    if step_count_width:
        correct_values = (1.0 + float(alpha_r) * probe_delta_prob).clamp_min(float(epsilon))
        error_values = (1.0 - float(alpha_d) * probe_delta_prob).clamp_min(0.0)
        correct_step_gate = torch.where(probed_step_mask, correct_values, correct_step_gate)
        error_step_gate = torch.where(probed_step_mask, error_values, error_step_gate)

    correct_token_multiplier = torch.ones(
        (batch_size, response_width), dtype=probe_probabilities.dtype, device=response_mask.device
    )
    error_token_multiplier = torch.ones_like(correct_token_multiplier)
    for row in range(batch_size):
        for step_index in range(max(0, int(step_counts[row].item()) - 1)):
            start = int(semantic_step_ranges[row, step_index, 0].item())
            end = int(semantic_step_ranges[row, step_index, 1].item())
            correct_token_multiplier[row, start:end] = correct_step_gate[row, step_index]
            error_token_multiplier[row, start:end] = error_step_gate[row, step_index]

    return R2OPLTokenModulation(
        probe_probabilities=probe_probabilities,
        probe_branch_mask=probe_branch_mask,
        semantic_step_mask=semantic_step_mask,
        probed_step_mask=probed_step_mask,
        probe_delta_prob=probe_delta_prob,
        correct_step_gate=correct_step_gate,
        error_step_gate=error_step_gate,
        correct_token_multiplier=correct_token_multiplier,
        error_token_multiplier=error_token_multiplier,
        probed_token_mask=probed_token_mask,
        last_step_token_mask=last_step_token_mask,
    )


def _ordered_groups(prompt_group_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(group_id) for group_id in prompt_group_ids))


def _sequence_rewards(verifier_rewards: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if verifier_rewards.ndim == 2:
        if verifier_rewards.shape[0] != batch_size:
            raise ValueError("R2OPL verifier rewards must share the response batch size.")
        rewards = verifier_rewards.float().sum(dim=-1)
    elif verifier_rewards.ndim == 1:
        rewards = verifier_rewards.float()
    else:
        raise ValueError("R2OPL verifier_rewards must be one- or two-dimensional.")
    if rewards.shape[0] != batch_size:
        raise ValueError("R2OPL verifier rewards must share the response batch size.")
    return rewards


def compute_r2opl_batch(
    response_mask: torch.Tensor,
    verifier_rewards: torch.Tensor,
    prompt_group_ids: Sequence[str],
    *,
    genuine_trajectory_mask: torch.Tensor | None = None,
) -> R2OPLBatchResult:
    """Compute binary outcomes and prompt-level difficulty ``1 - mean(R)``."""

    if response_mask.ndim != 2:
        raise ValueError("R2OPL response_mask must be two-dimensional.")
    response_mask = response_mask.bool()
    batch_size = response_mask.shape[0]
    if len(prompt_group_ids) != batch_size:
        raise ValueError("R2OPL prompt_group_ids must have one value per trajectory.")
    if genuine_trajectory_mask is None:
        genuine_trajectory_mask = torch.ones(
            batch_size, dtype=torch.bool, device=response_mask.device
        )
    elif genuine_trajectory_mask.ndim != 1 or genuine_trajectory_mask.shape[0] != batch_size:
        raise ValueError("R2OPL genuine_trajectory_mask must have one value per trajectory.")
    else:
        genuine_trajectory_mask = genuine_trajectory_mask.to(
            device=response_mask.device, dtype=torch.bool
        )
    token_counts = (response_mask & genuine_trajectory_mask.unsqueeze(-1)).sum(dim=-1)
    if bool((genuine_trajectory_mask & token_counts.eq(0)).any()):
        rows = (genuine_trajectory_mask & token_counts.eq(0)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "R2OPL cannot optimize a genuine trajectory with no response tokens; "
            f"empty rows: {rows[:5]}."
        )
    rewards = _sequence_rewards(verifier_rewards, batch_size=batch_size).to(response_mask.device)
    if not bool(torch.isfinite(rewards[genuine_trajectory_mask]).all()):
        raise ValueError("R2OPL verifier rewards must be finite.")
    binary = rewards.eq(0.0) | rewards.eq(1.0)
    if not bool(binary[genuine_trajectory_mask].all()):
        invalid = rewards[genuine_trajectory_mask & ~binary].detach().cpu().tolist()
        raise ValueError(f"R2OPL requires binary verifier rewards in {{0,1}}, got {invalid[:5]}.")

    correctness = rewards.eq(1.0) & genuine_trajectory_mask
    difficulty = torch.zeros(batch_size, dtype=torch.float32, device=response_mask.device)
    group_success: list[torch.Tensor] = []
    for group_id in _ordered_groups(prompt_group_ids):
        indices = [
            index
            for index, current_id in enumerate(prompt_group_ids)
            if str(current_id) == group_id and bool(genuine_trajectory_mask[index])
        ]
        if not indices:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=response_mask.device)
        success = correctness[index_tensor].float().mean()
        difficulty[index_tensor] = 1.0 - success
        group_success.append(success)
    if not group_success:
        raise ValueError("R2OPL requires at least one genuine prompt group.")

    correct_mask = response_mask & correctness.unsqueeze(-1)
    error_mask = response_mask & genuine_trajectory_mask.unsqueeze(-1) & ~correctness.unsqueeze(-1)
    group_success_tensor = torch.stack(group_success)
    genuine_rewards = correctness[genuine_trajectory_mask].float()
    genuine_count = int(genuine_trajectory_mask.sum().item())
    metrics = {
        "r2opl/train/mean_score": float(genuine_rewards.mean().item()),
        "r2opl/train/group_success_rate": float(group_success_tensor.mean().item()),
        "r2opl/train/prompt_difficulty_mean": float((1.0 - group_success_tensor).mean().item()),
        "r2opl/train/correct_trajectory_count": float(correctness.sum().item()),
        "r2opl/train/error_trajectory_count": float((genuine_trajectory_mask & ~correctness).sum().item()),
        "r2opl/train/correct_trajectory_ratio": float(genuine_rewards.mean().item()),
        "r2opl/train/error_trajectory_ratio": float((~correctness[genuine_trajectory_mask]).float().mean().item()),
        "r2opl/train/correct_token_count": float(correct_mask.sum().item()),
        "r2opl/train/error_token_count": float(error_mask.sum().item()),
        "r2opl/train/genuine_trajectory_count": float(genuine_count),
    }
    return R2OPLBatchResult(
        correct_mask=correct_mask,
        error_mask=error_mask,
        difficulty=difficulty,
        correctness=correctness,
        group_success=group_success_tensor,
        genuine_trajectory_mask=genuine_trajectory_mask,
        metrics=metrics,
    )


def _difficulty_tokens(difficulty: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if difficulty.ndim == 1 and difficulty.shape[0] == shape[0]:
        return difficulty.float().unsqueeze(-1).expand(shape)
    if difficulty.ndim == 2 and tuple(difficulty.shape) == tuple(shape):
        return difficulty.float()
    if difficulty.ndim == 2 and difficulty.shape == (shape[0], 1):
        return difficulty.float().expand(shape)
    raise ValueError(
        "R2OPL difficulty must have shape [batch], [batch,1], or [batch,response_length]."
    )


def _multiplier_or_one(
    value: torch.Tensor | None,
    *,
    shape: torch.Size,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.float32, device=device)
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"R2OPL {name} must have shape {tuple(shape)}, got {tuple(value.shape)}.")
    if value.device != device or not value.is_floating_point():
        raise ValueError(f"R2OPL {name} must be a floating tensor on the response device.")
    return value.float()


def r2opl_token_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    correct_mask: torch.Tensor,
    error_mask: torch.Tensor,
    difficulty: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    correct_multiplier: torch.Tensor | None = None,
    error_multiplier: torch.Tensor | None = None,
    lambda_: float = R2OPL_DEFAULT_LAMBDA,
    miu: float = R2OPL_DEFAULT_MIU,
) -> torch.Tensor:
    """Construct R²OPL's detached-ready sampled-token advantage.

    Correct tokens use ``miu * d_x * max(epsilon, 1 + alpha_R * delta)``;
    incorrect tokens use ``lambda * d_x * (log pi_T - log pi_S) *
    max(0, 1 - alpha_D * delta)``.  Passing a multiplier of one yields the
    standard RL/OPD signal, which is exactly how the final semantic step is
    represented.
    """

    _validate_hyperparameters(lambda_=lambda_, miu=miu)
    tensors = {
        "student_log_probs": student_log_probs,
        "teacher_log_probs": teacher_log_probs,
        "correct_mask": correct_mask,
        "error_mask": error_mask,
        "response_mask": response_mask,
    }
    if any(value.ndim != 2 for value in tensors.values()):
        raise ValueError("R2OPL token tensors must all be two-dimensional.")
    shapes = {tuple(value.shape) for value in tensors.values()}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={tuple(value.shape)}" for name, value in tensors.items())
        raise ValueError(f"R2OPL token tensors must share one shape; got {details}.")
    if student_log_probs.device != teacher_log_probs.device:
        raise ValueError("R2OPL Student and Teacher log-probabilities must share a device.")
    if response_mask.device != student_log_probs.device:
        raise ValueError("R2OPL response_mask must share the log-probability device.")
    if not student_log_probs.is_floating_point() or not teacher_log_probs.is_floating_point():
        raise ValueError("R2OPL Student and Teacher log-probabilities must be floating point.")

    response_mask = response_mask.bool()
    correct_mask = correct_mask.to(device=response_mask.device, dtype=torch.bool)
    error_mask = error_mask.to(device=response_mask.device, dtype=torch.bool)
    if bool((correct_mask & error_mask).any()):
        raise ValueError("R2OPL correct and error masks must be disjoint.")
    if bool(((correct_mask | error_mask) & ~response_mask).any()):
        raise ValueError("R2OPL correct/error masks must be subsets of response_mask.")
    if not bool(torch.isfinite(student_log_probs[response_mask]).all()):
        raise ValueError("R2OPL Student log-probabilities must be finite on response tokens.")
    if not bool(torch.isfinite(teacher_log_probs[response_mask]).all()):
        raise ValueError("R2OPL Teacher log-probabilities must be finite on response tokens.")

    shape = student_log_probs.shape
    difficulty_tokens = _difficulty_tokens(difficulty.to(student_log_probs.device), shape)
    correct_multiplier = _multiplier_or_one(
        correct_multiplier,
        shape=shape,
        device=student_log_probs.device,
        name="correct_multiplier",
    )
    error_multiplier = _multiplier_or_one(
        error_multiplier,
        shape=shape,
        device=student_log_probs.device,
        name="error_multiplier",
    )
    if not bool(torch.isfinite(correct_multiplier[correct_mask]).all()):
        raise ValueError("R2OPL correct multipliers must be finite on correct tokens.")
    if not bool(torch.isfinite(error_multiplier[error_mask]).all()):
        raise ValueError("R2OPL error multipliers must be finite on error tokens.")

    teacher_minus_student = teacher_log_probs.float() - student_log_probs.float()
    correct_advantage = difficulty_tokens * float(miu) * correct_multiplier
    error_advantage = (
        difficulty_tokens * float(lambda_) * teacher_minus_student * error_multiplier
    )
    advantage = torch.where(correct_mask, correct_advantage, torch.zeros_like(correct_advantage))
    advantage = torch.where(error_mask, error_advantage, advantage)
    return advantage * response_mask.to(advantage.dtype)


def r2opl_reinforce_loss(
    student_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """One-backward token-sum REINFORCE loss with ``stop_gradient(A)``.

    R²OPL's objective sums token contributions for each sampled trajectory,
    then averages trajectories.  In particular, this must not divide each
    rollout by its own response length: doing so would change the relative
    weight of otherwise identical short and long trajectories.
    """

    if (
        student_log_probs.ndim != 2
        or advantage.shape != student_log_probs.shape
        or response_mask.shape != student_log_probs.shape
    ):
        raise ValueError("R2OPL loss tensors must share shape [batch, response_length].")
    mask = response_mask.to(dtype=student_log_probs.dtype)
    token_loss = -(advantage.detach().to(mask.dtype) * student_log_probs) * mask
    per_sequence = token_loss.sum(dim=-1)
    active_sequences = mask.sum(dim=-1).gt(0)
    if not bool(active_sequences.any()):
        return token_loss.sum() * 0.0
    return per_sequence[active_sequences].mean()


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.float().sum().item()) if selected.numel() else 0.0


def r2opl_probe_metrics(
    *,
    modulation: R2OPLTokenModulation,
    correctness: torch.Tensor,
    response_mask: torch.Tensor,
    genuine_trajectory_mask: torch.Tensor | None = None,
    token_advantage: torch.Tensor | None = None,
    student_log_probs: torch.Tensor | None = None,
    teacher_log_probs: torch.Tensor | None = None,
    alpha_r: float = R2OPL_DEFAULT_ALPHA_R,
    alpha_d: float = R2OPL_DEFAULT_ALPHA_D,
    epsilon: float = R2OPL_DEFAULT_EPSILON,
    lambda_: float = R2OPL_DEFAULT_LAMBDA,
    miu: float = R2OPL_DEFAULT_MIU,
    prefix: str = "r2opl/train",
) -> dict[str, float]:
    """Produce separate correct/error probe diagnostics without grad norms.

    Returns pooled numerator/denominator statistics rather than local means.
    The trainer converts them into correct/error conditional means after the
    actor reducer has combined all micro-batches and DP ranks.  Step-level
    values use probed-step denominators; modulation uses probed-token
    denominators, so a final step's fixed multiplier cannot dilute either
    debugging signal.
    """

    _validate_hyperparameters(
        alpha_r=alpha_r,
        alpha_d=alpha_d,
        epsilon=epsilon,
        lambda_=lambda_,
        miu=miu,
    )
    if response_mask.ndim != 2 or correctness.ndim != 1:
        raise ValueError("R2OPL metrics require response_mask [batch,tokens] and correctness [batch].")
    batch_size = response_mask.shape[0]
    if correctness.shape[0] != batch_size:
        raise ValueError("R2OPL correctness must have one value per trajectory.")
    response_mask = response_mask.bool()
    correctness = correctness.to(device=response_mask.device, dtype=torch.bool)
    if genuine_trajectory_mask is None:
        genuine_trajectory_mask = torch.ones_like(correctness)
    if genuine_trajectory_mask.ndim != 1 or genuine_trajectory_mask.shape[0] != batch_size:
        raise ValueError("R2OPL genuine_trajectory_mask must have one value per trajectory.")
    genuine_trajectory_mask = genuine_trajectory_mask.to(
        device=response_mask.device, dtype=torch.bool
    )

    token_shape = response_mask.shape
    required_token_fields = (
        modulation.correct_token_multiplier,
        modulation.error_token_multiplier,
        modulation.probed_token_mask,
        modulation.last_step_token_mask,
    )
    if any(tuple(value.shape) != tuple(token_shape) for value in required_token_fields):
        raise ValueError("R2OPL modulation token fields must match response_mask.")
    step_shape = modulation.semantic_step_mask.shape
    if (
        modulation.probed_step_mask.shape != step_shape
        or modulation.probe_delta_prob.shape != step_shape
        or modulation.correct_step_gate.shape != step_shape
        or modulation.error_step_gate.shape != step_shape
    ):
        raise ValueError("R2OPL modulation step fields must share shape [batch, steps].")
    if step_shape[0] != batch_size or modulation.probe_branch_mask.shape[0] != batch_size:
        raise ValueError("R2OPL modulation fields must share the response batch size.")

    correct_rows = genuine_trajectory_mask & correctness
    error_rows = genuine_trajectory_mask & ~correctness
    correct_step_mask = modulation.probed_step_mask & correct_rows.unsqueeze(-1)
    error_step_mask = modulation.probed_step_mask & error_rows.unsqueeze(-1)
    correct_probe_tokens = (
        modulation.probed_token_mask & response_mask & correct_rows.unsqueeze(-1)
    )
    error_probe_tokens = (
        modulation.probed_token_mask & response_mask & error_rows.unsqueeze(-1)
    )
    correct_last_tokens = (
        modulation.last_step_token_mask & response_mask & correct_rows.unsqueeze(-1)
    )
    error_last_tokens = (
        modulation.last_step_token_mask & response_mask & error_rows.unsqueeze(-1)
    )
    standard_tokens = response_mask & ~modulation.probed_token_mask
    correct_standard_tokens = standard_tokens & correct_rows.unsqueeze(-1)
    error_standard_tokens = standard_tokens & error_rows.unsqueeze(-1)
    correct_branch_mask = modulation.probe_branch_mask & correct_rows.unsqueeze(-1)
    error_branch_mask = modulation.probe_branch_mask & error_rows.unsqueeze(-1)

    correct_response_tokens = response_mask & correct_rows.unsqueeze(-1)
    error_response_tokens = response_mask & error_rows.unsqueeze(-1)
    metrics: dict[str, float] = {
        # These count fields are sufficient statistics.  The actor reducer
        # aggregates them with SUM, then R2OPLTrainer materializes every
        # corresponding *_mean after all micro-batches/DP ranks are combined.
        # That prevents a missing outcome or a short rollout from distorting
        # the requested correct/error debugging averages.
        f"{prefix}/correct_probe_branch_count": float(correct_branch_mask.sum().item()),
        f"{prefix}/error_probe_branch_count": float(error_branch_mask.sum().item()),
        f"{prefix}/correct_probe_step_count": float(correct_step_mask.sum().item()),
        f"{prefix}/error_probe_step_count": float(error_step_mask.sum().item()),
        f"{prefix}/correct_probe_token_count": float(correct_probe_tokens.sum().item()),
        f"{prefix}/error_probe_token_count": float(error_probe_tokens.sum().item()),
        f"{prefix}/correct_last_step_standard_token_count": float(correct_last_tokens.sum().item()),
        f"{prefix}/error_last_step_standard_token_count": float(error_last_tokens.sum().item()),
        f"{prefix}/correct_standard_token_count": float(correct_standard_tokens.sum().item()),
        f"{prefix}/error_standard_token_count": float(error_standard_tokens.sum().item()),
        f"{prefix}/correct_response_token_count": float(correct_response_tokens.sum().item()),
        f"{prefix}/error_response_token_count": float(error_response_tokens.sum().item()),
        f"{prefix}/alpha_r": float(alpha_r),
        f"{prefix}/alpha_d": float(alpha_d),
        f"{prefix}/epsilon": float(epsilon),
        f"{prefix}/lambda": float(lambda_),
        f"{prefix}/miu": float(miu),
    }

    def add_masked_statistics(name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        """Record a numerator/denominator pair for one pooled dashboard mean."""

        metrics[f"{prefix}/{name}_sum"] = _masked_sum(values, mask)
        metrics[f"{prefix}/{name}_count"] = float(mask.sum().item())

    # Branch-level P values, step-level deltas/gates, and token-level
    # modulation deliberately use their own denominators.
    add_masked_statistics(
        "correct_probe_probability", modulation.probe_probabilities, correct_branch_mask
    )
    add_masked_statistics(
        "error_probe_probability", modulation.probe_probabilities, error_branch_mask
    )
    add_masked_statistics(
        "correct_probe_delta_prob", modulation.probe_delta_prob, correct_step_mask
    )
    add_masked_statistics(
        "error_probe_delta_prob", modulation.probe_delta_prob, error_step_mask
    )
    add_masked_statistics(
        "correct_alpha_r_gate", modulation.correct_step_gate, correct_step_mask
    )
    add_masked_statistics(
        "error_alpha_r_gate", modulation.correct_step_gate, error_step_mask
    )
    add_masked_statistics(
        "correct_alpha_d_gate", modulation.error_step_gate, correct_step_mask
    )
    add_masked_statistics(
        "error_alpha_d_gate", modulation.error_step_gate, error_step_mask
    )
    add_masked_statistics(
        "correct_probe_modulation", modulation.correct_token_multiplier, correct_probe_tokens
    )
    add_masked_statistics(
        "error_probe_modulation", modulation.error_token_multiplier, error_probe_tokens
    )
    # Terminal tokens must remain unmodulated.  Export the values as pooled
    # statistics so a real run can audit the exact multiplier 1.0.
    add_masked_statistics(
        "correct_last_step_multiplier", modulation.correct_token_multiplier, correct_last_tokens
    )
    add_masked_statistics(
        "error_last_step_multiplier", modulation.error_token_multiplier, error_last_tokens
    )

    if token_advantage is not None:
        if tuple(token_advantage.shape) != tuple(token_shape):
            raise ValueError("R2OPL token_advantage must match response_mask.")
        add_masked_statistics("correct_advantage", token_advantage, correct_response_tokens)
        add_masked_statistics(
            "correct_advantage_abs", token_advantage.abs(), correct_response_tokens
        )
        add_masked_statistics("error_opd_advantage", token_advantage, error_response_tokens)
        add_masked_statistics(
            "error_opd_advantage_abs", token_advantage.abs(), error_response_tokens
        )
        # These two values directly show that the unprobed final step follows
        # the standard RL / OPD branch signal rather than an alpha gate.
        add_masked_statistics(
            "correct_last_step_rl_advantage", token_advantage, correct_last_tokens
        )
        add_masked_statistics(
            "error_last_step_opd_advantage", token_advantage, error_last_tokens
        )

    if student_log_probs is not None or teacher_log_probs is not None:
        if student_log_probs is None or teacher_log_probs is None:
            raise ValueError("R2OPL raw OPD diagnostics require both Student and Teacher log-probabilities.")
        if tuple(student_log_probs.shape) != tuple(token_shape) or tuple(teacher_log_probs.shape) != tuple(token_shape):
            raise ValueError("R2OPL raw OPD log-probabilities must match response_mask.")
        raw_opd = teacher_log_probs.float() - student_log_probs.float()
        add_masked_statistics("correct_raw_opd", raw_opd, correct_response_tokens)
        add_masked_statistics("error_raw_opd", raw_opd, error_response_tokens)
        add_masked_statistics("error_raw_opd_abs", raw_opd.abs(), error_response_tokens)
    return metrics


# The old generic names make migrating the independently tested OPD_Forge
# layout straightforward, while the R2OPL-prefixed names remain the canonical
# public API in this module.
build_masked_probe_layout = build_r2opl_probe_layout
validate_masked_probe_layout = validate_r2opl_probe_layout


__all__ = [
    "R2OPL_DEFAULT_ALPHA_D",
    "R2OPL_DEFAULT_ALPHA_R",
    "R2OPL_DEFAULT_EPSILON",
    "R2OPL_DEFAULT_LAMBDA",
    "R2OPL_DEFAULT_MIU",
    "R2OPLBatchResult",
    "R2OPLProbeBatch",
    "R2OPLProbeBranch",
    "R2OPLProbeLayout",
    "R2OPLTokenModulation",
    "build_dense_attention_mask",
    "build_masked_probe_layout",
    "build_r2opl_additive_attention_mask",
    "build_r2opl_probe_batch",
    "build_r2opl_probe_layout",
    "build_r2opl_probe_layout_from_step_ends",
    "compute_r2opl_batch",
    "compute_r2opl_packed_probe_probabilities",
    "compute_r2opl_probe_probabilities",
    "compute_r2opl_token_modulation",
    "layout_rpc_payload",
    "r2opl_bool_mask_to_additive",
    "r2opl_probe_metrics",
    "r2opl_reinforce_loss",
    "r2opl_token_advantage",
    "semantic_step_ranges_from_ends",
    "validate_masked_probe_layout",
    "validate_r2opl_probe_layout",
]
