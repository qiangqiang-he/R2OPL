"""Expected Reasoning-Step Return helpers shared by validation runtimes.

The functions in this module are deliberately model-runtime agnostic.  They
operate on the original rollout token IDs and accept small async generation
callbacks, which keeps step selection and ERSR arithmetic CPU-testable.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from utils.answer_verifier import verify_dataset_response_answer, verify_response_answer
from utils.conservative_step_splitter import split_rollout_steps


STUDENT_ACTION = "student_action"
TEACHER_REPLACE = "teacher_replace"

_ACTION_LABELS = {
    STUDENT_ACTION: "Student Action",
    TEACHER_REPLACE: "Teacher Replace",
}

GenerateTokens = Callable[..., Awaitable[Sequence[int]]]


@dataclass(frozen=True)
class ERSRSettings:
    datasets: tuple[str, ...]
    max_steps_per_dataset: int
    mc_k: int
    seed: int
    max_response_tokens: int
    max_teacher_step_tokens: int
    temperature: float
    top_p: float
    top_k: int


def _read(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def parse_ersr_settings(config: Any, *, max_response_tokens: int) -> ERSRSettings:
    """Parse and validate the runtime-independent ERSR configuration."""

    datasets_value = _read(config, "datasets")
    if datasets_value is None or isinstance(datasets_value, str):
        raise ValueError("ersr.datasets must be a non-empty list")
    datasets = tuple(str(name).strip() for name in datasets_value)
    if not datasets or any(not name for name in datasets):
        raise ValueError("ersr.datasets must be a non-empty list")
    if len(datasets) != len(set(datasets)):
        raise ValueError("ersr.datasets must not contain duplicate names")

    settings = ERSRSettings(
        datasets=datasets,
        max_steps_per_dataset=int(_read(config, "max_steps_per_dataset", 0)),
        mc_k=int(_read(config, "mc_k", 0)),
        seed=int(_read(config, "seed", 0)),
        max_response_tokens=int(max_response_tokens),
        max_teacher_step_tokens=int(
            _read(config, "max_teacher_step_tokens", 0)
        ),
        temperature=float(_read(config, "temperature", 0.6)),
        top_p=float(_read(config, "top_p", 1.0)),
        top_k=int(_read(config, "top_k", -1)),
    )
    if settings.max_steps_per_dataset <= 0:
        raise ValueError("ersr.max_steps_per_dataset must be positive")
    if settings.mc_k <= 0:
        raise ValueError("ersr.mc_k must be positive")
    if settings.max_response_tokens <= 0:
        raise ValueError("ERSR max_response_tokens must be positive")
    if settings.max_teacher_step_tokens <= 0:
        raise ValueError("ersr.max_teacher_step_tokens must be positive")
    if settings.temperature < 0.0:
        raise ValueError("ersr.temperature must be non-negative")
    if not 0.0 < settings.top_p <= 1.0:
        raise ValueError("ersr.top_p must lie in (0, 1]")
    return settings


def validate_ersr_dataset_subset(
    ersr_datasets: Iterable[str], val_datasets: Iterable[str]
) -> None:
    """Require every ERSR benchmark to be part of normal validation."""

    validation = {str(name) for name in val_datasets}
    missing = [str(name) for name in ersr_datasets if str(name) not in validation]
    if missing:
        raise ValueError(
            "ersr.datasets must be a subset of data.val_datasets; missing: "
            + ", ".join(missing)
        )


def _dataset_seed(seed: int, dataset: str, action_type: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\0{dataset}\0{action_type}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_ersr_step_cases(
    validation_records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    settings: ERSRSettings,
    *,
    include_student_action: bool,
    include_teacher_replace: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select capped semantic-step interventions from validation rollouts.

    The final segmented step is always excluded, independently of whether its
    text contains an answer marker.  Student actions come only from correct
    trajectories; Teacher replacements come only from incorrect trajectories.
    """

    if not include_student_action and not include_teacher_replace:
        raise ValueError("ERSR must enable at least one action type")

    wanted = set(settings.datasets)
    # Reservoir sampling enforces the cap while candidates are discovered, so
    # a long response with many steps never causes every full token sequence to
    # be copied into an unbounded intermediate list.
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    available_counts: dict[tuple[str, str], int] = defaultdict(int)
    sampling_rngs: dict[tuple[str, str], random.Random] = {}
    split_failures: dict[str, int] = defaultdict(int)
    trajectories: dict[str, int] = defaultdict(int)
    trajectories_without_eligible_steps: dict[str, int] = defaultdict(int)

    for record_index, record in enumerate(validation_records):
        dataset = str(record.get("dataset", ""))
        if dataset not in wanted:
            continue
        correct = bool(record.get("correct", False))
        if correct and not include_student_action:
            continue
        if not correct and not include_teacher_replace:
            continue
        action_type = STUDENT_ACTION if correct else TEACHER_REPLACE
        trajectories[dataset] += 1

        prompt_ids = [int(value) for value in record.get("prompt_ids", [])]
        response_ids = [int(value) for value in record.get("response_ids", [])]
        try:
            _, steps = split_rollout_steps(response_ids, tokenizer)
        except Exception:
            split_failures[dataset] += 1
            continue

        # A response's final step is terminal for this evaluation.  Even when
        # the splitter cannot prove that it is an answer step, ERSR must not
        # intervene after the trajectory has already ended.
        eligible_steps = steps[:-1]
        if not eligible_steps:
            trajectories_without_eligible_steps[dataset] += 1
            continue

        for step in eligible_steps:
            if step.token_end <= step.token_start:
                continue
            key = (dataset, action_type)
            available_counts[key] += 1
            candidate = {
                "dataset": dataset,
                "action_type": action_type,
                "record_index": int(record_index),
                "uid": str(record.get("uid", record_index)),
                "answer": str(record.get("answer", "")),
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "step_index": int(step.index),
                "step_token_start": int(step.token_start),
                "step_token_end": int(step.token_end),
                "step_text": str(step.text),
            }
            reservoir = candidates[key]
            if len(reservoir) < settings.max_steps_per_dataset:
                reservoir.append(candidate)
                continue
            rng = sampling_rngs.setdefault(
                key,
                random.Random(
                    _dataset_seed(settings.seed, dataset, action_type)
                ),
            )
            replacement_index = rng.randrange(available_counts[key])
            if replacement_index < settings.max_steps_per_dataset:
                reservoir[replacement_index] = candidate

    selected: list[dict[str, Any]] = []
    available: dict[str, dict[str, int]] = defaultdict(dict)
    selected_counts: dict[str, dict[str, int]] = defaultdict(dict)
    action_types = []
    if include_student_action:
        action_types.append(STUDENT_ACTION)
    if include_teacher_replace:
        action_types.append(TEACHER_REPLACE)

    case_index = 0
    for dataset in settings.datasets:
        for action_type in action_types:
            rows = list(candidates.get((dataset, action_type), []))
            available[dataset][action_type] = available_counts.get(
                (dataset, action_type), 0
            )
            random.Random(
                _dataset_seed(settings.seed + 1, dataset, action_type)
            ).shuffle(rows)
            selected_counts[dataset][action_type] = len(rows)
            for row in rows:
                row["case_index"] = case_index
                row["case_id"] = (
                    f"{dataset}:{row['uid']}:{row['record_index']}:"
                    f"{row['step_index']}:{action_type}"
                )
                row["seed"] = settings.seed + case_index * 10_000
                selected.append(row)
                case_index += 1

    return selected, {
        "trajectories": dict(trajectories),
        "split_failures": dict(split_failures),
        "trajectories_without_eligible_steps": dict(
            trajectories_without_eligible_steps
        ),
        "available_steps": {key: dict(value) for key, value in available.items()},
        "selected_steps": {
            key: dict(value) for key, value in selected_counts.items()
        },
    }


def first_replacement_step_ids(
    proposal_ids: Sequence[int], tokenizer: Any
) -> list[int]:
    """Return the exact token IDs of the Teacher's first visible step."""

    original_ids = [int(value) for value in proposal_ids]
    _, steps = split_rollout_steps(original_ids, tokenizer)
    if not steps:
        return []
    return original_ids[: int(steps[0].token_end)]


def reward_for_continuation(
    tokenizer: Any,
    *,
    response_prefix_ids: Sequence[int],
    continuation_ids: Sequence[int],
    answer: str,
    data_source: str | None = None,
) -> float:
    """Grade a continuation, allowing the continuation itself to be empty."""

    full_response_ids = [int(value) for value in response_prefix_ids] + [
        int(value) for value in continuation_ids
    ]
    response = tokenizer.decode(
        full_response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # Training callers retain their exact verifier. Standalone evaluation
    # explicitly supplies the dataset, including SciBench's 5% tolerance.
    if data_source is not None:
        return float(verify_dataset_response_answer(response, str(answer), data_source))
    return float(verify_response_answer(response, str(answer)))


async def _estimate_value(
    *,
    tokenizer: Any,
    case: Mapping[str, Any],
    response_prefix_ids: Sequence[int],
    settings: ERSRSettings,
    arm_offset: int,
    student_generate: GenerateTokens,
) -> tuple[float, int]:
    remaining = settings.max_response_tokens - len(response_prefix_ids)

    async def sample(sample_index: int) -> tuple[float, int]:
        # A fixed action may already finish the answer or exactly fill the
        # response budget.  An empty Student continuation is a valid sample,
        # not an exception and not a reason to stop training.
        if remaining <= 0:
            continuation_ids: Sequence[int] = []
        else:
            continuation_ids = await student_generate(
                prompt_ids=[int(value) for value in case["prompt_ids"]]
                + [int(value) for value in response_prefix_ids],
                max_tokens=remaining,
                seed=int(case["seed"]) + arm_offset + sample_index,
            )
        reward = reward_for_continuation(
            tokenizer,
            response_prefix_ids=response_prefix_ids,
            continuation_ids=continuation_ids,
            answer=str(case["answer"]),
        )
        return reward, int(len(continuation_ids) == 0)

    samples = await asyncio.gather(*(sample(index) for index in range(settings.mc_k)))
    return (
        sum(value for value, _ in samples) / settings.mc_k,
        sum(empty for _, empty in samples),
    )


async def evaluate_ersr_case(
    case: Mapping[str, Any],
    tokenizer: Any,
    settings: ERSRSettings,
    *,
    student_generate: GenerateTokens,
    teacher_generate: GenerateTokens | None = None,
) -> dict[str, Any]:
    """Estimate one Student-action or Teacher-replacement advantage."""

    response_ids = [int(value) for value in case["response_ids"]]
    start = int(case["step_token_start"])
    end = int(case["step_token_end"])
    if not 0 <= start < end <= len(response_ids):
        raise ValueError(
            f"Invalid ERSR step token span [{start}, {end}) for response length "
            f"{len(response_ids)}"
        )
    before = response_ids[:start]
    action_type = str(case["action_type"])

    if action_type == STUDENT_ACTION:
        action_prefix = response_ids[:end]
        replacement_ids: list[int] | None = None
    elif action_type == TEACHER_REPLACE:
        if teacher_generate is None:
            raise ValueError("Teacher-replacement ERSR requires teacher_generate")
        teacher_budget = min(
            settings.max_teacher_step_tokens,
            settings.max_response_tokens - len(before),
        )
        if teacher_budget <= 0:
            return {
                "case_id": str(case["case_id"]),
                "dataset": str(case["dataset"]),
                "action_type": action_type,
                "valid": False,
                "skip_reason": "no_teacher_budget",
            }
        proposal_ids = await teacher_generate(
            prompt_ids=[int(value) for value in case["prompt_ids"]] + before,
            max_tokens=teacher_budget,
            seed=int(case["seed"]) + 5_000,
        )
        replacement_ids = first_replacement_step_ids(proposal_ids, tokenizer)
        if not replacement_ids:
            return {
                "case_id": str(case["case_id"]),
                "dataset": str(case["dataset"]),
                "action_type": action_type,
                "valid": False,
                "skip_reason": "empty_teacher_replacement",
            }
        action_prefix = before + replacement_ids
    else:
        raise ValueError(f"Unknown ERSR action type {action_type!r}")

    baseline_task = _estimate_value(
        tokenizer=tokenizer,
        case=case,
        response_prefix_ids=before,
        settings=settings,
        arm_offset=0,
        student_generate=student_generate,
    )
    action_task = _estimate_value(
        tokenizer=tokenizer,
        case=case,
        response_prefix_ids=action_prefix,
        settings=settings,
        arm_offset=1_000,
        student_generate=student_generate,
    )
    (baseline_value, baseline_empty), (action_value, action_empty) = (
        await asyncio.gather(baseline_task, action_task)
    )
    return {
        "case_id": str(case["case_id"]),
        "dataset": str(case["dataset"]),
        "action_type": action_type,
        "valid": True,
        "baseline_value": float(baseline_value),
        "action_value": float(action_value),
        "advantage": float(action_value - baseline_value),
        "mc_k": settings.mc_k,
        "baseline_empty_continuations": int(baseline_empty),
        "action_empty_continuations": int(action_empty),
        "replacement_num_tokens": (
            len(replacement_ids) if replacement_ids is not None else None
        ),
    }


def ersr_series_label(dataset: str, action_type: str) -> str:
    try:
        action_label = _ACTION_LABELS[action_type]
    except KeyError as exc:
        raise ValueError(f"Unknown ERSR action type {action_type!r}") from exc
    return f"{dataset} / {action_label}"


def aggregate_ersr_advantages(
    results: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    action_types: Sequence[str],
) -> tuple[dict[str, float], dict[str, int]]:
    """Return ordered per-dataset means for one combined W&B chart.

    A configured series with no eligible or valid step is a legitimate zero,
    not a missing metric.  This covers fully correct validation samples,
    terminal-only responses, conservative split failures, and skipped Teacher
    replacements without interrupting training or leaving a W&B gap.
    """

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for result in results:
        if not bool(result.get("valid", False)):
            skipped[str(result.get("skip_reason", "unknown"))] += 1
            continue
        key = (str(result["dataset"]), str(result["action_type"]))
        grouped[key].append(float(result["advantage"]))

    means: dict[str, float] = {}
    for dataset in datasets:
        for action_type in action_types:
            values = grouped.get((str(dataset), str(action_type)), [])
            means[ersr_series_label(str(dataset), str(action_type))] = (
                sum(values) / len(values) if values else 0.0
            )
    return means, dict(skipped)


__all__ = [
    "ERSRSettings",
    "STUDENT_ACTION",
    "TEACHER_REPLACE",
    "aggregate_ersr_advantages",
    "build_ersr_step_cases",
    "ersr_series_label",
    "evaluate_ersr_case",
    "first_replacement_step_ids",
    "parse_ersr_settings",
    "reward_for_continuation",
    "validate_ersr_dataset_subset",
]
