"""Dataset adapter for the two R2OPL question/answer JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datasets
import numpy as np

from utils.prompts import normalize_model_family, render_chat_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset


def _validated_item(
    item: Any,
    *,
    source: str,
    index: int,
    dataset_name: str,
) -> dict[str, str]:
    if not isinstance(item, dict):
        raise ValueError(f"{source}: item {index} must be an object")
    question = item.get("question")
    answer = item.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{source}: item {index} has an empty question")
    if answer is None or not str(answer).strip():
        raise ValueError(f"{source}: item {index} has an empty answer")
    return {
        "question": question.strip(),
        "answer": str(answer).strip(),
        "data_source": dataset_name,
        "sample_id": str(item.get("id", "")),
        "task_type": str(item.get("type", "")),
    }


def _normalize_dataset_names(dataset_names: Any) -> list[str] | None:
    if dataset_names is None:
        return None
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    names = [str(name).strip() for name in dataset_names]
    if not names or any(not name for name in names):
        raise ValueError("dataset_names must contain at least one non-empty name")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "dataset_names contains duplicates: " + ", ".join(duplicates)
        )
    return names


def _select_dataset_records(
    records: list[dict[str, str]],
    dataset_names: Any,
    *,
    source: str,
) -> list[dict[str, str]]:
    requested = _normalize_dataset_names(dataset_names)
    if requested is None:
        return records

    by_name: dict[str, list[dict[str, str]]] = {}
    for record in records:
        by_name.setdefault(record["data_source"], []).append(record)
    missing = [name for name in requested if name not in by_name]
    if missing:
        available = ", ".join(by_name) or "<none>"
        raise ValueError(
            f"{source}: unknown dataset_names {missing}; available datasets: "
            f"{available}"
        )
    return [record for name in requested for record in by_name[name]]


def load_question_answer_records(
    path: str | Path,
    *,
    dataset_names: Any = None,
) -> list[dict[str, str]]:
    """Load records, optionally selecting explicitly named logical datasets."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")

    records: list[dict[str, str]] = []
    if payload and isinstance(payload[0], dict) and "items" in payload[0]:
        for group_index, group in enumerate(payload):
            if not isinstance(group, dict):
                raise ValueError(f"{path}: group {group_index} must be an object")
            dataset_name = str(group.get("dataset_name", "")).strip()
            items = group.get("items")
            if not dataset_name or not isinstance(items, list):
                raise ValueError(
                    f"{path}: group {group_index} requires dataset_name and items"
                )
            declared_count = int(group.get("num_samples", len(items)))
            if declared_count != len(items):
                raise ValueError(
                    f"{path}: {dataset_name} declares {declared_count} samples "
                    f"but contains {len(items)}"
                )
            records.extend(
                _validated_item(
                    item,
                    source=str(path),
                    index=item_index,
                    dataset_name=dataset_name,
                )
                for item_index, item in enumerate(items)
            )
        return _select_dataset_records(
            records,
            dataset_names,
            source=str(path),
        )

    for index, item in enumerate(payload):
        dataset_name = (
            str(item.get("type", "")).strip()
            if isinstance(item, dict)
            else ""
        ) or path.stem
        records.append(
            _validated_item(
                item,
                source=str(path),
                index=index,
                dataset_name=dataset_name,
            )
        )
    return _select_dataset_records(
        records,
        dataset_names,
        source=str(path),
    )


def render_length_limited_chat_prompt(
    tokenizer: Any,
    *,
    question: str,
    model_family: str,
    max_prompt_length: int,
) -> tuple[str, bool]:
    """Render a prompt, right-truncating only the question when necessary.

    Truncating the already-rendered prompt would remove the assistant generation
    suffix. Instead, keep a character-prefix of the question and render the
    complete native chat template again. The final token count is always checked
    with the same no-special-token encoding used by the rollout worker.
    """

    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")

    question = str(question).strip()

    def render(candidate: str) -> tuple[str, int]:
        prompt = render_chat_prompt(
            tokenizer,
            question=candidate,
            model_family=model_family,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        return prompt, token_count

    full_prompt, full_length = render(question)
    if full_length <= max_prompt_length:
        return full_prompt, False

    # The prompt template itself is small relative to the formal 2048-token
    # budget. Retain at least the first character so prompt validation remains
    # meaningful, then find the longest fitting question prefix.
    first_prompt, first_length = render(question[:1])
    if first_length > max_prompt_length:
        # This is a final fail-open guard for an unexpectedly tiny override.
        # It preserves the requested prompt prefix and lets the rollout layer's
        # right-truncation enforce the same limit without dropping the sample.
        return full_prompt, True

    best_prompt = first_prompt
    low, high = 1, len(question)
    while low <= high:
        middle = (low + high) // 2
        candidate_prompt, candidate_length = render(question[:middle])
        if candidate_length <= max_prompt_length:
            best_prompt = candidate_prompt
            low = middle + 1
        else:
            high = middle - 1
    return best_prompt, True


class CustomDataset(RLHFDataset):
    """Render native chat prompts without changing source JSON."""

    def _validation_dataset_names(self) -> Any:
        configured_val_files = self.config.get("val_files", [])
        if isinstance(configured_val_files, str):
            configured_val_files = [configured_val_files]
        original_files = [str(path) for path in self.original_data_files]
        validation_files = [str(path) for path in configured_val_files]
        if original_files != validation_files:
            return None

        selected = self.config.get("val_datasets")
        if selected is None:
            raise ValueError(
                "data.val_datasets must explicitly select datasets from "
                "data.val_files"
            )
        # Normalize here so an empty or duplicate selection fails even before
        # reading the combined evaluation container.
        return _normalize_dataset_names(selected)

    def _build_messages(self, example: dict, key: str):
        prompt = example[key]
        if not isinstance(prompt, str):
            raise TypeError(
                "CustomDataset expected a complete prompt string, got "
                f"{type(prompt).__name__}."
            )
        return prompt

    def _read_files_and_tokenize(self) -> None:
        records: list[dict[str, str]] = []
        validation_dataset_names = self._validation_dataset_names()
        for data_file in self.data_files:
            records.extend(
                load_question_answer_records(
                    data_file,
                    dataset_names=validation_dataset_names,
                )
            )
        if not records:
            raise ValueError("R2OPL received no question/answer records")

        dataframe = datasets.Dataset.from_list(records)
        model_family = normalize_model_family(str(self.config.model_family))

        def adapt(example: dict, index: int) -> dict:
            question = str(example["question"])
            answer = str(example["answer"])
            prompt, prompt_truncated = render_length_limited_chat_prompt(
                self.tokenizer,
                question=question,
                model_family=model_family,
                max_prompt_length=self.max_prompt_length,
            )
            return {
                # PG-OPD requires Student and Teacher to share a tokenizer
                # family so the sampled response token IDs keep their meaning.
                "prompt": prompt,
                "teacher_prompt_text": prompt,
                "oa_ground_truth_answer": answer,
                "data_source": str(example["data_source"]),
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": int(index),
                    "id": str(example.get("sample_id", "")),
                    "type": str(example.get("task_type", "")),
                    "question": question,
                    "answer": answer,
                    "prompt_truncated": prompt_truncated,
                },
            }

        dataframe = dataframe.map(
            adapt,
            with_indices=True,
            desc="Rendering R2OPL prompts",
        )
        total = len(dataframe)
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            dataframe = dataframe.select(indices.tolist())

        # Every record is retained. Long questions were right-truncated above
        # while preserving the complete chat-template generation suffix.
        self.dataframe = dataframe

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        """Retain every record; length limiting happens during prompt rendering."""

        return dataframe


__all__ = [
    "CustomDataset",
    "load_question_answer_records",
    "render_length_limited_chat_prompt",
]
