#!/usr/bin/env python3
"""Standalone R2OPL evaluation with one full vLLM model replica per GPU.

The evaluator is deliberately independent of the VERL training stack at
runtime: every GPU worker spawns its own process, pins one visible GPU, and
loads a complete model replica, so ``runtime.gpus`` (eight IDs on the formal
server) shard the question batches dynamically while tensor parallelism stays
at one.  Models are evaluated strictly sequentially; a model's workers exit
before the next model loads, which is the only way vLLM reliably returns GPU
memory.

Two mutually exclusive model sources are supported in one standalone YAML
contract (there is intentionally no ``eval_base.yaml``; every evaluation
config is fully self-contained):

``training_output`` (VERL checkpoint tree)::

    training_output: ./outputs/pg_opd_qwen3_4b_instruct_2507_to_4b_no_thinking_100steps

All ``global_step_<int>`` children are discovered and sorted by step.  A
checkpoint that already contains a complete Hugging Face model directory is
evaluated directly; an FSDP actor (``model_world_size_N_rank_K.pt`` shards) is
merged with ``python -m verl.model_merger`` into a temporary directory inside
``training_output`` that is deleted after that step's final Student phase.
All checkpoints finish base evaluation first. Teacher replacements are then
batched across checkpoints in one Teacher load, followed by Student phases.

``model_source`` (direct Hugging Face models)::

    model_source:
      type: hf_models
      models:
        - name: Qwen3-1.7B
          path: ./models/Qwen3-1.7B

Every configured model is evaluated on every configured dataset.

Optionally ``ersr.enabled: true`` adds Expected Reasoning-Step Return on top
of the standard validation rollouts.  Case selection, token-exact step spans,
MC-continuation seed arithmetic, and aggregation reuse ``utils/ersr.py`` so the
numbers stay comparable with training-time ERSR.  ``ersr.actions`` selects
``student_action`` (correct trajectories) and/or ``teacher_replace``
(incorrect trajectories); the latter requires ``ersr.teacher`` or a matching
entry in ``ersr.teachers`` for each model family. Teacher
proposals run on a second per-GPU replica phase; Student MC continuations
reload the Student replicas afterwards.

Prompt rendering reuses ``utils/prompts.py`` through
``prompt_template: explicit_step_prompt | cross_domain_prompt`` (project
default: ``explicit_step_prompt``) and each model family's native chat
template.  Overlong questions keep the beginning of the question and truncate
its ending, exactly like training-time validation; no sample is ever dropped
for prompt length.

All vLLM/CUDA imports live inside the spawned GPU workers, so
``--validate-only`` checks the YAML contract, model directories, checkpoints,
datasets, tokenizers, and context budgets entirely on CPU.

Launch with ``bash scripts/start_eval.sh configs/eval/CONFIG.yaml``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from omegaconf import OmegaConf

from utils.prompts import (
    EXPLICIT_STEP_PROMPT_NAME,
    PROMPT_TEMPLATES,
    normalize_model_family,
    render_chat_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATTERN = re.compile(r"^global_step_(\d+)$")
HF_MODELS_TYPE = "hf_models"
ERSR_STUDENT_ACTION = "student_action"
ERSR_TEACHER_REPLACE = "teacher_replace"
ERSR_ACTIONS = (ERSR_STUDENT_ACTION, ERSR_TEACHER_REPLACE)
DEFAULT_RESULT_DIR = "./eval_results"
DEFAULT_ROLLOUTS_PER_GPU_BATCH = 256
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Configuration contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    family: str
    chat_template_path: Path | None = None


@dataclass(frozen=True)
class ERSRConfig:
    actions: tuple[str, ...]
    teacher: ModelSpec | None
    settings: Any  # utils.ersr.ERSRSettings; lazy import keeps CPU mode light
    teachers: tuple[ModelSpec, ...] = ()  # Optional teacher per tokenizer family.
    gpu_memory_utilization: float = 0.85


@dataclass(frozen=True)
class EvaluationConfig:
    config_path: Path
    project_root: Path
    source_type: str  # "checkpoints" | "models"
    training_output: Path | None
    models: tuple[ModelSpec, ...]
    run_name: str
    val_files: tuple[Path, ...]
    val_datasets: tuple[str, ...]
    prompt_name: str
    seed: int
    temperature: float
    top_p: float
    top_k: int
    n: int
    max_new_tokens: int
    length_control: tuple[int, ...]
    gpus: tuple[int, ...]
    tensor_parallel_size: int
    rollouts_per_gpu_batch: int
    min_free_gib: float
    dtype: str
    trust_remote_code: bool
    gpu_memory_utilization: float
    max_model_len: int
    enable_prefix_caching: bool
    save_rollouts: bool
    result_dir: Path
    ersr: ERSRConfig | None

    @property
    def questions_per_gpu_batch(self) -> int:
        return self.rollouts_per_gpu_batch // self.n

    @property
    def result_path(self) -> Path:
        return self.result_dir / f"{self.run_name}.json"

    def model_result_dir(self) -> Path:
        return self.result_dir / self.run_name

    def model_result_path(self, model: ModelSpec | str) -> Path:
        name = model.name if isinstance(model, ModelSpec) else str(model)
        return self.model_result_dir() / f"{name}.json"

    def summary_result_path(self) -> Path:
        return self.model_result_dir() / "summary.json"


@dataclass(frozen=True)
class CheckpointSpec:
    step_num: int
    checkpoint_dir: Path
    source_kind: str  # "huggingface" | "fsdp"
    source_dir: Path
    tokenizer_dir: Path
    model_dir: Path | None


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a YAML mapping, got {value!r}.")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}, got {value}.")
    return int(value)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}.")
    return float(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false, got {value!r}.")
    return value


def _resolve_project_path(value: Any, project_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _read_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    try:
        with config_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model config {config_path}: {exc}") from exc


def resolve_model_family(model_dir: Path, *, explicit: Any = None) -> str:
    """Resolve a supported family from an explicit hint, name, or config.json."""

    for candidate in (explicit, model_dir.name, model_dir.parent.name):
        if isinstance(candidate, str) and candidate.strip():
            try:
                return normalize_model_family(candidate)
            except ValueError:
                continue
    try:
        return normalize_model_family(
            str(_read_model_config(model_dir).get("model_type", ""))
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot determine the model family of {model_dir}; set an explicit "
            f"family field in the evaluation config. ({exc})"
        ) from exc


def _parse_model_entry(
    item: Any,
    index: int,
    *,
    label: str,
    project_root: Path,
    default_family: Any = None,
) -> ModelSpec:
    model = _mapping(item, f"{label}[{index}]")
    name = model.get("name")
    if not isinstance(name, str) or not RUN_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"{label}[{index}].name must be a filesystem-safe non-empty name."
        )
    model_path = _resolve_project_path(
        model.get("path"), project_root, f"{label}[{index}].path"
    )
    if not is_huggingface_model_dir(model_path):
        raise ValueError(
            f"{label}[{index}].path is not a complete Hugging Face model "
            f"directory: {model_path}"
        )
    family = resolve_model_family(
        model_path,
        explicit=model.get("family", default_family),
    )
    chat_template_path = None
    if model.get("chat_template_path") is not None:
        chat_template_path = _resolve_project_path(
            model["chat_template_path"], project_root, f"{label}[{index}].chat_template_path"
        )
        if not chat_template_path.is_file() or not chat_template_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"chat_template_path must be a non-empty template file: {chat_template_path}")
    return ModelSpec(name=name, path=model_path, family=family, chat_template_path=chat_template_path)


def _parse_ersr(
    raw_ersr: Mapping[str, Any],
    *,
    val_datasets: Sequence[str],
    max_new_tokens: int,
    project_root: Path,
    default_family: Any = None,
) -> ERSRConfig | None:
    from utils.ersr import parse_ersr_settings, validate_ersr_dataset_subset

    if not _boolean(raw_ersr.get("enabled", False), "ersr.enabled"):
        return None

    settings = parse_ersr_settings(
        {"max_steps_per_dataset": 200, **raw_ersr},
        max_response_tokens=max_new_tokens,
    )
    validate_ersr_dataset_subset(settings.datasets, val_datasets)

    actions_value = raw_ersr.get("actions", list(ERSR_ACTIONS))
    if isinstance(actions_value, str) or not isinstance(actions_value, list):
        raise ValueError("ersr.actions must be a YAML list of action names.")
    actions = tuple(str(action).strip() for action in actions_value)
    if not actions or any(action not in ERSR_ACTIONS for action in actions):
        raise ValueError(
            "ersr.actions may only contain: " + ", ".join(ERSR_ACTIONS)
        )
    if len(actions) != len(set(actions)):
        raise ValueError("ersr.actions must not contain duplicates.")

    teacher: ModelSpec | None = None
    teachers: list[ModelSpec] = []
    teacher_gpu_memory_utilization = _number(
        raw_ersr.get("gpu_memory_utilization", 0.85),
        "ersr.gpu_memory_utilization",
    )
    if not 0 < teacher_gpu_memory_utilization < 1:
        raise ValueError("ersr.gpu_memory_utilization must lie in (0, 1).")
    if ERSR_TEACHER_REPLACE in actions:
        if "teacher" in raw_ersr and "teachers" in raw_ersr:
            raise ValueError("Use ersr.teacher or ersr.teachers, not both.")
        if "teachers" in raw_ersr:
            for family, entry in _mapping(raw_ersr["teachers"], "ersr.teachers").items():
                spec = _parse_model_entry(
                    entry, 0, label=f"ersr.teachers.{family}",
                    project_root=project_root,
                )
                if spec.family != normalize_model_family(family):
                    raise ValueError(f"ersr.teachers.{family} has family {spec.family}.")
                if any(item.family == spec.family for item in teachers):
                    raise ValueError(f"Duplicate ERSR teacher family: {spec.family}")
                teachers.append(spec)
            if not teachers:
                raise ValueError("ersr.teachers must not be empty.")
        else:
            teacher = _parse_model_entry(
                _mapping(raw_ersr.get("teacher"), "ersr.teacher"),
                0, label="ersr.teacher", project_root=project_root,
                default_family=default_family,
            )
    return ERSRConfig(
        actions=actions,
        teacher=teacher,
        settings=settings,
        teachers=tuple(teachers),
        gpu_memory_utilization=teacher_gpu_memory_utilization,
    )


def ersr_teacher_for_model(config: EvaluationConfig, model: ModelSpec) -> ModelSpec | None:
    if config.ersr is None or ERSR_TEACHER_REPLACE not in config.ersr.actions:
        return None
    teacher = config.ersr.teacher or next(
        (item for item in config.ersr.teachers if item.family == model.family), None
    )
    if teacher is None or teacher.family != model.family:
        raise ValueError(
            f"ERSR teacher_replace for {model.name} requires a {model.family} teacher; "
            "set ersr.teacher or ersr.teachers.<family>."
        )
    return teacher


def load_evaluation_config(
    config_path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> EvaluationConfig:
    """Load and strictly validate one standalone evaluation YAML contract."""

    project_root = project_root.resolve()
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {config_path}")

    raw_config = OmegaConf.load(config_path)
    OmegaConf.resolve(raw_config)
    raw = _mapping(OmegaConf.to_container(raw_config, resolve=True), "config")

    if "training_output" in raw and "model_source" in raw:
        raise ValueError(
            "Use either training_output or model_source in one eval config, "
            "not both."
        )

    training_output: Path | None = None
    models: tuple[ModelSpec, ...] = ()
    default_family = raw.get("model_family")
    if default_family is not None and not isinstance(default_family, str):
        raise ValueError("model_family must be a string when present.")

    if "training_output" in raw:
        source_type = "checkpoints"
        if "models" in raw:
            raise ValueError(
                "The checkpoint evaluation config must contain training_output "
                "only; checkpoints are discovered automatically."
            )
        training_output = _resolve_project_path(
            raw.get("training_output"), project_root, "training_output"
        )
        if not training_output.is_dir():
            raise NotADirectoryError(
                f"training_output is not a directory: {training_output}"
            )
        run_name = raw.get("run_name", training_output.name)
        if not isinstance(run_name, str) or not RUN_NAME_PATTERN.fullmatch(run_name):
            raise ValueError(
                "run_name must be filesystem-safe (letters, digits, dots, "
                "underscores, hyphens)."
            )
    elif "model_source" in raw:
        source_type = "models"
        model_source = _mapping(raw.get("model_source"), "model_source")
        if model_source.get("type") != HF_MODELS_TYPE:
            raise ValueError(
                f"model_source.type must be {HF_MODELS_TYPE!r} for direct-model "
                "evaluation."
            )
        raw_models = model_source.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError("model_source.models must be a non-empty YAML list.")
        if "models" in raw:
            raise ValueError(
                "models must be nested under model_source, not at the top level."
            )
        parsed: list[ModelSpec] = []
        seen_names: set[str] = set()
        seen_paths: set[tuple[Path, Path | None]] = set()
        for index, item in enumerate(raw_models):
            spec = _parse_model_entry(
                item,
                index,
                label="model_source.models",
                project_root=project_root,
                default_family=default_family,
            )
            if spec.name in seen_names:
                raise ValueError(f"Duplicate model name: {spec.name}")
            source_key = (spec.path, spec.chat_template_path)
            if source_key in seen_paths:
                raise ValueError(f"Duplicate model path: {spec.path}")
            seen_names.add(spec.name)
            seen_paths.add(source_key)
            parsed.append(spec)
        models = tuple(parsed)
        run_name = raw.get("run_name")
        if not isinstance(run_name, str) or not RUN_NAME_PATTERN.fullmatch(run_name):
            raise ValueError(
                "model_source configs require a filesystem-safe run_name."
            )
    else:
        raise ValueError(
            "The evaluation config requires either training_output or "
            "model_source."
        )

    data = _mapping(raw.get("data"), "data")
    raw_val_files = data.get("val_files")
    if isinstance(raw_val_files, str):
        raw_val_files = [raw_val_files]
    if not isinstance(raw_val_files, list) or not raw_val_files:
        raise ValueError("data.val_files must be a non-empty list of JSON files.")
    val_files = tuple(
        _resolve_project_path(value, project_root, f"data.val_files[{index}]")
        for index, value in enumerate(raw_val_files)
    )
    for path in val_files:
        if not path.is_file():
            raise FileNotFoundError(f"data.val_files entry does not exist: {path}")

    raw_val_datasets = data.get("val_datasets")
    if isinstance(raw_val_datasets, str):
        raw_val_datasets = [raw_val_datasets]
    if not isinstance(raw_val_datasets, list) or not raw_val_datasets:
        raise ValueError("data.val_datasets must be a non-empty list of names.")
    val_datasets = tuple(str(name).strip() for name in raw_val_datasets)
    if any(not name for name in val_datasets):
        raise ValueError("data.val_datasets entries must be non-empty.")
    if len(val_datasets) != len(set(val_datasets)):
        raise ValueError("data.val_datasets must not contain duplicate names.")

    prompt_name = raw.get("prompt_template", EXPLICIT_STEP_PROMPT_NAME)
    if not isinstance(prompt_name, str) or prompt_name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"prompt_template must be one of: {available}; got {prompt_name!r}."
        )

    seed = _integer(raw.get("seed", 42), "seed", minimum=0)

    sampling = _mapping(raw.get("sampling"), "sampling")
    temperature = _number(sampling.get("temperature", 0.6), "sampling.temperature")
    if temperature < 0:
        raise ValueError("sampling.temperature must be non-negative.")
    top_p = _number(sampling.get("top_p", 0.95), "sampling.top_p")
    if not 0 < top_p <= 1:
        raise ValueError("sampling.top_p must lie in (0, 1].")
    top_k = _integer(sampling.get("top_k", -1), "sampling.top_k")
    n = _integer(sampling.get("n", 16), "sampling.n", minimum=1)
    max_new_tokens = _integer(
        sampling.get("max_new_tokens", 20480), "sampling.max_new_tokens", minimum=1
    )

    raw_lengths = raw.get("length_control", [max_new_tokens])
    if not isinstance(raw_lengths, list) or not raw_lengths:
        raise ValueError("length_control must be a non-empty list of token limits.")
    length_control = tuple(
        _integer(value, f"length_control[{index}]", minimum=1)
        for index, value in enumerate(raw_lengths)
    )
    if tuple(sorted(set(length_control))) != length_control:
        raise ValueError(
            "length_control values must be unique and strictly ascending."
        )
    if length_control[-1] != max_new_tokens:
        raise ValueError(
            "The last length_control value must equal sampling.max_new_tokens."
        )

    runtime = _mapping(raw.get("runtime", {}), "runtime")
    raw_gpus = runtime.get("gpus", list(range(8)))
    if not isinstance(raw_gpus, list) or not raw_gpus:
        raise ValueError("runtime.gpus must be a non-empty list of GPU IDs.")
    gpus = tuple(
        _integer(value, f"runtime.gpus[{index}]", minimum=0)
        for index, value in enumerate(raw_gpus)
    )
    if len(set(gpus)) != len(gpus):
        raise ValueError("runtime.gpus must contain distinct GPU IDs.")
    tensor_parallel_size = _integer(
        runtime.get("tensor_parallel_size", 1),
        "runtime.tensor_parallel_size",
        minimum=1,
    )
    if tensor_parallel_size != 1:
        raise ValueError(
            "runtime.tensor_parallel_size must be 1: every GPU loads one "
            "complete model replica and shards the questions."
        )
    rollouts_per_gpu_batch = _integer(
        runtime.get("rollouts_per_gpu_batch", DEFAULT_ROLLOUTS_PER_GPU_BATCH),
        "runtime.rollouts_per_gpu_batch",
        minimum=1,
    )
    if rollouts_per_gpu_batch % n:
        raise ValueError(
            f"runtime.rollouts_per_gpu_batch ({rollouts_per_gpu_batch}) must be "
            f"divisible by sampling.n ({n})."
        )
    min_free_gib = _number(
        runtime.get("min_free_gib", 5.0), "runtime.min_free_gib"
    )
    if min_free_gib < 0:
        raise ValueError("runtime.min_free_gib must be non-negative.")

    engine = _mapping(raw.get("engine", {}), "engine")
    dtype = engine.get("dtype", "bfloat16")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("engine.dtype must be a non-empty string.")
    trust_remote_code = _boolean(
        engine.get("trust_remote_code", True), "engine.trust_remote_code"
    )
    gpu_memory_utilization = _number(
        engine.get("gpu_memory_utilization", 0.90), "engine.gpu_memory_utilization"
    )
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("engine.gpu_memory_utilization must lie in (0, 1).")
    max_model_len = _integer(
        engine.get("max_model_len", 2048 + max_new_tokens),
        "engine.max_model_len",
        minimum=1,
    )
    if max_model_len <= max_new_tokens:
        raise ValueError(
            "engine.max_model_len must exceed sampling.max_new_tokens to leave "
            "room for prompts."
        )
    enable_prefix_caching = _boolean(
        engine.get("enable_prefix_caching", True), "engine.enable_prefix_caching"
    )

    report = _mapping(raw.get("report", {}), "report")
    save_rollouts = _boolean(
        report.get("save_rollouts", False), "report.save_rollouts"
    )

    result_dir = _resolve_project_path(
        raw.get("result_dir", DEFAULT_RESULT_DIR), project_root, "result_dir"
    )

    ersr = None
    if raw.get("ersr") is not None:
        ersr = _parse_ersr(
            _mapping(raw.get("ersr"), "ersr"),
            val_datasets=val_datasets,
            max_new_tokens=max_new_tokens,
            project_root=project_root,
            default_family=default_family,
        )

    return EvaluationConfig(
        config_path=config_path,
        project_root=project_root,
        source_type=source_type,
        training_output=training_output,
        models=models,
        run_name=run_name,
        val_files=val_files,
        val_datasets=val_datasets,
        prompt_name=prompt_name,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=n,
        max_new_tokens=max_new_tokens,
        length_control=length_control,
        gpus=gpus,
        tensor_parallel_size=tensor_parallel_size,
        rollouts_per_gpu_batch=rollouts_per_gpu_batch,
        min_free_gib=min_free_gib,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        save_rollouts=save_rollouts,
        result_dir=result_dir,
        ersr=ersr,
    )


# ---------------------------------------------------------------------------
# Checkpoint discovery and FSDP merging
# ---------------------------------------------------------------------------


def _weight_index_is_complete(path: Path, index_name: str) -> bool:
    index_path = path / index_name
    if not index_path.is_file():
        return False
    try:
        with index_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        weight_map = payload["weight_map"]
        shard_names = set(weight_map.values())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(shard_names) and all(
        (path / shard).is_file() and (path / shard).stat().st_size > 0
        for shard in shard_names
    )


def is_huggingface_model_dir(path: Path) -> bool:
    """Return whether ``path`` holds a config plus complete HF weight files."""

    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    present_indexes = [name for name in index_names if (path / name).is_file()]
    if present_indexes:
        return any(_weight_index_is_complete(path, name) for name in present_indexes)
    # Unindexed shard fragments are not complete HF exports.
    return any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def _materialize_gemma4_vllm_compat_weights(model_dir: Path) -> int:
    """Add the unused Gemma4 shared-KV ``k_norm`` tensors for native vLLM.

    A current Transformers Gemma4 model omits ``k_norm`` from its shared-KV
    tail layers.  Some native vLLM EngineCore instances are spawned in a fresh
    interpreter, so they cannot inherit the in-worker compatibility patch and
    still require those tensors.  This writes a tiny extra safetensors shard
    plus an index alongside the *temporary merged model*; it never alters the
    FSDP checkpoint or any trainable tensor.  The weights are all ones because
    the shared-KV forward branch does not read them.
    """

    model_config = _read_model_config(model_dir)
    if model_config.get("model_type") != "gemma4":
        return 0
    text_config = model_config.get("text_config")
    if not isinstance(text_config, Mapping):
        return 0
    try:
        num_layers = int(text_config["num_hidden_layers"])
        num_kv_shared_layers = int(text_config.get("num_kv_shared_layers", 0))
    except (KeyError, TypeError, ValueError):
        return 0
    first_shared_layer = max(num_layers - num_kv_shared_layers, 0)
    if num_layers <= 0 or first_shared_layer >= num_layers:
        return 0

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "Gemma4 checkpoint evaluation requires safetensors to build the "
            "native-vLLM compatibility shard."
        ) from exc

    index_path = model_dir / "model.safetensors.index.json"
    index_payload: dict[str, Any] = {}
    weight_map: dict[str, str] = {}
    if index_path.is_file():
        try:
            with index_path.open(encoding="utf-8") as handle:
                index_payload = json.load(handle)
            raw_weight_map = index_payload["weight_map"]
            if not isinstance(raw_weight_map, Mapping):
                raise TypeError("weight_map is not a mapping")
            weight_map = {
                str(name): str(shard) for name, shard in raw_weight_map.items()
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read Gemma4 safetensors index {index_path}: {exc}"
            ) from exc
    else:
        for shard_path in sorted(model_dir.glob("*.safetensors")):
            with safe_open(
                str(shard_path), framework="pt", device="cpu"
            ) as handle:
                for weight_name in handle.keys():
                    if weight_name in weight_map:
                        raise RuntimeError(
                            "Duplicate tensor name while indexing Gemma4 model: "
                            f"{weight_name}"
                        )
                    weight_map[weight_name] = shard_path.name
    if not weight_map:
        raise RuntimeError(
            f"No safetensors weights found in Gemma4 model: {model_dir}"
        )

    prefixes = (
        "model.language_model.layers.",
        "language_model.model.layers.",
        "model.layers.",
        "language_model.layers.",
    )
    prefix = next(
        (
            candidate
            for candidate in prefixes
            if f"{candidate}0.self_attn.k_norm.weight" in weight_map
        ),
        None,
    )
    if prefix is None:
        return 0

    missing_names = [
        f"{prefix}{layer}.self_attn.k_norm.weight"
        for layer in range(first_shared_layer, num_layers)
        if f"{prefix}{layer}.self_attn.k_norm.weight" not in weight_map
    ]
    if not missing_names:
        return 0

    compatibility_weights: dict[str, Any] = {}
    for weight_name in missing_names:
        layer_prefix, _, _ = weight_name.rpartition(".self_attn.k_norm.weight")
        reference_name = f"{layer_prefix}.self_attn.q_norm.weight"
        reference_shard = weight_map.get(reference_name)
        if reference_shard is None:
            raise RuntimeError(
                "Cannot derive Gemma4 shared-KV k_norm shape; missing "
                f"reference tensor {reference_name}."
            )
        reference_path = model_dir / reference_shard
        with safe_open(
            str(reference_path), framework="pt", device="cpu"
        ) as handle:
            reference_tensor = handle.get_tensor(reference_name)
        compatibility_weights[weight_name] = torch.ones_like(reference_tensor)

    compatibility_shard = "r2opl_gemma4_shared_kv_compat.safetensors"
    save_file(
        compatibility_weights,
        str(model_dir / compatibility_shard),
        metadata={"format": "pt"},
    )
    weight_map.update({name: compatibility_shard for name in compatibility_weights})
    metadata = index_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    index_payload["metadata"] = {
        **metadata,
        "total_size": sum(
            (model_dir / shard).stat().st_size for shard in set(weight_map.values())
        ),
    }
    index_payload["weight_map"] = weight_map
    index_path.write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return len(compatibility_weights)


def _validate_fsdp_source(path: Path) -> None:
    config_path = path / "fsdp_config.json"
    metadata_dir = path / "huggingface"
    if not config_path.is_file() or not (metadata_dir / "config.json").is_file():
        raise ValueError(f"Incomplete FSDP actor metadata: {path}")
    try:
        with config_path.open(encoding="utf-8") as handle:
            world_size = _integer(
                json.load(handle).get("world_size"), "FSDP world_size", minimum=1
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    missing = [
        rank
        for rank in range(world_size)
        if not (path / f"model_world_size_{world_size}_rank_{rank}.pt").is_file()
        or (path / f"model_world_size_{world_size}_rank_{rank}.pt").stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"FSDP actor {path} is missing model shards for ranks: {missing}"
        )


def _choose_hf_model(checkpoint_dir: Path) -> Path | None:
    candidates: list[Path] = []
    # Never pick a critic/reference HF export while the actor is still sharded.
    root = checkpoint_dir / "actor" if (checkpoint_dir / "actor").is_dir() else checkpoint_dir
    preferred = (root / "merged_hf", root / "huggingface", root)
    for path in preferred:
        if path.is_dir() and is_huggingface_model_dir(path) and path not in candidates:
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(
        key=lambda path: (
            0 if "actor" in path.relative_to(checkpoint_dir).parts else 1,
            len(path.relative_to(checkpoint_dir).parts),
            str(path),
        )
    )
    return candidates[0]


def _choose_fsdp_source(checkpoint_dir: Path) -> Path:
    candidates: list[Path] = []
    preferred = (checkpoint_dir / "actor", checkpoint_dir)
    for path in preferred:
        if (path / "fsdp_config.json").is_file() and path not in candidates:
            candidates.append(path)
    if not candidates:
        raise ValueError(
            f"No loadable HF model or FSDP actor shards found in {checkpoint_dir}."
        )
    actor_candidates = [path for path in candidates if path.name == "actor"]
    if len(actor_candidates) == 1:
        source = actor_candidates[0]
    elif len(candidates) == 1:
        source = candidates[0]
    else:
        raise ValueError(
            f"Ambiguous FSDP model sources in {checkpoint_dir}: "
            + ", ".join(str(path) for path in candidates)
        )
    _validate_fsdp_source(source)
    return source


def discover_checkpoints(config: EvaluationConfig) -> list[CheckpointSpec]:
    """Discover every ``global_step_<int>`` child of ``training_output``."""

    if config.source_type != "checkpoints" or config.training_output is None:
        raise ValueError("Checkpoint discovery requires a training_output config.")
    step_directories: dict[int, Path] = {}
    for path in config.training_output.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if not match or not path.is_dir():
            continue
        step_num = int(match.group(1))
        if step_num in step_directories:
            raise ValueError(
                f"Duplicate checkpoint step {step_num}: "
                f"{step_directories[step_num]} and {path}"
            )
        step_directories[step_num] = path.resolve()
    if not step_directories:
        raise ValueError(
            "No global_step_<integer> checkpoint directories found in "
            f"{config.training_output}."
        )

    specs: list[CheckpointSpec] = []
    for step_num, checkpoint_dir in sorted(step_directories.items()):
        hf_model = _choose_hf_model(checkpoint_dir)
        if hf_model is not None:
            specs.append(
                CheckpointSpec(
                    step_num=step_num,
                    checkpoint_dir=checkpoint_dir,
                    source_kind="huggingface",
                    source_dir=hf_model,
                    tokenizer_dir=hf_model,
                    model_dir=hf_model,
                )
            )
            continue
        fsdp_source = _choose_fsdp_source(checkpoint_dir)
        specs.append(
            CheckpointSpec(
                step_num=step_num,
                checkpoint_dir=checkpoint_dir,
                source_kind="fsdp",
                source_dir=fsdp_source,
                tokenizer_dir=fsdp_source / "huggingface",
                model_dir=None,
            )
        )
    return specs


@contextmanager
def prepare_model(
    config: EvaluationConfig, checkpoint: CheckpointSpec
) -> Iterator[Path]:
    """Yield a loadable HF model directory, merging FSDP shards on demand."""

    if checkpoint.model_dir is not None:
        yield checkpoint.model_dir
        return
    if checkpoint.source_kind != "fsdp":
        raise ValueError(f"Unsupported checkpoint source: {checkpoint.source_kind}")
    if config.training_output is None:
        raise ValueError("FSDP checkpoint merging requires training_output.")

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".eval_merged_global_step_{checkpoint.step_num}.",
            dir=config.training_output,
        )
    )
    merged_model_dir = temporary_root / "model"
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint.source_dir),
        "--target_dir",
        str(merged_model_dir),
        "--trust-remote-code",
        "--use_cpu_initialization",
    ]
    environment = os.environ.copy()
    python_paths = [str(config.project_root / "verl"), str(config.project_root)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    try:
        print(
            f"MERGE step={checkpoint.step_num} source={checkpoint.source_dir} "
            f"target={merged_model_dir}",
            flush=True,
        )
        subprocess.run(command, cwd=config.project_root, env=environment, check=True)
        if not is_huggingface_model_dir(merged_model_dir):
            raise RuntimeError(
                "Model merger finished but did not create complete HF weights: "
                f"{merged_model_dir}"
            )
        added_compat_weights = _materialize_gemma4_vllm_compat_weights(
            merged_model_dir
        )
        if added_compat_weights:
            print(
                "GEMMA4_VLLM_COMPAT_WEIGHTS "
                f"step={checkpoint.step_num} count={added_compat_weights}",
                flush=True,
            )
        yield merged_model_dir
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
        print(
            f"MERGED_MODEL_REMOVED step={checkpoint.step_num} path={temporary_root}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Datasets and prompts
# ---------------------------------------------------------------------------


def load_question_records(
    config: EvaluationConfig,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Load the selected validation questions once, without a tokenizer."""

    from utils.custom_dataset import load_question_answer_records

    dataset_names = list(config.val_datasets)
    records: list[dict[str, str]] = []
    for path in config.val_files:
        records.extend(
            load_question_answer_records(path, dataset_names=dataset_names)
        )
    by_dataset: dict[str, list[dict[str, str]]] = {name: [] for name in dataset_names}
    for record in records:
        by_dataset.setdefault(record["data_source"], []).append(record)
    missing = [name for name in dataset_names if not by_dataset[name]]
    if missing:
        raise ValueError(
            f"data.val_datasets selected benchmarks with no records: {missing}"
        )

    questions: list[dict[str, str]] = []
    question_counts: dict[str, int] = {}
    for dataset_name in dataset_names:
        items = by_dataset[dataset_name]
        question_counts[dataset_name] = len(items)
        for question_index, item in enumerate(items):
            questions.append(
                {
                    "dataset": dataset_name,
                    "question_index": question_index,
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
    return questions, question_counts


def _render_length_limited_prompt(
    tokenizer: Any,
    *,
    question: str,
    model_family: str,
    prompt_name: str,
    max_prompt_length: int,
) -> tuple[str, bool]:
    """Render the configured prompt, right-truncating only the question.

    Truncating an already-rendered prompt would cut off the assistant
    generation suffix, so keep a character prefix of the question and render
    the complete native chat template again — identical to the training-time
    adapter in ``utils/custom_dataset.py``.
    """

    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")

    def render(candidate: str) -> tuple[str, int]:
        prompt = render_chat_prompt(
            tokenizer,
            question=candidate,
            model_family=model_family,
            prompt_name=prompt_name,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, len(
            tokenizer.encode(prompt, add_special_tokens=False)
        )

    full_prompt, full_length = render(question)
    if full_length <= max_prompt_length:
        return full_prompt, False

    first_prompt, first_length = render(question[:1])
    if first_length > max_prompt_length:
        # Fail-open guard for an unexpectedly tiny budget: keep the complete
        # prompt (with its generation suffix); vLLM enforces max_model_len.
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


def render_model_tasks(
    config: EvaluationConfig,
    *,
    tokenizer: Any,
    family: str,
    questions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Render and tokenize every question for one model's tokenizer."""

    max_prompt_length = config.max_model_len - config.max_new_tokens
    tasks: list[dict[str, Any]] = []
    for question in questions:
        prompt, truncated = _render_length_limited_prompt(
            tokenizer,
            question=str(question["question"]),
            model_family=family,
            prompt_name=config.prompt_name,
            max_prompt_length=max_prompt_length,
        )
        tasks.append(
            {
                "dataset": question["dataset"],
                "question_index": int(question["question_index"]),
                "question": str(question["question"]),
                "answer": str(question["answer"]),
                "prompt": prompt,
                "prompt_truncated": bool(truncated),
                "prompt_token_ids": [
                    int(token)
                    for token in tokenizer.encode(prompt, add_special_tokens=False)
                ],
            }
        )
    return tasks


def validate_model_context(model_dir: Path, max_model_len: int) -> None:
    model_config = _read_model_config(model_dir)
    context = model_config.get("max_position_embeddings")
    if isinstance(context, int) and max_model_len > context:
        raise ValueError(
            f"engine.max_model_len={max_model_len} exceeds the model context "
            f"max_position_embeddings={context} of {model_dir}."
        )


def make_batches(
    tasks: Sequence[Mapping[str, Any]], questions_per_batch: int
) -> list[list[dict[str, Any]]]:
    if questions_per_batch < 1:
        raise ValueError("questions_per_batch must be positive.")
    if not tasks:
        raise ValueError("Cannot batch an empty task list.")
    return [
        [
            {
                "dataset": task["dataset"],
                "question_index": task["question_index"],
                "answer": task["answer"],
                "prompt_token_ids": task["prompt_token_ids"],
            }
            for task in tasks[start : start + questions_per_batch]
        ]
        for start in range(0, len(tasks), questions_per_batch)
    ]


# ---------------------------------------------------------------------------
# Post-hoc length scoring and aggregation
# ---------------------------------------------------------------------------


def score_completion_at_lengths(
    *,
    token_ids: Sequence[int],
    full_text: str,
    finish_reason: str | None,
    answer: str,
    length_control: Sequence[int],
    decode: Callable[[Sequence[int]], str],
    verify: Callable[[str, str], float],
) -> dict[int, dict[str, int]]:
    """Score one completion at every post-hoc token limit."""

    token_ids = list(token_ids)
    full_correct: int | None = None
    values: dict[int, dict[str, int]] = {}
    for limit in length_control:
        cut_by_posthoc_limit = len(token_ids) > limit
        if cut_by_posthoc_limit:
            response = decode(token_ids[:limit])
            correct = int(bool(verify(response, answer)))
        else:
            if full_correct is None:
                full_correct = int(bool(verify(full_text, answer)))
            correct = full_correct
        values[int(limit)] = {
            "correct": correct,
            "tokens": min(len(token_ids), int(limit)),
            "truncated": int(cut_by_posthoc_limit or finish_reason == "length"),
        }
    return values


def aggregate_question_results(
    *,
    question_results: Sequence[Mapping[str, Any]],
    dataset_question_counts: Mapping[str, int],
    dataset_order: Sequence[str],
    length_control: Sequence[int],
    n: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate per-dataset Avg@n / Pass@n / length metrics by limit."""

    expected_keys = {
        (dataset, question_index)
        for dataset, count in dataset_question_counts.items()
        for question_index in range(count)
    }
    actual_keys = [
        (str(record["dataset"]), int(record["question_index"]))
        for record in question_results
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise RuntimeError("Evaluation returned duplicate question results.")
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys - set(actual_keys))[:10]
        unexpected = sorted(set(actual_keys) - expected_keys)[:10]
        raise RuntimeError(
            f"Incomplete question results: missing={missing}, "
            f"unexpected={unexpected}."
        )

    avg_label = f"Avg@{n}"
    pass_label = f"Pass@{n}"
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in dataset_order:
        records = sorted(
            (record for record in question_results if record["dataset"] == dataset),
            key=lambda record: int(record["question_index"]),
        )
        question_count = dataset_question_counts[dataset]
        rollout_count = question_count * n
        length_summaries: dict[str, dict[str, Any]] = {}
        for limit in length_control:
            correct_count = 0
            passed_questions = 0
            token_count = 0
            truncated_count = 0
            per_sample: dict[str, dict[str, float]] = {}
            for record in records:
                counters = record["lengths"][str(limit)]
                per_question_rollouts = int(counters["rollouts"])
                if per_question_rollouts != n:
                    raise RuntimeError(
                        f"{dataset}[{record['question_index']}] at length {limit} "
                        f"has {per_question_rollouts} rollouts, expected {n}."
                    )
                per_question_correct = int(counters["correct_count"])
                if not 0 <= per_question_correct <= n:
                    raise RuntimeError("Invalid correct_count in worker result.")
                per_sample[str(int(record["question_index"]))] = {
                    avg_label: per_question_correct / n
                }
                correct_count += per_question_correct
                passed_questions += int(per_question_correct > 0)
                token_count += int(counters["token_count"])
                truncated_count += int(counters["truncated_count"])
            length_summaries[str(limit)] = {
                avg_label: correct_count / rollout_count,
                pass_label: passed_questions / question_count,
                "mean_length": token_count / rollout_count,
                "truncation_rate": truncated_count / rollout_count,
                "per_sample": per_sample,
            }
        summaries[dataset] = length_summaries
    return summaries


# ---------------------------------------------------------------------------
# GPU workers (all vLLM/CUDA imports stay inside the spawned process)
# ---------------------------------------------------------------------------


def _apply_native_gemma4_vllm_kv_sharing_patch(
    *,
    attention_cls: type[Any] | None = None,
    conditional_generation_cls: type[Any] | None = None,
) -> bool:
    """Make native vLLM accept Gemma4 FSDP-merged shared-KV checkpoints.

    Transformers deliberately omits ``k_norm`` (and KV projections) from the
    final ``num_kv_shared_layers`` Gemma4 layers: those layers reuse the KV
    states from earlier layers and never execute those modules.  Current
    native vLLM constructs ``k_norm`` for every layer, then its strict loader
    rejects a valid merged checkpoint because the unused tail parameters have
    no tensors.  Remove only those unused modules and ignore their optional
    tensors when loading an older base checkpoint that still contains them.

    Returns ``False`` when this vLLM build has no importable native Gemma4
    implementation, allowing the older generic fallback below to handle it.
    """

    if attention_cls is None or conditional_generation_cls is None:
        try:
            from vllm.model_executor.models.gemma4 import Gemma4Attention
            from vllm.model_executor.models.gemma4_mm import (
                Gemma4ForConditionalGeneration,
            )
        except (ImportError, ModuleNotFoundError):
            return False
        attention_cls = Gemma4Attention
        conditional_generation_cls = Gemma4ForConditionalGeneration

    original_attention_init = attention_cls.__init__
    if not getattr(original_attention_init, "_r2opl_gemma4_shared_kv", False):

        def init_without_unused_shared_k_norm(self, *args, **kwargs):
            original_attention_init(self, *args, **kwargs)
            if getattr(self, "is_kv_shared_layer", False):
                # vLLM's shared-KV forward branch never reads k_norm.  Assign
                # None rather than a dummy Parameter so strict loading cannot
                # require a tensor that a merged Transformers model omits.
                self.k_norm = None

        init_without_unused_shared_k_norm._r2opl_gemma4_shared_kv = True
        attention_cls.__init__ = init_without_unused_shared_k_norm

    original_load_weights = conditional_generation_cls.load_weights
    if not getattr(original_load_weights, "_r2opl_gemma4_shared_kv", False):

        def load_weights_without_optional_shared_k_norm(self, weights):
            config = getattr(self, "config", None)
            text_config = getattr(config, "text_config", config)
            try:
                first_shared_layer = int(text_config.num_hidden_layers) - int(
                    getattr(text_config, "num_kv_shared_layers", 0)
                )
            except (AttributeError, TypeError, ValueError):
                return original_load_weights(self, weights)

            # Accept both the ordinary HF conditional-generation names and
            # the text-only/merged aliases handled by vLLM's weight mapper.
            name_pattern = re.compile(
                r"^(?:model\.language_model\.|language_model\.model\."
                r"|model\.|language_model\.)layers\.(\d+)\.self_attn\."
                r"k_norm\.weight$"
            )

            def relevant_weights():
                for name, weight in weights:
                    match = name_pattern.fullmatch(name)
                    if match is not None and int(match.group(1)) >= first_shared_layer:
                        continue
                    yield name, weight

            return original_load_weights(self, relevant_weights())

        load_weights_without_optional_shared_k_norm._r2opl_gemma4_shared_kv = True
        conditional_generation_cls.load_weights = (
            load_weights_without_optional_shared_k_norm
        )
    return True


def _apply_gemma4_vllm_compatibility_patches() -> None:
    """Apply the appropriate Gemma4 loading compatibility patch for vLLM."""

    native_gemma4_available = False
    try:
        import inspect
        from vllm.model_executor.models.registry import ModelRegistry

        # New registries require a full ModelConfig to resolve a class. The
        # native registry is sufficient here; actual loading stays in LLM.
        if "model_config" in inspect.signature(ModelRegistry.resolve_model_cls).parameters:
            native_gemma4_available = (
                "Gemma4ForConditionalGeneration"
                in ModelRegistry.get_supported_archs()
            )
        else:
            ModelRegistry.resolve_model_cls(("Gemma4ForConditionalGeneration",))
            native_gemma4_available = True
    except Exception:
        pass

    if native_gemma4_available:
        # Native Gemma4 still needs a narrow shared-KV fix.  Do not fall back
        # to the generic Transformers implementation when it is unavailable:
        # the registry has already selected the native model.
        _apply_native_gemma4_vllm_kv_sharing_patch()
        return

    from vllm.model_executor.models.utils import AutoWeightsLoader

    original = AutoWeightsLoader._add_loadable_non_param_tensors
    if not getattr(original, "_r2opl_loads_buffers", False):

        def add_loadable_buffers(self, module, child_params):
            original(self, module, child_params)
            for name, value in module.named_buffers(recurse=False):
                child_params.setdefault(name, value)

        add_loadable_buffers._r2opl_loads_buffers = True
        AutoWeightsLoader._add_loadable_non_param_tensors = add_loadable_buffers

    from vllm.model_executor.models.transformers.base import Base as TransformersBase
    from vllm.model_executor.models.transformers.multimodal import MultiModalMixin

    current_get_language_model = MultiModalMixin.get_language_model
    original_get_language_model = getattr(
        current_get_language_model,
        "_r2opl_original_get_language_model",
        current_get_language_model,
    )
    if not hasattr(current_get_language_model, "_r2opl_original_get_language_model"):

        def get_cached_language_model(self):
            return self._r2opl_language_model

        get_cached_language_model._r2opl_original_get_language_model = (
            original_get_language_model
        )
        MultiModalMixin.get_language_model = get_cached_language_model

    original_init = TransformersBase.__init__
    if not getattr(original_init, "_r2opl_gemma4_shared_kv", False):

        def init_with_gemma4_shared_kv(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if isinstance(self, MultiModalMixin):
                object.__setattr__(
                    self,
                    "_r2opl_language_model",
                    original_get_language_model(self),
                )
            if getattr(self.config, "model_type", None) == "gemma4":
                for suffix in ("k_norm", "v_norm", "k_proj", "v_proj"):
                    if suffix not in self.ignore_unexpected_suffixes:
                        self.ignore_unexpected_suffixes.append(suffix)

        init_with_gemma4_shared_kv._r2opl_gemma4_shared_kv = True
        TransformersBase.__init__ = init_with_gemma4_shared_kv


def _eos_token_ids(tokenizer: Any) -> set[int]:
    eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        return set()
    if isinstance(eos_ids, int):
        return {int(eos_ids)}
    return {int(value) for value in eos_ids}


def _trim_generated_ids(token_ids: list[int], tokenizer: Any) -> list[int]:
    """Remove batch padding while retaining the first generated EOS token."""

    eos_values = _eos_token_ids(tokenizer)
    for index, token_id in enumerate(token_ids):
        if int(token_id) in eos_values:
            return token_ids[: index + 1]
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        while token_ids and int(token_ids[-1]) == int(pad_id):
            token_ids.pop()
    return token_ids


def _gpu_free_gib(torch: Any) -> float:
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return float(free_bytes) / (1024**3)


def _assert_gpu_headroom(torch: Any, minimum_gib: float, *, gpu_id: int, stage: str) -> float:
    free_gib = _gpu_free_gib(torch)
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"Stopping at {stage} on GPU {gpu_id}: only {free_gib:.2f} GiB GPU "
            f"memory is free; the required reserve is {minimum_gib:.2f} GiB. "
            "Lower engine.gpu_memory_utilization or free the GPU first."
        )
    return free_gib


def _gpu_worker(
    gpu_id: int,
    model_dir: str,
    worker_config: Mapping[str, Any],
    job_queue: Any,
    result_queue: Any,
) -> None:
    """Load one full vLLM replica on one visible GPU and consume job batches."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    family = str(worker_config["family"])
    try:
        if family == "gemini4":
            _apply_gemma4_vllm_compatibility_patches()

        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        from utils.answer_verifier import verify_dataset_response_answer

        min_free_gib = float(worker_config["min_free_gib"])
        if min_free_gib > 0:
            _assert_gpu_headroom(
                torch, min_free_gib, gpu_id=gpu_id, stage="before model load"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=bool(worker_config["trust_remote_code"]),
            local_files_only=True,
        )
        engine_kwargs: dict[str, Any] = {
            "model": model_dir,
            "tokenizer": model_dir,
            "tensor_parallel_size": int(worker_config["tensor_parallel_size"]),
            "dtype": str(worker_config["dtype"]),
            "trust_remote_code": bool(worker_config["trust_remote_code"]),
            "max_model_len": int(worker_config["max_model_len"]),
            "max_num_seqs": int(worker_config["rollouts_per_gpu_batch"]),
            "gpu_memory_utilization": float(
                worker_config["gpu_memory_utilization"]
            ),
            "enable_prefix_caching": bool(
                worker_config["enable_prefix_caching"]
            ),
            "seed": int(worker_config["seed"]),
        }
        if family == "gemini4":
            # Text-only reasoning: skip the image/audio towers entirely.
            engine_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}
        llm = LLM(**engine_kwargs)
        if min_free_gib > 0:
            free_gib = _assert_gpu_headroom(
                torch, min_free_gib, gpu_id=gpu_id, stage="after model load"
            )
            print(
                f"GPU_HEADROOM gpu={gpu_id} free_gib={free_gib:.2f} "
                f"reserve_gib={min_free_gib:.2f}",
                flush=True,
            )

        def decode_prefix(ids: Sequence[int]) -> str:
            return tokenizer.decode(
                list(ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        result_queue.put(
            {"kind": "ready", "ok": True, "gpu": gpu_id, "time": utc_now()}
        )
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "ready",
                "ok": False,
                "gpu": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        job = job_queue.get()
        if job is None:
            return
        job_kind = str(job.get("kind", "eval_batch"))
        try:
            if job_kind == "eval_batch":
                message = _run_eval_batch_job(
                    llm=llm,
                    tokenizer=tokenizer,
                    decode_prefix=decode_prefix,
                    verify=verify_dataset_response_answer,
                    SamplingParams=SamplingParams,
                    TokensPrompt=TokensPrompt,
                    gpu_id=gpu_id,
                    job=job,
                    worker_config=worker_config,
                )
            elif job_kind == "generate":
                message = _run_generate_job(
                    llm=llm,
                    SamplingParams=SamplingParams,
                    TokensPrompt=TokensPrompt,
                    gpu_id=gpu_id,
                    job=job,
                )
            else:
                raise ValueError(f"Unknown worker job kind: {job_kind!r}")
            result_queue.put(message)
        except BaseException as exc:
            result_queue.put(
                {
                    "kind": job_kind,
                    "ok": False,
                    "gpu": gpu_id,
                    "batch_id": job.get("batch_id"),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            return


def _run_eval_batch_job(
    *,
    llm: Any,
    tokenizer: Any,
    decode_prefix: Callable[[Sequence[int]], str],
    verify: Callable[[str, str, str], float],
    SamplingParams: Any,
    TokensPrompt: Any,
    gpu_id: int,
    job: Mapping[str, Any],
    worker_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate real questions only, up to rollouts_per_gpu_batch rollouts."""

    real_tasks = job["tasks"]
    questions_per_batch = int(worker_config["questions_per_gpu_batch"])
    if not 1 <= len(real_tasks) <= questions_per_batch:
        raise ValueError(f"Invalid question batch size: {len(real_tasks)}")

    generated_rollouts = len(real_tasks) * int(worker_config["n"])
    if generated_rollouts > int(worker_config["rollouts_per_gpu_batch"]):
        raise AssertionError(
            "Each GPU generate call must contain at most "
            f"{worker_config['rollouts_per_gpu_batch']} rollouts; got "
            f"{generated_rollouts}."
        )

    sampling_params = SamplingParams(
        n=int(worker_config["n"]),
        temperature=float(worker_config["temperature"]),
        top_p=float(worker_config["top_p"]),
        top_k=int(worker_config["top_k"]),
        max_tokens=int(worker_config["max_new_tokens"]),
        seed=int(worker_config["seed"]),
        skip_special_tokens=True,
    )
    started = time.monotonic()
    outputs = llm.generate(
        [
            TokensPrompt(prompt_token_ids=list(task["prompt_token_ids"]))
            for task in real_tasks
        ],
        sampling_params,
        use_tqdm=False,
    )
    if len(outputs) != len(real_tasks):
        raise RuntimeError(
            f"Expected {len(real_tasks)} request outputs, got {len(outputs)}."
        )

    collect_ids = set(worker_config.get("collect_ids_datasets", ()))
    question_results: list[dict[str, Any]] = []
    for task, request_output in zip(
        real_tasks, outputs[: len(real_tasks)], strict=True
    ):
        completions = request_output.outputs
        if len(completions) != int(worker_config["n"]):
            raise RuntimeError(
                f"{task['dataset']}[{task['question_index']}] returned "
                f"{len(completions)} rollouts, expected {worker_config['n']}."
            )
        counters = {
            str(limit): {
                "rollouts": 0,
                "correct_count": 0,
                "token_count": 0,
                "truncated_count": 0,
            }
            for limit in worker_config["length_control"]
        }
        rollouts: list[dict[str, Any]] | None = (
            [] if str(task["dataset"]) in collect_ids else None
        )
        for completion in completions:
            raw_finish_reason = completion.finish_reason
            finish_reason = (
                None
                if raw_finish_reason is None
                else str(getattr(raw_finish_reason, "value", raw_finish_reason))
            )
            token_ids = [int(value) for value in completion.token_ids]
            scored = score_completion_at_lengths(
                token_ids=token_ids,
                full_text=completion.text,
                finish_reason=finish_reason,
                answer=str(task["answer"]),
                length_control=worker_config["length_control"],
                decode=decode_prefix,
                verify=lambda response, answer: verify(
                    response, answer, str(task["dataset"])
                ),
            )
            full_correct = scored[int(worker_config["length_control"][-1])]["correct"]
            for limit, values in scored.items():
                target = counters[str(limit)]
                target["rollouts"] += 1
                target["correct_count"] += values["correct"]
                target["token_count"] += values["tokens"]
                target["truncated_count"] += values["truncated"]
            if rollouts is not None:
                rollouts.append(
                    {
                        "response_token_ids": _trim_generated_ids(
                            list(token_ids), tokenizer
                        ),
                        "finish_reason": finish_reason,
                        "correct": bool(full_correct),
                    }
                )
        record: dict[str, Any] = {
            "dataset": task["dataset"],
            "question_index": int(task["question_index"]),
            "lengths": counters,
        }
        if rollouts is not None:
            record["rollouts"] = rollouts
        question_results.append(record)

    return {
        "kind": "eval_batch",
        "ok": True,
        "gpu": gpu_id,
        "batch_id": int(job["batch_id"]),
        "elapsed_seconds": time.monotonic() - started,
        "questions": question_results,
    }


def _run_generate_job(
    *,
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    gpu_id: int,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Token-exact continuations for ERSR; up to rollouts_per_gpu_batch at once.

    Every request carries its own SamplingParams so the per-arm seed
    arithmetic from ``utils/ersr.py`` stays reproducible. Requests are never
    padded with duplicate continuations.
    """

    requests = job["requests"]
    if not 1 <= len(requests) <= int(1e9):
        raise ValueError(f"Invalid generate job size: {len(requests)}")
    started = time.monotonic()
    outputs = llm.generate(
        [
            TokensPrompt(prompt_token_ids=list(request["ids"]))
            for request in requests
        ],
        [
            SamplingParams(
                n=1,
                temperature=float(request["temperature"]),
                top_p=float(request["top_p"]),
                top_k=int(request["top_k"]),
                max_tokens=int(request["max_tokens"]),
                seed=int(request["seed"]),
                skip_special_tokens=True,
            )
            for request in requests
        ],
        use_tqdm=False,
    )
    if len(outputs) != len(requests):
        raise RuntimeError(
            f"Expected {len(requests)} continuation outputs, got {len(outputs)}."
        )
    return {
        "kind": "generate",
        "ok": True,
        "gpu": gpu_id,
        "batch_id": int(job["batch_id"]),
        "elapsed_seconds": time.monotonic() - started,
        "outputs": [
            [int(value) for value in output.outputs[0].token_ids]
            for output in outputs
        ],
    }


def _unexpected_worker_exits(processes: Sequence[mp.Process]) -> str | None:
    exited = [
        f"pid={process.pid}, exitcode={process.exitcode}"
        for process in processes
        if process.exitcode is not None
    ]
    return "; ".join(exited) if exited else None


def _await_messages(
    *,
    result_queue: Any,
    processes: Sequence[mp.Process],
    expected_kind: str,
    count: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    while len(messages) < count:
        try:
            message = result_queue.get(timeout=15)
        except queue.Empty:
            unexpected_exits = _unexpected_worker_exits(processes)
            if unexpected_exits:
                raise RuntimeError(
                    f"GPU workers exited before all {expected_kind} messages "
                    f"arrived at {progress_label}: {unexpected_exits}"
                )
            if time.monotonic() - last_heartbeat >= 60:
                print(
                    f"HEARTBEAT {progress_label} kind={expected_kind} "
                    f"remaining={count - len(messages)} time={utc_now()}",
                    flush=True,
                )
                last_heartbeat = time.monotonic()
            continue
        if message.get("kind") != expected_kind:
            raise RuntimeError(
                f"Unexpected worker message kind: {message.get('kind')!r}"
            )
        if not message.get("ok"):
            raise RuntimeError(
                f"GPU {message.get('gpu')} failed at {progress_label}: "
                f"{message.get('error')}\n{message.get('traceback', '')}"
            )
        messages.append(message)
        if expected_kind == "ready":
            print(
                f"READY {progress_label} gpu={message['gpu']} "
                f"workers={len(messages)}/{count}",
                flush=True,
            )
        else:
            print(
                f"BATCH_DONE {progress_label} batch={message['batch_id']} "
                f"gpu={message['gpu']} elapsed={message['elapsed_seconds']:.1f}s "
                f"completed={len(messages)}/{count}",
                flush=True,
            )
    return messages


def _stop_workers(
    job_queue: Any, processes: Sequence[mp.Process], *, force: bool
) -> None:
    if force:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        return
    for _ in processes:
        job_queue.put(None)
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)


class GpuPool:
    """One spawned worker per configured GPU sharing a dynamic job queue.

    Jobs are consumed work-stealing style, so faster GPUs naturally take more
    batches.  Workers only exit (and release their GPU) when the pool closes.
    """

    def __init__(
        self,
        *,
        config: EvaluationConfig,
        model_dir: Path | str,
        family: str,
        collect_ids_datasets: Sequence[str] = (),
    ) -> None:
        self.config = config
        self.model_dir = str(model_dir)
        self.context = mp.get_context("spawn")
        self.job_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        worker_config: dict[str, Any] = {
            "family": family,
            "seed": config.seed,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "n": config.n,
            "max_new_tokens": config.max_new_tokens,
            "length_control": list(config.length_control),
            "tensor_parallel_size": config.tensor_parallel_size,
            "rollouts_per_gpu_batch": config.rollouts_per_gpu_batch,
            "questions_per_gpu_batch": config.questions_per_gpu_batch,
            "min_free_gib": config.min_free_gib,
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "enable_prefix_caching": config.enable_prefix_caching,
            "collect_ids_datasets": list(collect_ids_datasets),
        }
        self.processes = [
            self.context.Process(
                target=_gpu_worker,
                args=(
                    gpu_id,
                    self.model_dir,
                    worker_config,
                    self.job_queue,
                    self.result_queue,
                ),
                name=f"r2opl-eval-gpu{gpu_id}",
            )
            for gpu_id in config.gpus
        ]

    def __enter__(self) -> "GpuPool":
        print(
            f"GPU_POOL_START model={self.model_dir} gpus={list(self.config.gpus)}",
            flush=True,
        )
        started_ok = False
        for process in self.processes:
            process.start()
        try:
            _await_messages(
                result_queue=self.result_queue,
                processes=self.processes,
                expected_kind="ready",
                count=len(self.processes),
                progress_label=f"startup model={Path(self.model_dir).name}",
            )
            started_ok = True
        finally:
            if not started_ok:
                _stop_workers(self.job_queue, self.processes, force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _stop_workers(self.job_queue, self.processes, force=exc_type is not None)

    def run(
        self, jobs: Sequence[Mapping[str, Any]], *, progress_label: str
    ) -> list[dict[str, Any]]:
        if not jobs:
            return []
        kind = str(jobs[0].get("kind", "eval_batch"))
        for batch_id, job in enumerate(jobs):
            payload = dict(job)
            payload["batch_id"] = batch_id
            self.job_queue.put(payload)
        messages = _await_messages(
            result_queue=self.result_queue,
            processes=self.processes,
            expected_kind=kind,
            count=len(jobs),
            progress_label=progress_label,
        )
        messages.sort(key=lambda message: int(message["batch_id"]))
        return messages


def evaluate_model(
    *,
    config: EvaluationConfig,
    model: ModelSpec,
    tasks: Sequence[Mapping[str, Any]],
    dataset_question_counts: Mapping[str, int],
    collect_ids_datasets: Sequence[str] = (),
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    """Evaluate one model with one independent full replica per GPU.

    Returns the aggregated dataset summaries and, for the ERSR-eligible
    datasets, per-rollout validation records with the original token IDs.
    """

    batch_size = min(config.questions_per_gpu_batch, max(1, (len(tasks) + len(config.gpus) - 1) // len(config.gpus)))
    batches = make_batches(tasks, batch_size)
    jobs = [{"kind": "eval_batch", "tasks": batch} for batch in batches]
    with GpuPool(
        config=config,
        model_dir=model.path,
        family=model.family,
        collect_ids_datasets=collect_ids_datasets,
    ) as pool:
        messages = pool.run(jobs, progress_label=f"model={model.name}")

    question_results = [
        question for message in messages for question in message["questions"]
    ]
    datasets = aggregate_question_results(
        question_results=question_results,
        dataset_question_counts=dataset_question_counts,
        dataset_order=list(config.val_datasets),
        length_control=config.length_control,
        n=config.n,
    )

    tasks_by_key = {
        (task["dataset"], int(task["question_index"])): task for task in tasks
    }
    records: list[dict[str, Any]] = []
    for message in messages:
        for question in message["questions"]:
            rollouts = question.get("rollouts")
            if rollouts is None:
                continue
            key = (str(question["dataset"]), int(question["question_index"]))
            task = tasks_by_key[key]
            for rollout_index, rollout in enumerate(rollouts):
                records.append(
                    {
                        "dataset": str(question["dataset"]),
                        "question_index": int(question["question_index"]),
                        "uid": (
                            f"{question['dataset']}:"
                            f"{question['question_index']}:{rollout_index}"
                        ),
                        "record_index": len(records),
                        "answer": str(task["answer"]),
                        "correct": bool(rollout["correct"]),
                        "prompt_ids": list(task["prompt_token_ids"]),
                        "response_ids": list(rollout["response_token_ids"]),
                    }
                )
    return datasets, records


# ---------------------------------------------------------------------------
# ERSR evaluation phases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    ids: list[int]
    max_tokens: int
    seed: int
    temperature: float
    top_p: float
    top_k: int
    request_id: str


def run_generation_phase(
    *,
    config: EvaluationConfig,
    model: ModelSpec,
    requests: Sequence[GenerationRequest],
    progress_label: str,
) -> dict[str, list[int]]:
    """Distribute token-exact generation requests across the GPU pool."""

    if not requests:
        return {}
    # Small ERSR selections should still use every available worker.
    cap = min(config.rollouts_per_gpu_batch, (len(requests) + len(config.gpus) - 1) // len(config.gpus))
    jobs = [
        {
            "kind": "generate",
            "requests": [
                {
                    "ids": request.ids,
                    "max_tokens": request.max_tokens,
                    "seed": request.seed,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                }
                for request in requests[start : start + cap]
            ],
        }
        for start in range(0, len(requests), cap)
    ]
    with GpuPool(config=config, model_dir=model.path, family=model.family) as pool:
        messages = pool.run(jobs, progress_label=progress_label)
    outputs: dict[str, list[int]] = {}
    flat_outputs = [ids for message in messages for ids in message["outputs"]]
    if len(flat_outputs) != len(requests):
        raise RuntimeError(
            f"Generation phase returned {len(flat_outputs)} outputs for "
            f"{len(requests)} requests."
        )
    for request, ids in zip(requests, flat_outputs, strict=True):
        outputs[request.request_id] = ids
    return outputs


def _ersr_settings_document(config: EvaluationConfig) -> dict[str, Any]:
    assert config.ersr is not None
    settings = config.ersr.settings
    return {
        "actions": list(config.ersr.actions),
        "datasets": list(settings.datasets),
        "max_steps_per_dataset": settings.max_steps_per_dataset,
        "step_cap_scope": "per_dataset_per_action",
        "mc_k": settings.mc_k,
        "max_teacher_step_tokens": settings.max_teacher_step_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "seed": settings.seed,
        "max_response_tokens": settings.max_response_tokens,
        "teacher_gpu_memory_utilization": config.ersr.gpu_memory_utilization,
        "teacher": (
            None
            if config.ersr.teacher is None
            else {
                "name": config.ersr.teacher.name,
                "path": str(config.ersr.teacher.path),
                "family": config.ersr.teacher.family,
            }
        ),
        "teachers": {
            item.family: {"name": item.name, "path": str(item.path)}
            for item in config.ersr.teachers
        },
    }


@dataclass
class ERSRWork:
    model: ModelSpec
    tokenizer: Any
    cases: list[dict[str, Any]]
    document: dict[str, Any]
    replacements: dict[str, list[int]] = field(default_factory=dict)
    invalid: dict[str, dict[str, Any]] = field(default_factory=dict)
    teacher_complete: bool = False


def prepare_ersr(
    *,
    config: EvaluationConfig,
    model: ModelSpec,
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> ERSRWork:
    """Select interventions on CPU; retain only selected trajectories."""

    if config.ersr is None:
        raise ValueError("ERSR is not enabled in this evaluation config.")
    from utils.ersr import build_ersr_step_cases

    settings = config.ersr.settings
    include_student = ERSR_STUDENT_ACTION in config.ersr.actions
    include_teacher = ERSR_TEACHER_REPLACE in config.ersr.actions

    cases, selection = build_ersr_step_cases(
        records,
        tokenizer,
        settings,
        include_student_action=include_student,
        include_teacher_replace=include_teacher,
    )
    print(
        f"ERSR_CASES model={model.name} selected={len(cases)} available={selection['available_steps']}",
        flush=True,
    )

    document: dict[str, Any] = {
        "settings": _ersr_settings_document(config),
        "selection": selection,
        "cases": [
            {
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "action_type": case["action_type"],
                "uid": case["uid"],
                "step_index": case["step_index"],
                "step_token_start": case["step_token_start"],
                "step_token_end": case["step_token_end"],
                "seed": case["seed"],
            }
            for case in cases
        ],
    }
    teacher = ersr_teacher_for_model(config, model)
    document["teacher"] = None if teacher is None else {
        "name": teacher.name, "path": str(teacher.path), "family": teacher.family,
    }
    return ERSRWork(model=model, tokenizer=tokenizer, cases=cases, document=document)


def generate_ersr_replacements(config: EvaluationConfig, works: Sequence[ERSRWork]) -> None:
    """Load each teacher once for all checkpoints, then release all its GPUs."""
    from utils.ersr import first_replacement_step_ids

    assert config.ersr is not None
    settings = config.ersr.settings
    groups: dict[ModelSpec, list[tuple[ERSRWork, dict[str, Any], GenerationRequest]]] = {}
    for work_index, work in enumerate(works):
        teacher = ersr_teacher_for_model(config, work.model)
        if teacher is None:
            work.teacher_complete = True
            continue
        for case in work.cases:
            if case["action_type"] != ERSR_TEACHER_REPLACE:
                continue
            before = case["response_ids"][: case["step_token_start"]]
            budget = min(
                settings.max_teacher_step_tokens,
                settings.max_response_tokens - len(before),
            )
            if budget <= 0:
                work.invalid[case["case_id"]] = {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "action_type": case["action_type"],
                    "valid": False,
                    "skip_reason": "no_teacher_budget",
                }
                continue
            request = GenerationRequest(
                    ids=[int(v) for v in case["prompt_ids"]]
                    + [int(v) for v in before],
                    max_tokens=budget,
                    seed=int(case["seed"]) + 5_000,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    top_k=settings.top_k,
                    # A case ID can repeat in different checkpoints.
                    request_id=f"teacher:{work_index}:{work.model.name}:{case['case_id']}",
            )
            groups.setdefault(teacher, []).append((work, case, request))
    for teacher, rows in groups.items():
        print(f"ERSR_TEACHER_START teacher={teacher.name} requests={len(rows)} gpus={list(config.gpus)}", flush=True)
        teacher_config = replace(
            config,
            gpu_memory_utilization=config.ersr.gpu_memory_utilization,
        )
        proposal_outputs = run_generation_phase(
            config=teacher_config,
            model=teacher,
            requests=[request for _, _, request in rows],
            progress_label=f"ersr-teacher teacher={teacher.name}",
        )
        for work, case, request in rows:
            replacement = first_replacement_step_ids(
                proposal_outputs[request.request_id], work.tokenizer
            )
            if not replacement:
                work.invalid[case["case_id"]] = {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "action_type": case["action_type"],
                    "valid": False,
                    "skip_reason": "empty_teacher_replacement",
                }
            else:
                work.replacements[case["case_id"]] = replacement
        print(f"ERSR_TEACHER_DONE teacher={teacher.name}", flush=True)
    for work in works:
        work.teacher_complete = True


def evaluate_ersr_student(config: EvaluationConfig, work: ERSRWork) -> dict[str, Any]:
    """Compute both MC arms after all teacher replacements have finished.

    Seeds match training: baseline +j, action +1000+j, teacher +5000.
    Only standalone evaluation uses dataset-specific continuation grading.
    """
    from utils.ersr import aggregate_ersr_advantages, reward_for_continuation

    assert config.ersr is not None
    if not work.teacher_complete:
        raise RuntimeError("Teacher replacements must complete before student continuations.")
    settings = config.ersr.settings
    model, tokenizer, cases = work.model, work.tokenizer, work.cases
    replacements, invalid, document = work.replacements, work.invalid, work.document
    case_arms: dict[str, dict[str, dict[str, Any]]] = {}

    def arm_requests(
        case: Mapping[str, Any], *, prefix: Sequence[int], arm_offset: int
    ) -> list[GenerationRequest]:
        remaining = settings.max_response_tokens - len(prefix)
        if remaining <= 0:
            return []
        return [
            GenerationRequest(
                ids=[int(v) for v in case["prompt_ids"]]
                + [int(v) for v in prefix],
                max_tokens=remaining,
                seed=int(case["seed"]) + arm_offset + sample_index,
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
                request_id=f"{case['case_id']}:{arm_offset}:{sample_index}",
            )
            for sample_index in range(settings.mc_k)
        ]

    student_requests: list[GenerationRequest] = []
    for case in cases:
        case_id = case["case_id"]
        if case_id in invalid:
            continue
        response_ids = case["response_ids"]
        start, end = case["step_token_start"], case["step_token_end"]
        before = response_ids[:start]
        if case["action_type"] == ERSR_STUDENT_ACTION:
            action_prefix = response_ids[:end]
        else:
            replacement = replacements.get(case_id)
            if replacement is None:
                continue  # already recorded as invalid above
            action_prefix = before + replacement
        arms = {
            "baseline": {"prefix": list(before), "requests": arm_requests(
                case, prefix=before, arm_offset=0
            )},
            "action": {"prefix": list(action_prefix), "requests": arm_requests(
                case, prefix=action_prefix, arm_offset=1_000
            )},
        }
        case_arms[case_id] = arms
        student_requests.extend(arms["baseline"]["requests"])
        student_requests.extend(arms["action"]["requests"])

    continuation_outputs = run_generation_phase(
        config=config,
        model=model,
        requests=student_requests,
        progress_label=f"ersr-student model={model.name}",
    )

    results: list[dict[str, Any]] = list(invalid.values())
    for case in cases:
        case_id = case["case_id"]
        if case_id not in case_arms:
            continue
        arms = case_arms[case_id]
        values: dict[str, float] = {}
        empty_counts: dict[str, int] = {}
        for arm_name, arm in arms.items():
            rewards = [
                reward_for_continuation(
                    tokenizer,
                    response_prefix_ids=arm["prefix"],
                    continuation_ids=continuation_outputs[request.request_id],
                    answer=str(case["answer"]),
                    data_source=str(case["dataset"]),
                )
                for request in arm["requests"]
            ]
            if arm["requests"]:
                empty_counts[arm_name] = sum(
                    1
                    for request in arm["requests"]
                    if not continuation_outputs[request.request_id]
                )
            else:
                # Budget exhausted: every MC sample has an empty continuation
                # (utils.ersr counts these the same way).
                empty_counts[arm_name] = settings.mc_k
            if rewards:
                values[arm_name] = sum(rewards) / settings.mc_k
            else:
                # Budget exhausted: the fixed prefix alone is the sample.
                values[arm_name] = reward_for_continuation(
                    tokenizer,
                    response_prefix_ids=arm["prefix"],
                    continuation_ids=[],
                    answer=str(case["answer"]),
                    data_source=str(case["dataset"]),
                )
        results.append(
            {
                "case_id": case_id,
                "dataset": case["dataset"],
                "action_type": case["action_type"],
                "valid": True,
                "baseline_value": float(values["baseline"]),
                "action_value": float(values["action"]),
                "advantage": float(values["action"] - values["baseline"]),
                "mc_k": settings.mc_k,
                "baseline_empty_continuations": int(empty_counts["baseline"]),
                "action_empty_continuations": int(empty_counts["action"]),
                "replacement_num_tokens": (
                    len(replacements[case_id])
                    if case["action_type"] == ERSR_TEACHER_REPLACE
                    else None
                ),
            }
        )

    means, skipped = aggregate_ersr_advantages(
        results, settings.datasets, config.ersr.actions
    )
    document.update({
        "results": results, "means": means, "skipped": skipped,
        "replacements": replacements,
    })
    return document


def evaluate_ersr(
    *, config: EvaluationConfig, model: ModelSpec,
    records: Sequence[Mapping[str, Any]], tokenizer: Any,
) -> dict[str, Any]:
    """Direct-model mode completes all phases before loading the next model."""
    work = prepare_ersr(config=config, model=model, records=records, tokenizer=tokenizer)
    generate_ersr_replacements(config, [work])
    return evaluate_ersr_student(config, work)


# ---------------------------------------------------------------------------
# Result documents
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _evaluation_metadata(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "config": str(config.config_path),
        "prompt_template": config.prompt_name,
        "seed": config.seed,
        "data": {
            "val_files": [str(path) for path in config.val_files],
            "val_datasets": list(config.val_datasets),
        },
        "sampling": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "n": config.n,
            "max_new_tokens": config.max_new_tokens,
        },
        "length_control": list(config.length_control),
        "runtime": {
            "gpus": list(config.gpus),
            "tensor_parallel_size": config.tensor_parallel_size,
            "rollouts_per_gpu_batch": config.rollouts_per_gpu_batch,
            "questions_per_gpu_batch": config.questions_per_gpu_batch,
            "min_free_gib": config.min_free_gib,
            "full_model_replica_per_gpu": True,
        },
        "engine": {
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "enable_prefix_caching": config.enable_prefix_caching,
        },
    }


def _write_audit_rollouts(
    config: EvaluationConfig,
    model: ModelSpec,
    tasks: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> Path:
    """Persist the ERSR-eligible raw rollouts (with token IDs) for audits."""

    tasks_by_key = {
        (task["dataset"], int(task["question_index"])): task for task in tasks
    }
    if config.source_type == "models":
        path = config.model_result_dir() / f"{model.name}.ersr_rollouts.jsonl"
    else:
        path = (
            config.result_dir
            / f"{config.run_name}.{model.name}.ersr_rollouts.jsonl"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            task = tasks_by_key[(record["dataset"], record["question_index"])]
            handle.write(
                json.dumps(
                    {
                        "model": model.name,
                        "dataset": record["dataset"],
                        "question_index": record["question_index"],
                        "uid": record["uid"],
                        "prompt": task["prompt"],
                        "prompt_ids": record["prompt_ids"],
                        "response_ids": record["response_ids"],
                        "response": tokenizer.decode(
                            record["response_ids"],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        "answer": task["answer"],
                        "correct": record["correct"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    return path


def print_metric_summary(
    label: str, datasets: Mapping[str, Mapping[str, Mapping[str, Any]]], n: int
) -> None:
    final_limit = max(int(limit) for group in datasets.values() for limit in group)
    avg_label = f"Avg@{n}"
    print(f"METRICS {label} (full length {final_limit}):", flush=True)
    for dataset, lengths in datasets.items():
        summary = lengths[str(final_limit)]
        print(
            f"  {dataset}: {avg_label}={summary[avg_label]:.4f} "
            f"Pass@{n}={summary[f'Pass@{n}']:.4f} "
            f"mean_length={summary['mean_length']:.1f} "
            f"truncation_rate={summary['truncation_rate']:.4f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_validation_plan(
    config: EvaluationConfig,
    checkpoints: Sequence[CheckpointSpec],
    question_counts: Mapping[str, int],
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        **_evaluation_metadata(config),
        "source_type": config.source_type,
        "run_name": config.run_name,
        "datasets": dict(question_counts),
        "total_questions": sum(question_counts.values()),
    }
    if config.source_type == "checkpoints":
        plan["training_output"] = str(config.training_output)
        plan["checkpoints"] = [
            {
                "step_num": checkpoint.step_num,
                "source_kind": checkpoint.source_kind,
                "source_dir": str(checkpoint.source_dir),
            }
            for checkpoint in checkpoints
        ]
        plan["result_path"] = str(config.result_path)
    else:
        plan["models"] = [
            {
                "name": model.name,
                "path": str(model.path),
                "family": model.family,
                "chat_template_path": None if model.chat_template_path is None else str(model.chat_template_path),
                "result_path": str(config.model_result_path(model)),
            }
            for model in config.models
        ]
        plan["summary_path"] = str(config.summary_result_path())
    if config.ersr is not None:
        plan["ersr"] = _ersr_settings_document(config)
        plan["ersr"]["schedule"] = (
            "all_base_then_shared_teacher_then_students"
            if config.source_type == "checkpoints" else "per_model"
        )
    return plan


def _load_driver_tokenizer(config: EvaluationConfig, model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
    )


def load_prompt_tokenizer(config: EvaluationConfig, model: ModelSpec) -> Any:
    """Use model-native tokenization/decoding; optionally override only Jinja."""
    tokenizer = _load_driver_tokenizer(config, model.path)
    if model.chat_template_path is not None:
        tokenizer.chat_template = model.chat_template_path.read_text(encoding="utf-8")
    return tokenizer


def validate_tokenizers(config: EvaluationConfig, models: Sequence[ModelSpec]) -> None:
    """Validate token-exact teacher/student handoff before merging or using GPUs."""
    teacher_vocabularies: dict[Path, dict[str, int]] = {}
    for model in models:
        tokenizer = load_prompt_tokenizer(config, model)
        teacher = ersr_teacher_for_model(config, model)
        if teacher is None:
            continue
        if teacher.path not in teacher_vocabularies:
            validate_model_context(teacher.path, config.max_model_len)
            teacher_vocabularies[teacher.path] = _load_driver_tokenizer(config, teacher.path).get_vocab()
        if tokenizer.get_vocab() != teacher_vocabularies[teacher.path]:
            raise ValueError(
                f"ERSR teacher {teacher.name} and student {model.name} must have identical "
                "token-to-ID vocabularies for token-exact replacement."
            )


def _evaluate_base_model(
    *,
    config: EvaluationConfig,
    model: ModelSpec,
    questions: Sequence[Mapping[str, Any]],
    question_counts: Mapping[str, int],
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Generate base rollouts and persist them before any ERSR work starts."""

    tokenizer = load_prompt_tokenizer(config, model)
    validate_model_context(model.path, config.max_model_len)
    tasks = render_model_tasks(
        config, tokenizer=tokenizer, family=model.family, questions=questions
    )
    collect_ids_datasets: list[str] = (
        list(config.val_datasets) if config.save_rollouts else []
    )
    if config.ersr is not None:
        ersr_datasets = set(config.ersr.settings.datasets)
        collect_ids_datasets = list(
            dict.fromkeys(
                collect_ids_datasets
                + [name for name in config.val_datasets if name in ersr_datasets]
            )
        )
    datasets_summary, records = evaluate_model(
        config=config,
        model=model,
        tasks=tasks,
        dataset_question_counts=question_counts,
        collect_ids_datasets=collect_ids_datasets,
    )
    print_metric_summary(model.name, datasets_summary, config.n)

    result: dict[str, Any] = {
        "model_name": model.name,
        "model_path": str(model.path),
        "model_family": model.family,
        "chat_template_path": None if model.chat_template_path is None else str(model.chat_template_path),
        "base_completed_at": utc_now(),
        "status": "base_completed",
        "datasets": datasets_summary,
        "ersr": None,
    }
    if config.save_rollouts and records:
        result["rollouts_path"] = str(
            _write_audit_rollouts(config, model, tasks, records, tokenizer)
        )
    return result, tokenizer, records


def _evaluate_one_model(
    *, config: EvaluationConfig, model: ModelSpec,
    questions: Sequence[Mapping[str, Any]], question_counts: Mapping[str, int],
) -> dict[str, Any]:
    result, tokenizer, records = _evaluate_base_model(
        config=config, model=model, questions=questions, question_counts=question_counts,
    )
    # Keep Accuracy/length metrics even if the expensive ERSR phase fails.
    atomic_write_json(config.model_result_path(model), {
        "run_name": config.run_name, **_evaluation_metadata(config), **result,
    })
    if config.ersr is not None:
        result["ersr"] = evaluate_ersr(
            config=config,
            model=model,
            records=records,
            tokenizer=tokenizer,
        )
        for series, value in result["ersr"]["means"].items():
            print(f"  ERSR {series}: mean_advantage={value:.4f}", flush=True)
    result.update(status="completed", completed_at=utc_now())
    return result


def evaluate_checkpoints(
    config: EvaluationConfig, checkpoints: Sequence[CheckpointSpec],
    questions: Sequence[Mapping[str, Any]], question_counts: Mapping[str, int],
) -> None:
    if config.result_path.exists():
        raise FileExistsError(f"Result file already exists: {config.result_path}; choose a new run_name.")
    document: dict[str, Any] = {
        "run_name": config.run_name, "training_output": str(config.training_output),
        **_evaluation_metadata(config), "status": "running", "phase": "base",
        "started_at": utc_now(), "steps": [],
    }
    atomic_write_json(config.result_path, document)
    works: list[ERSRWork] = []
    try:
        # Retain merged weights on disk until their Student phase finishes:
        # each checkpoint is merged once, no GPU model remains resident.
        with ExitStack() as merged_models:
            cleanups: list[ExitStack] = []
            for checkpoint in checkpoints:
                cleanup = merged_models.enter_context(ExitStack())
                model_dir = cleanup.enter_context(prepare_model(config, checkpoint))
                model = _resolve_checkpoint_model(checkpoint, model_dir)
                print(f"STEP_BASE_START step={checkpoint.step_num} source={checkpoint.source_kind}", flush=True)
                result, tokenizer, records = _evaluate_base_model(
                    config=config, model=model, questions=questions, question_counts=question_counts,
                )
                result.update(
                    step_num=checkpoint.step_num, source_kind=checkpoint.source_kind,
                    source_dir=str(checkpoint.source_dir),
                )
                document["steps"].append(result)
                atomic_write_json(config.result_path, document)
                if config.ersr is not None:
                    works.append(prepare_ersr(config=config, model=model, records=records, tokenizer=tokenizer))
                    cleanups.append(cleanup)
                else:
                    result.update(status="completed", completed_at=utc_now())
                    cleanup.close()
                del records
                print(f"STEP_BASE_DONE step={checkpoint.step_num}", flush=True)

            if config.ersr is not None:
                document["phase"] = "teacher"
                atomic_write_json(config.result_path, document)
                generate_ersr_replacements(config, works)
                for result, work in zip(document["steps"], works, strict=True):
                    # Save proposals before loading any Student. This also
                    # makes interrupted runs auditable without regenerating.
                    result["ersr"] = {
                        **work.document, "status": "teacher_completed",
                        "replacements": work.replacements,
                        "invalid": list(work.invalid.values()),
                    }
                document["phase"] = "student"
                atomic_write_json(config.result_path, document)
                for result, work, cleanup in zip(document["steps"], works, cleanups, strict=True):
                    print(f"ERSR_STUDENT_START model={work.model.name} gpus={list(config.gpus)}", flush=True)
                    result["ersr"] = evaluate_ersr_student(config, work)
                    result.update(status="completed", completed_at=utc_now())
                    atomic_write_json(config.result_path, document)
                    cleanup.close()
                    print(f"STEP_DONE step={result['step_num']}", flush=True)
    except BaseException as exc:
        document.update(status="failed", error=repr(exc), updated_at=utc_now())
        atomic_write_json(config.result_path, document)
        raise
    document.update(status="completed", phase="completed", completed_at=utc_now())
    atomic_write_json(config.result_path, document)
    print(f"EVALUATION_DONE result={config.result_path}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate YAML, models, checkpoints, datasets, and budgets on CPU.",
    )
    return parser.parse_args(argv)


def _resolve_checkpoint_model(checkpoint: CheckpointSpec, model_dir: Path) -> ModelSpec:
    family = resolve_model_family(
        model_dir,
        explicit=(
            None if model_dir.name == "model" else model_dir.name
        ),
    )
    return ModelSpec(
        name=f"global_step_{checkpoint.step_num}",
        path=model_dir,
        family=family,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_evaluation_config(args.config)

    checkpoints: list[CheckpointSpec] = []
    tokenizer_dirs: list[Path]
    if config.source_type == "checkpoints":
        checkpoints = discover_checkpoints(config)
        tokenizer_dirs = [checkpoint.tokenizer_dir for checkpoint in checkpoints]
    else:
        tokenizer_dirs = [model.path for model in config.models]

    questions, question_counts = load_question_records(config)
    for tokenizer_dir in tokenizer_dirs:
        validate_model_context(tokenizer_dir, config.max_model_len)
    validate_tokenizers(config, (
        [_resolve_checkpoint_model(checkpoint, checkpoint.tokenizer_dir) for checkpoint in checkpoints]
        if config.source_type == "checkpoints" else config.models
    ))

    plan = build_validation_plan(config, checkpoints, question_counts)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        print(
            "VALIDATION_OK: no model was merged, loaded, or generated with.",
            flush=True,
        )
        return

    if config.source_type == "models":
        output_dir = config.model_result_dir()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Result directory is not empty: {output_dir}. Use a new "
                "run_name so results from different attempts cannot be mixed."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = config.summary_result_path()
        summary: dict[str, Any] = {
            "run_name": config.run_name,
            "source_type": HF_MODELS_TYPE,
            "status": "running",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            **_evaluation_metadata(config),
            "models": [
                {
                    "name": model.name,
                    "path": str(model.path),
                    "result_path": str(config.model_result_path(model)),
                    "status": "pending",
                }
                for model in config.models
            ],
        }
        atomic_write_json(summary_path, summary)
        for model in config.models:
            for entry in summary["models"]:
                if entry["name"] == model.name:
                    entry["status"] = "running"
            summary["updated_at"] = utc_now()
            atomic_write_json(summary_path, summary)
            print(
                f"MODEL_START run={config.run_name} model={model.name}", flush=True
            )
            try:
                result = _evaluate_one_model(
                    config=config,
                    model=model,
                    questions=questions,
                    question_counts=question_counts,
                )
                document = {
                    "run_name": config.run_name,
                    **_evaluation_metadata(config),
                    **result,
                }
                atomic_write_json(config.model_result_path(model), document)
            except BaseException as exc:
                for entry in summary["models"]:
                    if entry["name"] == model.name:
                        entry["status"] = "failed"
                        entry["error"] = repr(exc)
                summary["status"] = "failed"
                summary["updated_at"] = utc_now()
                atomic_write_json(summary_path, summary)
                raise
            for entry in summary["models"]:
                if entry["name"] == model.name:
                    entry["status"] = "completed"
            summary["updated_at"] = utc_now()
            atomic_write_json(summary_path, summary)
            print(
                f"MODEL_DONE run={config.run_name} model={model.name} "
                f"result={config.model_result_path(model)}",
                flush=True,
            )
        summary["status"] = "completed"
        summary["completed_at"] = utc_now()
        summary["updated_at"] = summary["completed_at"]
        atomic_write_json(summary_path, summary)
        print(f"EVALUATION_DONE summary={summary_path}", flush=True)
        return

    evaluate_checkpoints(config, checkpoints, questions, question_counts)


if __name__ == "__main__":
    main()
