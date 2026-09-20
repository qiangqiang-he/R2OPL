"""Staged one-GPU PG-OPD smoke test with real local checkpoints.

This is deliberately test-only.  It exercises two 4-question training batches
with four Student rollouts per question, scores those exact sampled token IDs
with the Teacher, performs two real Student optimizer steps, and evaluates
two AMC23 questions at Avg@4.  ERSR evaluation is retired and intentionally
absent.  Student and Teacher are never resident on CUDA at the same time.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


# Keep each vLLM phase isolated.  The process boundary, rather than vLLM sleep
# mode, releases GPU state and guarantees that Student and Teacher are never
# resident together.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The uv-cache Transformers/huggingface_hub copies were a fallback for the old
# `verl` conda env.  The r2opl env ships proper versions (transformers 5.10.4,
# required by vllm 0.29) and the cached transformers 5.9.0 would shadow them,
# so the fallback only applies when transformers is not installed at all.
try:
    import transformers  # noqa: F401

    _transformers_installed = True
except ImportError:
    _transformers_installed = False

if not _transformers_installed:
    for package_source in (
        Path("/home/qqh/.cache/uv/archive-v0/NX9sFckgqAAJCiWP"),
        Path("/home/qqh/.cache/uv/archive-v0/bb_MRsJKLYru8WZz"),
    ):
        if package_source.is_dir() and str(package_source) not in sys.path:
            sys.path.insert(0, str(package_source))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from algorithms import resolve_algorithm
from algorithms.pg_opd import signed_pg_opd_loss
from utils.answer_verifier import verify_response_answer
from utils.custom_dataset import (
    load_question_answer_records,
    render_length_limited_chat_prompt,
)
from utils.opd_runtime import (
    compute_pass_avg_metrics,
    configure_pg_opd_batch,
    configure_pg_opd_defaults,
    selected_avg_metrics,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "tests"
    / "configs"
    / "pg_opd"
    / "pg_opd_qwen3_4b_instruct_2507_to_1p7b_single_gpu_smoke.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "artifacts" / "pg_opd_single_gpu_smoke",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--learning-rate", type=float, default=1.0e-7)
    parser.add_argument(
        "--vllm-stage",
        choices=("student", "teacher"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def _compose_config(config_path: Path):
    config_path = config_path.resolve()
    config_root = (PROJECT_ROOT / "configs").resolve()
    test_config_root = (PROJECT_ROOT / "tests" / "configs").resolve()
    try:
        config_name = (
            config_path.relative_to(test_config_root).with_suffix("").as_posix()
        )
    except ValueError as error:
        raise ValueError(
            f"Smoke config must be under {test_config_root}"
        ) from error
    search_path = f"hydra.searchpath=[file://{config_root},pkg://verl.trainer.config]"
    with initialize_config_dir(version_base=None, config_dir=str(test_config_root)):
        config = compose(config_name=config_name, overrides=[search_path])
    OmegaConf.resolve(config)
    configure_pg_opd_defaults(config)
    configure_pg_opd_batch(config)
    resolve_algorithm(config).validate(config)
    if not bool(config.get("local_smoke", {}).get("staged_single_gpu", False)):
        raise ValueError("The selected config is not a staged single-GPU smoke config")
    return config


def _apply_gemma4_vllm_compatibility_patches() -> None:
    """Legacy Gemma 4 loading patches for vLLM's generic Transformers loader.

    vLLM >= 0.22 ships a native ``Gemma4ForConditionalGeneration`` backend that
    loads the local checkpoints directly (verified by
    ``tests/verify_vllm_model_families.py``), so when the registry resolves
    that architecture the legacy patches are skipped.  The patches below teach
    the old generic loader the local layout: persistent clipping-limit
    buffers, redundant per-layer K/V tensors for the KV-shared tail layers,
    and skipping the unused multimodal towers.
    """

    try:
        from vllm.model_executor.models.registry import ModelRegistry

        ModelRegistry.resolve_model_cls(("Gemma4ForConditionalGeneration",))
        return  # native Gemma 4 support present; no patches needed
    except Exception:
        pass

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
                # Build vLLM's dynamic language-model view once before
                # torch.compile starts; calling ``self.__class__.mro()`` from
                # inside the compiled forward is unsupported by Dynamo.
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


def _resolve_project_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cuda_free_gib() -> float:
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return float(free_bytes) / 1024**3


def _release_model(model: Any | None) -> float:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return _cuda_free_gib()


def _load_vllm_engine(model_path: Path, inference_config: Any, *, seed: int):
    # Gemma 4 stores its clipping limits as persistent buffers.  vLLM's
    # generic Transformers weight loader currently recognizes BatchNorm
    # buffers only, so teach it to load all direct persistent buffers.  This
    # keeps the actual checkpoint values instead of silently discarding them.
    from vllm.model_executor.models.utils import AutoWeightsLoader

    original = AutoWeightsLoader._add_loadable_non_param_tensors
    if not getattr(original, "_r2opl_loads_buffers", False):

        def add_loadable_buffers(self, module, child_params):
            original(self, module, child_params)
            for name, value in module.named_buffers(recurse=False):
                child_params.setdefault(name, value)

        add_loadable_buffers._r2opl_loads_buffers = True
        AutoWeightsLoader._add_loadable_non_param_tensors = add_loadable_buffers

    # Gemma 4 checkpoints retain the per-layer K/V tensors for the tail
    # layers, while the current Transformers implementation intentionally
    # reuses the last non-shared K/V states there and does not instantiate
    # those modules.  Hugging Face treats these checkpoint entries as
    # unexpected; make vLLM's otherwise-strict generic loader do the same.
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
                # Build vLLM's dynamic language-model view once before
                # torch.compile starts.  Calling ``self.__class__.mro()`` from
                # inside the compiled forward is unsupported by Dynamo.
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

    from vllm import LLM

    return LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        dtype=str(inference_config.get("dtype", "bfloat16")),
        tensor_parallel_size=1,
        seed=int(seed),
        gpu_memory_utilization=float(inference_config.gpu_memory_utilization),
        max_model_len=int(inference_config.max_model_len),
        max_num_seqs=int(inference_config.max_num_seqs),
        max_num_batched_tokens=int(inference_config.max_num_batched_tokens),
        enable_chunked_prefill=bool(inference_config.enable_chunked_prefill),
        enable_prefix_caching=bool(inference_config.enable_prefix_caching),
        enable_sleep_mode=bool(inference_config.get("enable_sleep_mode", False)),
        # R^2OPL currently evaluates text-only reasoning.  Gemma 4 exposes
        # image/audio towers, but profiling those unused modalities both wastes
        # memory and hits a vLLM generic-backend norm-shape incompatibility.
        limit_mm_per_prompt={"image": 0, "audio": 0},
        max_logprobs=16,
    )


def _release_vllm_engine(engine: Any | None) -> float:
    if engine is not None:
        # Never call engine.sleep(): production and smoke configurations both
        # disable it for compatibility with the deployment driver.
        engine.llm_engine.engine_core.shutdown()
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        # A never-initialized or already-destroyed single-rank group needs no
        # cleanup. CUDA release below is still mandatory and verified.
        pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return _cuda_free_gib()


def _vllm_generate_ids(
    engine: Any,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> list[int]:
    if max_tokens <= 0:
        return []
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(
        n=1,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_tokens=int(max_tokens),
        seed=int(seed),
        detokenize=False,
    )
    output = engine.generate(
        TokensPrompt(prompt_token_ids=[int(value) for value in prompt_ids]),
        params,
        use_tqdm=False,
    )[0]
    return [int(value) for value in output.outputs[0].token_ids]


def _vllm_score_response_log_probs(
    engine: Any,
    records: Sequence[dict[str, Any]],
) -> None:
    """Attach Teacher log p(sampled Student token) from vLLM prompt scores."""

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = [
        TokensPrompt(
            prompt_token_ids=[int(value) for value in record["prompt_ids"]]
            + [int(value) for value in record["response_ids"]]
        )
        for record in records
    ]
    params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )
    outputs = engine.generate(prompts, params, use_tqdm=False)
    for record, output in zip(records, outputs, strict=True):
        prompt_length = len(record["prompt_ids"])
        combined_ids = list(record["prompt_ids"]) + list(record["response_ids"])
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != len(combined_ids):
            raise RuntimeError("vLLM did not return aligned Teacher prompt logprobs")
        sampled_log_probs: list[float] = []
        for position in range(prompt_length, len(combined_ids)):
            token_id = int(combined_ids[position])
            token_map = prompt_logprobs[position]
            token_entry = None if token_map is None else token_map.get(token_id)
            if token_entry is None:
                raise RuntimeError(
                    "vLLM prompt_logprobs omitted the sampled response token at "
                    f"position {position}"
                )
            sampled_log_probs.append(float(token_entry.logprob))
        record["teacher_log_probs"] = sampled_log_probs


def _load_model(model_path: Path, family: str, *, training: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    common = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if family == "qwen3":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **common,
        )
    elif family == "gemini4":
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )

        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_path,
            **common,
        )
    else:
        raise ValueError(f"Unsupported smoke-test family: {family}")

    if training:
        model.train()
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    else:
        model.eval()
    return model, tokenizer


def _generation_eos_token_id(model: Any, tokenizer: Any) -> int | list[int] | None:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    if isinstance(value, tuple):
        return [int(token_id) for token_id in value]
    return value


def _eos_ids(tokenizer: Any, *, eos_token_id: Any = None) -> set[int]:
    value = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(token_id) for token_id in value}


def _trim_response_ids(
    token_ids: Sequence[int], tokenizer: Any, *, eos_token_id: Any = None
) -> list[int]:
    result = [int(token_id) for token_id in token_ids]
    eos_ids = _eos_ids(tokenizer, eos_token_id=eos_token_id)
    for index, token_id in enumerate(result):
        if token_id in eos_ids:
            return result[: index + 1]
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        while result and result[-1] == int(pad_id):
            result.pop()
    return result


def _generate_from_ids(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[int]:
    if max_new_tokens <= 0:
        return []
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    input_ids = torch.tensor(
        [[int(value) for value in prompt_ids]],
        dtype=torch.long,
        device="cuda:0",
    )
    attention_mask = torch.ones_like(input_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.inference_mode():
        eos_token_id = _generation_eos_token_id(model, tokenizer)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else None,
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_token_id,
        )
    response = output[0, input_ids.shape[1] :].tolist()
    del input_ids, attention_mask, output
    return _trim_response_ids(
        response,
        tokenizer,
        eos_token_id=eos_token_id,
    )


def _generate_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    tokenizer.padding_side = old_padding_side
    input_ids = encoded["input_ids"].to("cuda:0")
    attention_mask = encoded["attention_mask"].to("cuda:0")
    prompt_ids = [
        [
            int(token_id)
            for token_id, visible in zip(
                input_ids[row].tolist(), attention_mask[row].tolist()
            )
            if int(visible)
        ]
        for row in range(len(prompts))
    ]
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    eos_token_id = _generation_eos_token_id(model, tokenizer)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
    input_width = int(input_ids.shape[1])
    results = [
        (
            prompt_ids[row],
            _trim_response_ids(
                output[row, input_width:].tolist(),
                tokenizer,
                eos_token_id=eos_token_id,
            ),
        )
        for row in range(len(prompts))
    ]
    del encoded, input_ids, attention_mask, output
    return results


def _render_prompt(tokenizer: Any, config: Any, question: str) -> str:
    prompt, _ = render_length_limited_chat_prompt(
        tokenizer,
        question=question,
        model_family=str(config.student_model_family),
        max_prompt_length=int(config.data.max_prompt_length),
    )
    return prompt


def _generate_records(
    engine: Any,
    tokenizer: Any,
    config: Any,
    examples: Sequence[dict[str, str]],
    *,
    dataset: str,
    rollouts: int,
    max_new_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    requests: list[tuple[int, dict[str, str], list[int]]] = []
    for question_index, example in enumerate(examples):
        prompt = _render_prompt(tokenizer, config, str(example["question"]))
        requests.append(
            (
                question_index,
                example,
                tokenizer.encode(prompt, add_special_tokens=False),
            )
        )

    params = SamplingParams(
        n=int(rollouts),
        temperature=float(config.actor_rollout_ref.rollout.temperature),
        top_p=float(config.actor_rollout_ref.rollout.top_p),
        top_k=int(config.actor_rollout_ref.rollout.top_k),
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )
    outputs = engine.generate(
        [TokensPrompt(prompt_token_ids=request[2]) for request in requests],
        params,
        use_tqdm=True,
    )

    records: list[dict[str, Any]] = []
    for request, output in zip(requests, outputs, strict=True):
        question_index, example, prompt_ids = request
        for rollout_index, completion in enumerate(output.outputs):
            response_ids = [int(value) for value in completion.token_ids]
            response = str(completion.text)
            uid = str(example.get("sample_id") or f"{dataset}-{question_index}")
            records.append(
                {
                    "dataset": dataset,
                    "uid": uid,
                    "question_index": question_index,
                    "rollout_index": rollout_index,
                    "question": str(example["question"]),
                    "answer": str(example["answer"]),
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "response": response,
                    "correct": bool(
                        verify_response_answer(response, str(example["answer"]))
                    ),
                }
            )
    return records


def _response_log_probs(
    model: Any,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    *,
    require_grad: bool,
) -> torch.Tensor:
    if not prompt_ids or not response_ids:
        raise ValueError("PG-OPD log-prob scoring requires prompt and response IDs")
    all_ids = [int(value) for value in prompt_ids] + [
        int(value) for value in response_ids
    ]
    input_ids = torch.tensor([all_ids], dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        response_start = len(prompt_ids)
        logits = output.logits[:, response_start - 1 : -1, :].float()
        targets = input_ids[:, response_start:]
        log_probs = torch.log_softmax(logits, dim=-1).gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
    if require_grad:
        return log_probs
    result = log_probs.detach().cpu()
    del output, logits, targets, input_ids, attention_mask, log_probs
    return result


def _group_training_records(
    records: Sequence[dict[str, Any]], questions_per_batch: int
) -> list[list[dict[str, Any]]]:
    question_order: list[int] = []
    by_question: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        question_index = int(record["question_index"])
        if question_index not in by_question:
            question_order.append(question_index)
            by_question[question_index] = []
        by_question[question_index].append(record)
    batches = []
    for offset in range(0, len(question_order), questions_per_batch):
        batch_questions = question_order[offset : offset + questions_per_batch]
        batches.append(
            [record for index in batch_questions for record in by_question[index]]
        )
    return batches


# In-process free-memory floor asserted between staged phases.  The process
# boundary (stage subprocess exit) is the real Student/Teacher isolation
# mechanism; this floor is a within-process heuristic tuned for small local
# Teachers (Qwen3-4B).  A large local Teacher (e.g. gemma-4-E4B substituting
# for the 31B model) cannot fully return its memory inside a still-living
# vLLM process, so smoke configs may relax it via
# ``local_smoke.min_free_gib_after_stage``.
_STAGE_MIN_FREE_GIB = 35.0


def _assert_stage_boundary(stage: str) -> float:
    free_gib = _cuda_free_gib()
    if free_gib < _STAGE_MIN_FREE_GIB:
        raise RuntimeError(
            f"{stage} did not release the prior model: only {free_gib:.2f} GiB free"
        )
    return free_gib


def _load_smoke_examples(config: Any, seed: int):
    training_records = load_question_answer_records(
        _resolve_project_path(str(config.data.train_files[0]))
    )
    questions_per_batch = int(config.data.train_batch_size)
    training_batches = int(config.local_smoke.training_batches)
    train_examples = random.Random(seed).sample(
        training_records,
        questions_per_batch * training_batches,
    )
    all_validation_records = load_question_answer_records(
        _resolve_project_path(str(config.data.val_files[0])),
        dataset_names=["AMC23"],
    )
    validation_indices = [int(value) for value in config.local_smoke.validation_indices]
    validation_records = [all_validation_records[index] for index in validation_indices]
    return train_examples, validation_records, validation_indices


def _stage_report_path(output_dir: Path, stage: str) -> Path:
    return output_dir / f"{stage}_stage.json"


def _run_vllm_subprocess(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--vllm-stage",
        stage,
    ]
    environment = os.environ.copy()
    # A child runs exactly one vLLM engine.  In-process mode lets local model
    # compatibility patches apply inside that engine, while process exit is
    # the only mechanism used to release its GPU state.
    environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)
    return _read_json(_stage_report_path(args.output_dir, stage))


def _run_internal_vllm_stage(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The staged smoke test requires one CUDA GPU")
    config = _compose_config(args.config)
    global _STAGE_MIN_FREE_GIB
    _STAGE_MIN_FREE_GIB = float(
        config.get("local_smoke", {}).get("min_free_gib_after_stage", 35.0)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = str(args.vllm_stage)
    stage_started = time.perf_counter()
    from transformers import AutoTokenizer

    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    teacher_path = _resolve_project_path(
        str(config.distillation.teacher_models.teacher_model.model_path)
    )
    train_path = args.output_dir / "train_rollouts.jsonl"
    val_path = args.output_dir / "amc23_avg4_rollouts.jsonl"

    if stage == "student":
        train_examples, validation_records, _ = _load_smoke_examples(
            config, args.seed
        )
        student_tokenizer = AutoTokenizer.from_pretrained(
            student_path, trust_remote_code=True, local_files_only=True
        )
        engine = _load_vllm_engine(
            student_path,
            config.actor_rollout_ref.rollout,
            seed=args.seed,
        )
        train_rollouts = _generate_records(
            engine,
            student_tokenizer,
            config,
            train_examples,
            dataset="train",
            rollouts=int(config.actor_rollout_ref.rollout.n),
            max_new_tokens=int(config.rlvr_generation.train_max_new_tokens),
            seed=args.seed,
        )
        val_rollouts = _generate_records(
            engine,
            student_tokenizer,
            config,
            validation_records,
            dataset="AMC23",
            rollouts=int(config.actor_rollout_ref.rollout.val_kwargs.n),
            max_new_tokens=int(config.rlvr_generation.val_max_new_tokens),
            seed=args.seed + 100_000,
        )
        free_gib = _release_vllm_engine(engine)
        _assert_stage_boundary("Student rollout release")
        _write_jsonl(train_path, train_rollouts)
        _write_jsonl(val_path, val_rollouts)
        report = {
            "stage": "vllm_student_rollout_and_validation",
            "seconds": time.perf_counter() - stage_started,
            "free_gpu_gib_after_release": free_gib,
        }

    elif stage == "teacher":
        train_rollouts = _read_jsonl(train_path)
        student_tokenizer = AutoTokenizer.from_pretrained(
            student_path, trust_remote_code=True, local_files_only=True
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_path, trust_remote_code=True, local_files_only=True
        )
        first_prompt_text = student_tokenizer.decode(
            train_rollouts[0]["prompt_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if (
            teacher_tokenizer.encode(first_prompt_text, add_special_tokens=False)
            != train_rollouts[0]["prompt_ids"]
        ):
            raise RuntimeError("Student and Teacher tokenizers do not preserve prompt IDs")
        engine = _load_vllm_engine(
            teacher_path,
            config.distillation.teacher_models.teacher_model.inference,
            seed=args.seed + 200_000,
        )
        _vllm_score_response_log_probs(engine, train_rollouts)
        free_gib = _release_vllm_engine(engine)
        _assert_stage_boundary("Teacher release")
        _write_jsonl(train_path, train_rollouts)
        report = {
            "stage": "vllm_teacher_logprob_scoring",
            "seconds": time.perf_counter() - stage_started,
            "free_gpu_gib_after_release": free_gib,
        }
    else:
        raise ValueError(f"Unknown internal vLLM stage: {stage}")

    _atomic_json(_stage_report_path(args.output_dir, stage), report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The staged smoke test requires one CUDA GPU")
    config = _compose_config(args.config)
    global _STAGE_MIN_FREE_GIB
    _STAGE_MIN_FREE_GIB = float(
        config.get("local_smoke", {}).get("min_free_gib_after_stage", 35.0)
    )
    family = str(config.student_model_family)
    questions_per_batch = int(config.data.train_batch_size)
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    training_batches = int(config.local_smoke.training_batches)
    if questions_per_batch != 4 or rollout_n != 4 or training_batches != 2:
        raise RuntimeError("Smoke contract requires exactly 2 batches of 4x4")

    validation_indices = [int(value) for value in config.local_smoke.validation_indices]
    if len(validation_indices) != 2:
        raise RuntimeError("AMC23 smoke validation must contain exactly two questions")

    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    teacher_path = _resolve_project_path(
        str(config.distillation.teacher_models.teacher_model.model_path)
    )
    stage_log: list[dict[str, Any]] = []
    started = time.perf_counter()

    # Every vLLM stage runs in its own process.  Besides enforcing strict
    # Student/Teacher separation, this is required by vLLM's one-instance
    # sleep-mode allocator.
    stage_log.append(_run_vllm_subprocess(args, "student"))
    train_rollouts = _read_jsonl(args.output_dir / "train_rollouts.jsonl")
    val_rollouts = _read_jsonl(args.output_dir / "amc23_avg4_rollouts.jsonl")

    uid_counts = Counter(record["uid"] for record in train_rollouts)
    if len(train_rollouts) != 32 or set(uid_counts.values()) != {4}:
        raise RuntimeError("Training rollout grouping is not 8 questions x 4")
    val_metrics = compute_pass_avg_metrics(
        [record["dataset"] for record in val_rollouts],
        [record["uid"] for record in val_rollouts],
        [float(record["correct"]) for record in val_rollouts],
        expected_rollouts=4,
    )
    avg4 = selected_avg_metrics(
        val_metrics,
        ["AMC23"],
        expected_rollouts=4,
    )

    from transformers import AutoTokenizer

    student_tokenizer = AutoTokenizer.from_pretrained(
        student_path, trust_remote_code=True, local_files_only=True
    )

    stage_log.append(_run_vllm_subprocess(args, "teacher"))
    train_rollouts = _read_jsonl(args.output_dir / "train_rollouts.jsonl")

    first_prompt_text = student_tokenizer.decode(
        train_rollouts[0]["prompt_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    # Stage 4: load only the trainable Student and run two genuine optimizer
    # steps on the vLLM-sampled original token IDs and Teacher prompt scores.
    stage_started = time.perf_counter()
    student, reloaded_tokenizer = _load_model(student_path, family, training=True)
    if reloaded_tokenizer.encode(first_prompt_text, add_special_tokens=False) != train_rollouts[0]["prompt_ids"]:
        raise RuntimeError("Reloaded Student tokenizer changed the sampled IDs")
    optimizer = torch.optim.SGD(student.parameters(), lr=float(args.learning_rate))
    train_batches = _group_training_records(train_rollouts, questions_per_batch)
    if len(train_batches) != 2 or any(len(batch) != 16 for batch in train_batches):
        raise RuntimeError("Expected two optimizer batches of 16 trajectories")
    optimizer_metrics: list[dict[str, float]] = []
    for batch_index, batch in enumerate(train_batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for record in batch:
            student_log_probs = _response_log_probs(
                student, record["prompt_ids"], record["response_ids"], require_grad=True
            )
            teacher_log_probs = torch.tensor(
                [record["teacher_log_probs"]],
                dtype=student_log_probs.dtype,
                device=student_log_probs.device,
            )
            response_mask = torch.ones_like(student_log_probs, dtype=torch.bool)
            sample_loss = signed_pg_opd_loss(
                student_log_probs, teacher_log_probs, response_mask
            )
            (sample_loss / len(batch)).backward()
            loss_sum += float(sample_loss.detach().cpu())
            del student_log_probs, teacher_log_probs, response_mask, sample_loss
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        optimizer_metrics.append(
            {"batch": float(batch_index), "loss": loss_sum / len(batch), "grad_norm": float(grad_norm.detach().cpu())}
        )
    free_after_training = _release_model(student)
    student = None
    stage_log.append(
        {
            "stage": "student_two_batch_training",
            "seconds": time.perf_counter() - stage_started,
            "free_gpu_gib_after_release": free_after_training,
        }
    )
    _assert_stage_boundary("Trained Student release")

    report = {
        "status": "passed",
        "config": str(args.config.resolve()),
        "model_family": family,
        "student_model": str(student_path),
        "teacher_model": str(teacher_path),
        "prompt_template": str(config.prompt_template),
        "project_name": str(config.trainer.project_name),
        "group_name": str(config.group_name),
        "training": {
            "question_batch_size": questions_per_batch,
            "rollouts_per_question": rollout_n,
            "trajectory_batch_size": questions_per_batch * rollout_n,
            "optimizer_steps": len(optimizer_metrics),
            "metrics": optimizer_metrics,
        },
        "validation": {
            "dataset": "AMC23",
            "questions": 2,
            "sample_indices": validation_indices,
            "rollouts_per_question": 4,
            "metrics": val_metrics,
            "selected_avg": avg4,
        },
        "stage_log": stage_log,
        "total_seconds": time.perf_counter() - started,
    }
    _write_jsonl(args.output_dir / "train_rollouts.jsonl", train_rollouts)
    _write_jsonl(args.output_dir / "amc23_avg4_rollouts.jsonl", val_rollouts)
    _atomic_json(args.output_dir / "report.json", report)
    return report


def main() -> None:
    args = parse_args()
    if args.vllm_stage is not None:
        report = _run_internal_vllm_stage(args)
    else:
        report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
