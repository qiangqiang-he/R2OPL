"""Staged one-GPU GRPO smoke test with a vLLM Student.

This is deliberately test-only infrastructure mirroring
``run_pg_opd_single_gpu_smoke``. It samples two 2-question training batches,
generates each batch's four Student rollouts per question in one vLLM call,
runs two genuine GRPO optimizer steps with the production VERL
advantage/policy-loss kernels, evaluates two AMC23 questions at Avg@4 in a
single 8-rollout vLLM call.  ERSR evaluation is retired and intentionally
absent.
GRPO is Student-only, so no Teacher is ever loaded.
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
# mode, releases GPU state and guarantees that only one engine (or the
# trainable Student) is resident at any time.
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

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from algorithms import resolve_algorithm
from utils.answer_verifier import verify_response_answer
from utils.custom_dataset import (
    load_question_answer_records,
    render_length_limited_chat_prompt,
)
from utils.opd_runtime import (
    compute_pass_avg_metrics,
    selected_avg_metrics,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "tests"
    / "configs"
    / "grpo"
    / "grpo_qwen3_1p7b_single_gpu_smoke.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "artifacts" / "grpo_single_gpu_smoke",
    )
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the configured actor optimizer learning rate.",
    )
    parser.add_argument(
        "--vllm-stage",
        choices=("student",),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.learning_rate is not None and args.learning_rate <= 0:
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
    algorithm = resolve_algorithm(config)
    algorithm.configure_defaults(config)
    algorithm.configure_batch(config)
    algorithm.validate(config)
    if int(config.trainer.n_gpus_per_node) != 1 or int(config.trainer.nnodes) != 1:
        raise ValueError("The GRPO smoke test requires a single-GPU configuration")
    return config


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


def _assert_stage_boundary(stage: str, minimum_gib: float = 35.0) -> float:
    free_gib = _cuda_free_gib()
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"{stage} did not release the prior engine: only {free_gib:.2f} GiB free"
        )
    return free_gib


def _release_model(model: Any | None) -> float:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return _cuda_free_gib()


def _apply_gemma4_vllm_compatibility_patches() -> None:
    """Match the patches used by ``run_pg_opd_single_gpu_smoke`` for Gemma 4."""

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


def _load_vllm_engine(
    model_path: Path,
    inference_config: Any,
    *,
    seed: int,
    family: str = "qwen3",
):
    if family == "gemini4":
        _apply_gemma4_vllm_compatibility_patches()

    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": str(model_path),
        "tokenizer": str(model_path),
        "trust_remote_code": True,
        "dtype": str(inference_config.get("dtype", "bfloat16")),
        "tensor_parallel_size": 1,
        "seed": int(seed),
        # Hard safety cap: one engine may never claim more than half the
        # 48 GB card, keeping the required free-VRAM reserve at all times.
        "gpu_memory_utilization": min(
            float(inference_config.gpu_memory_utilization), 0.5
        ),
        "max_model_len": int(inference_config.max_model_len),
        "max_num_seqs": int(inference_config.max_num_seqs),
        "max_num_batched_tokens": int(inference_config.max_num_batched_tokens),
        "enable_chunked_prefill": bool(inference_config.enable_chunked_prefill),
        "enable_prefix_caching": bool(inference_config.enable_prefix_caching),
        "enable_sleep_mode": bool(inference_config.get("enable_sleep_mode", False)),
    }
    if family == "gemini4":
        kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}
    return LLM(**kwargs)


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


def _response_log_probs(
    model: Any,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    *,
    require_grad: bool,
) -> torch.Tensor:
    if not prompt_ids or not response_ids:
        raise ValueError("GRPO log-prob scoring requires prompt and response IDs")
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
        dataset_names=[str(name) for name in config.data.val_datasets],
    )
    validation_records = all_validation_records[: int(config.data.val_max_samples)]
    if len(validation_records) != int(config.data.val_max_samples):
        raise RuntimeError("The validation pool does not cover val_max_samples")
    return train_examples, validation_records


def _generate_records(
    engine: Any,
    tokenizer: Any,
    config: Any,
    examples: Sequence[dict[str, str]],
    *,
    dataset: str,
    rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    phase_label: str,
    batch_index: int | None = None,
    question_index_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One vLLM call per rollout batch: n completions per question at once."""

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    requests: list[tuple[int, dict[str, str], list[int]]] = []
    for question_index, example in enumerate(examples):
        prompt, _truncated = render_length_limited_chat_prompt(
            tokenizer,
            question=str(example["question"]),
            model_family=str(config.student_model_family),
            max_prompt_length=int(config.data.max_prompt_length),
        )
        requests.append(
            (
                question_index,
                example,
                tokenizer.encode(prompt, add_special_tokens=False),
            )
        )

    params = SamplingParams(
        n=int(rollouts),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )
    started = time.perf_counter()
    outputs = engine.generate(
        [TokensPrompt(prompt_token_ids=request[2]) for request in requests],
        params,
        use_tqdm=False,
    )
    seconds = time.perf_counter() - started
    if len(outputs) != len(requests):
        raise RuntimeError(
            f"Expected {len(requests)} vLLM outputs, received {len(outputs)}."
        )

    records: list[dict[str, Any]] = []
    total_tokens = 0
    for request, output in zip(requests, outputs, strict=True):
        question_index, example, prompt_ids = request
        global_question_index = question_index + int(question_index_offset)
        uid = f"{dataset}-{global_question_index}"
        if len(output.outputs) != int(rollouts):
            raise RuntimeError(
                f"Question {question_index}: expected {rollouts} rollouts, "
                f"received {len(output.outputs)}."
            )
        for rollout_index, completion in enumerate(output.outputs):
            response_ids = [int(value) for value in completion.token_ids]
            response = str(completion.text)
            total_tokens += len(response_ids)
            record = {
                "dataset": dataset,
                "uid": uid,
                "question_index": global_question_index,
                "rollout_index": rollout_index,
                "question": str(example["question"]),
                "answer": str(example["answer"]),
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response": response,
                "finish_reason": str(completion.finish_reason),
                "correct": bool(
                    verify_response_answer(response, str(example["answer"]))
                ),
            }
            if batch_index is not None:
                record["batch_index"] = int(batch_index)
            records.append(record)

    timing = {
        "phase": phase_label,
        "questions": len(requests),
        "rollouts": len(records),
        "vllm_generate_calls": 1,
        "generated_tokens": total_tokens,
        "seconds": round(seconds, 3),
        "tokens_per_second": (
            round(total_tokens / seconds, 1) if seconds > 0 else None
        ),
    }
    return records, timing


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
        "--vllm-stage",
        stage,
    ]
    if args.learning_rate is not None:
        command.extend(["--learning-rate", str(args.learning_rate)])
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = str(args.vllm_stage)
    stage_started = time.perf_counter()
    from transformers import AutoTokenizer

    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    family = str(config.student_model_family)
    train_path = args.output_dir / "train_rollouts.jsonl"
    val_path = args.output_dir / "amc23_avg4_rollouts.jsonl"

    if stage == "student":
        train_examples, validation_records = _load_smoke_examples(
            config, args.seed
        )
        student_tokenizer = AutoTokenizer.from_pretrained(
            student_path, trust_remote_code=True, local_files_only=True
        )
        engine = _load_vllm_engine(
            student_path,
            config.actor_rollout_ref.rollout,
            seed=args.seed,
            family=family,
        )
        rollout = config.actor_rollout_ref.rollout
        questions_per_batch = int(config.data.train_batch_size)
        train_records: list[dict[str, Any]] = []
        rollout_calls: list[dict[str, Any]] = []
        for batch_index in range(int(config.local_smoke.training_batches)):
            batch_examples = train_examples[
                batch_index * questions_per_batch : (batch_index + 1)
                * questions_per_batch
            ]
            records, timing = _generate_records(
                engine,
                student_tokenizer,
                config,
                batch_examples,
                dataset="train",
                rollouts=int(rollout.n),
                max_new_tokens=int(config.rlvr_generation.train_max_new_tokens),
                temperature=float(rollout.temperature),
                top_p=float(rollout.top_p),
                top_k=int(rollout.top_k),
                seed=args.seed + 1_000 * batch_index,
                phase_label=f"train_batch_{batch_index + 1}",
                batch_index=batch_index,
                question_index_offset=batch_index * questions_per_batch,
            )
            train_records.extend(records)
            rollout_calls.append(timing)
        val_rollouts, val_timing = _generate_records(
            engine,
            student_tokenizer,
            config,
            validation_records,
            dataset="AMC23",
            rollouts=int(rollout.val_kwargs.n),
            max_new_tokens=int(config.rlvr_generation.val_max_new_tokens),
            temperature=float(rollout.val_kwargs.temperature),
            top_p=float(rollout.val_kwargs.top_p),
            top_k=int(rollout.val_kwargs.top_k),
            seed=args.seed + 100_000,
            phase_label="amc23_avg4_validation",
        )
        rollout_calls.append(val_timing)
        free_gib = _release_vllm_engine(engine)
        # The in-process release is best-effort; the true release mechanism is
        # this child process exiting, verified by the driver after it returns.
        _write_jsonl(train_path, train_records)
        _write_jsonl(val_path, val_rollouts)
        report = {
            "stage": "vllm_student_rollout_and_validation",
            "seconds": time.perf_counter() - stage_started,
            "free_gpu_gib_after_release": free_gib,
            "rollout_calls": rollout_calls,
        }
    else:
        raise ValueError(f"Unknown internal vLLM stage: {stage}")

    _atomic_json(_stage_report_path(args.output_dir, stage), report)
    return report


def _grpo_training_stage(
    args: argparse.Namespace, config: Any, train_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Two genuine GRPO optimizer steps with the production VERL kernels."""

    from verl.trainer.ppo import core_algos

    stage_started = time.perf_counter()
    family = str(config.student_model_family)
    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    actor = config.actor_rollout_ref.actor
    model, _tokenizer = _load_model(student_path, family, training=True)

    learning_rate = (
        float(args.learning_rate)
        if args.learning_rate is not None
        else float(actor.optim.lr)
    )
    if family == "gemini4":
        # Local-only compromise: Gemma E2B's ~19 GiB of AdamW states would
        # leave under the required 5 GiB safety margin on the 48 GB smoke
        # GPU, so the Gemma smoke updates with SGD.  The server keeps AdamW.
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(actor.optim.weight_decay),
        )
    clip_value = float(actor.optim.clip_grad)
    questions_per_batch = int(config.data.train_batch_size)
    batches = _group_training_records(train_records, questions_per_batch)

    metrics: list[dict[str, float]] = []
    for batch_index, batch in enumerate(batches, start=1):
        rewards = torch.tensor(
            [[float(bool(record["correct"]))] for record in batch],
            dtype=torch.float32,
        )
        response_mask = torch.ones_like(rewards)
        index = np.asarray([str(record["uid"]) for record in batch], dtype=object)
        advantages, _ = core_algos.compute_grpo_outcome_advantage(
            rewards,
            response_mask,
            index,
            norm_adv_by_std_in_grpo=bool(config.algorithm.norm_adv_by_std_in_grpo),
        )

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        batch_clipfrac = 0.0
        batch_kl = 0.0
        for position, record in enumerate(batch):
            old_log_probs = _response_log_probs(
                model,
                record["prompt_ids"],
                record["response_ids"],
                require_grad=False,
            )
            new_log_probs = _response_log_probs(
                model,
                record["prompt_ids"],
                record["response_ids"],
                require_grad=True,
            )
            old_row = old_log_probs.to(device=new_log_probs.device).unsqueeze(0)
            new_row = new_log_probs.unsqueeze(0)
            advantage_row = torch.full_like(new_row, float(advantages[position, 0]))
            mask_row = torch.ones_like(new_row)
            pg_loss, pg_clipfrac, ppo_kl, _lower_clipfrac = core_algos.compute_policy_loss(
                old_row,
                new_row,
                advantage_row,
                mask_row,
                cliprange=float(actor.clip_ratio),
                cliprange_low=float(actor.clip_ratio_low),
                cliprange_high=float(actor.clip_ratio_high),
                loss_agg_mode=str(actor.loss_agg_mode),
            )
            (pg_loss / len(batch)).backward()
            batch_loss += float(pg_loss.detach().cpu())
            batch_clipfrac += float(pg_clipfrac.detach().cpu())
            batch_kl += float(ppo_kl.detach().cpu())
            del old_log_probs, new_log_probs, old_row, new_row
            del advantage_row, mask_row, pg_loss

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_value
        )
        optimizer.step()
        trajectory_count = len(batch)
        entry = {
            "batch": float(batch_index),
            "questions": float(questions_per_batch),
            "trajectories": float(trajectory_count),
            "loss": batch_loss / trajectory_count,
            "grad_norm": float(grad_norm.detach().cpu()),
            "mean_pg_clipfrac": batch_clipfrac / trajectory_count,
            "mean_ppo_kl": batch_kl / trajectory_count,
            "mean_reward": float(rewards.mean().item()),
        }
        for key in ("loss", "grad_norm", "mean_pg_clipfrac", "mean_ppo_kl"):
            if not np.isfinite(entry[key]):
                raise RuntimeError(
                    f"GRPO batch {batch_index} produced non-finite {key}: {entry[key]}"
                )
        metrics.append(entry)

    # AdamW keeps its exp_avg/exp_avg_sq states on the GPU through the
    # optimizer object; drop them before the release check.
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    free_gib = _release_model(model)
    _assert_stage_boundary("Trained Student release")
    return {
        "stage": "student_two_batch_grpo_training",
        "seconds": time.perf_counter() - stage_started,
        "free_gpu_gib_after_release": free_gib,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "metrics": metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The staged smoke test requires one CUDA GPU")
    config = _compose_config(args.config)
    family = str(config.student_model_family)
    questions_per_batch = int(config.data.train_batch_size)
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    training_batches = int(config.local_smoke.training_batches)
    if questions_per_batch != 2 or rollout_n != 4 or training_batches != 2:
        raise RuntimeError("Smoke contract requires exactly 2 batches of 2x4")
    if int(config.actor_rollout_ref.rollout.val_kwargs.n) != 4:
        raise RuntimeError("AMC23 smoke validation must sample four rollouts")
    if int(config.data.val_max_samples) != 2:
        raise RuntimeError("AMC23 smoke validation must contain exactly two questions")
    if [str(name) for name in config.data.val_datasets] != ["AMC23"]:
        raise RuntimeError("The smoke validation dataset must be AMC23 only")

    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))

    stage_log: list[dict[str, Any]] = []
    started = time.perf_counter()

    # Stage 1: one subprocess runs every vLLM rollout batch (Student only).
    student_report = _run_vllm_subprocess(args, "student")
    _assert_stage_boundary("Student stage subprocess exit")
    stage_log.append(student_report)
    train_rollouts = _read_jsonl(args.output_dir / "train_rollouts.jsonl")
    val_rollouts = _read_jsonl(args.output_dir / "amc23_avg4_rollouts.jsonl")

    uid_counts = Counter(record["uid"] for record in train_rollouts)
    if len(train_rollouts) != 16 or set(uid_counts.values()) != {4}:
        raise RuntimeError("Training rollout grouping is not 4 questions x 4")
    batch_sizes = Counter(int(record["batch_index"]) for record in train_rollouts)
    if set(batch_sizes) != {0, 1} or set(batch_sizes.values()) != {8}:
        raise RuntimeError("Training rollouts must form two batches of 8 trajectories")
    if len(val_rollouts) != 8:
        raise RuntimeError("AMC23 validation must contain exactly 8 rollouts")

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

    # Stage 2: load only the trainable Student and run two genuine GRPO
    # optimizer steps on the vLLM-sampled original token IDs.
    training_report = _grpo_training_stage(args, config, train_rollouts)
    stage_log.append(training_report)

    rollout_calls = student_report["rollout_calls"]
    if len(rollout_calls) != 3 or any(
        call["vllm_generate_calls"] != 1 for call in rollout_calls
    ):
        raise RuntimeError("Each rollout batch must be one vLLM generate call")
    for call in rollout_calls:
        if not call["generated_tokens"] or call["seconds"] <= 0:
            raise RuntimeError(f"Empty or untimed rollout call: {call}")

    report = {
        "status": "passed",
        "config": str(args.config),
        "algorithm": "grpo",
        "model_family": family,
        "student_model": str(student_path),
        "teacher_model": None,
        "seed": int(args.seed),
        "training": {
            "question_batch_size": questions_per_batch,
            "rollouts_per_question": rollout_n,
            "trajectory_batch_size": questions_per_batch * rollout_n,
            "optimizer_steps": len(training_report["metrics"]),
            "metrics": training_report["metrics"],
        },
        "validation": {
            "dataset": "AMC23",
            "questions": 2,
            "rollouts_per_question": 4,
            "metrics": val_metrics,
            "selected_avg": avg4,
        },
        "rollout_speed": {
            "engine": "vllm",
            "calls": rollout_calls,
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
