"""Staged one-GPU EOPD smoke test with real local checkpoints.

This is deliberately test-only.  It exercises two 4-question training batches
with four Student rollouts per question in one vLLM call per batch, scores
those exact sampled token IDs with the Teacher through vLLM's bounded
Top-k-plus-entropy EOPD output, and performs two real Student optimizer steps
with the entropy-gated OPD + FKL loss from ``algorithms.eopd``, and evaluates
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

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from algorithms import resolve_algorithm
from algorithms.eopd import (
    entropy_gated_topk_forward_kl_from_logits,
    eopd_total_loss,
)
from algorithms.pg_opd import signed_pg_opd_loss
from tests.run_pg_opd_single_gpu_smoke import (
    _assert_stage_boundary,
    _atomic_json,
    _generation_eos_token_id,
    _group_training_records,
    _load_model,
    _load_smoke_examples,
    _read_json,
    _read_jsonl,
    _release_model,
    _resolve_project_path,
    _response_log_probs,
    _stage_report_path,
    _trim_response_ids,
    _vllm_generate_ids,
    _write_jsonl,
)
from utils.answer_verifier import verify_response_answer
from utils.opd_runtime import compute_pass_avg_metrics, selected_avg_metrics


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "tests"
    / "configs"
    / "eopd"
    / "eopd_qwen3_4b_instruct_2507_to_1p7b_single_gpu_smoke.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "artifacts" / "eopd_single_gpu_smoke",
    )
    parser.add_argument("--seed", type=int, default=20260924)
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
    algorithm = resolve_algorithm(config)
    algorithm.configure_defaults(config)
    algorithm.configure_batch(config)
    algorithm.validate(config)
    if not bool(config.get("local_smoke", {}).get("staged_single_gpu", False)):
        raise ValueError("The selected config is not a staged single-GPU smoke config")
    return config


def _cuda_free_gib() -> float:
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return float(free_bytes) / 1024**3


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


def _apply_gemma4_vllm_compatibility_patches() -> None:
    from tests.run_pg_opd_single_gpu_smoke import (
        _apply_gemma4_vllm_compatibility_patches as apply,
    )

    apply()


def _find_vllm_sampler(engine: Any) -> tuple[str, Any]:
    """Locate the in-process vLLM sampler without hard-coding internals."""

    seen: set[int] = set()
    stack = [(engine, "engine")]
    while stack:
        obj, path = stack.pop()
        if id(obj) in seen or not hasattr(obj, "__dict__"):
            continue
        seen.add(id(obj))
        model_runner = getattr(obj, "model_runner", None)
        if model_runner is not None and hasattr(model_runner, "sampler"):
            return f"{path}.model_runner.sampler", model_runner.sampler
        for name in (
            "llm_engine",
            "engine_core",
            "model_executor",
            "driver_worker",
            "worker",
        ):
            child = getattr(obj, name, None)
            if child is not None and id(child) not in seen:
                stack.append((child, f"{path}.{name}"))
    raise RuntimeError("Could not locate the vLLM sampler for the EOPD patch")


def _load_eopd_vllm_engine(
    model_path: Path,
    inference_config: Any,
    *,
    seed: int,
    family: str,
    max_logprobs: int | None,
    eopd_entropy_topk: int | None = None,
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
        "gpu_memory_utilization": float(inference_config.gpu_memory_utilization),
        "max_model_len": int(inference_config.max_model_len),
        "max_num_seqs": int(inference_config.max_num_seqs),
        "max_num_batched_tokens": int(inference_config.max_num_batched_tokens),
        "enable_chunked_prefill": bool(inference_config.enable_chunked_prefill),
        "enable_prefix_caching": bool(inference_config.enable_prefix_caching),
        "enable_sleep_mode": bool(inference_config.get("enable_sleep_mode", False)),
    }
    if family == "gemini4":
        kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}
    if max_logprobs is not None:
        kwargs["max_logprobs"] = int(max_logprobs)
    engine = LLM(**kwargs)

    if eopd_entropy_topk is not None:
        from verl.workers.rollout.vllm_rollout.utils import enable_eopd_entropy_gather

        _path, sampler = _find_vllm_sampler(engine)
        enable_eopd_entropy_gather(sampler, int(eopd_entropy_topk))
    return engine


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

    from utils.custom_dataset import render_length_limited_chat_prompt
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


def _score_records_with_teacher(
    engine: Any,
    records: Sequence[dict[str, Any]],
    *,
    topk: int,
) -> dict[str, Any]:
    """Attach Teacher sampled logprobs, Top-k, and full-vocab entropy.

    Every record is scored in one vLLM call: the patched sampler computes the
    exact full-vocabulary entropy before truncating to the bounded Top-k plus
    one entropy-carrier slot.
    """

    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    num_prompt_logprobs = int(topk) + 1
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
        prompt_logprobs=num_prompt_logprobs,
        detokenize=False,
    )
    started = time.perf_counter()
    outputs = engine.generate(prompts, params, use_tqdm=False)
    seconds = time.perf_counter() - started
    if len(outputs) != len(records):
        raise RuntimeError(
            f"Expected {len(records)} Teacher outputs, received {len(outputs)}."
        )

    scored_tokens = 0
    high_entropy_tokens = 0
    entropy_sum = 0.0
    for record, output in zip(records, outputs, strict=True):
        prompt_length = len(record["prompt_ids"])
        total_length = prompt_length + len(record["response_ids"])
        result: dict[str, list] = {}
        extract_prompt_logprobs(
            output,
            num_prompt_logprobs=num_prompt_logprobs,
            result_dict=result,
            prompt_logprobs_topk=int(topk),
        )
        expected_length = total_length
        for key in (
            "prompt_sampled_logprobs",
            "prompt_logprobs",
            "prompt_ids",
            "prompt_entropies",
        ):
            if len(result[key]) != expected_length:
                raise RuntimeError(
                    f"Teacher extraction produced {len(result[key])} rows for "
                    f"{key}; expected {expected_length}."
                )
        # Index j of the extracted lists holds position j + 1, so the
        # response span is [prompt_length - 1, total_length - 1).
        start = prompt_length - 1
        end = total_length - 1
        record["teacher_log_probs"] = [
            float(row[0]) for row in result["prompt_sampled_logprobs"][start:end]
        ]
        record["teacher_topk_log_probs"] = [
            [float(value) for value in row]
            for row in result["prompt_logprobs"][start:end]
        ]
        record["teacher_topk_ids"] = [
            [int(value) for value in row]
            for row in result["prompt_ids"][start:end]
        ]
        record["teacher_entropy"] = [
            float(row[0]) for row in result["prompt_entropies"][start:end]
        ]
        scored_tokens += end - start
        entropy_sum += sum(record["teacher_entropy"])
    return {
        "records": len(records),
        "scored_tokens": scored_tokens,
        "vllm_generate_calls": 1,
        "seconds": round(seconds, 3),
        "tokens_per_second": (
            round(scored_tokens / seconds, 1) if seconds > 0 else None
        ),
        "mean_teacher_entropy": (
            round(entropy_sum / scored_tokens, 4) if scored_tokens else None
        ),
    }


def _response_logits(
    model: Any,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
) -> torch.Tensor:
    """Return the Student's full-vocabulary response logits with gradients."""

    all_ids = [int(value) for value in prompt_ids] + [
        int(value) for value in response_ids
    ]
    input_ids = torch.tensor([all_ids], dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    response_start = len(prompt_ids)
    logits = output.logits[:, response_start - 1 : -1, :]
    result = logits.squeeze(0)
    del output, logits, input_ids, attention_mask
    return result


def _eopd_training_stage(
    args: argparse.Namespace, config: Any, train_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Two genuine EOPD optimizer steps with the local reference kernels."""

    stage_started = time.perf_counter()
    family = str(config.student_model_family)
    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    loss = config.distillation.distillation_loss
    alpha = float(loss.eopd_alpha)
    threshold = float(loss.eopd_entropy_threshold)
    topk = int(loss.topk)
    actor = config.actor_rollout_ref.actor
    model, _tokenizer = _load_model(student_path, family, training=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(actor.optim.weight_decay),
    )
    clip_value = float(actor.optim.clip_grad)
    questions_per_batch = int(config.data.train_batch_size)
    batches = _group_training_records(train_records, questions_per_batch)

    metrics: list[dict[str, float]] = []
    for batch_index, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        batch_pg = 0.0
        batch_fkl = 0.0
        batch_high_entropy = 0.0
        batch_teacher_entropy = 0.0
        for record in batch:
            student_log_probs = _response_log_probs(
                model,
                record["prompt_ids"],
                record["response_ids"],
                require_grad=True,
            )
            teacher_log_probs = torch.tensor(
                [record["teacher_log_probs"]],
                dtype=student_log_probs.dtype,
                device=student_log_probs.device,
            )
            response_mask = torch.ones_like(student_log_probs, dtype=torch.bool)

            logits = _response_logits(
                model, record["prompt_ids"], record["response_ids"]
            )
            forward_kl = entropy_gated_topk_forward_kl_from_logits(
                logits,
                torch.tensor(record["teacher_topk_ids"], device=logits.device),
                torch.tensor(
                    record["teacher_topk_log_probs"],
                    dtype=torch.float32,
                    device=logits.device,
                ),
                torch.tensor(
                    record["teacher_entropy"],
                    dtype=torch.float32,
                    device=logits.device,
                ),
                entropy_threshold=threshold,
            )
            sample_loss = eopd_total_loss(
                student_log_probs,
                teacher_log_probs,
                forward_kl.unsqueeze(0),
                response_mask,
                alpha=alpha,
            )
            (sample_loss / len(batch)).backward()
            batch_loss += float(sample_loss.detach().cpu())
            batch_pg += float(
                signed_pg_opd_loss(
                    student_log_probs.detach(),
                    teacher_log_probs,
                    response_mask,
                ).cpu()
            )
            batch_fkl += float(
                (forward_kl.detach().float().mean()).cpu()
            )
            batch_high_entropy += float(
                (torch.tensor(record["teacher_entropy"]) > threshold)
                .float().mean().cpu()
            )
            batch_teacher_entropy += float(
                np.mean(record["teacher_entropy"])
            )
            del student_log_probs, teacher_log_probs, response_mask
            del logits, forward_kl, sample_loss

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_value
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trajectory_count = len(batch)
        entry = {
            "batch": float(batch_index),
            "questions": float(questions_per_batch),
            "trajectories": float(trajectory_count),
            "loss": batch_loss / trajectory_count,
            "pg_loss": batch_pg / trajectory_count,
            "forward_kl": batch_fkl / trajectory_count,
            "grad_norm": float(grad_norm.detach().cpu()),
            "high_entropy_token_ratio": batch_high_entropy / trajectory_count,
            "mean_teacher_entropy": batch_teacher_entropy / trajectory_count,
        }
        for key in ("loss", "pg_loss", "forward_kl", "grad_norm"):
            if not np.isfinite(entry[key]):
                raise RuntimeError(
                    f"EOPD batch {batch_index} produced non-finite {key}: {entry[key]}"
                )
        metrics.append(entry)

    # AdamW keeps its exp_avg/exp_avg_sq states on the GPU through the
    # optimizer object; drop them before the release check.
    del optimizer
    free_gib = _release_model(model)
    _assert_stage_boundary("Trained Student release")
    return {
        "stage": "student_two_batch_eopd_training",
        "seconds": time.perf_counter() - stage_started,
        "free_gpu_gib_after_release": free_gib,
        "optimizer": "AdamW",
        "learning_rate": float(args.learning_rate),
        "alpha": alpha,
        "entropy_threshold": threshold,
        "topk": topk,
        "metrics": metrics,
    }


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = str(args.vllm_stage)
    stage_started = time.perf_counter()
    from transformers import AutoTokenizer

    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    teacher_path = _resolve_project_path(
        str(config.distillation.teacher_models.teacher_model.model_path)
    )
    family = str(config.student_model_family)
    train_path = args.output_dir / "train_rollouts.jsonl"
    val_path = args.output_dir / "amc23_avg4_rollouts.jsonl"
    topk = int(config.distillation.distillation_loss.topk)

    if stage == "student":
        train_examples, validation_records, _validation_indices = (
            _load_smoke_examples(config, args.seed)
        )
        student_tokenizer = AutoTokenizer.from_pretrained(
            student_path, trust_remote_code=True, local_files_only=True
        )
        engine = _load_eopd_vllm_engine(
            student_path,
            config.actor_rollout_ref.rollout,
            seed=args.seed,
            family=family,
            max_logprobs=None,
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
        engine = _load_eopd_vllm_engine(
            teacher_path,
            config.distillation.teacher_models.teacher_model.inference,
            seed=args.seed + 200_000,
            family=family,
            max_logprobs=topk + 1,
            eopd_entropy_topk=topk,
        )
        scoring_report = _score_records_with_teacher(
            engine, train_rollouts, topk=topk
        )
        free_gib = _release_vllm_engine(engine)
        # Process exit below is the verified release mechanism.
        _write_jsonl(train_path, train_rollouts)
        report = {
            "stage": "vllm_teacher_eopd_scoring",
            "seconds": time.perf_counter() - stage_started,
            "free_gpu_gib_after_release": free_gib,
            "scoring": scoring_report,
        }
    else:
        raise ValueError(f"Unknown internal vLLM stage: {stage}")

    _atomic_json(_stage_report_path(args.output_dir, stage), report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The staged smoke test requires one CUDA GPU")
    config = _compose_config(args.config)
    family = str(config.student_model_family)
    questions_per_batch = int(config.data.train_batch_size)
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    training_batches = int(config.local_smoke.training_batches)
    if questions_per_batch != 4 or rollout_n != 4 or training_batches != 2:
        raise RuntimeError("Smoke contract requires exactly 2 batches of 4x4")

    validation_indices = [int(value) for value in config.local_smoke.validation_indices]
    if len(validation_indices) != 2:
        raise RuntimeError("AMC23 smoke validation must contain exactly two questions")
    if int(config.actor_rollout_ref.rollout.val_kwargs.n) != 4:
        raise RuntimeError("AMC23 smoke validation must sample four rollouts")

    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    teacher_path = _resolve_project_path(
        str(config.distillation.teacher_models.teacher_model.model_path)
    )

    stage_log: list[dict[str, Any]] = []
    started = time.perf_counter()

    # Stage 1: one subprocess runs every Student vLLM rollout batch.
    student_report = _run_vllm_subprocess(args, "student")
    _assert_stage_boundary("Student stage subprocess exit")
    stage_log.append(student_report)
    train_rollouts = _read_jsonl(args.output_dir / "train_rollouts.jsonl")
    val_rollouts = _read_jsonl(args.output_dir / "amc23_avg4_rollouts.jsonl")

    uid_counts = Counter(record["uid"] for record in train_rollouts)
    if len(train_rollouts) != 32 or set(uid_counts.values()) != {4}:
        raise RuntimeError("Training rollout grouping is not 8 questions x 4")
    batch_sizes = Counter(int(record["batch_index"]) for record in train_rollouts)
    if set(batch_sizes) != {0, 1} or set(batch_sizes.values()) != {16}:
        raise RuntimeError("Training rollouts must form two batches of 16 trajectories")
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

    # Stage 2: the Teacher subprocess scores every sampled token with the
    # bounded Top-k plus full-vocabulary entropy output.
    teacher_report = _run_vllm_subprocess(args, "teacher")
    _assert_stage_boundary("Teacher stage subprocess exit")
    stage_log.append(teacher_report)
    train_rollouts = _read_jsonl(args.output_dir / "train_rollouts.jsonl")

    # Stage 3: load only the trainable Student and run two genuine EOPD
    # optimizer steps on the vLLM-sampled original token IDs.
    training_report = _eopd_training_stage(args, config, train_rollouts)
    stage_log.append(training_report)

    rollout_calls = student_report["rollout_calls"]
    if len(rollout_calls) != 3 or any(
        call["vllm_generate_calls"] != 1 for call in rollout_calls
    ):
        raise RuntimeError("Each rollout batch must be one vLLM generate call")
    scoring = teacher_report["scoring"]
    if scoring["vllm_generate_calls"] != 1 or scoring["scored_tokens"] <= 0:
        raise RuntimeError("Teacher EOPD scoring must be one vLLM generate call")

    report = {
        "status": "passed",
        "config": str(args.config),
        "algorithm": "eopd",
        "model_family": family,
        "student_model": str(student_path),
        "teacher_model": str(teacher_path),
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
            "sample_indices": validation_indices,
            "rollouts_per_question": 4,
            "metrics": val_metrics,
            "selected_avg": avg4,
        },
        "rollout_speed": {
            "engine": "vllm",
            "calls": rollout_calls,
        },
        "teacher_scoring": scoring,
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
