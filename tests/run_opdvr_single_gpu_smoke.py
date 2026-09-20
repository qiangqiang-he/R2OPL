"""Staged one-GPU OPDVR smoke test with real local checkpoints.

Reuses the PG-OPD staged pipeline verbatim (Student vLLM rollouts + Teacher
prompt-logprob scoring in separate subprocesses); the only OPDVR-specific
stage is the training phase, which applies the ReLU correctness gate to the
sampled-token reverse-KL advantage before two genuine optimizer steps.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

from algorithms.opdvr import opdvr_gated_advantage, opdvr_reinforce_loss
from tests.run_pg_opd_single_gpu_smoke import (
    _assert_stage_boundary,
    _atomic_json,
    _compose_config,
    _group_training_records,
    _load_model,
    _read_jsonl,
    _release_model,
    _resolve_project_path,
    _response_log_probs,
    _run_vllm_subprocess,
    _stage_report_path,
    _write_jsonl,
)
from utils.opd_runtime import compute_pass_avg_metrics, selected_avg_metrics


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "tests"
    / "configs"
    / "opdvr"
    / "opdvr_qwen3_4b_instruct_2507_to_1p7b_single_gpu_smoke.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "artifacts" / "opdvr_single_gpu_smoke",
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

    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    teacher_path = _resolve_project_path(
        str(config.distillation.teacher_models.teacher_model.model_path)
    )
    stage_log: list[dict[str, Any]] = []
    started = time.perf_counter()

    # Stage 1: Student vLLM rollouts (subprocess).
    stage_log.append(_run_vllm_subprocess(args, "student"))
    _assert_stage_boundary("Student stage subprocess exit")
    train_rollouts = _read_jsonl(args.output_dir / "train_rollouts.jsonl")
    val_rollouts = _read_jsonl(args.output_dir / "amc23_avg4_rollouts.jsonl")

    uid_counts = Counter(record["uid"] for record in train_rollouts)
    if len(train_rollouts) != 32 or set(uid_counts.values()) != {4}:
        raise RuntimeError("Training rollout grouping is not 8 questions x 4")

    correct_trajectories = sum(1 for record in train_rollouts if record["correct"])
    incorrect_trajectories = len(train_rollouts) - correct_trajectories
    if correct_trajectories == 0 or incorrect_trajectories == 0:
        raise RuntimeError(
            "OPDVR smoke needs a correctness mix; got "
            f"{correct_trajectories} correct / {incorrect_trajectories} incorrect"
        )

    val_metrics = compute_pass_avg_metrics(
        [record["dataset"] for record in val_rollouts],
        [record["uid"] for record in val_rollouts],
        [float(record["correct"]) for record in val_rollouts],
        expected_rollouts=4,
    )
    avg4 = selected_avg_metrics(val_metrics, ["AMC23"], expected_rollouts=4)

    # Stage 2: Teacher scoring (subprocess).
    stage_log.append(_run_vllm_subprocess(args, "teacher"))
    _assert_stage_boundary("Teacher stage subprocess exit")
    train_rollouts = _read_jsonl(args.output_dir / "train_rollouts.jsonl")

    # Stage 3: gated OPDVR training on the sampled tokens.
    stage_started = time.perf_counter()
    student, _tokenizer = _load_model(student_path, family, training=True)
    optimizer = torch.optim.SGD(student.parameters(), lr=float(args.learning_rate))
    train_batches = _group_training_records(train_rollouts, questions_per_batch)
    if len(train_batches) != 2 or any(len(batch) != 16 for batch in train_batches):
        raise RuntimeError("Expected two optimizer batches of 16 trajectories")

    optimizer_metrics: list[dict[str, float]] = []
    gated_stats: list[dict[str, float]] = []
    for batch_index, batch in enumerate(train_batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        positives = 0
        negatives = 0
        zeros = 0
        for record in batch:
            student_log_probs = _response_log_probs(
                student, record["prompt_ids"], record["response_ids"], require_grad=True
            )
            teacher_log_probs = torch.tensor(
                [record["teacher_log_probs"]],
                dtype=student_log_probs.dtype,
                device=student_log_probs.device,
            )
            correct = torch.tensor(
                [bool(record["correct"])],
                device=student_log_probs.device,
            )
            response_mask = torch.ones_like(student_log_probs, dtype=torch.bool)
            advantage = opdvr_gated_advantage(
                student_log_probs, teacher_log_probs, correct, response_mask
            )
            positives += int((advantage > 0).sum())
            negatives += int((advantage < 0).sum())
            zeros += int((advantage == 0).sum())
            sample_loss = opdvr_reinforce_loss(
                student_log_probs, advantage, response_mask
            )
            (sample_loss / len(batch)).backward()
            loss_sum += float(sample_loss.detach().cpu())
            del student_log_probs, teacher_log_probs, response_mask, advantage, sample_loss
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_metrics.append(
            {
                "batch": float(batch_index),
                "loss": loss_sum / len(batch),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
        )
        gated_stats.append(
            {
                "batch": float(batch_index),
                "positive_advantage_tokens": float(positives),
                "negative_advantage_tokens": float(negatives),
                "zero_advantage_tokens": float(zeros),
            }
        )
    free_after_training = _release_model(student)
    student = None
    stage_log.append(
        {
            "stage": "student_two_batch_opdvr_training",
            "seconds": time.perf_counter() - stage_started,
            "free_gpu_gib_after_release": free_after_training,
        }
    )
    _assert_stage_boundary("Trained Student release")

    report = {
        "status": "passed",
        "config": str(args.config.resolve()),
        "algorithm": "opdvr",
        "model_family": family,
        "student_model": str(student_path),
        "teacher_model": str(teacher_path),
        "training": {
            "question_batch_size": questions_per_batch,
            "rollouts_per_question": rollout_n,
            "trajectory_batch_size": questions_per_batch * rollout_n,
            "correct_trajectories": float(correct_trajectories),
            "incorrect_trajectories": float(incorrect_trajectories),
            "optimizer_steps": len(optimizer_metrics),
            "metrics": optimizer_metrics,
            "gated_token_stats": gated_stats,
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
    if args.vllm_stage:
        from tests.run_pg_opd_single_gpu_smoke import _run_internal_vllm_stage

        _run_internal_vllm_stage(args)
        return
    report = run(args)
    import json

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
