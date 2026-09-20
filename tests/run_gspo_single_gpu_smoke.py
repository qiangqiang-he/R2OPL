"""Staged one-GPU GSPO smoke test with a vLLM Student.

Test-only infrastructure mirroring ``run_grpo_single_gpu_smoke``. It samples
two 2-question training batches, generates each batch's four Student rollouts
per question in one vLLM call, runs two genuine GSPO optimizer steps with the
production VERL sequence-level clipped policy loss, and evaluates two AMC23
questions at Avg@4 in a single 8-rollout vLLM call.  ERSR evaluation is
retired and intentionally absent.  GSPO is Student-only, so no Teacher is
ever loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Must precede the ``tests.*`` cross-imports below: running this file as a
# script puts ``tests/`` (not the project root) at sys.path[0].
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from tests.run_grpo_single_gpu_smoke import (
    _assert_stage_boundary,
    _atomic_json,
    _compose_config,
    _generate_records,
    _group_training_records,
    _load_model,
    _load_smoke_examples,
    _load_vllm_engine,
    _read_json,
    _read_jsonl,
    _release_model,
    _release_vllm_engine,
    _resolve_project_path,
    _response_log_probs,
    _stage_report_path,
    _write_jsonl,
)
from utils.opd_runtime import compute_pass_avg_metrics, selected_avg_metrics


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "tests"
    / "configs"
    / "gspo"
    / "gspo_qwen3_1p7b_single_gpu_smoke.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "artifacts" / "gspo_single_gpu_smoke",
    )
    parser.add_argument("--seed", type=int, default=20260923)
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


def _gspo_training_stage(
    args: argparse.Namespace, config: Any, train_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Two genuine GSPO optimizer steps with the production VERL kernel."""

    from verl.trainer.ppo import core_algos
    from verl.workers.config.actor import ActorConfig

    stage_started = time.perf_counter()
    family = str(config.student_model_family)
    student_path = _resolve_project_path(str(config.actor_rollout_ref.model.path))
    actor = config.actor_rollout_ref.actor
    model, _tokenizer = _load_model(student_path, family, training=True)

    loss_config = ActorConfig(
        strategy="fsdp",
        rollout_n=int(config.actor_rollout_ref.rollout.n),
        use_dynamic_bsz=True,
        clip_ratio=float(actor.clip_ratio),
        clip_ratio_low=float(actor.clip_ratio_low),
        clip_ratio_high=float(actor.clip_ratio_high),
        loss_agg_mode=str(actor.loss_agg_mode),
    )

    learning_rate = (
        float(args.learning_rate)
        if args.learning_rate is not None
        else float(actor.optim.lr)
    )
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
            pg_loss, pg_metrics = core_algos.compute_policy_loss_gspo(
                old_log_prob=old_row,
                log_prob=new_row,
                advantages=advantage_row,
                response_mask=mask_row,
                loss_agg_mode=str(actor.loss_agg_mode),
                config=loss_config,
            )
            (pg_loss / len(batch)).backward()
            batch_loss += float(pg_loss.detach().cpu())
            batch_clipfrac += float(pg_metrics["actor/pg_clipfrac"])
            batch_kl += float(pg_metrics["actor/ppo_kl"])
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
                    f"GSPO batch {batch_index} produced non-finite {key}: {entry[key]}"
                )
        metrics.append(entry)

    free_gib = _release_model(model)
    _assert_stage_boundary("Trained Student release")
    return {
        "stage": "student_two_batch_gspo_training",
        "seconds": time.perf_counter() - stage_started,
        "free_gpu_gib_after_release": free_gib,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "policy_loss": "gspo",
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

    # Stage 2: load only the trainable Student and run two genuine GSPO
    # optimizer steps on the vLLM-sampled original token IDs.
    training_report = _gspo_training_stage(args, config, train_rollouts)
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
        "algorithm": "gspo",
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
