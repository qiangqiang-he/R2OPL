"""CPU-only unit tests for the OPDVR algorithm (arXiv:2608.24696)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from omegaconf import OmegaConf

from algorithms.opdvr import (
    OPDVR_LOSS_MODE,
    OPDVR_VARIANT,
    grpd_group_advantage,
    opdvr_gated_advantage,
    opdvr_loss,
    opdvr_reinforce_loss,
    validate_opdvr_config,
)


def test_gated_advantage_sign_contract():
    """Correct trajectories only get >= 0; incorrect only get <= 0."""

    student = torch.tensor([[-0.1, -2.0, -0.3], [-0.5, -0.05, -1.0]])
    teacher = torch.tensor([[-0.2, -1.0, -0.1], [-0.1, -0.5, -2.0]])
    # reverse_kl = student - teacher: [[0.1, -1.0, -0.2], [-0.4, 0.45, 1.0]]
    correct = torch.tensor([True, False])
    gated = opdvr_gated_advantage(student, teacher, correct)
    assert gated.shape == student.shape
    # Row 0 correct: relu(-rkl) = relu([-0.1, 1.0, 0.2]) = [0, 1.0, 0.2]
    assert torch.allclose(gated[0], torch.tensor([0.0, 1.0, 0.2]), atol=1e-6)
    # Row 1 incorrect: -relu(rkl) = -relu([-0.4, 0.45, 1.0]) = [0, -0.45, -1.0]
    assert torch.allclose(gated[1], torch.tensor([0.0, -0.45, -1.0]), atol=1e-6)
    # Sign contract
    assert (gated[0] >= 0).all()
    assert (gated[1] <= 0).all()


def test_gated_advantage_masks_conflicting_directions():
    """Tokens whose ratio sign conflicts with the verifier are zeroed."""

    student = torch.tensor([[-0.1, -3.0]])
    teacher = torch.tensor([[-0.2, -1.0]])
    # rkl = [0.1, -2.0]
    correct = torch.tensor([True])
    gated = opdvr_gated_advantage(student, teacher, correct)
    # correct wants >=0: relu(-rkl) = [0, 2.0]  -- the +0.1 conflicting token is zeroed
    assert torch.allclose(gated, torch.tensor([[0.0, 2.0]]), atol=1e-6)

    incorrect = torch.tensor([False])
    gated_bad = opdvr_gated_advantage(student, teacher, incorrect)
    # incorrect wants <=0: -relu(rkl) = [-0.1, 0]  -- the -2.0 conflicting token is zeroed
    assert torch.allclose(gated_bad, torch.tensor([[-0.1, 0.0]]), atol=1e-6)


def test_gated_advantage_response_mask():
    student = torch.randn(2, 5)
    teacher = torch.randn(2, 5)
    correct = torch.tensor([True, False])
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, :3] = True
    gated = opdvr_gated_advantage(student, teacher, correct, mask)
    assert (gated[:, 3:] == 0).all()
    assert gated.shape == student.shape


def test_gated_advantage_matches_vanilla_opd_on_aligned_tokens():
    """On non-conflicting tokens the gated magnitude equals |rkl|."""

    rkl = torch.tensor([[2.0, -3.0]])
    student = torch.zeros(1, 2)
    teacher = -rkl  # student - teacher = rkl
    gated_correct = opdvr_gated_advantage(student, teacher, torch.tensor([True]))
    assert torch.allclose(gated_correct, torch.tensor([[0.0, 3.0]]), atol=1e-6)
    gated_incorrect = opdvr_gated_advantage(student, teacher, torch.tensor([False]))
    assert torch.allclose(gated_incorrect, torch.tensor([[-2.0, 0.0]]), atol=1e-6)


def test_reinforce_loss_manual_value():
    student = torch.tensor([[-0.5, -1.0]])
    advantage = torch.tensor([[0.2, -0.4]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = opdvr_reinforce_loss(student, advantage, mask)
    expected = -(((0.2 * -0.5) + (-0.4 * -1.0)) / 2.0)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_full_loss_gradient_direction():
    """A positive gated advantage must increase the sampled log-prob."""

    student = torch.tensor([[-0.5, -1.0]], requires_grad=True)
    teacher = torch.tensor([[-2.0, -0.1]])
    correct = torch.tensor([True])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = opdvr_loss(student, teacher, correct, mask)
    loss.backward()
    # rkl = [1.5, -0.9] -> gated (correct) = [0, 0.9]; loss = -(0.9 * -1.0)/1 = 0.9
    # d loss / d student[1] = -0.9 * d(logp)/d(...) < 0 means gradient ascent on logp
    assert student.grad is not None
    assert student.grad[0, 1] < 0  # minimizing loss pushes logp up when adv > 0
    assert torch.allclose(student.grad[0, 0], torch.tensor(0.0), atol=1e-7)


def test_grpd_group_advantage_dr_grpo_style():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    groups = ["q1", "q1", "q2", "q2"]
    adv = grpd_group_advantage(rewards, groups)
    # q1 mean = 0.5 -> [0.5, -0.5]; q2 mean = 1.0 -> [0.0, 0.0]
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 0.0, 0.0]), atol=1e-6)


def test_verl_registry_contains_opdvr():
    from verl.trainer.distillation.losses import get_distillation_loss_fn

    fn = get_distillation_loss_fn(OPDVR_LOSS_MODE)
    from verl.trainer.distillation.losses import (
        compute_opdvr_gated_sampled_token_reverse_kl,
    )

    assert fn is compute_opdvr_gated_sampled_token_reverse_kl


def test_verl_gate_helper_matches_reference_kernel():
    from verl.trainer.distillation.losses import opdvr_correctness_gate

    rkl = torch.tensor([[1.0, -2.0], [0.5, -0.5]])
    correct_rows = torch.tensor([True, False])
    correct_token_mask = (
        correct_rows.to(dtype=torch.bool).unsqueeze(-1).expand_as(rkl)
    )
    reference = opdvr_gated_advantage(
        torch.zeros(2, 2), -rkl, correct_rows
    )  # student=0, teacher=-rkl -> reverse_kl = rkl
    from_verl = opdvr_correctness_gate(rkl, correct_token_mask)
    assert torch.allclose(from_verl, reference, atol=1e-6)


def test_verl_group_advantage_matches_reference():
    from verl.trainer.distillation.losses import opdvr_group_relative_advantage

    rewards = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    groups = ["a", "a", "a", "b", "b", "b"]
    assert torch.allclose(
        opdvr_group_relative_advantage(rewards, groups),
        grpd_group_advantage(rewards, groups),
        atol=1e-6,
    )


def test_config_schema_has_opdvr_grpd():
    from verl.workers.config.distillation import DistillationLossConfig

    cfg = DistillationLossConfig(loss_mode=OPDVR_LOSS_MODE)
    assert hasattr(cfg, "opdvr_grpd")
    assert cfg.opdvr_grpd is False


def _minimal_opdvr_config(**overrides):
    from omegaconf import OmegaConf

    base = {
        "algorithm": {"name": OPDVR_VARIANT},
        "prompt_template": "explicit_step_prompt",
        "student_model_family": "qwen3",
        "teacher_model_family": "qwen3",
        "trainer": {
            "project_name": "R^2OPL",
            "n_gpus_per_node": 4,
            "nnodes": 1,
        },
        "group_name": "OPDVR",
        "data": {
            "train_files": ["x.json"],
            "val_files": ["y.json"],
            "val_datasets": ["AMC23"],
            "max_prompt_length": 2048,
            "max_response_length": 20480,
            "train_batch_size": 64,
        },
        "rlvr_generation": {
            "train_max_new_tokens": 8192,
            "val_max_new_tokens": 20480,
        },
        "actor_rollout_ref": {
            "model": {"path": "/tmp/student"},
            "actor": {"ppo_max_token_len_per_gpu": 10241, "ppo_mini_batch_size": 64},
            "rollout": {
                "n": 4,
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 4,
                "max_model_len": 22528,
                "log_prob_max_token_len_per_gpu": 10241,
                "val_kwargs": {"n": 4},
            },
            "ref": {"log_prob_max_token_len_per_gpu": 10241},
        },
        "distillation": {
            "enabled": True,
            "n_gpus_per_node": 4,
            "nnodes": 1,
            "teacher_models": {
                "teacher_model": {
                    "model_path": "/tmp/teacher",
                    "inference": {
                        "prompt_length": 2048,
                        "response_length": 8192,
                        "max_model_len": 22528,
                        "tensor_model_parallel_size": 1,
                        "data_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    },
                }
            },
            "distillation_loss": {
                "loss_mode": OPDVR_LOSS_MODE,
                "topk": None,
                "diagnostic_topk": 16,
                "selection_ratio": 1.0,
                "selection_method": "random",
                "use_policy_gradient": True,
                "policy_loss_mode": "reinforce",
                "use_task_rewards": False,
            },
        },
    }
    config = OmegaConf.create(base)
    OmegaConf.update(config, "actor_rollout_ref.actor.ulysses_sequence_parallel_size", 1)
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    return config


def test_validate_accepts_minimal_config():
    validate_opdvr_config(_minimal_opdvr_config())


def test_validate_rejects_wrong_loss_mode():
    config = _minimal_opdvr_config()
    OmegaConf.update(config, "distillation.distillation_loss.loss_mode", "reverse_kl")
    with pytest.raises(ValueError, match="loss_mode"):
        validate_opdvr_config(config)


def test_validate_rejects_wrong_algorithm_name():
    config = _minimal_opdvr_config()
    OmegaConf.update(config, "algorithm.name", "pg_opd")
    with pytest.raises(ValueError, match="opdvr"):
        validate_opdvr_config(config)


def test_registry_resolves_opdvr():
    from algorithms import resolve_algorithm

    spec = resolve_algorithm(_minimal_opdvr_config())
    assert spec.name == OPDVR_VARIANT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
