"""Objective/probe regression cases ported from OPD_Forge's v2 implementation."""
import asyncio
import math
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from algorithms.r2opl_base import R2OPL_BASE_DEFAULT_LAMBDA, R2OPLBaseTrainer, compute_r2opl_base_batch
from utils.r2opl_v2 import compute_r2opl_v2_probe, find_tail_split_token_end
from verl.trainer import main_ppo_sync as verl_sync

class CharacterTokenizer:
    """Tiny reversible tokenizer sufficient for probe construction tests."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [ord(character) + 10 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(token_id) - 10) for token_id in token_ids)

def _loss_config(lambda_: float = R2OPL_BASE_DEFAULT_LAMBDA):
    return SimpleNamespace(
        loss_mode="r2opl_base",
        loss_max_clamp=20.0,
        use_policy_gradient=True,
        policy_loss_mode="reinforce",
        use_task_rewards=False,
        global_batch_info={},
        r2opl_lambda=lambda_,
    )

def _run_r2opl_v2_loss(student_log_probs, teacher_log_probs, response_mask, result, *, lambda_, miu, branch="total"):
    from verl.trainer.distillation import losses

    actor_config = SimpleNamespace(
        loss_agg_mode="seq-mean-token-mean",
        loss_scale_factor=None,
        global_batch_info={},
    )
    data = {
        "teacher_logprobs": teacher_log_probs.unsqueeze(-1),
        "response_mask": response_mask,
        "r2opl_base_correct_mask": result.correct_mask,
        "r2opl_base_error_mask": result.error_mask,
        "r2opl_base_difficulty": result.difficulty.unsqueeze(-1).expand_as(response_mask),
        "r2opl_base_lambda": torch.full(
            (student_log_probs.shape[0],), lambda_, dtype=torch.float32
        ),
        "r2opl_base_miu": torch.full(
            (student_log_probs.shape[0],), miu, dtype=torch.float32
        ),
        "r2opl_base_gradient_branch": branch,
        "old_log_probs": student_log_probs.detach().clone(),
        "dp_size": 1,
        "batch_num_tokens": int(response_mask.sum().item()),
        "global_batch_size": student_log_probs.shape[0],
    }
    with patch.object(losses, "no_padding_2_padding", lambda tensor, _data: tensor):
        return losses.distillation_loss(
            actor_config,
            SimpleNamespace(distillation_loss=_loss_config(lambda_)),
            {"log_probs": student_log_probs},
            data,
        )

def _expected_loss(student_log_probs, teacher_log_probs, response_mask, result, *, lambda_, miu, branch):
    difficulty = result.difficulty.unsqueeze(-1)
    correct_advantage = difficulty.expand_as(student_log_probs) * miu
    error_advantage = difficulty * lambda_ * (teacher_log_probs - student_log_probs).detach()
    if branch == "correct":
        advantage = correct_advantage * result.correct_mask
    elif branch == "error":
        advantage = error_advantage * result.error_mask
    elif branch == "total":
        advantage = torch.where(result.correct_mask, correct_advantage, error_advantage) * response_mask
    else:
        raise ValueError(branch)
    token_counts = response_mask.sum(dim=-1).clamp_min(1)
    return (-(advantage.detach() * student_log_probs) * response_mask).sum(dim=-1).div(token_counts).mean()

def test_miu_scales_correct_branch_and_probe_rescues_truncated():
    group_size, question_count, width = 8, 2, 5
    batch_size = group_size * question_count
    response_mask = torch.ones((batch_size, width), dtype=torch.bool)
    old = torch.full((batch_size, width), -2.0)
    teacher = old + 0.4
    # Question 0: 4 correct, 4 wrong.  Question 1: all 8 wrong.
    rewards = torch.zeros((batch_size, width))
    rewards[0:4, 0] = 1.0
    group_ids = ["q0"] * group_size + ["q1"] * group_size

    # Two of q1's wrong trajectories are truncated and rescued by the probe.
    truncated = torch.zeros(batch_size, dtype=torch.bool)
    truncated[8:10] = True
    probe_correct = torch.zeros(batch_size, dtype=torch.bool)
    probe_correct[8:10] = True

    result = compute_r2opl_base_batch(
        old, teacher, response_mask, rewards, group_ids, lambda_=0.05, miu=2.0,
        truncated_mask=truncated, probe_correct_mask=probe_correct,
    )

    # q0 success 4/8 -> difficulty 0.5.  q1 success 2/8 (two rescued) -> difficulty 0.75.
    assert result.difficulty[0].item() == pytest.approx(0.5)
    assert result.difficulty[8].item() == pytest.approx(0.75)
    # Rescued rows are correct and no longer error rows.
    assert result.correctness[8:10].all()
    assert not result.error_mask[8:10].any()
    assert result.correct_mask[8:10].all()
    assert result.probe_rescued[8:10].all()
    assert result.probe_rescued.sum().item() == 2
    # q0's four correct rows contribute 0.5*miu=1.0 each; q1's two rescued
    # rows contribute 0.75*miu=1.5 each; trajectory mean = (4*1.0+2*1.5)/6 = 7/6.
    assert result.metrics["r2opl_base/train/correct_advantage_mean"] == pytest.approx(
        7.0 / 6.0
    )
    assert result.metrics["r2opl_base/train/correct_raw_advantage_mean"] == pytest.approx(2.0)
    assert result.metrics["r2opl_base/train/miu"] == pytest.approx(2.0)
    # Rewards are corrected to 1 for rescued trajectories.
    assert result.metrics["r2opl_base/train/mean_score"] == pytest.approx(6.0 / 16.0)
    assert result.metrics["r2opl_base/train/probe_rescued_trajectory_count"] == pytest.approx(2.0)
    assert result.metrics["r2opl_base/train/response_truncated_ratio"] == pytest.approx(2.0 / 16.0)

def test_probe_cannot_rescue_non_truncated_or_verifier_correct():
    response_mask = torch.ones((4, 3), dtype=torch.bool)
    old = torch.zeros((4, 3))
    teacher = torch.ones((4, 3))
    rewards = torch.tensor([0.0, 1.0, 0.0, 0.0])
    truncated = torch.tensor([False, False, True, True])
    probe_correct = torch.tensor([True, True, False, True])

    result = compute_r2opl_base_batch(
        old, teacher, response_mask, rewards, ["g"] * 4, lambda_=0.05, miu=1.0,
        truncated_mask=truncated, probe_correct_mask=probe_correct,
    )
    # Row 0 non-truncated + probe flag -> still error.  Row 1 verifier correct -> correct.
    # Row 2 truncated but probe failed -> error.  Row 3 truncated + probe pass -> correct.
    assert result.correctness.tolist() == [False, True, False, True]
    assert result.probe_rescued.tolist() == [False, False, False, True]

def test_correct_and_error_branches_match_requested_objective_with_miu():
    lambda_, miu = 0.05, 4.0
    response_mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool
    )
    teacher = torch.tensor(
        [[-0.3, -0.8, -1.1], [-1.0, -1.4, 0.0], [-0.5, -1.2, -0.9], [-2.0, 0.0, 0.0]]
    )
    old = torch.tensor(
        [[-0.8, -1.0, -1.3], [-1.4, -1.6, 0.0], [-0.7, -1.0, -1.5], [-1.5, 0.0, 0.0]]
    )
    result = compute_r2opl_base_batch(
        old, teacher, response_mask, torch.tensor([1.0, 0.0, 0.0, 0.0]),
        ["a", "a", "b", "b"], lambda_=lambda_, miu=miu,
    )
    for branch in ("correct", "error", "total"):
        student = old.clone().requires_grad_(True)
        actual, _ = _run_r2opl_v2_loss(
            student, teacher, response_mask, result, lambda_=lambda_, miu=miu, branch=branch
        )
        expected_student = old.clone().requires_grad_(True)
        expected = _expected_loss(
            expected_student, teacher, response_mask, result, lambda_=lambda_, miu=miu, branch=branch
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(student.grad, expected_student.grad)

def test_find_tail_split_token_end_uses_last_sentence_delimiter():
    tokenizer = CharacterTokenizer()
    text = "Step one. Then continue."
    ids = tokenizer.encode(text)
    # The last delimiter is the final period (EOS lookahead), so the recovered
    # prefix reproduces the whole text through the original token IDs.
    token_end = find_tail_split_token_end(tokenizer, ids)
    assert token_end is not None
    assert tokenizer.decode(ids[:token_end]) == text
    # No delimiter -> None.
    assert find_tail_split_token_end(tokenizer, tokenizer.encode("no delimiter here")) is None

def _probe_sequence_lengths(tokenizer, prompt_ids, response_ids, answer):
    from utils.answer_probe import build_answer_probe

    probe = build_answer_probe(tokenizer, answer)
    tail_token_end = find_tail_split_token_end(tokenizer, response_ids)
    return {
        "head": len(prompt_ids) + len(probe.token_ids),
        "tail": len(prompt_ids) + tail_token_end + len(probe.token_ids),
    }

def test_compute_r2opl_v2_probe_rescues_only_when_improving():
    tokenizer = CharacterTokenizer()
    prompt_ids = tokenizer.encode("prompt")
    response_ids = tokenizer.encode("partial reasoning. cut off")
    lengths = _probe_sequence_lengths(tokenizer, prompt_ids, response_ids, "4")

    async def confident_tail(*, sequence_ids, answer_token_positions, routing_key):
        del routing_key
        # The answer is the single character "4" (token id ord("4")+10).
        assert sequence_ids[answer_token_positions[0]] == ord("4") + 10
        return math.log(0.6) if len(sequence_ids) == lengths["tail"] else math.log(0.1)

    result = asyncio.run(
        compute_r2opl_v2_probe(
            tokenizer=tokenizer,
            student_probe=confident_tail,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            answer="4",
            max_new_tokens=len(response_ids),
        )
    )
    assert result.truncated is True
    assert result.probe_correct is True

    async def low_absolute_tail(*, sequence_ids, answer_token_positions, routing_key):
        del routing_key
        # Absolute tail confidence stays below 0.5, but the improvement over
        # the head prior (0.45 - 0.05 = 0.4) exceeds the delta threshold.
        assert sequence_ids[answer_token_positions[0]] == ord("4") + 10
        return math.log(0.45) if len(sequence_ids) == lengths["tail"] else math.log(0.05)

    result = asyncio.run(
        compute_r2opl_v2_probe(
            tokenizer=tokenizer,
            student_probe=low_absolute_tail,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            answer="4",
            max_new_tokens=len(response_ids),
        )
    )
    assert result.truncated is True
    assert result.probe_correct is True

    async def weak_improvement(*, sequence_ids, answer_token_positions, routing_key):
        del routing_key
        assert sequence_ids[answer_token_positions[0]] == ord("4") + 10
        # tail - head = 0.4 - 0.1 = 0.3, which does not EXCEED 0.3 -> fail.
        return math.log(0.4) if len(sequence_ids) == lengths["tail"] else math.log(0.1)

    result = asyncio.run(
        compute_r2opl_v2_probe(
            tokenizer=tokenizer,
            student_probe=weak_improvement,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            answer="4",
            max_new_tokens=len(response_ids),
        )
    )
    assert result.truncated is True
    assert result.probe_correct is False

    # Non-truncated response is never probed.
    async def never_called(**kwargs):
        raise AssertionError("must not probe a non-truncated trajectory")

    result = asyncio.run(
        compute_r2opl_v2_probe(
            tokenizer=tokenizer,
            student_probe=never_called,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            answer="4",
            max_new_tokens=len(response_ids) + 1,
        )
    )
    assert result.truncated is False
    assert result.probe_correct is False

def test_controller_old_log_prob_handles_two_questions_with_probe(monkeypatch):
    from verl.trainer import main_ppo_sync as sync

    question_count, rollouts_per_question = 2, 8
    batch_size, response_len, prompt_len = question_count * rollouts_per_question, 3, 2
    sequence_len = prompt_len + response_len

    class FakeBatch:
        keys = [f"sample-{index}" for index in range(batch_size)]
        partition_id = 0
        tags = [{} for _ in range(batch_size)]

        def __len__(self):
            return batch_size

    prompts = torch.nested.as_nested_tensor(
        [torch.arange(prompt_len) + index for index in range(batch_size)], layout=torch.jagged
    )
    responses = torch.nested.as_nested_tensor(
        [torch.arange(response_len) + 10 + index for index in range(batch_size)], layout=torch.jagged
    )
    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(response_len, dtype=torch.bool) for _ in range(batch_size)], layout=torch.jagged
    )
    teacher_full_sequence = torch.nested.as_nested_tensor(
        [torch.arange(sequence_len, dtype=torch.float32) + index for index in range(batch_size)],
        layout=torch.jagged,
    )
    # One truncated+rescued trajectory in question 1.
    truncated = torch.zeros(batch_size)
    probe_correct = torch.zeros(batch_size)
    truncated[8] = 1.0
    probe_correct[8] = 1.0
    fields = TensorDict(
        {
            "uid": [f"question-{index // rollouts_per_question}" for index in range(batch_size)],
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "rm_scores": torch.tensor([1.0] * rollouts_per_question + [0.0] * rollouts_per_question),
            "teacher_logprobs": teacher_full_sequence,
            "old_log_probs": torch.zeros((batch_size, response_len)),
            "r2opl_v2_truncated": truncated,
            "r2opl_v2_probe_correct": probe_correct,
            "r2opl_v2_probe_attempted": truncated,
        },
        batch_size=[batch_size],
    )
    requested_fields = []
    stored_fields = {}

    def fake_get(*, select_fields, **_kwargs):
        requested_fields.extend(select_fields)
        return fields.select(*select_fields)

    def fake_put(*, fields, **_kwargs):
        stored_fields.update({key: value for key, value in fields.items()})
        return FakeBatch()

    monkeypatch.setattr(sync.tq, "kv_batch_get", fake_get)
    monkeypatch.setattr(sync.tq, "kv_batch_put", fake_put)
    monkeypatch.setattr(
        sync.PPOTrainer, "_compute_old_log_prob", lambda _self, batch, _metrics: batch
    )

    trainer = object.__new__(R2OPLBaseTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": question_count},
            "actor_rollout_ref": {"rollout": {"n": rollouts_per_question}},
            "algorithm": {"r2opl_base": {"lambda": 0.05, "miu": 2.0}},
        }
    )
    trainer._compute_old_log_prob(FakeBatch(), metrics={})

    assert "r2opl_v2_truncated" in requested_fields
    assert "r2opl_v2_probe_correct" in requested_fields
    assert stored_fields["r2opl_base_correct_mask"].is_nested
    assert stored_fields["r2opl_base_error_mask"].is_nested
    assert stored_fields["r2opl_base_difficulty"].is_nested
    assert stored_fields["r2opl_base_lambda"].shape == (batch_size,)
    assert stored_fields["r2opl_base_miu"].shape == (batch_size,)
    assert float(stored_fields["r2opl_base_miu"][0]) == pytest.approx(2.0)

def test_agent_loop_compute_teacher_logprobs_sets_v2_probe_fields():
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopOutput,
        AgentLoopWorker,
    )

    tokenizer = CharacterTokenizer()
    prompt_ids = tokenizer.encode("prompt")
    response_ids = tokenizer.encode("reasoning step. cut off")
    answer = "4"
    diagnostic_topk = 2

    from utils.answer_probe import build_answer_probe

    probe = build_answer_probe(tokenizer, answer)
    tail_end = find_tail_split_token_end(tokenizer, response_ids)
    head_len = len(prompt_ids) + len(probe.token_ids)
    tail_len = len(prompt_ids) + tail_end + len(probe.token_ids)

    class FakeTeacherManager:
        def __init__(self):
            self.probe_calls = []

        async def compute_teacher_logprobs_single(self, **kwargs):
            prompt_length = int(kwargs["student_prompt_length"])
            response_length = int(kwargs["response_length"])
            seq_len = prompt_length + response_length
            causal_start = prompt_length - 1
            ids = torch.zeros((seq_len, 1), dtype=torch.int32)
            ids[causal_start : causal_start + response_length, 0] = torch.tensor(
                kwargs["sequence_ids"][-response_length:], dtype=torch.int32
            )
            logprobs = torch.zeros((seq_len, 1), dtype=torch.float32)
            logprobs[causal_start : causal_start + response_length, 0] = -0.5
            topk_ids = ids.expand(-1, diagnostic_topk).clone()
            topk_logprobs = torch.full((seq_len, diagnostic_topk), -2.0, dtype=torch.float32)
            topk_logprobs[causal_start : causal_start + response_length, 0] = -0.1
            return (
                ids, logprobs, topk_ids, topk_logprobs, logprobs.clone(), None,
                {"teacher_engine_s": 0.0, "teacher_logprob_extract_s": 0.0},
            )

        async def compute_answer_probe_mean_logprob_single(
            self, *, sequence_ids, answer_token_positions, routing_key
        ):
            raise AssertionError("The Teacher must not classify Student truncations")

    class FakeStudentClient:
        def __init__(self):
            self.probe_calls = []

        async def generate(self, *, prompt_ids, **kwargs):
            if kwargs['sampling_params']['prompt_logprobs'] == 0:
                self.probe_calls.append(prompt_ids)
                score = math.log(0.6) if len(prompt_ids) == tail_len else math.log(0.1)
                return SimpleNamespace(extra_fields={
                    'prompt_sampled_ids': [[token] for token in prompt_ids[1:]] + [[0]],
                    'prompt_sampled_logprobs': [[score] for token in prompt_ids],
                })
            seq_len = len(prompt_ids)
            resp_len = len(response_ids)
            causal_start = seq_len - resp_len - 1
            ids = torch.zeros((seq_len, diagnostic_topk), dtype=torch.int32)
            ids[causal_start : causal_start + resp_len, :] = torch.tensor(
                response_ids, dtype=torch.int32
            ).unsqueeze(-1)
            logprobs = torch.full((seq_len, diagnostic_topk), -2.0, dtype=torch.float32)
            logprobs[causal_start : causal_start + resp_len, 0] = -0.1
            return SimpleNamespace(
                extra_fields={"prompt_ids": ids.tolist(), "prompt_logprobs": logprobs.tolist()}
            )

    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True
    worker.teacher_key = "data_source"
    worker.tokenizer = tokenizer
    worker.config = OmegaConf.create(
        {
            "algorithm": {"name": "r2opl_base"},
            "teacher_prompt": "qwen3_no_thinking_prompt",
            "rlvr_generation": {"train_max_new_tokens": len(response_ids)},
            "actor_rollout_ref": {"rollout": {"max_model_len": 1024}},
            "distillation": {"distillation_loss": {"diagnostic_topk": diagnostic_topk}},
        }
    )
    manager = FakeTeacherManager()
    worker.teacher_server_manager = manager
    worker.llm_client = FakeStudentClient()

    async def tokenize_preformatted_prompt(text):
        del text
        return prompt_ids

    worker.tokenize_preformatted_prompt = tokenize_preformatted_prompt
    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        metrics=AgentLoopMetrics(),
    )
    asyncio.run(
        worker._compute_teacher_logprobs(
            output,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            validate=False,
            sample_kwargs={
                "data_source": "math",
                "teacher_prompt_text": "prompt",
                "oa_ground_truth_answer": answer,
            },
        )
    )

    assert output.extra_fields["r2opl_v2_truncated"] == 1.0
    assert output.extra_fields["r2opl_v2_probe_correct"] == 1.0
    assert manager.probe_calls == []
    assert len(worker.llm_client.probe_calls) == 2
    assert {len(seq) for seq in worker.llm_client.probe_calls} == {head_len, tail_len}

def test_agent_loop_output_as_dict_pops_v2_probe_fields():
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopOutput,
    )

    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={"r2opl_v2_truncated": 1.0, "r2opl_v2_probe_correct": 0.0},
    )
    result = output.as_dict()
    assert torch.equal(result["r2opl_v2_truncated"], torch.tensor(1.0, dtype=torch.float32))
    assert torch.equal(result["r2opl_v2_probe_correct"], torch.tensor(0.0, dtype=torch.float32))
    assert "r2opl_v2_truncated" not in result["extra_fields"]
    assert "r2opl_v2_probe_correct" not in result["extra_fields"]
