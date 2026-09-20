"""Correctness-based objectives must receive real scores under every prompt."""
import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from utils.answer_verifier import verify_response_answer
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.trainer import main_ppo_sync as sync


@pytest.mark.parametrize('algorithm', ['r2opl_base', 'opdvr', 'pg_opd'])
@pytest.mark.parametrize('validate', [False, True])
@pytest.mark.parametrize('answer', ['72', '73'])
def test_postprocess_preserves_verifier_outcomes(monkeypatch, algorithm, validate, answer):
    config = OmegaConf.create({'algorithm': {'name': algorithm},
                               'student_prompt': 'explicit_step_prompt',
                               'teacher_prompt': 'explicit_step_prompt'})
    expected_scoring = validate or algorithm in {'r2opl_base', 'opdvr'}
    expected_score = verify_response_answer(r'\boxed{' + answer + '}', '72') if expected_scoring else 0.
    calls, writes = [], []

    async def score(outputs, kwargs):
        calls.append('score')
        outputs[-1].reward_score = verify_response_answer(r'\boxed{' + answer + '}', '72')

    async def teacher(output, **kwargs):
        assert output.reward_score == expected_score

    async def put(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(sync.tq, 'async_kv_batch_put', put)
    worker = SimpleNamespace(config=config, distillation_enabled=True,
                             _compute_score=score, _compute_teacher_logprobs=teacher,
                             _compute_multi_modal_inputs=lambda *args: {},
                             _compute_position_ids=lambda ids, *args: torch.arange(ids.shape[-1]).unsqueeze(0))
    output = AgentLoopOutput(prompt_ids=[1, 2], response_ids=[3, 4, 5], response_mask=[1, 1, 1],
                             metrics=AgentLoopMetrics())
    worker_cls = sync.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    asyncio.run(worker_cls._agent_loop_postprocess(worker, output, validate=validate,
                                                   uid='question', session_id=0, global_steps=1))
    assert bool(calls) == expected_scoring
    assert len(writes) == 1
    assert writes[0]['partition_id'] == ('val' if validate else 'train')
    stored_rewards = writes[0]['fields']['rm_scores'][0]
    torch.testing.assert_close(stored_rewards, torch.tensor([0., 0., expected_score]))


def test_optional_no_thinking_diagnostics_are_preserved():
    config = OmegaConf.create({'algorithm': {'name': 'pg_opd'},
                               'student_prompt': 'qwen3_no_thinking_prompt',
                               'teacher_prompt': 'qwen3_no_thinking_prompt'})
    assert sync.should_track_opd_reward_metrics(config)
