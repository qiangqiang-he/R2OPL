"""Fail visibly on a broken probe, while preserving the original rescue rule."""
import asyncio
import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from algorithms.r2opl_base import R2OPLBaseTrainer, R2OPL_BASE_PROBE_RESCUE_MARGIN
from test_r2opl_probe_migration import CharacterTokenizer
from utils.r2opl_v2 import compute_r2opl_v2_probe, R2OPL_V2_DELTA_THRESHOLD
from utils.answer_probe import score_answer_probe


@pytest.mark.parametrize('failure', ['rpc', 'nan', 'missing_answer'])
def test_probe_does_not_hide_runtime_failure(failure):
    tokenizer = CharacterTokenizer()
    response = tokenizer.encode('Reasoning. unfinished')

    async def score(**kwargs):
        if failure == 'rpc':
            raise TypeError('incompatible API')
        return float('nan')

    with pytest.raises((TypeError, RuntimeError, ValueError)):
        asyncio.run(compute_r2opl_v2_probe(
            tokenizer=tokenizer, student_probe=score,
            prompt_ids=tokenizer.encode('Question'), response_ids=response,
            answer='' if failure == 'missing_answer' else '72', max_new_tokens=len(response)))


def test_missing_controller_metadata_is_not_defaulted_to_false():
    with pytest.raises(RuntimeError, match='missing'):
        R2OPLBaseTrainer._scalar_field(TensorDict({}, batch_size=[1]), 'r2opl_v2_truncated', 1)
    assert R2OPL_BASE_PROBE_RESCUE_MARGIN == R2OPL_V2_DELTA_THRESHOLD == 0.3


@pytest.mark.parametrize('invalid', [None, 'alignment', 'nonfinite', 'context'])
def test_student_probe_uses_causal_answer_tokens_and_checks_capacity(invalid):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        assert kwargs['sampling_params']['prompt_logprobs'] == 0
        return SimpleNamespace(extra_fields={
            'prompt_sampled_ids': [[20], [31 if invalid == 'alignment' else 30], [40], [0]],
            'prompt_sampled_logprobs': [[-9.], [float('nan') if invalid == 'nonfinite' else -0.2], [-0.4], [0.]],
        })

    coro = score_answer_probe(
        client=SimpleNamespace(generate=generate), max_model_len=4 if invalid == 'context' else 5,
        sequence_ids=[10, 20, 30, 40], answer_token_positions=[2, 3])
    if invalid:
        with pytest.raises((ValueError, RuntimeError)):
            asyncio.run(coro)
        if invalid == 'context':
            assert not calls
    else:
        assert asyncio.run(coro) == pytest.approx(-0.3)
