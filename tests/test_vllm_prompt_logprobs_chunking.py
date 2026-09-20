"""Compare bounded scoring against the installed vLLM V1 implementation."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import vllm
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.sampler import Sampler

from verl.workers.rollout.vllm_rollout.prompt_logprobs import enable_chunked_prompt_logprobs


def load_function(path, name):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, LogprobsTensors=LogprobsTensors)
    namespace['async_tensor_h2d'] = lambda values, device: torch.tensor(values, device=device)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


UPSTREAM = load_function(Path(vllm.__file__).parent / 'v1/worker/gpu_model_runner.py',
                         '_get_prompt_logprobs_dict')
ENTROPY = load_function(Path(__file__).resolve().parents[1] /
                        'verl/verl/workers/rollout/vllm_rollout/utils.py', 'enable_eopd_entropy_gather')


def make_runner(prompts, weight, mode='raw_logprobs', entropy=False, patched=False):
    calls = []

    def project(hidden):
        calls.append(len(hidden))
        return hidden @ weight

    sampler = SimpleNamespace(compute_logprobs=Sampler.compute_logprobs, gather_logprobs=Sampler.gather_logprobs)
    if entropy:
        ENTROPY(sampler, 3)
    runner = SimpleNamespace(
        requests={key: SimpleNamespace(prompt_token_ids=value, num_computed_tokens=0,
                                       in_progress_prompt_logprobs_cpu=None) for key, value in prompts.items()},
        num_prompt_logprobs={key: 4 if entropy else 3 for key in prompts},
        input_batch=SimpleNamespace(req_id_to_index={}), query_start_loc=SimpleNamespace(np=None),
        model=SimpleNamespace(compute_logits=project), sampler=sampler,
        model_config=SimpleNamespace(logprobs_mode=mode), device=weight.device,
        calls=calls, syncs=0,
    )
    # Match the actual installed version, including its partial-prefill cache.
    if 'in_progress_dict' in UPSTREAM.__code__.co_varnames:
        runner.input_batch.in_progress_prompt_logprobs_cpu = {}
    def sync():
        runner.syncs += 1
        if weight.is_cuda:
            torch.cuda.synchronize()
    runner._sync_device = sync
    runner._get_prompt_logprobs_dict = MethodType(UPSTREAM, runner)
    if patched:
        enable_chunked_prompt_logprobs(runner, chunk_size=4)
    return runner


def run_schedule(runner, schedule, states):
    runner.input_batch.req_id_to_index = {key: idx for idx, key in enumerate(schedule)}
    runner.query_start_loc.np = np.cumsum([0, *schedule.values()])
    hidden = torch.cat([states[key][runner.requests[key].num_computed_tokens:
                                  runner.requests[key].num_computed_tokens + count]
                        for key, count in schedule.items()])
    result = runner._get_prompt_logprobs_dict(hidden, schedule)
    for key, count in schedule.items():
        runner.requests[key].num_computed_tokens += count
    return result


@pytest.mark.parametrize('mode', ['raw_logprobs', 'processed_logprobs', 'raw_logits', 'processed_logits'])
@pytest.mark.parametrize('entropy', [False, True])
@pytest.mark.parametrize('staged', [False, True])
def test_prompt_scores_equal_upstream(mode, entropy, staged):
    if entropy and 'logits' in mode:
        pytest.skip('EOPD uses normalized logprobs only')
    torch.manual_seed(4)
    prompts = {'a': list(range(19)), 'b': list(range(11))}
    weight = torch.randn(7, 32)
    states = {key: torch.randn(len(value), 7) for key, value in prompts.items()}
    original = make_runner(prompts, weight, mode, entropy)
    chunked = make_runner(prompts, weight, mode, entropy, patched=True)
    # b is unscheduled/preempted on the first step; a hits the exact prompt-1
    # boundary and must return only when the final prompt token is scheduled.
    schedules = [{'a': 9}, {'a': 9, 'b': 4}, {'a': 1, 'b': 7}] if staged else [{'a': 19, 'b': 11}]
    for schedule in schedules:
        expected = run_schedule(original, schedule, states)
        actual = run_schedule(chunked, schedule, states)
        assert actual.keys() == expected.keys()
        for key in actual:
            for field in ('logprob_token_ids', 'logprobs', 'selected_token_ranks'):
                torch.testing.assert_close(getattr(actual[key], field), getattr(expected[key], field))
    assert not chunked.num_prompt_logprobs
    assert original.syncs == chunked.syncs
    assert max(chunked.calls) <= 4 < max(original.calls)
    cache = getattr(chunked.input_batch, 'in_progress_prompt_logprobs_cpu', None)
    assert cache == {} if cache is not None else all(
        r.in_progress_prompt_logprobs_cpu is None for r in chunked.requests.values())


def test_empty_embeddings_and_idempotent_install():
    runner = make_runner({'a': None}, torch.randn(7, 32), patched=True)
    installed = runner._get_prompt_logprobs_dict
    assert enable_chunked_prompt_logprobs(runner)
    assert runner._get_prompt_logprobs_dict is installed
    assert installed(torch.empty(0, 7), {'a': 1}) == {}
    runner.num_prompt_logprobs.clear()
    assert installed(torch.empty(0, 7), {}) == {}
    assert not enable_chunked_prompt_logprobs(SimpleNamespace())  # native V2
    with pytest.raises(ValueError):
        enable_chunked_prompt_logprobs(runner, chunk_size=0)


def test_one_token_prompt():
    runner = make_runner({'a': [1]}, torch.randn(7, 32), patched=True)
    actual = run_schedule(runner, {'a': 1}, {'a': torch.randn(1, 7)})
    assert actual['a'].logprobs.shape == (0, 4)
    assert not runner.calls and runner.syncs == 1
