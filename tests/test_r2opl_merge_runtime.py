"""Exercise merged production methods on CPU, without Ray actors or model files."""
from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer import main_ppo_sync as sync
from verl.utils import tensordict_utils as tu


@pytest.mark.parametrize('branches', [False, True])
def test_fsdp_real_backward_preserves_branch_gradients(monkeypatch, branches):
    from verl.workers.engine.fsdp import transformer_impl as impl

    class CPUEngine(impl.FSDPEngine):
        def __init__(self):
            self.module = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.constant_(self.module.weight, 2.)
            self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.1)
            self.ulysses_sequence_parallel_size = 1
            self.scaler = None
            self.sync_calls = []

        def get_data_parallel_group(self):
            return None

        def get_data_parallel_size(self):
            return 4

        def _gradient_sync_context(self, *, is_last_micro_batch):
            self.sync_calls.append(is_last_micro_batch)
            return nullcontext()

        def forward_step(self, data, loss_function, forward_only):
            branch = tu.get_non_tensor_data(data, 'r2opl_base_gradient_branch', 'total')
            coefficient = {'correct': 2., 'error': 3., 'total': 5.}[branch]
            loss = self.module(data['x']).sum() * coefficient
            return loss, {'loss': loss.detach().item(), 'metrics': {}, 'model_output': {'unused': loss.detach()}}

    engine = CPUEngine()
    monkeypatch.setattr(impl, 'get_device_id', lambda: 'cpu')
    monkeypatch.setattr(impl.torch.distributed, 'all_reduce', lambda *a, **kw: None)
    monkeypatch.setattr(impl, 'prepare_micro_batches', lambda data, **kw: ([data[:1], data[1:]], None))
    monkeypatch.setattr(impl, 'postprocess_batch_func', lambda output_lst, **kw: output_lst)
    data = TensorDict({'x': torch.tensor([[1.], [2.]]), 'loss_mask': torch.ones(2, 1)}, batch_size=[2])
    if branches:
        data['r2opl_base_correct_mask'] = torch.ones(2, 1)
        data['r2opl_base_error_mask'] = torch.ones(2, 1)
    output = engine.forward_backward_batch(data, loss_function=object())
    assert engine.module.weight.grad.item() == pytest.approx(15.)
    assert sum(o['loss'] for o in output) == pytest.approx(30.)
    assert len(output) == 2
    assert all('model_output' not in o for o in output)
    if branches:
        assert output[0]['metrics']['r2opl_base/correct_grad_norm'] == pytest.approx(6.)
        assert output[0]['metrics']['r2opl_base/error_grad_norm'] == pytest.approx(9.)
        assert engine.sync_calls == [False, True, False, True]


@pytest.mark.parametrize('routing_key', [None, 'Math'])
def test_llm_client_dispatch_accepts_actual_vllm_rpc(monkeypatch, routing_key):
    from verl.workers.rollout.llm_server import LLMServerClient
    from verl.workers.rollout.replica import TokenOutput
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    server_class = vLLMHttpServer
    signature = inspect.signature(server_class.generate)
    calls = []

    async def generate(**kwargs):
        signature.bind(None, **kwargs)
        calls.append(kwargs)
        return TokenOutput(token_ids=[1], log_probs=[-0.1], extra_fields={})

    client = LLMServerClient(OmegaConf.create({'actor_rollout_ref': {'rollout': {'name': 'vllm'}}}))
    async def acquire(*args, **kwargs):
        return 'server', SimpleNamespace(generate=SimpleNamespace(remote=generate))
    monkeypatch.setattr(client, '_acquire_server', acquire)
    monkeypatch.setattr(client, '_release_server', lambda *a, **kw: None)
    asyncio.run(client.generate('id', prompt_ids=[1, 2], sampling_params={'prompt_logprobs': 16},
                                routing_key=routing_key, mm_processor_output=None))
    assert len(calls) == 1
    assert 'routing_key' not in calls[0]


def test_prompt_failure_settles_all_sessions_before_terminal_status(monkeypatch):
    events = []
    async def put(**kwargs):
        events.append(kwargs['tag'])
    monkeypatch.setattr(sync.tq, 'async_kv_put', put)

    async def session(*args, session_id, **kwargs):
        if session_id == 0:
            raise TypeError("unexpected keyword argument 'routing_key'")
        await asyncio.sleep(.01)
        events.append({'status': 'sibling_finished'})

    worker = SimpleNamespace(config=OmegaConf.create({'actor_rollout_ref': {'rollout': {'n': 2}}}),
                             _run_agent_loop=session)
    worker_class = sync.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    asyncio.run(worker_class._run_prompt(worker, {'uid': 'q', 'global_steps': 1}, {}, {'validate': False}))
    assert events[-1]['status'] == 'failure'
    assert any(e['status'] == 'sibling_finished' for e in events[:-1])
    assert 'routing_key' in events[-1]['error']


def test_failed_replay_keeps_original_error_instead_of_empty_batch():
    replay = sync.ReplayBuffer.__new__(sync.ReplayBuffer)
    replay.partitions = defaultdict(dict, {'train': {
        'q': {'global_steps': 1, 'status': 'failure', 'error': "TypeError: routing_key"},
        'q_0_0': {'global_steps': 1, 'status': 'success'},
    }})
    replay.lock = threading.Lock()
    replay.poll_interval = 0
    replay._poll_error = None
    replay._stop_event = threading.Event()
    with pytest.raises(RuntimeError, match='routing_key'):
        replay.sample('train', global_steps=1)


def test_fused_qwen_forward_with_installed_transformers():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from verl.models.transformers.dense_common import forward_with_torch_backend
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=8, attn_implementation='eager'))
    tokens = torch.tensor([[1, 2, 3, 4]])
    output = forward_with_torch_backend(model, input_ids=tokens, temperature=1.0,
                                        return_dict=True, use_cache=False)
    assert output.log_probs.shape == tokens.shape
    output.log_probs.sum().backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_r2opl_teacher_config_materialization():
    from tests.test_formal_config_compose import compose_formal, formal_experiment_configs
    from algorithms import resolve_algorithm
    from verl.utils.config import omega_conf_to_dataclass
    for name in formal_experiment_configs():
        if not name.startswith("r2opl_base/"):
            continue
        cfg = compose_formal(name)
        algorithm = resolve_algorithm(cfg)
        algorithm.configure_defaults(cfg)
        algorithm.configure_batch(cfg)
        actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
        rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
        assert actor.strategy == 'fsdp'
        assert rollout.data_parallel_size == cfg.trainer.n_gpus_per_node
        distillation = omega_conf_to_dataclass(cfg.distillation)
        if distillation.enabled:
            teacher = distillation.teacher_models['default']
            assert teacher.world_size == 4
            if 'gemma4_31b' in name:
                assert teacher.inference.tensor_model_parallel_size == 4
                assert teacher.num_replicas == 1


def test_disabled_sleep_never_wakes_vllm():
    from unittest.mock import AsyncMock
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    cls = vLLMHttpServer
    server = SimpleNamespace(node_rank=0, config=SimpleNamespace(free_cache_engine=False),
                             rollout_mode=RolloutMode.HYBRID, engine=SimpleNamespace(wake_up=AsyncMock(),
                             reset_prefix_cache=AsyncMock()), _get_wake_up_tags=lambda: ['weights', 'kv_cache'])
    asyncio.run(cls.wake_up(server))
    server.engine.wake_up.assert_not_awaited()






def test_bypass_old_logprobs_returns_batch_and_keeps_rollout_values(monkeypatch):
    original = torch.tensor([[-1., -2.]])
    data = TensorDict({'rollout_log_probs': original.clone()}, batch_size=[1])
    writes = []
    monkeypatch.setattr(sync.tq, 'kv_batch_get', lambda **kwargs: data.clone())
    monkeypatch.setattr(sync.tq, 'kv_batch_put', lambda **kwargs: writes.append(kwargs))
    trainer = SimpleNamespace(config=OmegaConf.create({'algorithm': {'rollout_correction': {'bypass_mode': True}}}))
    batch = SimpleNamespace(keys=['q'], partition_id='train')
    assert sync.PPOTrainer._compute_old_log_prob(trainer, batch, {}) is batch
    torch.testing.assert_close(writes[0]['fields']['old_log_probs'], original)
    assert 'rollout_log_probs' not in writes[0]['fields']


def test_fused_gemma_forward_with_installed_transformers():
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
    from verl.models.transformers.dense_common import forward_with_torch_backend

    config = Gemma4Config(text_config=Gemma4TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8,
        num_global_key_value_heads=1, vocab_size_per_layer_input=32, hidden_size_per_layer_input=4,
        num_kv_shared_layers=0, layer_types=['full_attention'], attn_implementation='eager'),
        vision_config=None, audio_config=None)
    model = Gemma4ForConditionalGeneration(config)
    tokens = torch.tensor([[2, 3, 4, 5]])
    expected = model(tokens, use_cache=False).logits.log_softmax(-1).gather(
        -1, tokens.roll(-1, dims=-1).unsqueeze(-1)).squeeze(-1)
    output = forward_with_torch_backend(model, input_ids=tokens, temperature=1.0,
                                        return_dict=True, use_cache=False)
    torch.testing.assert_close(output.log_probs, expected)
    output.log_probs.sum().backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_non_collecting_worker_returns_valid_transferqueue_metadata():
    from verl.utils.transferqueue_utils import _postprocess_common, BatchMeta
    output = _postprocess_common(TensorDict({}, batch_size=[]), put_data=True, need_collect=False)
    assert isinstance(output, BatchMeta)
    assert output.size == 0


@pytest.mark.parametrize('receiver_fails', [False, True])
def test_shared_memory_weight_transfer_round_trip_without_cuda_ipc(monkeypatch, tmp_path, receiver_fails):
    from concurrent.futures import ThreadPoolExecutor
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as transfer

    def unexpected_ipc():
        raise AssertionError('shared-memory transport must not invoke CUDA IPC cleanup')
    monkeypatch.setattr(transfer, 'get_torch_device', lambda: SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None, ipc_collect=unexpected_ipc))
    monkeypatch.setattr(transfer, 'is_support_ipc', lambda: True)
    endpoint = f'ipc://{tmp_path}/weights.sock'
    weights = [('first', torch.arange(100, dtype=torch.float32)),
               ('second', torch.arange(200, dtype=torch.bfloat16))]
    received = {}
    receiver = transfer.BucketedWeightReceiver(endpoint, device=torch.device('cpu'), use_shm=True)
    sender = transfer.BucketedWeightSender(endpoint, bucket_size_mb=1, use_shm=True)
    def receive_bucket(bucket, last):
        if receiver_fails:
            raise ValueError('incompatible weight API')
        received.update({k: v.clone() for k, v in bucket})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(receiver.receive_weights, receive_bucket)
        if receiver_fails:
            with pytest.raises(RuntimeError, match='incompatible weight API'):
                asyncio.run(sender.async_send_weights(iter(weights)))
            with pytest.raises(ValueError, match='incompatible weight API'):
                future.result(timeout=5)
            return
        else:
            asyncio.run(sender.async_send_weights(iter(weights)))
            future.result(timeout=5)
    for name, expected in weights:
        torch.testing.assert_close(received[name], expected)


def test_vllm_tied_weights_can_arrive_in_separate_buckets():
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from verl.workers.rollout.vllm_rollout.utils import drop_tied_weight_aliases
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(4, 2)
    model.lm_head = model.model.embed_tokens
    weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    first = drop_tied_weight_aliases(model, [('model.embed_tokens.weight', weight)])
    second = drop_tied_weight_aliases(model, [('lm_head.weight', weight)])
    AutoWeightsLoader(model).load_weights(first)
    AutoWeightsLoader(model).load_weights(second)
    assert second == []
    torch.testing.assert_close(model.lm_head.weight, weight)
