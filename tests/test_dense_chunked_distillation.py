"""Dense SDPA must preserve chunked distillation values and gradients."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.models.transformers.dense_common import forward_with_torch_backend
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@pytest.mark.parametrize('combined', [False, True])
@pytest.mark.parametrize('entropy_column', [False, True])
def test_dense_variable_length_chunked_loss_and_backward(combined, entropy_column):
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, attn_implementation='sdpa'))
    offsets = torch.tensor([0, 3, 8])
    nested = lambda values: torch.nested.nested_tensor_from_jagged(values, offsets)
    teacher_logits = torch.randn(8, 32)
    teacher_logps, teacher_ids = teacher_logits.log_softmax(-1).topk(3, dim=-1)
    teacher_entropy = torch.tensor([0.2, 1.2, 0.6, 0.9, 0.1, 1.4, 0.3, 1.6])
    batch = TensorDict({
        'input_ids': nested(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])),
        'position_ids': nested(torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])),
        'teacher_ids': nested(teacher_ids), 'teacher_logprobs': nested(teacher_logps),
        'teacher_entropy': nested(teacher_entropy.unsqueeze(-1) if entropy_column else teacher_entropy),
    }, batch_size=[2])
    tu.assign_non_tensor(batch, temperature=1.0, use_remove_padding=False, use_fused_kernels=True,
                        distillation_use_topk=True, distillation_direct_chunked=not combined,
                        distillation_combined_topk=combined, distillation_chunk_size=2)
    engine = SimpleNamespace(pad_to_length=False, use_ulysses_sp=False)
    inputs, output_args = FSDPEngineWithLMHead.prepare_model_inputs(engine, batch)
    raw = forward_with_torch_backend(model, **inputs, use_cache=False)
    actual = FSDPEngineWithLMHead.prepare_model_outputs(engine, raw, output_args, batch, None)
    assert actual['distillation_losses'].offsets().tolist() == offsets.tolist()
    assert ('log_probs' in actual) == combined

    logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                   position_ids=inputs['position_ids'], use_cache=False).logits
    valid = torch.arange(5) < offsets.diff().unsqueeze(1)
    logps = logits[valid].log_softmax(-1)
    student_teacher_logps = logps.gather(-1, teacher_ids)
    teacher_targets = teacher_logps.log_softmax(-1) if combined else teacher_logps
    expected = (teacher_targets.exp() * (teacher_targets - student_teacher_logps)).sum(-1)
    if combined:
        expected = expected * (teacher_entropy > 0.8)
        sampled_logps = logps.gather(-1, output_args['input_ids_rmpad_rolled'].unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual['log_probs'].values(), sampled_logps)
    torch.testing.assert_close(actual['distillation_losses'].values(), expected)
    actual_loss = actual['distillation_losses'].values().sum()
    reference_loss = expected.sum()
    if combined:
        actual_loss = actual_loss + actual['log_probs'].values().sum()
        reference_loss = reference_loss + sampled_logps.sum()
    actual_grad = torch.autograd.grad(actual_loss, model.lm_head.weight)[0]
    reference_grad = torch.autograd.grad(reference_loss, model.lm_head.weight)[0]
    torch.testing.assert_close(actual_grad, reference_grad, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
