"""Exercise the actual vLLM Student diagnostic RPC path with a 14K prompt."""
import json
import os
import signal
import time

os.environ.setdefault('VLLM_USE_V2_MODEL_RUNNER', '0')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class LongPromptProbeExtension(vLLMColocateWorkerExtension):
    def instrument_prompt_probe(self):
        return instrument(self)

    def report_prompt_probe(self):
        return report(self)


def instrument(worker):
    import torch
    runner = worker.model_runner
    assert runner._verl_prompt_logprobs_chunk_size == 128
    original = runner.model.compute_logits
    runner._probe_projection_rows = []

    def project(hidden, *args, **kwargs):
        runner._probe_projection_rows.append(len(hidden))
        return original(hidden, *args, **kwargs)

    runner.model.compute_logits = project
    free, total = torch.cuda.mem_get_info()
    # A hard allocator limit leaves ample desktop memory even if the probe fails.
    torch.cuda.set_per_process_memory_fraction(min(0.55, (free - 4 * 1024**3) / total))
    torch.cuda.reset_peak_memory_stats()


def report(worker):
    import torch
    torch.cuda.synchronize()
    rows = worker.model_runner._probe_projection_rows
    return dict(max_projection_rows=max(rows), projection_calls=len(rows),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                free_mib=torch.cuda.mem_get_info()[0] / 1024**2)


def main():
    import torch
    from vllm import LLM, SamplingParams

    def timeout(*_):
        raise TimeoutError('No diagnostic result within 180 seconds; stop and investigate')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    assert torch.cuda.mem_get_info()[0] > 30 * 1024**3
    started = time.monotonic()
    llm = LLM(model='./models/Qwen3-0.6B', max_model_len=14337, max_num_batched_tokens=22528,
              max_num_seqs=8, enforce_eager=True, gpu_memory_utilization=0.25,
              kv_cache_memory_bytes=3 * 1024**3, enable_prefix_caching=False,
              worker_extension_cls='probe_r2opl_long_prompt.LongPromptProbeExtension')
    llm.collective_rpc('monkey_patch_model', kwargs={'vocab_size': len(llm.get_tokenizer())})
    llm.collective_rpc('instrument_prompt_probe')
    tokenizer = llm.get_tokenizer()
    base = tokenizer.encode('Question: If Alice has 12 apples and buys 3 more, how many apples does she have?\n')
    prompt = (base * (14336 // len(base) + 1))[:14336]
    # Same full-sequence diagnostic request as the R2OPL Student top-k path.
    for attempt in range(2):
        output = llm.generate([{'prompt_token_ids': prompt}],
                              SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, prompt_logprobs=16),
                              use_tqdm=False)[0]
        assert len(output.prompt_logprobs) == len(prompt)
        assert output.prompt_logprobs[0] is None
        assert all(16 <= len(row) <= 17 for row in output.prompt_logprobs[1:])
        assert len(output.outputs[0].token_ids) == 1
        stats = llm.collective_rpc('report_prompt_probe')[0]
        assert stats['max_projection_rows'] <= 128
        assert stats['free_mib'] >= 4096
        print(json.dumps(dict(attempt=attempt + 1, prompt_tokens=len(prompt), elapsed_s=time.monotonic()-started,
                              **stats)), flush=True)
        signal.alarm(180)
    signal.alarm(0)
    print('PASS: two real 14K Student diagnostic requests completed with bounded projection', flush=True)
    llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
