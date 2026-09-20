"""Measure long, Qwen-sized prompt projection with a hard desktop VRAM cap."""
import gc
import json
import time

import torch

from test_vllm_prompt_logprobs_chunking import make_runner, run_schedule


def main():
    free, total = torch.cuda.mem_get_info()
    assert free > 30 * 1024**3, 'Run this probe alone, with at least 30 GiB free'
    torch.cuda.set_per_process_memory_fraction(min(0.55, (free - 4 * 1024**3) / total))
    torch.manual_seed(19)
    # Same vocabulary and hidden dimension as the server's Qwen3-1.7B head.
    weight = torch.randn(2048, 151936, device='cuda', dtype=torch.bfloat16) * 0.02
    records = []
    for length in (4096, 14336):
        prompts = {'a': [i % 151936 for i in range(length + 1)]}
        states = {'a': torch.randn(length + 1, 2048, device='cuda', dtype=torch.bfloat16)}
        reference = None
        for patched in (False, True):
            gc.collect()
            torch.cuda.empty_cache()
            runner = make_runner(prompts, weight, patched=patched)
            runner.num_prompt_logprobs['a'] = 16
            if patched:
                runner._verl_prompt_logprobs_chunk_size = 128
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.inference_mode():
                output = run_schedule(runner, {'a': length + 1}, states)['a']
            if reference is None:
                reference = output
            else:
                for field in ('logprob_token_ids', 'logprobs', 'selected_token_ranks'):
                    torch.testing.assert_close(getattr(output, field), getattr(reference, field))
                assert max(runner.calls) <= 128
            record = dict(tokens=length, chunked=patched, max_projection_rows=max(runner.calls),
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                          free_mib=torch.cuda.mem_get_info()[0] / 1024**2,
                          elapsed_s=time.monotonic() - started)
            assert record['free_mib'] >= 4096
            print(json.dumps(record), flush=True)
            records.append(record)
            del runner, output
        del states, reference
    for old, new in zip(records[::2], records[1::2]):
        assert new['peak_allocated_mib'] < old['peak_allocated_mib'] / 3
    print('PASS: top-16 scores/ranks unchanged, projection bounded, desktop reserve preserved', flush=True)


if __name__ == '__main__':
    main()
