"""Quick family check for the r2opl-cu12 server-variant env (vllm 0.19.1 +
torch 2.10.0+cu128): Qwen3-0.6B, 2 questions x 4 rollouts in ONE generate."""

import json
import time

from vllm import LLM, SamplingParams

MODEL = "/mnt/c/Users/qqian/Desktop/OPD_Forge/models/Qwen3-0.6B"

QUESTIONS = [
    "What is 17 * 23? Show your reasoning briefly, then give the final answer.",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
]


def main() -> None:
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.42,
        max_num_seqs=8,
        enable_prefix_caching=True,
        seed=20260920,
    )
    load_s = time.perf_counter() - t0
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        for q in QUESTIONS
    ]
    t1 = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(n=4, max_tokens=256, temperature=0.6, top_p=0.95))
    gen_s = time.perf_counter() - t1
    n_tok = sum(len(c.token_ids) for o in outs for c in o.outputs)
    assert all(len(o.outputs) == 4 for o in outs)
    assert all(c.text.strip() for o in outs for c in o.outputs)
    print(
        json.dumps(
            {
                "family": "qwen3",
                "env": "r2opl-cu12 vllm 0.19.1 + torch 2.10.0+cu128",
                "sequences": 8,
                "total_new_tokens": n_tok,
                "tokens_per_second": round(n_tok / gen_s, 1),
                "load_seconds": round(load_s, 1),
                "generate_seconds": round(gen_s, 2),
                "status": "ok",
            }
        )
    )


if __name__ == "__main__":
    main()
