"""Single-GPU vLLM rollout verification for each supported model family.

Loads the smallest available model of one family with vLLM and produces
2 questions x 4 rollouts = 8 sequences in ONE ``LLM.generate`` call, then
reports basic sanity statistics.  Each family runs in its own process so
GPU memory is fully released between models.

Usage (from /home/qqh/R2OPL with the r2opl env active):
    python tests/verify_vllm_model_families.py --family qwen3
    python tests/verify_vllm_model_families.py --family qwen35
    python tests/verify_vllm_model_families.py --family qwen36
    python tests/verify_vllm_model_families.py --family gemma4
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "qwen3": {
        "path": "/mnt/c/Users/qqian/Desktop/OPD_Forge/models/Qwen3-0.6B",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.55,
        "questions": [
            "What is 17 * 23? Show your reasoning briefly, then give the final answer.",
            "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
        ],
    },
    "qwen35": {
        "path": "/mnt/c/Users/qqian/Desktop/OPD_Forge/models/Qwen3.5-2B",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.55,
        "questions": [
            "What is 17 * 23? Show your reasoning briefly, then give the final answer.",
            "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
        ],
    },
    "qwen36": {
        "path": "/mnt/c/Users/qqian/Desktop/OPD_Forge/models/Qwen3.6-27B-FP8",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        # Hybrid Mamba attention: each decode sequence needs one Mamba cache
        # block, and only ~169 fit beside the 27B FP8 weights on 48 GB.
        "max_num_seqs": 128,
        "questions": [
            "What is 17 * 23? Show your reasoning briefly, then give the final answer.",
            "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
        ],
    },
    "gemma4": {
        "path": "/mnt/c/Users/qqian/Desktop/OPD_Forge/models/gemma-4-E2B-it",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.55,
        "questions": [
            "What is 17 * 23? Show your reasoning briefly, then give the final answer.",
            "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tests" / "artifacts" / "vllm_family_checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = FAMILIES[args.family]
    model_path = spec["path"]
    if not Path(model_path, "config.json").is_file():
        raise SystemExit(f"model config not found under {model_path}")

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": model_path,
        "max_model_len": spec["max_model_len"],
        "gpu_memory_utilization": spec["gpu_memory_utilization"],
        "enable_prefix_caching": True,
    }
    if "max_num_seqs" in spec:
        llm_kwargs["max_num_seqs"] = spec["max_num_seqs"]
    started = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - started

    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in spec["questions"]
    ]
    # ONE generate call: 2 prompts x n=4 rollouts -> 8 sequences in a single batch.
    sampling = SamplingParams(
        n=4,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gen_started = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    gen_s = time.perf_counter() - gen_started

    sequences = []
    total_tokens = 0
    for prompt_idx, output in enumerate(outputs):
        if len(output.outputs) != 4:
            raise SystemExit(f"expected 4 rollouts per prompt, got {len(output.outputs)}")
        for rollout_idx, completion in enumerate(output.outputs):
            text = completion.text
            n_tokens = len(completion.token_ids)
            total_tokens += n_tokens
            finish = completion.finish_reason
            if n_tokens == 0 or not text.strip():
                raise SystemExit(f"empty rollout: prompt {prompt_idx} rollout {rollout_idx}")
            sequences.append(
                {
                    "prompt_index": prompt_idx,
                    "rollout_index": rollout_idx,
                    "finish_reason": finish,
                    "num_tokens": n_tokens,
                    "preview": text.strip()[:160],
                }
            )

    report = {
        "family": args.family,
        "model_path": model_path,
        "num_prompts": len(prompts),
        "rollouts_per_prompt": 4,
        "total_sequences": len(sequences),
        "total_new_tokens": total_tokens,
        "tokens_per_second_overall": total_tokens / gen_s if gen_s > 0 else math.inf,
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "sequences": sequences,
        "status": "ok",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{args.family}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sequences"}, indent=2))
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
