"""Feasibility check: run Gemma 4 rollouts with SGLang on the cu12 stack.

Target environment: ``r2opl-sgl`` (sglang 0.5.10 + torch 2.9.1+cu128), the
variant for hosts whose NVIDIA driver cannot run the CUDA-13 stack (e.g.
driver 535 / CUDA 12.2) — vLLM has no cu12 build with Gemma 4 support, but
SGLang 0.5.10 (2026-04-05) ships on an entirely cu12 dependency set.

Rollouts are produced by the SGLang engine in ONE ``generate`` call:
2 questions x n=4 rollouts = 8 sequences.

Usage (r2opl-sgl env active, from the repo root):
    python tests/verify_sglang_gemma4.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_text_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/mnt/c/Users/qqian/Desktop/OPD_Forge/models/gemma-4-E2B-it",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--mem-fraction", type=float, default=0.55)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "tests"
        / "artifacts"
        / "vllm_family_checks"
        / "gemma4_sglang_cu12.json",
    )
    args = parser.parse_args()

    if not Path(args.model, "config.json").is_file():
        raise SystemExit(f"model config not found under {args.model}")

    import sglang as sgl
    from transformers import AutoTokenizer

    print("sglang", sgl.__version__)

    # Feasibility-test only: sglang 0.5.11 asserts sglang-kernel>=0.4.2, but
    # 0.4.2+ wheels are CUDA-13 builds while 0.4.1 is the last cu12 build.
    # Neutralize the pure-Python version assertion so we can learn whether the
    # cu12 kernel actually lacks any operator the Gemma 4 path needs.
    for _modname in (
        "sglang.srt.entrypoints.engine",
        "sglang.srt.utils.common",
    ):
        try:
            import importlib

            _mod = importlib.import_module(_modname)
            if hasattr(_mod, "assert_pkg_version"):
                _mod.assert_pkg_version = lambda *a, **k: None
                print(f"patched assert_pkg_version in {_modname}")
        except Exception as exc:  # noqa: BLE001
            print(f"patch skipped for {_modname}: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    questions = [
        "What is 17 * 23? Show your reasoning briefly, then give the final answer.",
        "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    engine_kwargs = {
        "model_path": args.model,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction,
    }
    started = time.perf_counter()
    llm = sgl.Engine(**engine_kwargs)
    load_s = time.perf_counter() - started

    # SGLang's SamplingParams has no vLLM-style ``n``: the standard way (also
    # used by verl's sglang rollout) is to duplicate each prompt n times and
    # submit ALL sequences in ONE ``generate`` call -> one batched call, 8
    # rollouts total for 2 questions x 4.
    rollout_n = 4
    batched_prompts = [p for p in prompts for _ in range(rollout_n)]
    sampling = {
        "max_new_tokens": args.max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    gen_started = time.perf_counter()
    outputs = llm.generate(batched_prompts, sampling_params=sampling)  # ONE batched call
    gen_s = time.perf_counter() - gen_started
    llm.shutdown()

    if len(outputs) != len(batched_prompts):
        raise SystemExit(
            f"expected {len(batched_prompts)} outputs, got {len(outputs)}"
        )

    sequences = []
    total_tokens = 0
    for idx, output in enumerate(outputs):
        prompt_idx, rollout_idx = divmod(idx, rollout_n)
        texts = _as_text_list(
            output.get("text") if isinstance(output, dict) else getattr(output, "text", None)
        )
        if len(texts) != 1:
            raise SystemExit(f"unexpected text shape at output {idx}: {len(texts)}")
        text = texts[0]
        if not text.strip():
            raise SystemExit(f"empty rollout: prompt {prompt_idx} rollout {rollout_idx}")
        n_tokens = len(tokenizer(text).input_ids)
        total_tokens += n_tokens
        sequences.append(
            {
                "prompt_index": prompt_idx,
                "rollout_index": rollout_idx,
                "num_tokens": n_tokens,
                "preview": text.strip()[:160],
            }
        )

    report = {
        "mode": "sglang-0.5.11-cu12-kernel0.4.1",
        "model_path": args.model,
        "num_prompts": len(prompts),
        "rollouts_per_prompt": 4,
        "total_sequences": len(sequences),
        "total_new_tokens": total_tokens,
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "sequences": sequences,
        "status": "ok",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "sequences"}, indent=2))
    print(f"report written to {args.output}")


if __name__ == "__main__":
    main()
