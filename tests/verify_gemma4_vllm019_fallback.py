"""Can vllm 0.19.1 (cu12-built, the max stack for NVIDIA driver 535) load Gemma 4
and produce real vLLM rollouts through the generic Transformers fallback?

This mirrors the repo's legacy recipe from ``tests/run_cross_domain_rollouts.py``
(patches + plain LLM construction with MM towers disabled), then issues ONE
generate call producing 2 questions x 4 rollouts.

Run inside the r2opl-cu12 env:
    python tests/verify_gemma4_vllm021_fallback.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

MODEL_PATH = "/home/qqh/OPD_Forge/models/gemma-4-E2B-it"
OUT_PATH = Path("/home/qqh/R2OPL/tests/artifacts/vllm_family_checks/gemma4_vllm021_fallback.json")

QUESTIONS = [
    "What is 17 * 23? Show your reasoning briefly, then give the final answer.",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
]


def _apply_gemma4_vllm_compatibility_patches() -> None:
    """Legacy patches (verbatim from tests/run_cross_domain_rollouts.py).

    Teach vLLM's generic Transformers loader the local Gemma 4 layout:
    persistent clipping-limit buffers, redundant per-layer K/V tensors for the
    KV-shared tail layers, and skipping the unused multimodal towers.
    """

    from vllm.model_executor.models.utils import AutoWeightsLoader

    original = AutoWeightsLoader._add_loadable_non_param_tensors
    if not getattr(original, "_r2opl_loads_buffers", False):

        def add_loadable_buffers(self, module, child_params):
            original(self, module, child_params)
            for name, value in module.named_buffers(recurse=False):
                child_params.setdefault(name, value)

        add_loadable_buffers._r2opl_loads_buffers = True
        AutoWeightsLoader._add_loadable_non_param_tensors = add_loadable_buffers

    from vllm.model_executor.models.transformers.base import Base as TransformersBase
    from vllm.model_executor.models.transformers.multimodal import MultiModalMixin

    current_get_language_model = MultiModalMixin.get_language_model
    original_get_language_model = getattr(
        current_get_language_model,
        "_r2opl_original_get_language_model",
        current_get_language_model,
    )
    if not hasattr(current_get_language_model, "_r2opl_original_get_language_model"):

        def get_cached_language_model(self):
            return self._r2opl_language_model

        get_cached_language_model._r2opl_original_get_language_model = (
            original_get_language_model
        )
        MultiModalMixin.get_language_model = get_cached_language_model

    original_init = TransformersBase.__init__
    if not getattr(original_init, "_r2opl_gemma4_shared_kv", False):

        def init_with_gemma4_shared_kv(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if isinstance(self, MultiModalMixin):
                object.__setattr__(
                    self,
                    "_r2opl_language_model",
                    original_get_language_model(self),
                )
            if getattr(self.config, "model_type", None) == "gemma4":
                for suffix in ("k_norm", "v_norm", "k_proj", "v_proj"):
                    if suffix not in self.ignore_unexpected_suffixes:
                        self.ignore_unexpected_suffixes.append(suffix)

        init_with_gemma4_shared_kv._r2opl_gemma4_shared_kv = True
        TransformersBase.__init__ = init_with_gemma4_shared_kv


def main() -> None:
    _apply_gemma4_vllm_compatibility_patches()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in QUESTIONS
    ]

    started = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=20260920,
        gpu_memory_utilization=0.42,
        max_model_len=4096,
        max_num_seqs=8,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    load_s = time.perf_counter() - started

    sampling = SamplingParams(n=4, max_tokens=256, temperature=0.6, top_p=0.95)
    gen_started = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    gen_s = time.perf_counter() - gen_started

    sequences = []
    total_tokens = 0
    for prompt_idx, output in enumerate(outputs):
        if len(output.outputs) != 4:
            raise SystemExit(f"expected 4 rollouts, got {len(output.outputs)}")
        for rollout_idx, completion in enumerate(output.outputs):
            if not completion.text.strip():
                raise SystemExit(f"empty rollout {prompt_idx}/{rollout_idx}")
            total_tokens += len(completion.token_ids)
            sequences.append(
                {
                    "prompt_index": prompt_idx,
                    "rollout_index": rollout_idx,
                    "finish_reason": completion.finish_reason,
                    "num_tokens": len(completion.token_ids),
                    "preview": completion.text.strip()[:160],
                }
            )

    report = {
        "env": "r2opl-cu12 (vllm 0.19.1 + torch 2.10.0+cu128, cu12 server variant)",
        "model_path": MODEL_PATH,
        "num_prompts": 2,
        "rollouts_per_prompt": 4,
        "total_sequences": len(sequences),
        "total_new_tokens": total_tokens,
        "tokens_per_second": total_tokens / gen_s if gen_s else 0,
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "sequences": sequences,
        "status": "ok",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sequences"}, indent=2))
    print(f"report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
