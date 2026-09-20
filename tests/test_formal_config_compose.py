"""CPU-only composition tests for every FORMAL training config.

These tests reproduce exactly what ``bash scripts/start_train.sh <config>``
does up to (but not including) any GPU/Ray/model work:

1. compose the config through the formal path, including the
   ``pkg://verl.trainer.config`` searchpath (this is what pulls upstream
   verl's schema defaults such as ``data.gen_batch_size: null`` into the
   composed config -- the smoke-only compose path never sees them, which is
   how the 2026-09-20 server crash slipped through);
2. run ``configure_vllm_no_sleep`` + ``resolve_algorithm`` +
   ``configure_defaults`` + ``configure_batch`` + ``OmegaConf.resolve`` +
   ``algorithm.validate`` -- the identical CPU pipeline the entrypoint runs
   before touching CUDA;
3. assert the derived dataloader batch size is a positive integer (the
   gen_batch_size null regression), the sleep-mode contract holds, the
   evaluation sets are exactly the three approved ones, and ERSR stays
   disabled in formal runs.

No GPU, no Ray init, no tokenizer/model loading.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VLLM_USE_V1_MULTIPROCESSING", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from algorithms import resolve_algorithm  # noqa: E402
from utils.training_entrypoint import configure_vllm_no_sleep  # noqa: E402

CONFIG_ROOT = (PROJECT_ROOT / "configs").resolve()
ALGORITHM_DIRS = ("pg_opd", "eopd", "grpo", "gspo", "r2opl_base")
APPROVED_VAL_DATASETS = ["AMC23", "SciBench", "LogicBench"]


def formal_experiment_configs() -> list[str]:
    names: list[str] = []
    for algorithm_dir in ALGORITHM_DIRS:
        folder = CONFIG_ROOT / algorithm_dir
        for yaml_path in sorted(folder.glob("*.yaml")):
            names.append(f"{algorithm_dir}/{yaml_path.stem}")
    assert names, "no formal experiment configs found"
    return names


def compose_formal(config_name: str):
    """Mirror start_train.sh's formal branch exactly."""
    search_path = "hydra.searchpath=[file://{},pkg://verl.trainer.config]".format(
        CONFIG_ROOT
    )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        config = compose(config_name=config_name, overrides=[search_path])
    return config


def test_formal_configs_exist():
    names = formal_experiment_configs()
    for algorithm_dir in ALGORITHM_DIRS:
        assert any(name.startswith(algorithm_dir + "/") for name in names)


def _exercise_formal_pipeline(config_name: str) -> None:
    config = compose_formal(config_name)

    configured_sleep = configure_vllm_no_sleep(config)
    algorithm = resolve_algorithm(config)
    algorithm.configure_defaults(config)
    algorithm.configure_batch(config)
    OmegaConf.resolve(config)
    algorithm.validate(config)

    # --- dataloader batch-size regression (server crash 2026-09-20) ---
    # Upstream's schema stores data.gen_batch_size = null; the trainer must
    # treat that exactly like "unset" and fall back to train_batch_size.
    raw_gen = config.data.get("gen_batch_size", None)
    effective_batch_size = raw_gen or config.data.train_batch_size
    assert effective_batch_size is not None and int(effective_batch_size) > 0, (
        f"{config_name}: effective dataloader batch size must be a positive "
        f"int (gen_batch_size={raw_gen!r}, "
        f"train_batch_size={config.data.train_batch_size!r})"
    )

    # --- sleep-mode contract (hard constraint from the project owner) ---
    assert config.actor_rollout_ref.rollout.enable_sleep_mode is False
    assert config.actor_rollout_ref.rollout.free_cache_engine is False
    assert configured_sleep >= 1

    # --- evaluation contract: exactly the three approved datasets ---
    val_datasets = list(config.data.val_datasets)
    assert val_datasets == APPROVED_VAL_DATASETS, (
        f"{config_name}: val_datasets must be {APPROVED_VAL_DATASETS}, "
        f"got {val_datasets}"
    )

    # --- formal runs keep ERSR disabled ---
    if "ersr" in config and "enabled" in config.ersr:
        assert config.ersr.enabled is False, (
            f"{config_name}: formal runs must keep ersr.enabled=false"
        )


def test_formal_pipeline_for_every_experiment():
    for config_name in formal_experiment_configs():
        _exercise_formal_pipeline(config_name)


def test_upstream_schema_actually_injects_gen_batch_size_null():
    """Guard the guard: prove the formal path really merges upstream schema.

    If this ever stops holding, the dataloader regression above loses its
    teeth and should be revisited.
    """
    config = compose_formal("grpo/grpo_qwen3_1p7b_len12k_500steps")
    assert "gen_batch_size" in config.data
    assert config.data.gen_batch_size is None


if __name__ == "__main__":
    names = formal_experiment_configs()
    print(f"exercising {len(names)} formal configs through the full CPU pipeline")
    for name in names:
        _exercise_formal_pipeline(name)
        print(f"  OK  {name}")
    print("ALL FORMAL CONFIGS OK")
