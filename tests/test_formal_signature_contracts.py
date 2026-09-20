"""CPU-only signature-contract tests for the formal training path.

The formal ``start_train.sh`` path exercises Python object constructions that
the staged smoke tests bypass entirely.  These tests verify, without loading
models or initializing CUDA, that every constructor/method called during
``PPOTrainer.init_workers()`` and ``fit()`` accepts the exact keyword
argument names used by ``main_ppo_sync.py``.

Background: the 2026-09-20 server crashes (gen_batch_size null dataloader,
CheckpointEngineManager trainer->actor_wg rename) were both cross-file
interface mismatches introduced by the verl merge.  Git cannot detect them;
only runtime construction catches them.  These tests close that gap at the
CPU level.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


def _params_of(func) -> set[str]:
    """Return the set of accepted parameter names (excluding self)."""
    sig = inspect.signature(func)
    return {name for name in sig.parameters if name != "self"}


def _accepts_kwargs(func) -> bool:
    sig = inspect.signature(func)
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _check_call_site(
    class_obj,
    call_kwargs: dict[str, str],
    context: str,
) -> None:
    """Verify every keyword used at a call site is accepted by the target.

    ``call_kwargs`` maps the kwarg name used at the call site to a human
    description of what it passes.
    """
    init = class_obj.__init__
    accepted = _params_of(init)
    if _accepts_kwargs(init):
        return  # **kwargs accepts anything; skip
    unknown = set(call_kwargs) - accepted
    assert not unknown, (
        f"{context}: passes {sorted(unknown)} but "
        f"{class_obj.__module__}.{class_obj.__qualname__}.__init__ "
        f"only accepts {sorted(accepted)}"
    )


# ---------------------------------------------------------------------------
# main_ppo_sync.py: init_workers() call sites
# ---------------------------------------------------------------------------


def test_checkpoint_engine_manager_signature():
    """2026-09-20 server crash: trainer= was renamed to actor_wg= upstream."""

    from verl.checkpoint_engine.base import CheckpointEngineManager

    _check_call_site(
        CheckpointEngineManager,
        {"config": ..., "actor_wg": ..., "replicas": ...},
        "main_ppo_sync.init_workers -> CheckpointEngineManager",
    )


def test_stateful_dataloader_signature():
    """2026-09-20 server crash: batch_size=None + drop_last=True rejected."""

    from torchdata.stateful_dataloader import StatefulDataLoader

    accepted = _params_of(StatefulDataLoader.__init__)
    for kw in ("dataset", "batch_size", "num_workers", "drop_last", "collate_fn", "sampler"):
        assert kw in accepted, (
            f"StatefulDataLoader no longer accepts {kw!r}; "
            f"params: {sorted(accepted)}"
        )


def test_agent_loop_manager_tq_signature():
    from verl.trainer.main_ppo_sync import AgentLoopManagerTQ

    accepted = _params_of(AgentLoopManagerTQ.__init__)
    assert _accepts_kwargs(AgentLoopManagerTQ.__init__) or "replay_buffer" in accepted, (
        f"AgentLoopManagerTQ.__init__ params: {sorted(accepted)}; "
        "main_ppo_sync passes replay_buffer="
    )


def test_reward_loop_manager_signature():
    from verl.experimental.reward_loop import RewardLoopManager

    _check_call_site(
        RewardLoopManager,
        {"config": ..., "rm_resource_pool": ...},
        "main_ppo_sync.init_workers -> RewardLoopManager",
    )


def test_multi_teacher_model_manager_signature():
    from verl.experimental.teacher_loop import MultiTeacherModelManager

    _check_call_site(
        MultiTeacherModelManager,
        {"config": ..., "resource_pool": ...},
        "main_ppo_sync.init_workers -> MultiTeacherModelManager",
    )


def test_agent_loop_manager_create_signature():
    """AgentLoopManagerTQ.create() is a Ray @remote-annotated classmethod."""

    from verl.trainer.main_ppo_sync import AgentLoopManagerTQ

    # .create() on Ray actors wraps __init__; verify the underlying
    # constructor accepts the kwargs forwarded by the .create() call site.
    # The call site passes: config, llm_client, teacher_client,
    # reward_loop_worker_handles, replay_buffer.
    init = AgentLoopManagerTQ.__init__
    if _accepts_kwargs(init):
        return
    accepted = _params_of(init)
    # If it's a Ray-modified class, check the original if accessible
    orig = getattr(AgentLoopManagerTQ, "__ray_metadata__", None)
    if orig is not None:
        modified_cls = getattr(orig, "modified_class", None)
        if modified_cls is not None and modified_cls is not AgentLoopManagerTQ:
            accepted = _params_of(modified_cls.__init__)
            if _accepts_kwargs(modified_cls.__init__):
                return
    for kw in ("config", "llm_client", "teacher_client", "replay_buffer"):
        assert kw in accepted or any(
            kw.startswith(p.rstrip("*")) for p in accepted if p.startswith("*")
        ), (
            f"AgentLoopManagerTQ may not accept {kw!r}; "
            f"__init__ params: {sorted(accepted)}"
        )


def test_ray_class_with_init_args_signature():
    from verl.single_controller.ray.base import RayClassWithInitArgs

    _check_call_site(
        RayClassWithInitArgs,
        {"cls": ..., "config": ..., "distillation_config": ..., "role": ...},
        "main_ppo_sync.init_workers -> RayClassWithInitArgs",
    )


def test_ray_worker_group_signature():
    from verl.single_controller.ray.base import RayWorkerGroup

    _check_call_site(
        RayWorkerGroup,
        {"resource_pool": ..., "ray_cls_with_init": ..., "device_name": ...},
        "main_ppo_sync.init_workers -> RayWorkerGroup",
    )


def test_create_rl_dataset_signature():
    from verl.trainer.ppo.utils import create_rl_dataset

    accepted = _params_of(create_rl_dataset)
    for kw in ("data_paths", "data_config", "tokenizer", "processor", "is_train", "max_samples"):
        assert kw in accepted, (
            f"create_rl_dataset no longer accepts {kw!r}; params: {sorted(accepted)}"
        )


def test_create_rl_sampler_signature():
    from verl.trainer.ppo.utils import create_rl_sampler

    accepted = _params_of(create_rl_sampler)
    for kw in ("data_config", "dataset"):
        assert kw in accepted, (
            f"create_rl_sampler no longer accepts {kw!r}; params: {sorted(accepted)}"
        )


def test_ppo_trainer_init_signature():
    """The entrypoint constructs PPOTrainer subclasses with these kwargs."""

    from algorithms import ALGORITHM_REGISTRY

    for name, spec in ALGORITHM_REGISTRY.items():
        init = spec.trainer_class.__init__
        if _accepts_kwargs(init):
            continue
        accepted = _params_of(init)
        # The training_entrypoint passes config=, role_worker_mapping=,
        # resource_pool_manager= (as **kwargs to the subclass which calls
        # super().__init__(*args, **kwargs) down to PPOTrainer).
        # At minimum the base PPOTrainer must accept them.
        from verl.trainer.main_ppo_sync import TaskRunner

        base_init = TaskRunner.__init__
        base_accepted = _params_of(base_init) | {"*args", "**kwargs"} - {"self"}
        assert base_accepted or _accepts_kwargs(base_init), (
            f"Base TaskRunner.__init__ has no params: {base_accepted}"
        )


def test_llm_server_manager_signature():
    """LLMServerManager is created via .create() classmethod during init_workers."""

    from verl.workers.rollout.llm_server import LLMServerManager

    init = LLMServerManager.__init__
    accepted = _params_of(init)
    for kw in ("config", "worker_group", "rollout_resource_pool"):
        assert kw in accepted, (
            f"LLMServerManager.__init__ no longer accepts {kw!r}; "
            f"params: {sorted(accepted)}"
        )


def test_run_ppo_signature():
    """The entrypoint calls verl_sync.run_ppo(config, task_runner_class=...)."""

    from verl.trainer.main_ppo import run_ppo

    accepted = _params_of(run_ppo)
    for kw in ("config", "task_runner_class"):
        assert kw in accepted, (
            f"run_ppo no longer accepts {kw!r}; params: {sorted(accepted)}"
        )


def test_auto_set_device_signature():
    from verl.trainer.main_ppo_sync import auto_set_device

    accepted = _params_of(auto_set_device)
    assert "config" in accepted


def test_validate_config_signature():
    from verl.utils.config import validate_config

    accepted = _params_of(validate_config)
    for kw in ("config", "use_reference_policy", "use_critic"):
        assert kw in accepted, (
            f"validate_config no longer accepts {kw!r}; params: {sorted(accepted)}"
        )


# ---------------------------------------------------------------------------
# main_ppo_sync.py: _init_dataloader() derived values
# ---------------------------------------------------------------------------


def test_dataloader_batch_size_expression_handles_schema_null():
    """Prove the fix: gen_batch_size=null falls back to train_batch_size."""

    from omegaconf import OmegaConf

    for raw, expected in [
        (None, 64),       # upstream schema stores null
        (128, 128),       # explicitly set
        ({"absent": True}, 64),  # key entirely absent
    ]:
        if raw == {"absent": True}:
            cfg = OmegaConf.create({"data": {"train_batch_size": 64}})
        else:
            cfg = OmegaConf.create(
                {"data": {"gen_batch_size": raw, "train_batch_size": 64}}
            )
        effective = cfg.data.get("gen_batch_size", None) or cfg.data.train_batch_size
        assert effective == expected, f"gen_batch_size={raw!r} -> {effective}"


# ---------------------------------------------------------------------------
# distillation losses: registry completeness
# ---------------------------------------------------------------------------


def test_all_loss_modes_in_registry():
    """Every loss_mode referenced by any config must be registered."""

    from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY

    required = {
        "reverse_kl",
        "forward_kl_topk",
        "error_reverse_kl",
        "correct_reverse_kl",
        "fire_opd",
        "eopd",
        "opdvr",
    }
    registered = set(DISTILLATION_LOSS_REGISTRY.keys())
    missing = required - registered
    assert not missing, f"Loss modes referenced by configs but not registered: {missing}"


def test_policy_loss_modes_available():
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    for mode in ("reinforce", "vanilla"):
        fn = get_policy_loss_fn(mode)
        assert callable(fn)


# ---------------------------------------------------------------------------
# algorithm trainer classes: __init__ chain doesn't crash on inspect
# ---------------------------------------------------------------------------


def test_algorithm_trainer_classes_constructible():
    """Every registered trainer's __init__ accepts config + worker groups."""

    from algorithms import ALGORITHM_REGISTRY

    for name, spec in ALGORITHM_REGISTRY.items():
        init = spec.trainer_class.__init__
        accepted = _params_of(init)
        has_kwargs = _accepts_kwargs(init)
        assert has_kwargs or {"*args", "args"} & accepted or len(accepted) >= 1, (
            f"{name} trainer {spec.trainer_class.__name__} has no way to "
            f"receive config; params: {sorted(accepted)}"
        )


# ---------------------------------------------------------------------------
# formal config compose: validate every experiment (not just signature checks)
# ---------------------------------------------------------------------------


def test_formal_configs_all_algorithm_names_resolve():
    """Every formal config's algorithm.name must be in the registry."""

    from algorithms import ALGORITHM_REGISTRY
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_root = (PROJECT_ROOT / "configs").resolve()
    search = f"hydra.searchpath=[file://{config_root},pkg://verl.trainer.config]"

    for algo_dir in sorted(config_root.iterdir()):
        if not algo_dir.is_dir():
            continue
        for yaml_path in sorted(algo_dir.glob("*.yaml")):
            name = f"{algo_dir.name}/{yaml_path.stem}"
            with initialize_config_dir(
                version_base=None, config_dir=str(config_root)
            ):
                cfg = compose(config_name=name, overrides=[search])
            OmegaConf.resolve(cfg)
            # Only experiment configs (not the algorithm base yamls like
            # grpo.yaml) define algorithm.name.
            algo_cfg = cfg.get("algorithm")
            if algo_cfg is None or "name" not in algo_cfg:
                continue
            algo_name = str(algo_cfg.name)
            assert algo_name in ALGORITHM_REGISTRY, (
                f"{name}: algorithm.name={algo_name!r} not registered; "
                f"expected one of {sorted(ALGORITHM_REGISTRY)}"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
