"""Hydra/Ray entrypoint shared by R2OPL algorithms."""

from __future__ import annotations

import hydra
import ray

from algorithms import resolve_algorithm
from verl.trainer import main_ppo_sync as verl_sync


_BaseTaskRunner = verl_sync.TaskRunner.__ray_metadata__.modified_class


def configure_vllm_no_sleep(config) -> int:
    """Disable unsupported vLLM sleep paths for every configured engine.

    Teacher configuration is deliberately optional so RL-only algorithms can
    use the same entrypoint without defining ``distillation`` or
    ``teacher_models``.
    """

    configured = 0

    def disable(engine_config) -> None:
        nonlocal configured
        if engine_config is None:
            return
        if "enable_sleep_mode" in engine_config:
            engine_config["enable_sleep_mode"] = False
        if "free_cache_engine" in engine_config:
            # VERL gates sleep()/wake_up() calls with this field.
            engine_config["free_cache_engine"] = False
        configured += 1

    actor_rollout_ref = config.get("actor_rollout_ref")
    if actor_rollout_ref is not None:
        disable(actor_rollout_ref.get("rollout"))

    distillation = config.get("distillation")
    teacher_models = (
        None if distillation is None else distillation.get("teacher_models")
    )
    if teacher_models is not None:
        for teacher_config in teacher_models.values():
            if teacher_config is not None:
                disable(teacher_config.get("inference"))
    return configured


@ray.remote
class R2OPLTaskRunner(_BaseTaskRunner):
    """Build only the resource pools required by the selected algorithm."""

    def run(self, config):
        configure_vllm_no_sleep(config)
        algorithm = resolve_algorithm(config)
        algorithm.configure_defaults(config)
        algorithm.configure_batch(config)
        verl_sync.pprint(verl_sync.OmegaConf.to_container(config, resolve=True))
        verl_sync.OmegaConf.resolve(config)
        algorithm.validate(config)

        verl_sync.tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)
            if verl_sync.need_teacher_policy(config):
                self.resource_pool_manager.max_colocate_count = 2
            trainer = algorithm.trainer_class(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            replay_buffer = getattr(trainer, "replay_buffer", None)
            if replay_buffer is not None:
                replay_buffer.close()
            verl_sync.tq.close()


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    configure_vllm_no_sleep(config)
    algorithm = resolve_algorithm(config)
    algorithm.configure_defaults(config)
    algorithm.configure_batch(config)
    algorithm.validate(config)
    verl_sync.auto_set_device(config)
    config.transfer_queue.enable = True
    verl_sync.validate_config(
        config=config,
        use_reference_policy=verl_sync.need_reference_policy(config),
        use_critic=verl_sync.need_critic(config),
    )
    verl_sync.run_ppo(config, task_runner_class=R2OPLTaskRunner)


# Compatibility alias for the existing PG-OPD CPU contracts.
PGOPDTaskRunner = R2OPLTaskRunner


if __name__ == "__main__":
    main()
