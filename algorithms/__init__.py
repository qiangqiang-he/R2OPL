"""Algorithm registry for R2OPL's OPD and RL methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from algorithms.grpo import (
    GRPOTrainer,
    configure_grpo_batch,
    configure_grpo_defaults,
    validate_grpo_config,
)
from algorithms.pg_opd import PGOPDTrainer, validate_pg_opd_config
from utils.opd_runtime import configure_pg_opd_batch, configure_pg_opd_defaults


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    trainer_class: Type
    validate: Callable
    configure_defaults: Callable
    configure_batch: Callable


ALGORITHM_REGISTRY = {
    "pg_opd": AlgorithmSpec(
        name="pg_opd",
        trainer_class=PGOPDTrainer,
        validate=validate_pg_opd_config,
        configure_defaults=configure_pg_opd_defaults,
        configure_batch=configure_pg_opd_batch,
    ),
    "grpo": AlgorithmSpec(
        name="grpo",
        trainer_class=GRPOTrainer,
        validate=validate_grpo_config,
        configure_defaults=configure_grpo_defaults,
        configure_batch=configure_grpo_batch,
    ),
}


def resolve_algorithm(config) -> AlgorithmSpec:
    name = str(config.algorithm.name)
    try:
        return ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algorithm.name={name!r}; expected one of: "
            f"{', '.join(sorted(ALGORITHM_REGISTRY))}"
        ) from exc


__all__ = ["ALGORITHM_REGISTRY", "AlgorithmSpec", "resolve_algorithm"]
