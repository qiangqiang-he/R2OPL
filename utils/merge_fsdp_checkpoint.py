#!/usr/bin/env python3
"""Merge a VERL FSDP actor checkpoint into a standard Hugging Face model.

VERL stores one ``model_world_size_<N>_rank_<R>.pt`` file per FSDP rank.
This command validates that all model shards are present and delegates the
actual DTensor/FSDP reconstruction to VERL's model merger.  The output is a
normal Transformers model directory (model weights, config, generation config
and tokenizer/processor files), suitable for ``from_pretrained``.

Examples:

    # ``--source-ckpt`` may be the global-step directory ...
    python utils/merge_fsdp_checkpoint.py \\
        --source-ckpt outputs/run/checkpoints/global_step_500 \\
        --output-dir outputs/run/merged_global_step_500

    # ... or its actor subdirectory directly.
    python utils/merge_fsdp_checkpoint.py \\
        --source-ckpt outputs/run/checkpoints/global_step_500/actor \\
        --output-dir /path/to/merged_model

Use ``--dry-run`` to validate a checkpoint before allocating memory or writing
the merged model.  The destination must not exist: this prevents accidentally
mixing old output files with a newly exported model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_MODEL_SHARD_RE = re.compile(r"model_world_size_(?P<world_size>\d+)_rank_(?P<rank>\d+)\.pt$")


@dataclass(frozen=True)
class FSDPCheckpoint:
    """Validated location and layout metadata for one actor checkpoint."""

    actor_dir: Path
    world_size: int
    shard_paths: tuple[Path, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_vendored_verl_to_import_path() -> None:
    """Make ``python utils/merge_fsdp_checkpoint.py`` work from any CWD."""

    verl_root = _repo_root() / "verl"
    if not verl_root.is_dir():
        raise RuntimeError(f"Vendored VERL directory is missing: {verl_root}")
    verl_root_text = str(verl_root)
    if verl_root_text not in sys.path:
        sys.path.insert(0, verl_root_text)


def resolve_actor_checkpoint(source_ckpt: str | Path) -> Path:
    """Resolve either a global-step directory or its ``actor`` subdirectory."""

    source = Path(source_ckpt).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {source}")

    actor_dir = source / "actor" if (source / "actor").is_dir() else source
    if not actor_dir.is_dir():  # Defensive, retained for a clearer future error.
        raise FileNotFoundError(f"Actor checkpoint directory does not exist: {actor_dir}")
    return actor_dir


def inspect_fsdp_checkpoint(source_ckpt: str | Path) -> FSDPCheckpoint:
    """Verify a complete VERL FSDP actor checkpoint and return its shard layout."""

    actor_dir = resolve_actor_checkpoint(source_ckpt)
    matches: list[tuple[int, int, Path]] = []
    for path in actor_dir.iterdir():
        match = _MODEL_SHARD_RE.fullmatch(path.name)
        if match is not None and path.is_file():
            matches.append((int(match["world_size"]), int(match["rank"]), path))

    if not matches:
        raise FileNotFoundError(
            "No VERL FSDP model shards found in "
            f"{actor_dir}; expected model_world_size_<N>_rank_<R>.pt files."
        )

    world_sizes = {world_size for world_size, _, _ in matches}
    if len(world_sizes) != 1:
        raise ValueError(
            f"Checkpoint has mixed FSDP world sizes {sorted(world_sizes)} in {actor_dir}."
        )
    world_size = world_sizes.pop()
    if world_size < 1:
        raise ValueError(f"Invalid FSDP world size {world_size} in {actor_dir}.")

    by_rank = {rank: path for _, rank, path in matches}
    if len(by_rank) != len(matches):
        raise ValueError(f"Checkpoint contains duplicate FSDP ranks in {actor_dir}.")
    expected_ranks = set(range(world_size))
    actual_ranks = set(by_rank)
    if actual_ranks != expected_ranks:
        raise FileNotFoundError(
            f"Incomplete FSDP checkpoint in {actor_dir}: expected ranks "
            f"0..{world_size - 1}, found {sorted(actual_ranks)}."
        )

    fsdp_config_path = actor_dir / "fsdp_config.json"
    if not fsdp_config_path.is_file():
        raise FileNotFoundError(f"Missing FSDP metadata file: {fsdp_config_path}")
    try:
        fsdp_config = json.loads(fsdp_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid FSDP metadata file {fsdp_config_path}: {exc}") from exc
    configured_world_size = fsdp_config.get("world_size")
    if configured_world_size != world_size:
        raise ValueError(
            f"FSDP metadata says world_size={configured_world_size!r}, but shard names "
            f"say world_size={world_size} in {actor_dir}."
        )

    hf_config_path = actor_dir / "huggingface" / "config.json"
    if not hf_config_path.is_file():
        raise FileNotFoundError(
            f"Missing Hugging Face config required for export: {hf_config_path}"
        )

    return FSDPCheckpoint(
        actor_dir=actor_dir,
        world_size=world_size,
        shard_paths=tuple(by_rank[rank] for rank in range(world_size)),
    )


def merge_fsdp_checkpoint(
    checkpoint: FSDPCheckpoint,
    output_dir: str | Path,
    *,
    trust_remote_code: bool = False,
) -> None:
    """Merge ``checkpoint`` and export it as a complete Hugging Face directory."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Output directory already exists: {destination}. Choose a new path to avoid "
            "mixing old weights with this export."
        )
    if destination.parent.exists() and not destination.parent.is_dir():
        raise NotADirectoryError(f"Output parent is not a directory: {destination.parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    _add_vendored_verl_to_import_path()
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger
    from verl.model_merger.output_validation import validate_hf_model_output

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(checkpoint.actor_dir),
        hf_model_config_path=str(checkpoint.actor_dir / "huggingface"),
        target_dir=str(destination),
        trust_remote_code=trust_remote_code,
    )
    merger = FSDPModelMerger(config)
    try:
        merger.merge_and_save()
        validate_hf_model_output(destination)
    finally:
        merger.cleanup()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a VERL FSDP actor checkpoint into Hugging Face format."
    )
    parser.add_argument(
        "--source-ckpt",
        required=True,
        help="Path to global_step_<N>/ or global_step_<N>/actor/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for the merged standard Hugging Face model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow a model configuration that requires Transformers remote code.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source checkpoint and destination without merging or writing files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = inspect_fsdp_checkpoint(args.source_ckpt)
    destination = Path(args.output_dir).expanduser().resolve()

    if destination.exists():
        raise FileExistsError(
            f"Output directory already exists: {destination}. Choose a new output path."
        )
    if args.dry_run:
        print(
            "Validated FSDP checkpoint: "
            f"actor_dir={checkpoint.actor_dir}, world_size={checkpoint.world_size}, "
            f"shards={len(checkpoint.shard_paths)}. Would export to {destination}."
        )
        return 0

    merge_fsdp_checkpoint(
        checkpoint,
        destination,
        trust_remote_code=bool(args.trust_remote_code),
    )
    print(
        f"Merged {checkpoint.world_size} FSDP shards from {checkpoint.actor_dir} "
        f"into standard Hugging Face model directory: {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
