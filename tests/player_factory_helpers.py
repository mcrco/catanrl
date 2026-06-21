"""Helpers for player-factory tests: on-disk experiment stubs (no torch)."""

from __future__ import annotations

from pathlib import Path

import yaml

from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.experiment_store import (
    KIND_POLICY,
    KIND_VALUE,
    CheckpointRegistry,
    ExperimentMetadata,
    GameConfig,
    NetworkSpec,
    save_checkpoint_registry,
    save_metadata,
)
from catanrl.models.backbones import BackboneConfig, MLPBackboneConfig
from catanrl.utils.catanatron_action_space import get_action_space_size


def write_experiment_stub(
    root: Path,
    name: str,
    *,
    observation_level: str = "private",
    num_players: int = 2,
    map_type: str = "MINI",
    include_critic: bool = False,
    hidden_dims: tuple[int, ...] = (8, 8),
) -> Path:
    """Write experiment metadata + checkpoint paths on disk without building networks."""
    exp_path = root / name
    ckpt_dir = exp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    actor_dims = compute_single_agent_dims(
        num_players, map_type, actor_observation_level=observation_level  # type: ignore[arg-type]
    )
    critic_dims = compute_single_agent_dims(
        num_players, map_type, critic_observation_level="full"  # type: ignore[arg-type]
    )
    num_actions = get_action_space_size(num_players, map_type)

    policy_backbone = BackboneConfig(
        architecture="mlp",
        args=MLPBackboneConfig(
            input_dim=actor_dims["actor_dim"],
            hidden_dims=list(hidden_dims),
        ),
    )
    policy_file = "policy_best.pt"
    (ckpt_dir / policy_file).write_bytes(b"")

    networks: dict[str, NetworkSpec] = {
        "policy": NetworkSpec(
            kind=KIND_POLICY,
            backbone=policy_backbone,
            model_type="flat",
            observation_level=observation_level,
            num_actions=num_actions,
        )
    }

    if include_critic:
        critic_backbone = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(
                input_dim=critic_dims["critic_dim"],
                hidden_dims=list(hidden_dims),
            ),
        )
        critic_file = "critic_best.pt"
        (ckpt_dir / critic_file).write_bytes(b"")
        networks["critic"] = NetworkSpec(
            kind=KIND_VALUE,
            backbone=critic_backbone,
            observation_level="full",
        )

    metadata = ExperimentMetadata(
        name=name,
        algorithm="test",
        game=GameConfig(num_players=num_players, map_type=map_type),
        networks=networks,
    )
    save_metadata(str(exp_path), metadata)

    selectors: dict[str, dict[str, str]] = {"best": {"policy": f"checkpoints/{policy_file}"}}
    if include_critic:
        selectors["best"]["critic"] = f"checkpoints/{critic_file}"
    registry = CheckpointRegistry(selectors=selectors)
    save_checkpoint_registry(str(exp_path), registry)
    return exp_path


def write_player_spec_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)
