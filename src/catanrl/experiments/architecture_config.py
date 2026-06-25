"""Load model architecture presets from YAML files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml

ObservationLevel = Literal["private", "public", "full"]
NetworkMode = Literal["shared", "separate"]
BackboneType = Literal["mlp", "xdim", "xdim_res"]
MapType = Literal["BASE", "MINI", "TOURNAMENT"]

OBSERVATION_LEVEL_CHOICES = ("private", "public", "full")
NETWORK_MODE_CHOICES = ("shared", "separate")
BACKBONE_TYPE_CHOICES = ("mlp", "xdim", "xdim_res")
MAP_TYPE_CHOICES = ("BASE", "MINI", "TOURNAMENT")
MODEL_TYPE_CHOICES = ("flat", "hierarchical")


@dataclass(frozen=True)
class ArchitecturePreset:
    model_type: str
    backbone_type: str
    policy_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    policy_mode: str
    critic_mode: str
    network_mode: str
    xdim_cnn_channels: tuple[int, ...]
    xdim_cnn_kernel_size: tuple[int, int]
    xdim_policy_fusion_hidden_dim: int | None
    xdim_critic_fusion_hidden_dim: int | None
    map_type: str
    vps_to_win: int
    discard_limit: int
    num_players: int | None = None

    @property
    def actor_observation_level(self) -> str:
        return self.policy_mode

    @property
    def critic_observation_level(self) -> str:
        return self.critic_mode


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a model architecture YAML preset under configs/models/. "
            "Required for a fresh run; optional when warm-starting with "
            "--load-from-experiment (architecture is read from experiment metadata). "
            "When both are set, the preset must match the source experiment."
        ),
    )


def _require_mapping(data: object, *, field: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Expected '{field}' to be a mapping.")
    return data


def _parse_int_list(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{field}' must be a non-empty list of integers.")
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int) or item <= 0:
            raise ValueError(f"'{field}' must contain positive integers.")
        parsed.append(item)
    return tuple(parsed)


def _parse_optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{field}' must be a positive integer when set.")
    return value


def _parse_choice(value: object, *, field: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"'{field}' must be one of: {allowed}.")
    return value


def validate_architecture_source(
    config_path: str | None,
    load_from_experiment: str | None,
) -> None:
    """Require at least one architecture source for training."""
    if not config_path and not load_from_experiment:
        raise ValueError(
            "Provide --config (architecture preset) or --load-from-experiment (warm start)."
        )


def load_architecture_preset(path: str | Path) -> ArchitecturePreset:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Architecture preset not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Architecture preset must be a YAML mapping: {config_path}")

    model = _require_mapping(raw.get("model"), field="model")
    game = _require_mapping(raw.get("game"), field="game")

    kernel = _parse_int_list(model.get("xdim_cnn_kernel_size"), field="model.xdim_cnn_kernel_size")
    if len(kernel) != 2:
        raise ValueError("model.xdim_cnn_kernel_size must contain exactly two integers.")

    num_players = game.get("num_players")
    if num_players is not None and (not isinstance(num_players, int) or num_players < 2):
        raise ValueError("game.num_players must be an integer >= 2 when set.")

    return ArchitecturePreset(
        model_type=_parse_choice(
            model.get("model_type"), field="model.model_type", choices=MODEL_TYPE_CHOICES
        ),
        backbone_type=_parse_choice(
            model.get("backbone_type"),
            field="model.backbone_type",
            choices=BACKBONE_TYPE_CHOICES,
        ),
        policy_hidden_dims=_parse_int_list(
            model.get("policy_hidden_dims"), field="model.policy_hidden_dims"
        ),
        critic_hidden_dims=_parse_int_list(
            model.get("critic_hidden_dims"), field="model.critic_hidden_dims"
        ),
        policy_mode=_parse_choice(
            model.get("policy_mode"), field="model.policy_mode", choices=OBSERVATION_LEVEL_CHOICES
        ),
        critic_mode=_parse_choice(
            model.get("critic_mode"), field="model.critic_mode", choices=OBSERVATION_LEVEL_CHOICES
        ),
        network_mode=_parse_choice(
            model.get("network_mode"), field="model.network_mode", choices=NETWORK_MODE_CHOICES
        ),
        xdim_cnn_channels=_parse_int_list(
            model.get("xdim_cnn_channels"), field="model.xdim_cnn_channels"
        ),
        xdim_cnn_kernel_size=(kernel[0], kernel[1]),
        xdim_policy_fusion_hidden_dim=_parse_optional_int(
            model.get("xdim_policy_fusion_hidden_dim"),
            field="model.xdim_policy_fusion_hidden_dim",
        ),
        xdim_critic_fusion_hidden_dim=_parse_optional_int(
            model.get("xdim_critic_fusion_hidden_dim"),
            field="model.xdim_critic_fusion_hidden_dim",
        ),
        map_type=_parse_choice(game.get("map_type"), field="game.map_type", choices=MAP_TYPE_CHOICES),
        vps_to_win=_parse_optional_int(game.get("vps_to_win"), field="game.vps_to_win") or 15,
        discard_limit=_parse_optional_int(game.get("discard_limit"), field="game.discard_limit") or 9,
        num_players=num_players,
    )


def architecture_train_config_fields(arch: ArchitecturePreset) -> dict[str, Any]:
    """Expand a preset into train_config keys (never includes the YAML path)."""
    fields: dict[str, Any] = {
        "model_type": arch.model_type,
        "backbone_type": arch.backbone_type,
        "policy_hidden_dims": list(arch.policy_hidden_dims),
        "critic_hidden_dims": list(arch.critic_hidden_dims),
        "policy_mode": arch.policy_mode,
        "critic_mode": arch.critic_mode,
        "network_mode": arch.network_mode,
        "actor_observation_level": arch.actor_observation_level,
        "critic_observation_level": arch.critic_observation_level,
        "xdim_cnn_channels": list(arch.xdim_cnn_channels),
        "xdim_cnn_kernel_size": list(arch.xdim_cnn_kernel_size),
        "xdim_policy_fusion_hidden_dim": arch.xdim_policy_fusion_hidden_dim,
        "xdim_critic_fusion_hidden_dim": arch.xdim_critic_fusion_hidden_dim,
        "map_type": arch.map_type,
        "vps_to_win": arch.vps_to_win,
        "discard_limit": arch.discard_limit,
    }
    if arch.num_players is not None:
        fields["num_players"] = arch.num_players
    return fields


def validate_player_count(arch: ArchitecturePreset, num_players: int) -> None:
    if arch.num_players is not None and arch.num_players != num_players:
        raise ValueError(
            f"Architecture preset expects game.num_players={arch.num_players}, "
            f"but this run uses {num_players} players."
        )
