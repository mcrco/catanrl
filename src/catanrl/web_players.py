from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import yaml
from catanatron.models.player import Color

from catanrl.algorithms.common.backbone_builder import build_backbone_config
from catanrl.envs.puffer.common import BOARD_HEIGHT, BOARD_WIDTH, compute_single_agent_dims
from catanrl.models.models import build_flat_policy_network, build_hierarchical_policy_network
from catanrl.players.nn_policy_player import NNPolicyPlayer
from catanrl.utils.catanatron_action_space import get_action_space_size


DEFAULT_RUN_NAME = "marl-cc-f-xdim-flat-shaped-pretrained-dagger"
DEFAULT_WEIGHTS_RELATIVE_PATH = f"weights/{DEFAULT_RUN_NAME}/policy_best.pt"
DEFAULT_WANDB_CONFIG_RELATIVE_PATH = "wandb/latest-run/files/config.yaml"


def register_players(registry) -> None:
    """Register locally trained CatanRL players with the Catanatron web UI."""

    repo_root = _repo_root()
    config_path = Path(
        os.environ.get(
            "CATANRL_WEB_PLAYER_CONFIG",
            str(repo_root / DEFAULT_WANDB_CONFIG_RELATIVE_PATH),
        )
    )
    config = _load_wandb_config(config_path)
    weights_path = Path(
        os.environ.get(
            "CATANRL_WEB_PLAYER_WEIGHTS",
            _config_value(config, "save_path", repo_root / DEFAULT_WEIGHTS_RELATIVE_PATH),
        )
    )
    if not weights_path.is_absolute():
        weights_path = repo_root / weights_path

    map_type = str(_config_value(config, "map_type", "BASE"))
    num_players = int(_config_value(config, "num_players", 2))
    model_type = str(_config_value(config, "model_type", "flat"))

    registry.register(
        key="CATANRL_LATEST_NN",
        label="CatanRL Latest NN",
        description=(
            f"Latest local CatanRL policy from {_display_path(weights_path, repo_root)} "
            f"({num_players}p {map_type}, {model_type})."
        ),
        min_players=num_players,
        max_players=num_players,
        map_templates=(map_type,),
        factory=lambda color, context: _create_policy_player(
            color=color,
            config_path=str(config_path),
            weights_path=str(weights_path),
            map_type=context.map_template,
            num_players=context.num_players,
        ),
    )
    logging.info("Registered CatanRL web player from %s", weights_path)


def _repo_root() -> Path:
    return Path(os.environ.get("CATANRL_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_wandb_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    config: dict[str, Any] = {}
    for key, value in raw_config.items():
        if isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
        else:
            config[key] = value

    save_path = _wandb_arg_value(raw_config, "--save-path")
    if save_path is not None and "save_path" not in config:
        config["save_path"] = str(Path(save_path) / "policy_best.pt")

    return config


def _wandb_arg_value(raw_config: dict[str, Any], flag: str) -> str | None:
    wandb_config = raw_config.get("_wandb", {})
    if isinstance(wandb_config, dict) and "value" in wandb_config:
        wandb_config = wandb_config["value"]
    entries = wandb_config.get("e", {}) if isinstance(wandb_config, dict) else {}
    if not isinstance(entries, dict):
        return None

    for entry in entries.values():
        args = entry.get("args", []) if isinstance(entry, dict) else []
        if flag in args:
            index = args.index(flag)
            if index + 1 < len(args):
                return str(args[index + 1])
    return None


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _int_tuple(values: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    if isinstance(values, str):
        return tuple(int(part.strip()) for part in values.split(",") if part.strip())
    return tuple(int(value) for value in values)


def _create_policy_player(
    *,
    color: Color,
    config_path: str,
    weights_path: str,
    map_type: str,
    num_players: int,
) -> NNPolicyPlayer:
    model, model_type = _load_policy_model(config_path, weights_path, map_type, num_players)
    return NNPolicyPlayer(
        color=color,
        model_type=model_type,
        model=model,
        map_type=map_type,
    )


@lru_cache(maxsize=8)
def _load_policy_model(
    config_path: str,
    weights_path: str,
    map_type: str,
    num_players: int,
):
    config = _load_wandb_config(Path(config_path))
    expected_map = str(_config_value(config, "map_type", map_type))
    expected_players = int(_config_value(config, "num_players", num_players))
    if map_type != expected_map or num_players != expected_players:
        raise ValueError(
            f"CatanRL model expects {expected_players} players on {expected_map}, "
            f"got {num_players} players on {map_type}."
        )

    model_type = str(_config_value(config, "model_type", "flat"))
    backbone_type = str(_config_value(config, "backbone_type", "mlp"))
    hidden_dims = _int_tuple(_config_value(config, "policy_hidden_dims", (256, 256)), (256, 256))
    xdim_cnn_channels = _int_tuple(
        _config_value(config, "xdim_cnn_channels", (64, 128, 128)),
        (64, 128, 128),
    )
    xdim_cnn_kernel_size = _int_tuple(
        _config_value(config, "xdim_cnn_kernel_size", (3, 5)),
        (3, 5),
    )
    xdim_fusion_hidden_dim = _config_value(config, "xdim_policy_fusion_hidden_dim", None)

    dims = compute_single_agent_dims(num_players, map_type)
    backbone_config = build_backbone_config(
        backbone_type=backbone_type,
        hidden_dims=hidden_dims,
        input_dim=dims["actor_dim"],
        board_height=BOARD_HEIGHT,
        board_width=BOARD_WIDTH,
        board_channels=dims["board_channels"],
        numeric_dim=dims["numeric_dim"],
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,  # type: ignore[arg-type]
        xdim_fusion_hidden_dim=xdim_fusion_hidden_dim,
    )

    if model_type == "flat":
        model = build_flat_policy_network(
            backbone_config=backbone_config,
            num_actions=get_action_space_size(num_players, map_type),
        )
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_network(
            backbone_config=backbone_config,
            num_players=num_players,
            map_type=map_type,
        )
    else:
        raise ValueError(f"Unsupported CatanRL model_type '{model_type}'.")

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("Loaded CatanRL policy model from %s", weights_path)
    return model, model_type
