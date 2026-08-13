"""Preflight guards for reproducible Canopy-parity experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

from catanrl.experiment_store import Experiment
from catanrl.experiments.architecture_config import ArchitecturePreset
from catanrl.models.backbones import CrossDimensionalBackboneConfig


class CanopyContractError(ValueError):
    """Raised when a parity run would change the agreed Catan contract."""


def _raise_if_violations(violations: Sequence[str]) -> None:
    if violations:
        details = "\n  - ".join(violations)
        raise CanopyContractError(f"Canopy parity contract violations:\n  - {details}")


def validate_canopy_architecture(arch: ArchitecturePreset) -> None:
    """Require the architecture/game contract used by fresh parity training."""
    violations: list[str] = []
    expected = {
        "game.map_type": (arch.map_type, "BASE"),
        "game.num_players": (arch.num_players, 2),
        "game.vps_to_win": (arch.vps_to_win, 15),
        "game.discard_limit": (arch.discard_limit, 9),
        "model.policy_mode": (arch.policy_mode, "full"),
        "model.critic_mode": (arch.critic_mode, "full"),
    }
    for field, (actual, required) in expected.items():
        if actual != required:
            violations.append(f"{field}={actual!r}; required {required!r}")
    _raise_if_violations(violations)


def validate_canopy_experiment(
    experiment: Experiment,
    *,
    require_terminal_dagger: bool = False,
) -> None:
    """Verify a warm-start/evaluation experiment from authoritative metadata."""
    metadata = experiment.metadata
    game = metadata.game
    policy = experiment.policy_spec
    violations: list[str] = []

    expected_game = {
        "game.map_type": (game.map_type, "BASE"),
        "game.num_players": (game.num_players, 2),
        "game.vps_to_win": (game.vps_to_win, 15),
        "game.discard_limit": (game.discard_limit, 9),
    }
    for field, (actual, required) in expected_game.items():
        if actual != required:
            violations.append(f"{field}={actual!r}; required {required!r}")

    if policy.observation_level != "full":
        violations.append(f"policy.observation_level={policy.observation_level!r}; required 'full'")
    critic = metadata.networks.get("critic")
    if critic is not None and critic.observation_level != "full":
        violations.append(f"critic.observation_level={critic.observation_level!r}; required 'full'")
    for role, network in metadata.networks.items():
        backbone_args = network.backbone.args
        if (
            isinstance(backbone_args, CrossDimensionalBackboneConfig)
            and backbone_args.board_layout != "width_height"
        ):
            violations.append(
                f"{role}.backbone.board_layout={backbone_args.board_layout!r}; "
                "required 'width_height'"
            )

    if require_terminal_dagger:
        train_config = metadata.train_config
        if metadata.algorithm != "dagger":
            violations.append(f"algorithm={metadata.algorithm!r}; required 'dagger'")
        if train_config.get("env_backend") != "cppanatron":
            violations.append(
                f"train_config.env_backend={train_config.get('env_backend')!r}; "
                "required 'cppanatron'"
            )
        if train_config.get("reward_function") != "win":
            violations.append(
                f"train_config.reward_function={train_config.get('reward_function')!r}; "
                "required 'win'"
            )
        gamma = train_config.get("gamma")
        if not isinstance(gamma, (int, float)) or not math.isclose(float(gamma), 1.0):
            violations.append(f"train_config.gamma={gamma!r}; required 1.0")

    _raise_if_violations(violations)


def canopy_checkpoint_global_iteration(
    experiment: Experiment,
    checkpoint_step: int,
) -> int:
    """Map a branch-local checkpoint number to its global self-play iteration."""

    if isinstance(checkpoint_step, bool) or checkpoint_step < 0:
        raise ValueError("checkpoint_step must be a non-negative integer")
    raw_offset = experiment.metadata.train_config.get("self_play_iteration_offset", 0)
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, int) or raw_offset < 0:
        raise CanopyContractError(
            "train_config.self_play_iteration_offset must be a non-negative integer"
        )
    return raw_offset + checkpoint_step


__all__ = [
    "CanopyContractError",
    "canopy_checkpoint_global_iteration",
    "validate_canopy_architecture",
    "validate_canopy_experiment",
]
