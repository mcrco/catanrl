"""YAML player agent specs: how to use experiment checkpoints at inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from catanrl.experiment_store import list_experiments, load_experiment

PlayerType = Literal["policy", "mcts", "belief"]
PLAYER_TYPE_CHOICES = ("policy", "mcts", "belief")


@dataclass(frozen=True)
class MCTSPlayerConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    ismcts_determinizations: int = 1
    num_search_workers: int = 1
    adversarial_policy: str = "self"
    prunning: bool = False
    inference_batch_size: int = 32
    inference_wait_ms: float = 1.0
    virtual_loss: float = 1.0


@dataclass(frozen=True)
class BeliefPlayerConfig:
    num_samples: int = 100
    sample: bool = False


@dataclass(frozen=True)
class PlayerSpec:
    id: str
    type: PlayerType
    experiment: str
    checkpoint: str = "best"
    label: str = ""
    description: str = ""
    device: str | None = None
    actor_observation_level: str | None = None
    critic_observation_level: str | None = None
    mcts: MCTSPlayerConfig = field(default_factory=MCTSPlayerConfig)
    belief: BeliefPlayerConfig = field(default_factory=BeliefPlayerConfig)

    def display_label(self) -> str:
        return self.label or self.id

    def display_description(self) -> str:
        if self.description:
            return self.description
        try:
            exp = load_experiment(self.experiment)
            obs = self.actor_observation_level or exp.policy_spec.observation_level or "private"
            return (
                f"{exp.metadata.algorithm} | {exp.num_players}p {exp.map_type} | "
                f"{self.type} | obs={obs} | ckpt={self.checkpoint}"
            )
        except (FileNotFoundError, KeyError, ValueError):
            return f"{self.type} agent from {self.experiment}"

    def game_num_players(self) -> int:
        return load_experiment(self.experiment).num_players

    def game_map_type(self) -> str:
        return load_experiment(self.experiment).map_type


def player_config_dir() -> Path:
    env = os.environ.get("CATANRL_PLAYER_CONFIG_DIR")
    if env:
        return Path(env).resolve()
    return Path("configs/players").resolve()


def _parse_mcts_block(raw: object) -> MCTSPlayerConfig:
    if raw is None:
        return MCTSPlayerConfig()
    if not isinstance(raw, dict):
        raise ValueError("'mcts' must be a mapping when set.")
    return MCTSPlayerConfig(
        num_simulations=int(raw.get("num_simulations", 64)),
        c_puct=float(raw.get("c_puct", 1.5)),
        ismcts_determinizations=int(raw.get("ismcts_determinizations", 1)),
        num_search_workers=int(raw.get("num_search_workers", 1)),
        adversarial_policy=str(raw.get("adversarial_policy", "self")),
        prunning=bool(raw.get("prunning", False)),
        inference_batch_size=int(raw.get("inference_batch_size", 32)),
        inference_wait_ms=float(raw.get("inference_wait_ms", 1.0)),
        virtual_loss=float(raw.get("virtual_loss", 1.0)),
    )


def _parse_belief_block(raw: object) -> BeliefPlayerConfig:
    if raw is None:
        return BeliefPlayerConfig()
    if not isinstance(raw, dict):
        raise ValueError("'belief' must be a mapping when set.")
    return BeliefPlayerConfig(
        num_samples=int(raw.get("num_samples", 100)),
        sample=bool(raw.get("sample", False)),
    )


def player_spec_from_dict(data: dict[str, Any], *, source: str = "") -> PlayerSpec:
    player_id = data.get("id")
    if not isinstance(player_id, str) or not player_id.strip():
        raise ValueError(f"Player spec must have a non-empty 'id'{f' in {source}' if source else ''}.")

    player_type = data.get("type")
    if player_type not in PLAYER_TYPE_CHOICES:
        allowed = ", ".join(PLAYER_TYPE_CHOICES)
        raise ValueError(f"Player '{player_id}' type must be one of: {allowed}.")

    experiment = data.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError(f"Player '{player_id}' must specify a non-empty 'experiment'.")

    checkpoint = data.get("checkpoint", "best")
    if not isinstance(checkpoint, (str, int)):
        raise ValueError(f"Player '{player_id}' checkpoint must be a string or integer.")

    device = data.get("device")
    if device is not None and not isinstance(device, str):
        raise ValueError(f"Player '{player_id}' device must be a string when set.")

    actor_obs = data.get("actor_observation_level")
    critic_obs = data.get("critic_observation_level")
    for name, value in (
        ("actor_observation_level", actor_obs),
        ("critic_observation_level", critic_obs),
    ):
        if value is not None and value not in ("private", "public", "full"):
            raise ValueError(
                f"Player '{player_id}' {name} must be private, public, or full when set."
            )

    label = data.get("label", "")
    description = data.get("description", "")
    if not isinstance(label, str) or not isinstance(description, str):
        raise ValueError(f"Player '{player_id}' label and description must be strings.")

    return PlayerSpec(
        id=player_id.strip(),
        type=player_type,
        experiment=experiment.strip(),
        checkpoint=str(checkpoint),
        label=label,
        description=description,
        device=device,
        actor_observation_level=actor_obs,
        critic_observation_level=critic_obs,
        mcts=_parse_mcts_block(data.get("mcts")),
        belief=_parse_belief_block(data.get("belief")),
    )


def load_player_spec(path: str | Path) -> PlayerSpec:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Player spec not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Player spec must be a YAML mapping: {config_path}")
    return player_spec_from_dict(raw, source=str(config_path))


def _auto_discover_enabled() -> bool:
    value = os.environ.get("CATANRL_WEB_AUTO_DISCOVER", "true").strip().lower()
    return value not in ("0", "false", "no", "off")


def _default_auto_discover_checkpoint() -> str:
    return os.environ.get("CATANRL_WEB_CHECKPOINT", "best")


def _experiments_covered_by_specs(specs: list[PlayerSpec]) -> set[str]:
    covered: set[str] = set()
    for spec in specs:
        covered.add(spec.experiment)
        covered.add(spec.id)
    return covered


def load_all_player_specs(
    *,
    config_dir: Path | None = None,
    experiments_root: str | None = None,
    auto_discover: bool | None = None,
) -> list[PlayerSpec]:
    """Load YAML player specs and optionally synthesize default policy agents."""
    directory = config_dir or player_config_dir()
    specs: list[PlayerSpec] = []
    seen_ids: set[str] = set()

    if directory.is_dir():
        for path in sorted(directory.glob("*.yaml")):
            spec = load_player_spec(path)
            if spec.id in seen_ids:
                raise ValueError(f"Duplicate player spec id '{spec.id}' in {directory}.")
            seen_ids.add(spec.id)
            specs.append(spec)

    should_discover = _auto_discover_enabled() if auto_discover is None else auto_discover
    if should_discover:
        covered = _experiments_covered_by_specs(specs)
        checkpoint = _default_auto_discover_checkpoint()
        for name in list_experiments(experiments_root):
            if name in covered:
                continue
            specs.append(
                PlayerSpec(
                    id=name,
                    type="policy",
                    experiment=name,
                    checkpoint=checkpoint,
                    label=name,
                )
            )

    return specs


def get_player_spec(spec_id: str, specs: list[PlayerSpec] | None = None) -> PlayerSpec:
    pool = specs if specs is not None else load_all_player_specs()
    for spec in pool:
        if spec.id == spec_id:
            return spec
    raise KeyError(f"No player spec with id '{spec_id}'.")
