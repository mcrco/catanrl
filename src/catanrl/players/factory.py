"""Build CatanRL inference players from PlayerSpec + experiment_store."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import torch
from catanatron.models.player import Color, Player

from catanrl.experiment_store import KIND_POLICY_VALUE, Experiment, load_experiment
from catanrl.experiments.network_config import validate_ismcts_observation_levels
from catanrl.features.catanatron_utils import ActorObservationLevel, CriticObservationLevel
from catanrl.players.belief_policy_player import BeliefAveragedPolicyPlayer
from catanrl.players.nn_mcts_player import NNMCTSPlayer
from catanrl.players.nn_policy_player import NNPolicyPlayer
from catanrl.players.player_config import PlayerSpec, get_player_spec


def resolve_device(spec: PlayerSpec | None = None) -> torch.device:
    if spec is not None and spec.device:
        return torch.device(spec.device)
    requested = os.environ.get("CATANRL_WEB_PLAYER_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _actor_observation_level(
    spec: PlayerSpec, exp: Experiment
) -> ActorObservationLevel:
    if spec.actor_observation_level is not None:
        return spec.actor_observation_level  # type: ignore[return-value]
    level = exp.policy_spec.observation_level
    if level in ("private", "public", "full"):
        return level  # type: ignore[return-value]
    tc = exp.metadata.train_config
    fallback = tc.get("actor_observation_level") or tc.get("policy_mode")
    if fallback in ("private", "public", "full"):
        return fallback  # type: ignore[return-value]
    return "private"


def _critic_observation_level(spec: PlayerSpec, exp: Experiment) -> CriticObservationLevel:
    if spec.critic_observation_level is not None:
        return spec.critic_observation_level  # type: ignore[return-value]

    critic = exp.metadata.networks.get("critic")
    if critic is not None and critic.observation_level in ("private", "public", "full"):
        return critic.observation_level  # type: ignore[return-value]

    tc = exp.metadata.train_config
    for key in ("critic_observation_level", "critic_mode"):
        value = tc.get(key)
        if value in ("private", "public", "full"):
            return value  # type: ignore[return-value]

    if exp.policy_spec.kind == KIND_POLICY_VALUE:
        return _actor_observation_level(spec, exp)

    return "full"


def validate_player_spec(spec: PlayerSpec, exp: Experiment) -> None:
    """Validate a spec against its experiment before building a player."""
    exp.resolve_checkpoint(spec.checkpoint, "policy")

    if spec.type == "belief":
        actor_level = _actor_observation_level(spec, exp)
        if actor_level != "full":
            raise ValueError(
                f"Belief-averaged player '{spec.id}' requires a full-info policy, but "
                f"observation level is '{actor_level}'."
            )

    if spec.type == "mcts":
        policy_spec = exp.policy_spec
        uses_joint = policy_spec.kind == KIND_POLICY_VALUE
        if not uses_joint and "critic" not in exp.metadata.networks:
            raise ValueError(
                f"MCTS player '{spec.id}' requires a critic network in experiment "
                f"'{exp.metadata.name}', but none is recorded in metadata."
            )
        if not uses_joint:
            exp.resolve_checkpoint(spec.checkpoint, "critic")

        mcts = spec.mcts
        if mcts.ismcts_determinizations > 1:
            validate_ismcts_observation_levels(
                ismcts_determinizations=mcts.ismcts_determinizations,
                num_players=exp.num_players,
                actor_observation_level=_actor_observation_level(spec, exp),
                critic_observation_level=_critic_observation_level(spec, exp),
            )


def validate_game_context(
    spec: PlayerSpec,
    *,
    num_players: int,
    map_type: str,
) -> None:
    exp = load_experiment(spec.experiment)
    if exp.num_players != num_players:
        raise ValueError(
            f"Player '{spec.id}' expects {exp.num_players} players, got {num_players}."
        )
    if exp.map_type != map_type:
        raise ValueError(
            f"Player '{spec.id}' expects map {exp.map_type}, got {map_type}."
        )


def build_player_from_spec(
    spec: PlayerSpec,
    color: Color,
    *,
    num_players: int | None = None,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"] | None = None,
) -> Player:
    exp = load_experiment(spec.experiment)
    validate_player_spec(spec, exp)

    if num_players is not None and map_type is not None:
        validate_game_context(spec, num_players=num_players, map_type=map_type)

    device = resolve_device(spec)
    model_type = exp.model_type or "flat"
    resolved_map: Literal["BASE", "MINI", "TOURNAMENT"] = exp.map_type  # type: ignore[assignment]
    actor_level = _actor_observation_level(spec, exp)

    if spec.type == "policy":
        policy = exp.build_policy(which=spec.checkpoint, device=device)
        return NNPolicyPlayer(
            color=color,
            model_type=model_type,
            model=policy,
            map_type=resolved_map,
            actor_observation_level=actor_level,
        )

    if spec.type == "belief":
        policy = exp.build_policy(which=spec.checkpoint, device=device)
        return BeliefAveragedPolicyPlayer(
            color=color,
            model_type=model_type,
            model=policy,
            map_type=resolved_map,
            num_samples=spec.belief.num_samples,
            sample=spec.belief.sample,
        )

    if spec.type == "mcts":
        policy_spec = exp.policy_spec
        uses_joint = policy_spec.kind == KIND_POLICY_VALUE
        if uses_joint:
            policy_model = exp.build_policy(
                which=spec.checkpoint, device=device, as_policy_only=False
            )
            critic_model = None
        else:
            policy_model = exp.build_policy(which=spec.checkpoint, device=device)
            critic_model = exp.build_critic(which=spec.checkpoint, device=device)

        mcts = spec.mcts
        return NNMCTSPlayer(
            color=color,
            model_type=model_type,
            policy_model=policy_model,
            critic_model=critic_model,
            map_type=resolved_map,
            num_simulations=mcts.num_simulations,
            c_puct=mcts.c_puct,
            ismcts_determinizations=mcts.ismcts_determinizations,
            prunning=mcts.prunning,
            opponent_policy=mcts.adversarial_policy,
            actor_observation_level=actor_level,
            critic_observation_level=_critic_observation_level(spec, exp),
            num_search_workers=mcts.num_search_workers,
            inference_batch_size=mcts.inference_batch_size,
            inference_wait_ms=mcts.inference_wait_ms,
            virtual_loss=mcts.virtual_loss,
            device=device,
        )

    raise ValueError(f"Unsupported player type '{spec.type}'.")


@lru_cache(maxsize=32)
def get_cached_player(spec_id: str, color_value: str) -> Player:
    """Return a process-local cached player for web games (not pickled)."""
    spec = get_player_spec(spec_id)
    color = Color[color_value]
    return build_player_from_spec(spec, color)
