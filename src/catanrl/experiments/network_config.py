"""Shared CLI helpers for policy/critic observation levels and network layout."""

from __future__ import annotations

import argparse
from typing import Literal

ObservationLevel = Literal["private", "public", "full"]
NetworkMode = Literal["shared", "separate"]

OBSERVATION_LEVEL_CHOICES: tuple[str, ...] = ("private", "public", "full")
NETWORK_MODE_CHOICES: tuple[str, ...] = ("shared", "separate")


def add_observation_network_arguments(
    parser: argparse.ArgumentParser,
    *,
    policy_mode_default: str = "private",
    critic_mode_default: str = "full",
    network_mode_default: str = "separate",
) -> None:
    """Register canonical ``--policy-mode``, ``--critic-mode``, and ``--network-mode`` flags."""
    parser.add_argument(
        "--policy-mode",
        type=str,
        default=None,
        choices=OBSERVATION_LEVEL_CHOICES,
        help=(
            "Policy (actor) information level: private, public (1v1 opponent resources), "
            "or full."
        ),
    )
    parser.add_argument(
        "--critic-mode",
        type=str,
        default=None,
        choices=OBSERVATION_LEVEL_CHOICES,
        help=(
            "Critic (value) information level: private, public (1v1 opponent resources), "
            "or full."
        ),
    )
    parser.add_argument(
        "--network-mode",
        type=str,
        default=network_mode_default,
        choices=NETWORK_MODE_CHOICES,
        help="Use a shared policy-value backbone (shared) or separate networks (separate).",
    )
    parser.add_argument(
        "--actor-observation-level",
        type=str,
        default=None,
        choices=OBSERVATION_LEVEL_CHOICES,
        help="Deprecated alias for --policy-mode.",
    )
    parser.add_argument(
        "--critic-observation-level",
        type=str,
        default=None,
        choices=OBSERVATION_LEVEL_CHOICES,
        help="Deprecated alias for --critic-mode.",
    )


def resolve_observation_network_args(
    args: argparse.Namespace,
    *,
    policy_mode_default: str = "private",
    critic_mode_default: str = "full",
) -> None:
    """Normalize observation/network flags and validate shared-backbone constraints."""
    policy_mode = (
        getattr(args, "policy_mode", None)
        or getattr(args, "actor_observation_level", None)
        or policy_mode_default
    )
    critic_mode = (
        getattr(args, "critic_mode", None)
        or getattr(args, "critic_observation_level", None)
        or critic_mode_default
    )
    network_mode = getattr(args, "network_mode", "separate")

    if getattr(args, "policy_mode", None) and getattr(args, "actor_observation_level", None):
        if args.policy_mode != args.actor_observation_level:
            raise ValueError(
                "--policy-mode and --actor-observation-level disagree: "
                f"{args.policy_mode!r} vs {args.actor_observation_level!r}."
            )
    if getattr(args, "critic_mode", None) and getattr(args, "critic_observation_level", None):
        if args.critic_mode != args.critic_observation_level:
            raise ValueError(
                "--critic-mode and --critic-observation-level disagree: "
                f"{args.critic_mode!r} vs {args.critic_observation_level!r}."
            )

    args.policy_mode = policy_mode
    args.critic_mode = critic_mode
    args.network_mode = network_mode
    # Legacy names used throughout training code and experiment metadata.
    args.actor_observation_level = policy_mode
    args.critic_observation_level = critic_mode

    assert_shared_network_obs_levels(network_mode, policy_mode, critic_mode)

    ismcts_determinizations = getattr(args, "ismcts_determinizations", 1)
    if ismcts_determinizations > 1:
        validate_ismcts_observation_levels(
            ismcts_determinizations=ismcts_determinizations,
            num_players=getattr(args, "num_players", None),
            actor_observation_level=policy_mode,
            critic_observation_level=critic_mode,
        )


def assert_shared_network_obs_levels(
    network_mode: str,
    policy_mode: str,
    critic_mode: str,
) -> None:
    """Require matching observation levels when using a shared policy-value backbone."""
    if network_mode != "shared":
        return
    if policy_mode != critic_mode:
        raise ValueError(
            f"network_mode='shared' requires policy and critic information levels to match, "
            f"but got policy_mode={policy_mode!r} and critic_mode={critic_mode!r}. "
            f"Either set both to the same level (e.g. 'private') or use network_mode='separate'."
        )


def validate_ismcts_observation_levels(
    *,
    ismcts_determinizations: int,
    num_players: int | None,
    actor_observation_level: ObservationLevel,
    critic_observation_level: ObservationLevel,
) -> None:
    """Reject private observation levels when IS-MCTS determinization is enabled.

    IS-MCTS samples concrete hidden worlds (opponent dev-card hands). Private
    features discard that information, so running IS-MCTS with a private actor
    or critic is inconsistent.

    For 1v1, ``public`` or ``full`` are allowed. For 3–4 players only ``full``
    is valid (``public`` is not defined outside 1v1).
    """
    if ismcts_determinizations <= 1:
        return

    if num_players is not None and num_players != 2:
        allowed: tuple[ObservationLevel, ...] = ("full",)
        allowed_label = "'full'"
    else:
        allowed = ("public", "full")
        allowed_label = "'public' or 'full'"

    for role, level in (
        ("actor", actor_observation_level),
        ("critic", critic_observation_level),
    ):
        if level not in allowed:
            raise ValueError(
                f"Information-Set MCTS (ismcts_determinizations > 1) requires "
                f"{role}_observation_level to be {allowed_label}, not '{level}'. "
                "Private features discard determinized hidden state."
            )
