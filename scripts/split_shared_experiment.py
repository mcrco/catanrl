"""Convert a shared policy-value experiment into separate policy and critic checkpoints."""

from __future__ import annotations

import argparse
import os

import torch

from catanrl.experiment_store import (
    KIND_POLICY,
    KIND_POLICY_VALUE,
    KIND_VALUE,
    NetworkSpec,
    build_network,
    default_checkpoints_dir,
    experiment_dir,
    load_experiment,
    save_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Shared policy-value experiment name or path")
    parser.add_argument("destination", help="Name of the converted experiment")
    parser.add_argument("--which", default="best", help="Source checkpoint selector")
    args = parser.parse_args()

    destination_path = experiment_dir(args.destination)
    if os.path.exists(destination_path):
        raise FileExistsError(f"Destination experiment already exists: {destination_path}")

    source = load_experiment(args.source)
    shared_spec = source.policy_spec
    if shared_spec.kind != KIND_POLICY_VALUE:
        raise ValueError(
            f"Source policy must have kind={KIND_POLICY_VALUE!r}, got {shared_spec.kind!r}"
        )

    policy_spec = NetworkSpec(
        kind=KIND_POLICY,
        backbone=shared_spec.backbone,
        model_type=shared_spec.model_type,
        observation_level=shared_spec.observation_level,
        num_actions=shared_spec.num_actions,
    )
    critic_spec = NetworkSpec(
        kind=KIND_VALUE,
        backbone=shared_spec.backbone,
        observation_level=shared_spec.observation_level,
    )

    source_checkpoint = source.resolve_checkpoint(args.which, "policy")
    shared_state = torch.load(source_checkpoint, map_location="cpu")
    policy_state = {
        key: value for key, value in shared_state.items() if not key.startswith("value_head.")
    }
    critic_state = {
        key: value for key, value in shared_state.items() if not key.startswith("policy_head.")
    }

    # Strict loading verifies that the filtered state dicts exactly match the target models.
    build_network(policy_spec, source.metadata.game).load_state_dict(policy_state)
    build_network(critic_spec, source.metadata.game).load_state_dict(critic_state)

    checkpoint_dir = default_checkpoints_dir(args.destination)
    os.makedirs(checkpoint_dir)
    torch.save(policy_state, os.path.join(checkpoint_dir, "policy_best.pt"))
    torch.save(critic_state, os.path.join(checkpoint_dir, "critic_best.pt"))

    train_config = dict(source.metadata.train_config)
    train_config.update(
        {
            "network_mode": "separate",
            "critic_mode": shared_spec.observation_level,
            "critic_observation_level": shared_spec.observation_level,
            "converted_from_experiment": source.metadata.name,
            "converted_from_which": args.which,
        }
    )
    path = save_experiment(
        args.destination,
        checkpoint_dir,
        algorithm=f"converted-{source.metadata.algorithm}",
        game=source.metadata.game,
        networks={"policy": policy_spec, "critic": critic_spec},
        train_config=train_config,
        notes=(
            f"Split from {source.metadata.name!r} checkpoint {args.which!r}; policy and "
            "critic start with independent copies of the shared backbone."
        ),
    )
    print(f"Created converted experiment: {path}")


if __name__ == "__main__":
    main()
