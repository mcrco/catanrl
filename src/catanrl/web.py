"""Play against trained catanrl policies in the Catanatron Web UI.

Catanatron's web server (``catanatron[web]``) keeps a registry of the players
selectable in its UI. This module adds one entry per saved experiment so the
trained models show up in the player dropdown next to the built-in bots.

Run the API server with one of::

    uv run catanrl-web                                  # every experiment under experiments/
    uv run catanrl-web --experiments name-a,name-b      # only these
    uv run flask --app catanrl.web run --port 5001      # same, configured via CATANRL_WEB_* env vars

and the UI with ``npm start`` inside ``catanatron/ui``.

Catanatron pickles the whole ``Game`` (players included) into its database on
every move and unpickles it on every request. The players defined here
therefore never pickle the network: only the reference needed to find it again
in a process-level cache, so a game's rows stay small and no request rebuilds
a model.
"""

from __future__ import annotations

import argparse
import os
import threading
from collections.abc import Sequence
from functools import partial
from typing import Literal, cast

from catanatron.models.map import build_map
from catanatron.models.player import Color
from catanatron.web import create_app as create_catanatron_app
from catanatron.web import register_web_player

from catanrl.experiment_store import (
    METADATA_FILENAME,
    Experiment,
    experiments_root,
    load_experiment,
)
from catanrl.features.catanatron_utils import ActorObservationLevel
from catanrl.models import PolicyNetworkWrapper
from catanrl.players import BeliefAveragedPolicyPlayer, NNPolicyPlayer

DEFAULT_WHICH = "best"
DEFAULT_DEVICE = "cpu"
KEY_PREFIX = "CATANRL"
ORACLE_KEY_PREFIX = "CATANRL_ORACLE"

ENV_EXPERIMENTS = "CATANRL_WEB_EXPERIMENTS"
ENV_WHICH = "CATANRL_WEB_WHICH"
ENV_DEVICE = "CATANRL_WEB_DEVICE"

MapType = Literal["BASE", "MINI", "TOURNAMENT"]


# --------------------------------------------------------------------------- #
# Process-level caches
# --------------------------------------------------------------------------- #
_cache_lock = threading.RLock()
_experiments: dict[str, Experiment] = {}
_policies: dict[tuple[str, str, str], PolicyNetworkWrapper] = {}


def cached_experiment(name: str) -> Experiment:
    with _cache_lock:
        exp = _experiments.get(name)
        if exp is None:
            exp = load_experiment(name)
            _experiments[name] = exp
        return exp


def cached_policy(experiment: str, which: str, device: str) -> PolicyNetworkWrapper:
    """Load an experiment's policy once per process and reuse it across games."""
    key = (experiment, str(which), device)
    with _cache_lock:
        model = _policies.get(key)
        if model is None:
            model = cast(
                PolicyNetworkWrapper,
                cached_experiment(experiment).build_policy(which=which, device=device),
            )
            _policies[key] = model
        return model


def clear_caches() -> None:
    with _cache_lock:
        _experiments.clear()
        _policies.clear()


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #
class _ExperimentBackedPlayer:
    """Mixin: pickle by reference and refuse games the model was not trained for."""

    experiment: str
    which: str
    device: str

    def _remember_spec(self, experiment: str, which: str, device: str) -> None:
        self.experiment = experiment
        self.which = which
        self.device = device

    def __getstate__(self):
        return {
            "color": self.color,  # type: ignore[attr-defined]
            "experiment": self.experiment,
            "which": self.which,
            "device": self.device,
        }

    def __setstate__(self, state):
        self.__init__(**state)  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.experiment}):{self.color.value}"  # type: ignore[attr-defined]

    def decide(self, game, playable_actions):
        exp = cached_experiment(self.experiment)
        num_players = len(game.state.colors)
        if num_players != exp.num_players:
            raise ValueError(
                f"Experiment '{self.experiment}' was trained for {exp.num_players} players "
                f"but this game has {num_players}."
            )
        expected_land_tiles = len(build_map(exp.map_type).land_tiles)
        if len(game.state.board.map.land_tiles) != expected_land_tiles:
            raise ValueError(
                f"Experiment '{self.experiment}' was trained on the {exp.map_type} map; "
                f"this game uses a different map template."
            )
        return super().decide(game, playable_actions)  # type: ignore[misc]


class ExperimentPolicyPlayer(_ExperimentBackedPlayer, NNPolicyPlayer):
    """Greedy policy from a saved experiment.

    Observes the game at the level the policy was trained on, so a ``full``
    policy sees the opponents' hidden cards.
    """

    def __init__(
        self,
        color: Color,
        experiment: str,
        which: str = DEFAULT_WHICH,
        device: str = DEFAULT_DEVICE,
    ):
        exp = cached_experiment(experiment)
        NNPolicyPlayer.__init__(
            self,
            color,
            model_type=exp.model_type or "flat",
            model=cached_policy(experiment, which, device),
            map_type=cast(MapType, exp.map_type),
            actor_observation_level=cast(
                ActorObservationLevel, exp.policy_spec.observation_level or "private"
            ),
            device=device,
        )
        self._remember_spec(experiment, which, device)


class ExperimentBeliefPlayer(_ExperimentBackedPlayer, BeliefAveragedPolicyPlayer):
    """Full-information policy from a saved experiment, marginalized over the
    belief about opponents' hidden dev cards so it does not cheat."""

    def __init__(
        self,
        color: Color,
        experiment: str,
        which: str = DEFAULT_WHICH,
        device: str = DEFAULT_DEVICE,
    ):
        exp = cached_experiment(experiment)
        BeliefAveragedPolicyPlayer.__init__(
            self,
            color,
            model_type=exp.model_type or "flat",
            model=cached_policy(experiment, which, device),
            map_type=cast(MapType, exp.map_type),
            device=device,
        )
        self._remember_spec(experiment, which, device)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def discover_experiments() -> list[str]:
    """Names of every experiment directory under :func:`experiments_root`."""
    root = experiments_root()
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, METADATA_FILENAME))
    )


def register_experiment_players(
    experiments: Sequence[str] | None = None,
    which: str = DEFAULT_WHICH,
    device: str = DEFAULT_DEVICE,
) -> list[str]:
    """Register web players for the given experiments (default: all discovered).

    Full-information policies get two entries: the belief-averaged player (the
    fair one to play against) and the raw oracle. Others get a single entry.
    Models are loaded lazily, on the first game that uses them.
    Returns the registered player keys.
    """
    names = list(experiments) if experiments is not None else discover_experiments()
    keys: list[str] = []
    for name in names:
        exp = cached_experiment(name)
        try:
            exp.resolve_checkpoint(which, "policy")
        except FileNotFoundError:
            if experiments is not None:
                raise
            continue  # e.g. a run that is still training; skip it silently.

        spec = {"experiment": name, "which": which, "device": device}
        if exp.policy_spec.observation_level == "full":
            entries = [
                (f"{KEY_PREFIX}:{name}", ExperimentBeliefPlayer, f"{name} (belief-averaged)"),
                (
                    f"{ORACLE_KEY_PREFIX}:{name}",
                    ExperimentPolicyPlayer,
                    f"{name} (oracle: sees hidden cards)",
                ),
            ]
        else:
            entries = [(f"{KEY_PREFIX}:{name}", ExperimentPolicyPlayer, name)]
        for key, player_cls, label in entries:
            register_web_player(key, partial(player_cls, **spec), label)
            keys.append(key)
    return keys


def _experiments_from_env() -> list[str] | None:
    raw = os.environ.get(ENV_EXPERIMENTS)
    if not raw:
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]


def create_app(test_config=None):
    """Flask app factory: catanatron's web app plus the catanrl players.

    Configured through ``CATANRL_WEB_EXPERIMENTS`` (comma-separated, default: all
    experiments), ``CATANRL_WEB_WHICH`` (checkpoint selector, default ``best``)
    and ``CATANRL_WEB_DEVICE`` (default ``cpu``).
    """
    register_experiment_players(
        _experiments_from_env(),
        which=os.environ.get(ENV_WHICH, DEFAULT_WHICH),
        device=os.environ.get(ENV_DEVICE, DEFAULT_DEVICE),
    )
    return create_catanatron_app(test_config)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve the Catanatron web API with catanrl experiments as players."
    )
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated experiment names (default: every experiment under experiments/).",
    )
    parser.add_argument("--which", default=DEFAULT_WHICH, help="Checkpoint selector.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device for inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args(argv)

    experiments = (
        [name.strip() for name in args.experiments.split(",") if name.strip()]
        if args.experiments
        else None
    )
    keys = register_experiment_players(experiments, which=args.which, device=args.device)
    print(f"Registered {len(keys)} catanrl players:")
    for key in keys:
        print(f"  {key}")
    print("Start the UI with `npm start` in catanatron/ui, then open http://localhost:3000")

    app = create_catanatron_app()
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
