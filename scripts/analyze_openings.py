"""Opening-placement quality by seat: a trained policy against F, and F against F.

Plays deterministic native (cppanatron) games and records each player's two
initial settlements, scoring them by production pips (dice-probability x 36),
resource coverage, and port access. At each of the policy's settlement picks it
also records what F (ValueFunctionPlayer) would have played from the same
position and the best pips still available.

All three matchups (policy first vs F, F vs policy second, F vs F) play the same
map and game seeds. F is deterministic, so the policy's first pick in each seat
is made from exactly the position F faced in the F-vs-F game on that map, which
pairs the comparisons pick by pick.

Usage:
    uv run scripts/analyze_openings.py --experiment NAME --which best \\
        --games 2000 --seed 67 --out openings.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from catanatron.gym.board_tensor_features import get_node_and_edge_maps
from catanatron.models.enums import ActionType

from catanrl.envs.cppanatron import FlatGame
from catanrl.envs.cppanatron.puffer_env import TURNS_LIMIT
from catanrl.envs.puffer.common import BOARD_HEIGHT, BOARD_WIDTH, compute_single_agent_dims
from catanrl.experiment_store import load_experiment
from catanrl.utils.catanatron_action_space import get_action_array
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

NUM_PLAYERS = 2
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


@dataclass
class Pick:
    order: int  # 1-4 across both players (snake: seat 0, seat 1, seat 1, seat 0)
    seat: int
    node: int
    pips: float
    best_available_pips: float
    resources: list[str]
    port: bool
    f_node: int  # what F would have played here (equals node when F is the mover)
    f_pips: float


@dataclass
class GameRecord:
    matchup: str  # "policy_first", "policy_second", "f_vs_f"
    map_index: int
    policy_seat: int | None
    winner: int | None
    turns: int
    picks: list[Pick] = field(default_factory=list)


class NodeTable:
    """Static per-node production and port data read from an observation's board planes."""

    def __init__(self, observation: np.ndarray, numeric_dim: int, channels: int):
        board = observation[numeric_dim:].reshape(BOARD_WIDTH, BOARD_HEIGHT, channels)
        node_positions, _ = get_node_and_edge_maps()
        resource_offset = 2 * NUM_PLAYERS
        port_offset = resource_offset + 5 + 1
        self.pips: dict[int, float] = {}
        self.resources: dict[int, frozenset[str]] = {}
        self.port: dict[int, bool] = {}
        for node, (x, y) in node_positions.items():
            planes = board[x, y]
            production = planes[resource_offset : resource_offset + 5]
            self.pips[node] = float(np.round(production.sum() * 36.0, 3))
            self.resources[node] = frozenset(r for r, p in zip(RESOURCES, production) if p > 0)
            self.port[node] = bool(planes[port_offset : port_offset + 6].any())


def _settlement_nodes() -> dict[int, int]:
    actions = get_action_array(NUM_PLAYERS, "BASE")
    return {
        index: int(value)  # type: ignore[arg-type]
        for index, (action_type, value) in enumerate(actions)
        if action_type == ActionType.BUILD_SETTLEMENT
    }


def play_games(
    matchup: str,
    seeds: list[tuple[int, int]],
    policy: torch.nn.Module | None,
    device: torch.device,
    numeric_dim: int,
    channels: int,
    f_opening: bool = False,
) -> list[GameRecord]:
    """Play one matchup on every seed; ``f_opening`` lets F make the policy's initial placements."""
    policy_seat = {"policy_first": 0, "policy_second": 1, "f_vs_f": None}[matchup]
    settlement_node = _settlement_nodes()
    games, tables, records = [], [], []
    for index, (map_seed, game_seed) in enumerate(seeds):
        game = FlatGame(
            NUM_PLAYERS,
            "BASE",
            seed=game_seed,
            map_seed=map_seed,
            number_placement="official_spiral",
            discard_limit=9,
            vps_to_win=15,
        )
        games.append(game)
        tables.append(NodeTable(game.observation(0), numeric_dim, channels))
        records.append(GameRecord(matchup, index, policy_seat, None, 0))
    pick_counts = [0] * len(games)
    active = set(range(len(games)))

    while active:
        policy_rows: list[int] = []
        for i in list(active):
            game = games[i]
            if game.winner is not None or game.num_turns >= TURNS_LIMIT:
                records[i].winner = game.winner
                records[i].turns = game.num_turns
                active.discard(i)
            elif game.current_player == policy_seat and not (
                f_opening and game.game.is_initial_build_phase
            ):
                policy_rows.append(i)
            else:
                _play(game, game.expert_action(), records[i], tables[i], pick_counts, i, settlement_node)
        if not policy_rows:
            continue
        observations = np.stack([games[i].observation(policy_seat) for i in policy_rows])
        masks = np.stack([games[i].action_mask() for i in policy_rows]).astype(bool)
        with torch.no_grad():
            logits = policy(torch.from_numpy(observations).float().to(device))
            logits = logits.masked_fill(~torch.from_numpy(masks).to(device), float("-inf"))
            actions = logits.argmax(dim=-1).cpu().numpy()
        for i, action in zip(policy_rows, actions):
            _play(games[i], int(action), records[i], tables[i], pick_counts, i, settlement_node)
    return records


def _play(game, action, record, table, pick_counts, index, settlement_node) -> None:
    if game.game.is_initial_build_phase and action in settlement_node:
        mask = game.action_mask()
        available = [settlement_node[a] for a in settlement_node if mask[a]]
        f_action = game.expert_action()
        f_node = settlement_node.get(f_action, -1)
        node = settlement_node[action]
        pick_counts[index] += 1
        record.picks.append(
            Pick(
                order=pick_counts[index],
                seat=game.current_player,
                node=node,
                pips=table.pips[node],
                resources=sorted(table.resources[node]),
                port=table.port[node],
                best_available_pips=max(table.pips[n] for n in available),
                f_node=f_node,
                f_pips=table.pips[f_node] if f_node >= 0 else float("nan"),
            )
        )
    game.step(action)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--which", default="best")
    parser.add_argument("--games", type=int, default=2000, help="Maps per matchup")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--f-opening",
        action="store_true",
        help=(
            "Counterfactual: F makes the policy's initial settlements and roads, then the "
            "policy plays on. Compared on the same seeds, this isolates the opening's effect."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_experiment(args.experiment)
    if exp.num_players != NUM_PLAYERS or exp.map_type != "BASE":
        raise ValueError("Opening analysis supports 2-player BASE experiments only")
    if exp.policy_spec.observation_level != "full":
        raise ValueError("Opening analysis feeds full observations; use a full-info policy")
    policy = exp.build_policy(which=args.which, device=device)
    dims = compute_single_agent_dims(NUM_PLAYERS, "BASE", actor_observation_level="full")
    numeric_dim, channels = dims["actor_numeric_dim"], dims["board_channels"]

    seeds = [
        derive_map_and_game_seeds(derive_seed(args.seed, "openings", i)) for i in range(args.games)
    ]
    records: list[GameRecord] = []
    matchups = ("policy_first", "policy_second") if args.f_opening else (
        "policy_first",
        "policy_second",
        "f_vs_f",
    )
    for matchup in matchups:
        played = play_games(
            matchup, seeds, policy, device, numeric_dim, channels, f_opening=args.f_opening
        )
        records.extend(played)
        print(f"{matchup}: {len(played)} games")

    with open(args.out, "w") as handle:
        json.dump(
            {
                "experiment": args.experiment,
                "which": args.which,
                "seed": args.seed,
                "games_per_matchup": args.games,
                "f_opening": args.f_opening,
                "records": [asdict(r) for r in records],
            },
            handle,
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
