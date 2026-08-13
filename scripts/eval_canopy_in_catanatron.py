#!/usr/bin/env python3
"""Evaluate the released Canopy checkpoint directly inside Catanatron.

The Python Catanatron game is authoritative for map generation, dice, rules,
state transitions, legality, scoring, and the F opponent.  Only Canopy's move
selection is delegated to the Rust bridge.  Multiple live games are advanced
in lockstep so Canopy neural leaf evaluations remain GPU-batched.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from catanatron.game import Game
from catanatron.models.player import Color, SimplePlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points
from tqdm import tqdm

from catanrl.envs.cppanatron import NativeGame
from catanrl.envs.cppanatron.puffer_env import TURNS_LIMIT
from catanrl.eval.canopy_catanatron_bridge import CanopyBridgeProcess
from catanrl.eval.reporting import EvalResult, summarize_eval_results
from catanrl.utils.catanatron_action_space import (
    canopy_action_count_increment,
    to_action_space,
)
from catanrl.utils.catanatron_game import force_player_order
from catanrl.utils.catanatron_map import build_catan_map_from_native_game
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed


@dataclass
class LiveGame:
    game: Game
    episode_seed: int
    seat: str
    rng_state: object
    action_count: int = 0

    @property
    def done(self) -> bool:
        return self.game.winning_color() is not None or self.game.state.num_turns >= TURNS_LIMIT


def _new_game(episode_seed: int, seat: str) -> LiveGame:
    map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
    with NativeGame(
        2,
        "BASE",
        seed=game_seed,
        map_seed=map_seed,
        number_placement="random",
        discard_limit=9,
        vps_to_win=15,
    ) as native_layout:
        catan_map = build_catan_map_from_native_game(native_layout, "BASE")
    canopy = SimplePlayer(Color.RED)
    opponent = ValueFunctionPlayer(Color.BLUE)
    players = [canopy, opponent] if seat == "first" else [opponent, canopy]
    process_rng_state = random.getstate()
    try:
        game = Game(
            players=players,
            catan_map=catan_map,
            seed=game_seed,
            discard_limit=9,
            vps_to_win=15,
        )
        force_player_order(game, players)
        game_rng_state = random.getstate()
    finally:
        random.setstate(process_rng_state)
    return LiveGame(
        game=game,
        episode_seed=episode_seed,
        seat=seat,
        rng_state=game_rng_state,
    )


@contextmanager
def _game_random_state(live: LiveGame):
    """Isolate Catanatron's module-global RNG for each interleaved game."""

    process_rng_state = random.getstate()
    random.setstate(live.rng_state)
    try:
        yield
    finally:
        live.rng_state = random.getstate()
        random.setstate(process_rng_state)


def _execute(live: LiveGame, action) -> None:
    with _game_random_state(live):
        live.game.execute(action)
    action_index = to_action_space(action, 2, "BASE", live.game.state.colors)
    live.action_count += canopy_action_count_increment(action_index, 2, "BASE")


def _is_done(live: LiveGame, max_actions: int) -> bool:
    return live.done or (max_actions > 0 and live.action_count >= max_actions)


def _play_batch(
    bridge: CanopyBridgeProcess,
    games: list[LiveGame],
    *,
    max_actions: int,
) -> None:
    while any(not _is_done(live, max_actions) for live in games):
        # Advance F and all forced Canopy moves locally.  Stop only at genuine
        # Canopy decisions so every bridge request carries a useful GPU batch.
        for live in games:
            while not _is_done(live, max_actions):
                game = live.game
                if game.state.current_color() == Color.RED:
                    if len(game.playable_actions) > 1:
                        break
                    _execute(live, game.playable_actions[0])
                else:
                    with _game_random_state(live):
                        opponent_action = game.state.current_player().decide(
                            game, game.playable_actions
                        )
                    _execute(live, opponent_action)

        decisions = [
            (live.game, tuple(live.game.playable_actions))
            for live in games
            if not _is_done(live, max_actions)
        ]
        if not decisions:
            continue
        actions = bridge.decide_many(decisions)
        decision_index = 0
        for live in games:
            if _is_done(live, max_actions):
                continue
            if live.game.state.current_color() != Color.RED:
                raise RuntimeError("scheduler did not stop on the Canopy seat")
            _execute(live, actions[decision_index])
            decision_index += 1


def _record(live: LiveGame, max_actions: int) -> dict[str, object]:
    winner = live.game.winning_color()
    canopy_vps = get_actual_victory_points(live.game.state, Color.RED)
    total_vps = sum(
        get_actual_victory_points(live.game.state, color) for color in live.game.state.colors
    )
    return {
        "seat": live.seat,
        "episode_seed": live.episode_seed,
        "win": winner == Color.RED,
        "draw": winner is None,
        "truncated": winner is None
        and (
            live.game.state.num_turns >= TURNS_LIMIT
            or (max_actions > 0 and live.action_count >= max_actions)
        ),
        "winner": None if winner is None else winner.value,
        "vps": int(canopy_vps),
        "total_vps": int(total_vps),
        "turns": int(live.game.state.num_turns),
        "actions": live.action_count,
    }


def _summary(records: list[dict[str, object]], checkpoint: Path) -> dict[str, object]:
    by_seat: dict[str, EvalResult] = {}
    for seat in ("first", "second"):
        rows = [row for row in records if row["seat"] == seat]
        by_seat[seat] = EvalResult(
            wins=sum(bool(row["win"]) for row in rows),
            vps=[int(row["vps"]) for row in rows],
            total_vps=[int(row["total_vps"]) for row in rows],
            turns=[int(row["turns"]) for row in rows],
        )
    return summarize_eval_results(
        "cullback/canopy nexus-v3 search",
        checkpoint.name,
        by_seat,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=1600)
    parser.add_argument("--games-per-seat", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12043)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simulations < 1:
        raise ValueError("--simulations must be positive")
    if args.games_per_seat < 1:
        raise ValueError("--games-per-seat must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_actions < 0:
        raise ValueError("--max-actions cannot be negative")
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    scenarios = [
        (seat, derive_seed(args.seed, "native_budget_episode", game_index))
        for seat in ("first", "second")
        for game_index in range(args.games_per_seat)
    ]
    records: list[dict[str, object]] = []
    progress = tqdm(total=len(scenarios), desc="Canopy search vs F in Catanatron")
    with CanopyBridgeProcess(
        args.binary,
        args.checkpoint,
        simulations=args.simulations,
        seed=args.seed,
    ) as bridge:
        for start in range(0, len(scenarios), args.batch_size):
            batch = [
                _new_game(episode_seed, seat)
                for seat, episode_seed in scenarios[start : start + args.batch_size]
            ]
            _play_batch(bridge, batch, max_actions=args.max_actions)
            records.extend(_record(live, args.max_actions) for live in batch)
            progress.update(len(batch))
    progress.close()

    summary = _summary(records, args.checkpoint)
    payload = {
        "implementation": "cullback/canopy adapted into Catanatron",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_name": args.checkpoint.name,
        "game_engine": "Catanatron",
        "map_layout_source": "cppanatron layout imported into Catanatron",
        "opponent": "F",
        "map_type": "BASE",
        "number_placement": "random",
        "num_players": 2,
        "vps_to_win": 15,
        "discard_limit": 9,
        "simulations": args.simulations,
        "root_noise": 0.0,
        "games_per_seat": args.games_per_seat,
        "seed": args.seed,
        "max_actions": args.max_actions,
        "summary": summary,
        "game_records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
