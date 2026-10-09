"""Paired native games between search agents built on one frozen policy/critic.

Agents are described by spec strings:

* ``raw``: the policy's argmax;
* ``F``: native Catanatron ValueFunctionPlayer;
* ``mcts:sims=256,c_puct=1.5``: native PUCT MCTS (``run_native_search_policy``);
* ``2ffs:budget=1024,depth=3,leaf=0.1,slope=0.05,rollout=none,batch=1``:
  two-fidelity fast-slow search (``catanrl.algorithms.twoffs``).

Every game index is played once with agent A in each seat on the same map and
game seed, and all workers share one batched GPU inference server. Search cost
is reported in network state evaluations per searched decision.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

OBSERVATION_SIZE = 3770


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    params: dict[str, str] = field(default_factory=dict)
    text: str = ""

    @property
    def label(self) -> str:
        return self.text or self.kind


def parse_agent(text: str) -> AgentSpec:
    kind, _, rest = text.partition(":")
    kind = kind.strip().lower()
    if kind == "f":
        kind = "value"
    if kind not in {"raw", "value", "mcts", "2ffs"}:
        raise ValueError(f"unknown agent kind in {text!r}")
    params: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in rest.split(","))):
        key, _, value = item.partition("=")
        if not value:
            raise ValueError(f"agent parameter {item!r} in {text!r} needs key=value")
        params[key.strip()] = value.strip()
    return AgentSpec(kind=kind, params=params, text=text)


def twoffs_config_from_params(params: dict[str, str]):
    from catanrl.algorithms.twoffs.catan import CatanTwoFFSConfig

    config = CatanTwoFFSConfig()
    rollout = params.get("rollout", "none").lower()
    mapping: dict[str, Any] = {
        "budget": ("budget", float),
        "depth": ("depth", int),
        "leaf": ("fast_bias_leaf", float),
        "slope": ("fast_bias_slope", float),
        "sigma": ("slow_sigma", float),
        "slow_cost": ("initial_slow_cost", float),
        "batch": ("slow_batch", int),
        "delta": ("delta", float),
        "eps": ("epsilon", float),
        "max_nodes": ("max_nodes", int),
        "alpha_power": ("alpha_power", float),
        "fallback": ("fallback", str),
    }
    updates: dict[str, Any] = {}
    for key, value in params.items():
        if key == "rollout":
            continue
        if key not in mapping:
            raise ValueError(f"unknown 2ffs parameter {key!r}")
        name, cast = mapping[key]
        updates[name] = cast(value)
    updates["rollout_decisions"] = None if rollout in {"none", "full", "end"} else int(rollout)
    return replace(config, **updates)


@dataclass
class AgentStats:
    decisions: int = 0
    searched: int = 0
    evaluations: int = 0
    seconds: float = 0.0
    max_depth: int = 0
    disagreements_with_raw: int = 0
    # 2FFS only.
    rollout_evaluations: int = 0
    stopped: int = 0
    slow_queries: int = 0
    fast_queries: int = 0
    nodes: int = 0

    def add(self, other: AgentStats) -> None:
        for name in self.__dataclass_fields__:
            if name == "max_depth":
                self.max_depth = max(self.max_depth, other.max_depth)
            else:
                setattr(self, name, getattr(self, name) + getattr(other, name))

    def summary(self, prefix: str) -> dict[str, float]:
        searched = max(1, self.searched)
        result = {
            f"{prefix}/decisions": float(self.decisions),
            f"{prefix}/searched_decisions": float(self.searched),
            f"{prefix}/evals_per_search": self.evaluations / searched,
            f"{prefix}/seconds_per_search": self.seconds / searched,
        }
        if self.max_depth:
            result[f"{prefix}/max_depth"] = float(self.max_depth)
        if self.fast_queries or self.slow_queries or self.stopped:
            result.update(
                {
                    f"{prefix}/stopped_fraction": self.stopped / searched,
                    f"{prefix}/fast_per_search": self.fast_queries / searched,
                    f"{prefix}/slow_per_search": self.slow_queries / searched,
                    f"{prefix}/rollout_evals_per_search": self.rollout_evaluations / searched,
                    f"{prefix}/nodes_per_search": self.nodes / searched,
                }
            )
        if self.disagreements_with_raw:
            result[f"{prefix}/raw_disagreement"] = self.disagreements_with_raw / searched
        return result


class _Agent:
    def __init__(self, spec: AgentSpec, evaluator, seed: int) -> None:
        self.spec = spec
        self.evaluator = evaluator
        self.seed = seed
        self.full_indices = np.arange(OBSERVATION_SIZE)
        if spec.kind == "2ffs":
            self.twoffs_config = twoffs_config_from_params(spec.params)
        if spec.kind == "mcts":
            self.sims = int(spec.params.get("sims", 256))
            self.c_puct = float(spec.params.get("c_puct", 1.5))

    def _raw_action(self, game, legal: np.ndarray) -> int:
        logits, _ = self.evaluator.evaluate(game.observation(game.current_player)[None])
        return int(legal[np.argmax(logits[0][legal])])

    def act(self, game, legal: np.ndarray, episode_seed: int, decision: int, stats: AgentStats) -> int:
        stats.decisions += 1
        kind = self.spec.kind
        if kind == "value":
            return game.expert_action()
        started = time.perf_counter()
        stats.searched += 1
        if kind == "raw":
            action = self._raw_action(game, legal)
            stats.evaluations += 1
        elif kind == "mcts":
            from catanrl.algorithms.alphazero.native_search import run_native_search_policy

            result = run_native_search_policy(
                game=game,
                inference_backend=self.evaluator,
                actor_indices=self.full_indices,
                critic_indices=self.full_indices,
                num_simulations=self.sims,
                c_puct=self.c_puct,
                search_seed=derive_seed(episode_seed, "matchup_mcts", decision),
                add_noise=False,
                dirichlet_alpha=0.3,
                dirichlet_frac=0.0,
                action_temperature=0.0,
                target_temperature=1.0,
                rng=np.random.default_rng(derive_seed(episode_seed, "matchup_mcts_rng", decision)),
            )
            action = result.action
            stats.evaluations += result.diagnostics.neural_evaluations
            stats.max_depth = max(stats.max_depth, result.diagnostics.maximum_depth)
            if result.diagnostics.prior_top1_action != action:
                stats.disagreements_with_raw += 1
        elif kind == "2ffs":
            from catanrl.algorithms.twoffs.catan import search_catan

            outcome = search_catan(
                game.game.copy(),
                game.space,
                self.evaluator,
                self.twoffs_config,
                np.random.default_rng(derive_seed(episode_seed, "matchup_2ffs", decision)),
            )
            result = outcome.result
            action = outcome.action
            stats.evaluations += outcome.network_evaluations
            stats.rollout_evaluations += outcome.rollout_evaluations
            stats.stopped += int(result.stopped)
            stats.fast_queries += result.num_fast_queries
            stats.slow_queries += result.num_slow_queries
            stats.nodes += result.num_nodes
            stats.max_depth = max(stats.max_depth, result.max_depth)
            # Diagnostic only, outside the search budget.
            if self._raw_action(game, legal) != action:
                stats.disagreements_with_raw += 1
        else:  # pragma: no cover - parse_agent validates kinds
            raise ValueError(kind)
        stats.seconds += time.perf_counter() - started
        return action


def _play_game(
    *,
    agents: Sequence[_Agent],
    a_seat: int,
    episode_seed: int,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], list[AgentStats]]:
    from catanrl.envs.cppanatron import FlatGame

    map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
    game = FlatGame(
        2,
        settings["map_type"],
        seed=game_seed,
        map_seed=map_seed,
        number_placement=settings["number_placement"],
        discard_limit=int(settings["discard_limit"]),
        vps_to_win=int(settings["vps_to_win"]),
    )
    seat_agents = {a_seat: 0, 1 - a_seat: 1}
    stats = [AgentStats(), AgentStats()]
    decision = 0
    while game.winner is None and game.num_turns < int(settings["turns_limit"]):
        legal = np.flatnonzero(game.action_mask())
        if legal.size == 0:
            raise RuntimeError("native matchup reached a position without legal actions")
        if legal.size == 1:
            game.step(int(legal[0]))
            continue
        agent_index = seat_agents[game.current_player]
        action = agents[agent_index].act(game, legal, episode_seed, decision, stats[agent_index])
        game.step(action)
        decision += 1
    winner = game.winner
    return (
        {
            "episode_seed": episode_seed,
            "a_seat": a_seat,
            "seat": "first" if a_seat == 0 else "second",
            "winner": winner,
            "win": winner == a_seat,
            "draw": winner is None,
            "turns": game.num_turns,
            "a_vps": game.player(a_seat).actual_victory_points,
            "b_vps": game.player(1 - a_seat).actual_victory_points,
        },
        stats,
    )


def _worker_main(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    scenarios: Sequence[tuple[int, int]],
    agent_texts: tuple[str, str],
    settings: dict[str, Any],
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        from catanrl.eval.batched_inference import RemoteEvaluator

        evaluator = RemoteEvaluator(
            worker_id,
            request_queue,
            response_queue,
            OBSERVATION_SIZE,
            value_transform=settings["value_transform"],
        )
        agents = [
            _Agent(parse_agent(text), evaluator, derive_seed(settings["seed"], "agent", index))
            for index, text in enumerate(agent_texts)
        ]
        for a_seat, episode_seed in scenarios:
            record, stats = _play_game(
                agents=agents, a_seat=a_seat, episode_seed=episode_seed, settings=settings
            )
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "record": record,
                    "stats": [vars(item) for item in stats],
                }
            )
        result_queue.put({"worker_id": worker_id, "done": True})
    except BaseException:
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})


@dataclass
class MatchupResult:
    agent_a: str
    agent_b: str
    records: list[dict[str, Any]] = field(default_factory=list)
    stats_a: AgentStats = field(default_factory=AgentStats)
    stats_b: AgentStats = field(default_factory=AgentStats)
    seconds: float = 0.0
    server_rows: int = 0
    server_batches: int = 0

    def summary(self) -> dict[str, float]:
        from catanrl.eval.reporting import wilson_interval

        games = len(self.records)
        wins = sum(bool(record["win"]) for record in self.records)
        first = [record for record in self.records if record["a_seat"] == 0]
        second = [record for record in self.records if record["a_seat"] == 1]
        low, high = wilson_interval(wins, games) if games else (0.0, 0.0)
        result = {
            "games": float(games),
            "win_rate": wins / games if games else 0.0,
            "win_rate_ci95_low": low,
            "win_rate_ci95_high": high,
            "first_win_rate": sum(r["win"] for r in first) / len(first) if first else 0.0,
            "second_win_rate": sum(r["win"] for r in second) / len(second) if second else 0.0,
            "draw_rate": sum(r["draw"] for r in self.records) / games if games else 0.0,
            "mean_turns": sum(r["turns"] for r in self.records) / games if games else 0.0,
            "wall_seconds": self.seconds,
            "server_rows_per_second": self.server_rows / self.seconds if self.seconds else 0.0,
            "server_mean_batch_rows": self.server_rows / max(1, self.server_batches),
        }
        result.update(self.stats_a.summary("a"))
        result.update(self.stats_b.summary("b"))
        return result


def run_matchup(
    *,
    policy_model,
    critic_model,
    model_type: str,
    device: str,
    agent_a: str,
    agent_b: str,
    games_per_seat: int,
    num_workers: int,
    seed: int,
    settings: dict[str, Any],
    max_rows: int = 2048,
    max_wait_ms: float = 1.0,
    progress: bool = True,
    on_record=None,
) -> MatchupResult:
    from tqdm import tqdm

    from catanrl.eval.batched_inference import BatchInferenceServer

    parse_agent(agent_a)
    parse_agent(agent_b)
    scenarios = [
        (seat, derive_seed(seed, "search_matchup_episode", game_index))
        for game_index in range(games_per_seat)
        for seat in range(2)
    ]
    num_workers = max(1, min(num_workers, len(scenarios)))
    assignments = [scenarios[index::num_workers] for index in range(num_workers)]
    settings = {**settings, "seed": seed}

    context = mp.get_context("spawn")
    request_queue: mp.Queue = context.Queue()
    response_queues = [context.Queue() for _ in range(num_workers)]
    result_queue: mp.Queue = context.Queue()
    server = BatchInferenceServer(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=device,
        observation_size=OBSERVATION_SIZE,
        actor_indices=None,
        critic_indices=None,
        request_queue=request_queue,
        response_queues=response_queues,
        max_rows=max_rows,
        max_wait_ms=max_wait_ms,
    )
    processes = [
        context.Process(
            target=_worker_main,
            args=(
                worker_id,
                request_queue,
                response_queues[worker_id],
                result_queue,
                assignments[worker_id],
                (agent_a, agent_b),
                settings,
            ),
            daemon=True,
        )
        for worker_id in range(num_workers)
    ]
    result = MatchupResult(agent_a=agent_a, agent_b=agent_b)
    server.start()
    server.ready.wait()
    if server.startup_error:
        print(server.startup_error, flush=True)
    started = time.perf_counter()
    error: str | None = None
    bar = tqdm(total=len(scenarios), disable=not progress, desc=f"{agent_a} vs {agent_b}")
    try:
        for process in processes:
            process.start()
        pending = set(range(num_workers))
        while pending:
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                for worker_id in list(pending):
                    if processes[worker_id].exitcode is not None:
                        error = f"worker {worker_id} exited with {processes[worker_id].exitcode}"
                        pending.discard(worker_id)
                if error:
                    break
                continue
            worker_id = int(message["worker_id"])
            if "error" in message:
                error = message["error"]
                break
            if message.get("done"):
                pending.discard(worker_id)
                continue
            result.records.append(message["record"])
            result.stats_a.add(AgentStats(**message["stats"][0]))
            result.stats_b.add(AgentStats(**message["stats"][1]))
            bar.update(1)
            if on_record is not None:
                on_record(result)
    finally:
        bar.close()
        for process in processes:
            if error is not None and process.is_alive():
                process.terminate()
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
        server.stop()
    result.seconds = time.perf_counter() - started
    result.server_rows = server.rows_evaluated
    result.server_batches = server.batches
    if error is not None:
        raise RuntimeError(f"search matchup worker failed:\n{error}")
    return result


__all__ = [
    "AgentSpec",
    "AgentStats",
    "MatchupResult",
    "parse_agent",
    "run_matchup",
    "twoffs_config_from_params",
]
