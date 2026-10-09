"""2FFS over cppanatron positions.

The tree alternates the root player's decisions (``MAX``) and the opponent's
(``MIN``), with ``CHANCE`` nodes for dice rolls, development-card draws and
robber steals. Their outcome distributions mirror the native MCTS
(``action_outcomes`` in ``cppanatron/src/mcts.cpp``): rolls by dice total, cards
by the remaining deck composition, steals by the victim's hand. Single legal
deterministic moves are played through without spending depth, and a forced
stochastic move above the frontier becomes a chance node.

Fidelities, both from the root player's perspective in ``[-1, 1]``:

* fast: the critic of the frozen policy/critic pair, with bias envelope
  ``B(h) = fast_bias_leaf + fast_bias_slope * h`` for remaining depth ``h``
  (the paper's depth-shrinking envelope, Remark 2.2 / Theorem 3.8);
* slow: argmax-policy self-play rollouts with sampled chance outcomes, either
  to the end of the game (unbiased for the policy's value) or truncated after
  ``rollout_decisions`` network decisions and bootstrapped by the critic.

Costs are counted in network state evaluations, so a fast query costs 1 and a
slow sample costs the number of positions its rollout evaluated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from cppanatron.catanatron import observation_layout

import cppanatron as cpp
from catanrl.beliefs.native_dev_card_belief import development_deck_counts

from .core import CHANCE, LEAF, MAX, MIN, ChildSpec, TwoFFS, TwoFFSConfig, TwoFFSResult

_DICE_MULTIPLICITIES = (1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1)
_DICE_OUTCOMES = tuple(
    ((1, total - 1) if total <= 7 else (6, total - 6), multiplicity / 36.0)
    for total, multiplicity in zip(range(2, 13), _DICE_MULTIPLICITIES)
)
# DevelopmentCard order, matching development_deck_counts: knight, year of
# plenty, monopoly, road building, victory point.
_DEV_CARDS = tuple(cpp.DevelopmentCard.__members__.values())
_RESOURCES = tuple(cpp.Resource.__members__.values())

_ROLL = cpp.ActionType.ROLL
_BUY_DEV = cpp.ActionType.BUY_DEVELOPMENT_CARD
_MOVE_ROBBER = cpp.ActionType.MOVE_ROBBER


def chance_outcomes(
    game: cpp.Game, action: cpp.Action
) -> list[tuple[dict[str, Any], float]] | None:
    """``(execute kwargs, probability)`` per outcome, or ``None`` if deterministic."""
    action_type = action.action_type
    if action_type == _ROLL:
        return [({"dice": dice}, probability) for dice, probability in _DICE_OUTCOMES]
    if action_type == _BUY_DEV:
        counts = development_deck_counts(game)
        total = float(sum(counts))
        return [
            ({"development_card": card}, count / total)
            for card, count in zip(_DEV_CARDS, counts)
            if count > 0
        ]
    if action_type == _MOVE_ROBBER:
        victim = action.value[1]
        if victim is None:
            return None
        resources = game.player(victim).resources
        total = float(sum(resources))
        if total <= 0:
            return None
        return [
            ({"stolen_resource": resource}, count / total)
            for resource, count in zip(_RESOURCES, resources)
            if count > 0
        ]
    return None


def sample_chance(
    game: cpp.Game, action: cpp.Action, rng: np.random.Generator
) -> dict[str, Any]:
    """Execute kwargs for one sampled chance outcome (empty if deterministic).

    Rollouts must force outcomes from the search's own generator: copies of a
    game share its engine RNG state, so unforced copies would replay the real
    game's future dice.
    """
    action_type = action.action_type
    if action_type == _ROLL:
        return {"dice": (int(rng.integers(1, 7)), int(rng.integers(1, 7)))}
    if action_type == _BUY_DEV:
        counts = development_deck_counts(game)
        pick = int(rng.integers(sum(counts)))
        for card, count in zip(_DEV_CARDS, counts):
            if pick < count:
                return {"development_card": card}
            pick -= count
        raise AssertionError("unreachable")
    if action_type == _MOVE_ROBBER:
        victim = action.value[1]
        if victim is None:
            return {}
        resources = game.player(victim).resources
        total = sum(resources)
        if total <= 0:
            return {}
        pick = int(rng.integers(total))
        for resource, count in zip(_RESOURCES, resources):
            if pick < count:
                return {"stolen_resource": resource}
            pick -= count
        raise AssertionError("unreachable")
    return {}


class PositionEvaluator(Protocol):
    """Batched frozen-network inference on full observations."""

    def evaluate(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(policy logits [n, actions], values [n])``, values zero-sum for the observer."""


@dataclass(frozen=True)
class CatanTwoFFSConfig:
    depth: int = 3
    fast_bias_leaf: float = 0.1
    fast_bias_slope: float = 0.05
    # None plays rollouts to the end of the game.
    rollout_decisions: int | None = None
    slow_sigma: float = 1.0
    initial_slow_cost: float = 100.0
    turns_limit: int = 1000
    delta: float = 0.05
    epsilon: float = 0.0
    budget: float = 1024.0
    slow_batch: int = 1
    max_nodes: int = 200_000
    confidence_nodes: int = 200_000
    alpha_power: float = 2.0
    latch_all: bool = False
    fallback: str = "leader"


class CatanOracle:
    """Lazily expanded Catan tree with critic and rollout fidelities."""

    fast_query_cost = 1.0

    def __init__(
        self,
        game: cpp.Game,
        space: cpp.FlatActionSpace,
        evaluator: PositionEvaluator,
        config: CatanTwoFFSConfig,
        rng: np.random.Generator,
    ) -> None:
        self.game = game
        self.space = space
        self.evaluator = evaluator
        self.config = config
        self.rng = rng
        self.root_player = game.current_player_index
        self.layout = observation_layout()
        self.observation_size = cpp.full_observation_size(game.num_players, self.layout)
        self.slow_sigma = float(config.slow_sigma)
        self._slow_cost_total = 0.0
        self._slow_samples = 0
        self.network_evaluations = 0
        self.rollout_evaluations = 0
        self._root_logits: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Tree generation
    # ------------------------------------------------------------------

    def root(self) -> tuple[Any, int]:
        return self.game, int(self.config.depth)

    def _terminal_value(self, game: cpp.Game) -> float | None:
        winner = game.winning_player_index
        if winner is not None:
            return 1.0 if winner == self.root_player else -1.0
        if game.num_turns >= self.config.turns_limit:
            return 0.0
        return None

    def _state_spec(
        self, game: cpp.Game, remaining_depth: int, probability: float, key: Any
    ) -> ChildSpec:
        """Play through forced deterministic moves, then classify the position.

        A forced stochastic move (typically a roll) stays a state node with its
        own fast value; expanding it reveals the chance node without spending
        depth, so outcomes are only paid for where the search looks.
        """
        while True:
            value = self._terminal_value(game)
            if value is not None:
                return ChildSpec(LEAF, remaining_depth, game, probability, value, key=key)
            legal = np.flatnonzero(self.space.mask(game))
            if legal.size != 1:
                break
            action = self.space.decode(int(legal[0]), game)
            if chance_outcomes(game, action) is not None:
                break
            game.execute(action)
        kind = MAX if game.current_player_index == self.root_player else MIN
        return ChildSpec(kind, remaining_depth, game, probability, key=key)

    def _chance_spec(
        self,
        game: cpp.Game,
        action: cpp.Action,
        outcomes: list[tuple[dict[str, Any], float]],
        remaining_depth: int,
        probability: float,
        key: Any,
    ) -> ChildSpec:
        children = []
        for kwargs, outcome_probability in outcomes:
            child = game.copy()
            child.execute(action, **kwargs)
            children.append(self._state_spec(child, remaining_depth, outcome_probability, None))
        return ChildSpec(
            CHANCE,
            remaining_depth,
            (game, action, outcomes),
            probability,
            outcomes=children,
            key=key,
        )

    def expand(self, payload: Any, kind: int, remaining_depth: int) -> list[ChildSpec]:
        game: cpp.Game = payload
        legal = np.flatnonzero(self.space.mask(game))
        # A forced move is not a decision and does not consume search depth.
        child_depth = remaining_depth if legal.size == 1 else remaining_depth - 1
        specs = []
        for index in legal:
            index = int(index)
            action = self.space.decode(index, game)
            outcomes = chance_outcomes(game, action)
            if outcomes is None:
                child = game.copy()
                child.execute(action)
                specs.append(self._state_spec(child, child_depth, 1.0, index))
            else:
                specs.append(self._chance_spec(game, action, outcomes, child_depth, 1.0, index))
        return specs

    # ------------------------------------------------------------------
    # Fidelities
    # ------------------------------------------------------------------

    def fast_bound(self, kind: int, remaining_depth: int) -> float:
        return self.config.fast_bias_leaf + self.config.fast_bias_slope * max(0, remaining_depth)

    def _observations(self, games: Sequence[cpp.Game]) -> np.ndarray:
        observations = np.empty((len(games), self.observation_size), dtype=np.float32)
        for row, game in enumerate(games):
            cpp.write_full_observation(game, game.current_player_index, self.layout, observations[row])
        return observations

    def _evaluate(self, games: Sequence[cpp.Game]) -> tuple[np.ndarray, np.ndarray]:
        logits, values = self.evaluator.evaluate(self._observations(games))
        self.network_evaluations += len(games)
        return logits, values

    def _root_values(self, games: Sequence[cpp.Game], values: np.ndarray) -> list[float]:
        return [
            float(value) if game.current_player_index == self.root_player else -float(value)
            for game, value in zip(games, np.clip(values, -1.0, 1.0))
        ]

    def fast_evaluate(self, specs: Sequence[ChildSpec]) -> list[tuple[float, float]]:
        games = [spec.payload for spec in specs]
        _, values = self._evaluate(games)
        return [
            (value, self.fast_bound(spec.kind, spec.remaining_depth))
            for spec, value in zip(specs, self._root_values(games, values))
        ]

    def prior_scores(self, keys: Sequence[int]) -> list[float]:
        """Root policy logits for flat actions ``keys`` (one network evaluation)."""
        if self._root_logits is None:
            logits, _ = self._evaluate([self.game])
            self._root_logits = logits[0]
        return [float(self._root_logits[int(key)]) for key in keys]

    def slow_cost_estimate(self) -> float:
        if self._slow_samples == 0:
            return float(self.config.initial_slow_cost)
        return self._slow_cost_total / self._slow_samples

    def slow_sample(self, payload: Any, kind: int, count: int) -> tuple[list[float], float]:
        starts = []
        for _ in range(count):
            if kind == CHANCE:
                game, action, outcomes = payload
                pick = self.rng.choice(len(outcomes), p=[p for _, p in outcomes])
                start = game.copy()
                start.execute(action, **outcomes[int(pick)][0])
            else:
                start = payload.copy()
            starts.append(start)
        before = self.network_evaluations
        values = self.rollout(starts)
        cost = float(self.network_evaluations - before)
        self.rollout_evaluations += int(cost)
        self._slow_cost_total += cost
        self._slow_samples += count
        return values, cost

    def rollout(self, games: list[cpp.Game]) -> list[float]:
        """Argmax-policy self-play from each game (mutated), with sampled chance."""
        results: list[float | None] = [None] * len(games)
        decisions = [0] * len(games)
        limit = self.config.rollout_decisions
        active = list(range(len(games)))
        while active:
            waiting = []
            for index in active:
                game = games[index]
                while True:
                    value = self._terminal_value(game)
                    if value is not None:
                        results[index] = value
                        break
                    if limit is not None and decisions[index] >= limit:
                        break
                    legal = np.flatnonzero(self.space.mask(game))
                    if legal.size != 1:
                        waiting.append(index)
                        break
                    action = self.space.decode(int(legal[0]), game)
                    game.execute(action, **sample_chance(game, action, self.rng))
            waiting_set = set(waiting)
            truncated = [
                index
                for index in active
                if results[index] is None and index not in waiting_set
            ]
            if truncated:
                bootstrap_games = [games[index] for index in truncated]
                _, values = self._evaluate(bootstrap_games)
                for index, value in zip(truncated, self._root_values(bootstrap_games, values)):
                    results[index] = value
            if waiting:
                deciding = [games[index] for index in waiting]
                logits, _ = self._evaluate(deciding)
                for index, game, row in zip(waiting, deciding, logits):
                    mask = self.space.mask(game)
                    masked = np.where(mask, row, -np.inf)
                    action = self.space.decode(int(np.argmax(masked)), game)
                    game.execute(action, **sample_chance(game, action, self.rng))
                    decisions[index] += 1
            active = waiting
        return [float(value) for value in results]


@dataclass(frozen=True)
class CatanSearchOutcome:
    action: int
    result: TwoFFSResult
    network_evaluations: int
    rollout_evaluations: int


def search_catan(
    game: cpp.Game,
    space: cpp.FlatActionSpace,
    evaluator: PositionEvaluator,
    config: CatanTwoFFSConfig,
    rng: np.random.Generator,
) -> CatanSearchOutcome:
    """Run 2FFS at ``game`` (which must have at least two legal actions)."""
    oracle = CatanOracle(game, space, evaluator, config, rng)
    search = TwoFFS(
        oracle,
        TwoFFSConfig(
            delta=config.delta,
            epsilon=config.epsilon,
            confidence_nodes=config.confidence_nodes,
            value_range=(-1.0, 1.0),
            slow_batch=config.slow_batch,
            budget=config.budget,
            max_nodes=config.max_nodes,
            alpha_power=config.alpha_power,
            latch_all=config.latch_all,
            fallback=config.fallback,  # type: ignore[arg-type]
        ),
    )
    result = search.run()
    if result.recommendation_key is None:
        raise RuntimeError("2FFS returned no recommendation")
    return CatanSearchOutcome(
        action=int(result.recommendation_key),
        result=result,
        network_evaluations=oracle.network_evaluations,
        rollout_evaluations=oracle.rollout_evaluations,
    )


__all__ = [
    "CatanOracle",
    "CatanSearchOutcome",
    "CatanTwoFFSConfig",
    "PositionEvaluator",
    "chance_outcomes",
    "sample_chance",
    "search_catan",
]
