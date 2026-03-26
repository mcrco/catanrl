from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
from catanatron.features import create_sample
from catanatron.game import Game
from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY, normalize_action
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import DEVELOPMENT_CARDS
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.tree_search_utils import execute_spectrum, list_prunned_actions
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catanrl.features.catanatron_utils import (
    full_game_to_features,
    game_to_features,
    get_full_numeric_feature_names,
)
from catanrl.models import PolicyNetworkWrapper, ValueNetworkWrapper

EPSILON = 1e-8
CRITIC_MODE_ALIASES = {
    "full": "full",
    "guess": "guessed_dev_cards",
    "guessed": "guessed_dev_cards",
    "guessed_dev_cards": "guessed_dev_cards",
}


@dataclass
class _Node:
    game: Game
    parent: "_Node | None"
    prior_action: object | None = None
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self):
        self.children: dict[object, list[tuple["_Node", float]]] = defaultdict(list)
        self.action_priors: dict[object, float] = {}

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class NNMCTSPlayer(Player):
    """
    MCTS player guided by policy priors and critic value estimates.
    """

    def __init__(
        self,
        color: Color,
        model_type: str,
        policy_model: PolicyNetworkWrapper,
        critic_model: ValueNetworkWrapper,
        map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE",
        num_simulations: int = 64,
        c_puct: float = 1.5,
        prunning: bool = False,
        opponent_policy: str = "self",
        opponent_factory: Callable[[Color], Player] | None = None,
        critic_mode: Literal["full", "guessed_dev_cards"] = "full",
        **kwargs,
    ):
        super().__init__(color, is_bot=True, **kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.map_type = map_type
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.prunning = bool(prunning)
        self.opponent_policy = opponent_policy.lower()
        self.opponent_factory = opponent_factory
        self.critic_mode = self._normalize_critic_mode(critic_mode)
        self._opponents_by_color: dict[Color, Player] = {}
        self._critic_index_cache: dict[int, dict[str, int]] = {}
        self._starting_dev_counter = Counter(starting_devcard_bank())

        if self.opponent_factory is None and self.opponent_policy != "self":
            self.opponent_factory = self._build_opponent_factory(self.opponent_policy)
        self.policy_model = policy_model.to(self.device)
        self.critic_model = critic_model.to(self.device)
        self.policy_model.eval()
        self.critic_model.eval()

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        root = _Node(game=game.copy(), parent=None)
        self._expand(root)

        if not root.children:
            return playable_actions[0]

        for _ in range(self.num_simulations):
            self._run_simulation(root)

        best_action = max(
            root.children.keys(),
            key=lambda action: sum(child.visits for child, _ in root.children[action]),
        )
        return self._resolve_root_action(best_action, playable_actions)

    def _run_simulation(self, root: _Node) -> None:
        node = root
        path = [node]

        while node.expanded and node.game.winning_color() is None:
            node = self._select(node)
            path.append(node)

        if node.game.winning_color() is not None:
            winner = node.game.winning_color()
            leaf_value = 1.0 if winner == self.color else -1.0
        else:
            self._expand(node)
            leaf_value = self._critic_value(node.game)

        for path_node in reversed(path):
            path_node.visits += 1
            path_node.value_sum += leaf_value

    def _expand(self, node: _Node) -> None:
        if node.game.winning_color() is not None or node.expanded:
            return

        actions = (
            list_prunned_actions(node.game)
            if self.prunning
            else list(node.game.state.playable_actions)
        )
        if not actions:
            return

        current_color = node.game.state.current_color()
        if current_color != self.color and self.opponent_factory is not None:
            opponent_action = self._opponent_action(node.game, actions)
            node.action_priors = {opponent_action: 1.0}
            actions = [opponent_action]
        else:
            node.action_priors = self._policy_priors(node.game, current_color, actions)

        for action in actions:
            outcomes = execute_spectrum(node.game, action)
            for next_game, probability in outcomes:
                node.children[action].append((_Node(game=next_game, parent=node), probability))

    def _opponent_action(self, game, actions):
        if self.opponent_factory is None:
            raise RuntimeError("Opponent factory is not configured.")
        color = game.state.current_color()
        opponent = self._opponents_by_color.get(color)
        if opponent is None:
            opponent = self.opponent_factory(color)
            self._opponents_by_color[color] = opponent
        return opponent.decide(game, actions)

    @staticmethod
    def _build_opponent_factory(policy: str) -> Callable[[Color], Player]:
        name = policy.lower()
        if name in {"random", "r"}:
            return lambda color: RandomPlayer(color)
        if name in {"weighted", "w"}:
            return lambda color: WeightedRandomPlayer(color)
        if name in {"value", "valuefunction", "f"}:
            return lambda color: ValueFunctionPlayer(color)
        if name in {"victorypoint", "vp"}:
            return lambda color: VictoryPointPlayer(color)
        if name in {"alphabeta", "ab"}:
            return lambda color: AlphaBetaPlayer(color, depth=2, prunning=True)
        if name in {"sameturnalphabeta", "sab"}:
            return lambda color: SameTurnAlphaBetaPlayer(color)
        if name in {"mcts", "m"}:
            return lambda color: MCTSPlayer(color, 100)
        if name in {"playouts", "greedyplayouts", "g"}:
            return lambda color: GreedyPlayoutsPlayer(color, 25)
        raise ValueError(
            f"Unknown opponent_policy '{policy}'. "
            "Use 'self' or one of: random, weighted, value, victorypoint, alphabeta, "
            "sameturnalphabeta, mcts, playouts."
        )

    @staticmethod
    def _normalize_critic_mode(critic_mode: str) -> str:
        normalized = CRITIC_MODE_ALIASES.get(critic_mode.lower())
        if normalized is None:
            valid_modes = ", ".join(sorted(CRITIC_MODE_ALIASES))
            raise ValueError(f"Unknown critic_mode '{critic_mode}'. Use one of: {valid_modes}.")
        return normalized

    def _select(self, node: _Node) -> _Node:
        total_visits = sum(child.visits for kids in node.children.values() for child, _ in kids)
        maximizing = node.game.state.current_color() == self.color
        best_score = -float("inf")
        best_action = None

        for action, children in node.children.items():
            action_visits = sum(child.visits for child, _ in children)
            if action_visits > 0:
                q_value = sum(probability * child.value for child, probability in children) / sum(
                    probability for _, probability in children
                )
            else:
                q_value = 0.0

            if not maximizing:
                q_value = -q_value

            prior = node.action_priors.get(action, 0.0)
            u_value = self.c_puct * prior * math.sqrt(total_visits + 1.0) / (1.0 + action_visits)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action

        assert best_action is not None
        children = node.children[best_action]
        outcomes = [child for child, _ in children]
        probs = [probability for _, probability in children]
        return random.choices(outcomes, weights=probs, k=1)[0]

    def _policy_priors(self, game, color: Color, actions) -> dict[object, float]:
        with torch.no_grad():
            actor_features = game_to_features(
                game,
                color,
                len(game.state.colors),
                self.map_type,
            ).reshape(1, -1)
            actor_tensor = torch.from_numpy(actor_features).to(self.device)
            if self.model_type == "flat":
                logits = self.policy_model(actor_tensor)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits = self.policy_model(actor_tensor)
                logits = self.policy_model.get_flat_action_logits(action_type_logits, param_logits)
            else:
                raise ValueError(f"Unknown model_type '{self.model_type}'")
            logits_np = logits.squeeze(0).detach().cpu().numpy()

        normalized_actions = [normalize_action(action) for action in actions]
        possibilities = [(action.action_type, action.value) for action in normalized_actions]
        indices = [ACTIONS_ARRAY.index(possibility) for possibility in possibilities]
        action_logits = np.array([logits_np[idx] for idx in indices], dtype=np.float32)
        action_logits = action_logits - float(np.max(action_logits))
        exp_logits = np.exp(action_logits)
        probs = exp_logits / max(float(exp_logits.sum()), EPSILON)
        return {action: float(prob) for action, prob in zip(actions, probs)}

    def _critic_value(self, game) -> float:
        with torch.no_grad():
            if self.critic_mode == "full":
                critic_features = full_game_to_features(
                    game,
                    len(game.state.colors),
                    self.map_type,
                    base_color=self.color,
                )
            else:
                critic_features = self._guessed_dev_cards_features(game)
            critic_features = critic_features.reshape(1, -1)
            critic_tensor = torch.from_numpy(critic_features).to(self.device)
            value = float(self.critic_model(critic_tensor).squeeze(0).item())
        return float(np.clip(value, -1.0, 1.0))

    def _guessed_dev_cards_features(self, game) -> np.ndarray:
        num_players = len(game.state.colors)
        features = full_game_to_features(
            game,
            num_players,
            self.map_type,
            base_color=self.color,
        ).astype(np.float32, copy=True)
        sample = create_sample(game, self.color)
        dev_guess = self._sample_opponent_dev_cards(sample, num_players)
        numeric_name_to_index = self._get_full_numeric_index(num_players)

        for player_idx, card_counter in dev_guess.items():
            vp_feature = f"P{player_idx}_VICTORY_POINT_IN_HAND"
            old_vp = float(features[numeric_name_to_index[vp_feature]])
            new_vp = float(card_counter.get("VICTORY_POINT", 0))
            features[numeric_name_to_index[vp_feature]] = new_vp
            for dev_card in DEVELOPMENT_CARDS:
                feature_name = f"P{player_idx}_{dev_card}_IN_HAND"
                features[numeric_name_to_index[feature_name]] = float(card_counter.get(dev_card, 0))
            actual_vps_feature = f"P{player_idx}_ACTUAL_VPS"
            if actual_vps_feature in numeric_name_to_index:
                features[numeric_name_to_index[actual_vps_feature]] += new_vp - old_vp
        return features

    def _sample_opponent_dev_cards(self, sample, num_players: int) -> dict[int, Counter]:
        known_cards = Counter()
        for dev_card in DEVELOPMENT_CARDS:
            known_cards[dev_card] += int(sample.get(f"P0_{dev_card}_IN_HAND", 0))
            for player_idx in range(num_players):
                known_cards[dev_card] += int(sample.get(f"P{player_idx}_{dev_card}_PLAYED", 0))

        unseen = self._starting_dev_counter - known_cards
        unseen_cards = [card for card, count in unseen.items() for _ in range(max(0, int(count)))]
        random.shuffle(unseen_cards)

        opponent_card_counts = {
            player_idx: int(sample.get(f"P{player_idx}_NUM_DEVS_IN_HAND", 0))
            for player_idx in range(1, num_players)
        }
        guessed: dict[int, Counter] = {}
        cursor = 0
        for player_idx in range(1, num_players):
            num_cards = max(0, opponent_card_counts[player_idx])
            draw = unseen_cards[cursor : cursor + num_cards]
            cursor += num_cards
            hand_counter = Counter(draw)
            if len(draw) < num_cards:
                self._fill_missing_dev_cards(hand_counter, num_cards - len(draw), unseen)
            guessed[player_idx] = hand_counter
        return guessed

    @staticmethod
    def _fill_missing_dev_cards(hand_counter: Counter, missing: int, unseen: Counter) -> None:
        if missing <= 0:
            return
        fallback_pool = [card for card, count in unseen.items() for _ in range(max(0, int(count)))]
        if not fallback_pool:
            fallback_pool = ["KNIGHT"]
        for _ in range(missing):
            hand_counter[random.choice(fallback_pool)] += 1

    def _get_full_numeric_index(self, num_players: int) -> dict[str, int]:
        index = self._critic_index_cache.get(num_players)
        if index is not None:
            return index
        numeric_names = get_full_numeric_feature_names(num_players, self.map_type)
        index = {name: idx for idx, name in enumerate(numeric_names)}
        self._critic_index_cache[num_players] = index
        return index

    @staticmethod
    def _resolve_root_action(best_action, playable_actions):
        best_normalized = normalize_action(best_action)
        best_tuple = (best_normalized.action_type, best_normalized.value)
        for action in playable_actions:
            normalized_action = normalize_action(action)
            if (normalized_action.action_type, normalized_action.value) == best_tuple:
                return action
        raise ValueError("NN MCTS Player chose unavailable action.")

    def __repr__(self) -> str:
        return super().__repr__().replace("Player", "NNMCTSPlayer")


class NNMCTSGuessedDevCardsPlayer(NNMCTSPlayer):
    """
    NNMCTS player that hides exact opponent dev-card types from the critic.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("critic_mode", "guessed_dev_cards")
        super().__init__(*args, **kwargs)
