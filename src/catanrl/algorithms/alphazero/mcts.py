from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, to_action_space
from typing import Optional, Dict, Tuple, List
from catanatron.models.enums import Action
import numpy as np
import math

from .trainer import AlphaZeroTrainer

class MCTSNode:
    """Single node in the neural-guided Monte Carlo Tree Search."""

    def __init__(
        self,
        game: Game,
        to_play: Color,
        parent: Optional["MCTSNode"],
        prior: float,
    ):
        self.game = game
        self.to_play = to_play
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.priors: Dict[int, float] = {}
        self.action_map: Dict[int, Action] = {}
        self.children: Dict[int, "MCTSNode"] = {}

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def is_terminal(self) -> bool:
        return self.game.winning_color() is not None

    def expanded(self) -> bool:
        return len(self.priors) > 0

    def expand(self, action_priors: np.ndarray) -> None:
        if self.expanded():
            return
        valid_actions = self.game.state.playable_actions
        if not valid_actions:
            return

        priors = action_priors.copy()
        total = priors.sum()
        if total <= 0:
            uniform = 1.0 / len(valid_actions)
            for action in valid_actions:
                idx = to_action_space(action)
                self.priors[idx] = uniform
                self.action_map[idx] = action
        else:
            for action in valid_actions:
                idx = to_action_space(action)
                self.priors[idx] = float(priors[idx])
                self.action_map[idx] = action
            norm = sum(self.priors.values())
            if norm > 0:
                for idx in list(self.priors.keys()):
                    self.priors[idx] /= norm

    def add_exploration_noise(self, alpha: float, frac: float) -> None:
        if not self.priors:
            return
        action_indices = list(self.priors.keys())
        noise = np.random.dirichlet([alpha] * len(action_indices))
        for idx, noise_val in zip(action_indices, noise):
            self.priors[idx] = self.priors[idx] * (1 - frac) + noise_val * frac

    def select_child(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        assert self.priors, "Cannot select a child from an unexpanded node."
        best_score = -float("inf")
        best_action = -1
        exploration = math.sqrt(self.visit_count + 1)

        for action_idx, prior in self.priors.items():
            child = self.children.get(action_idx)
            child_visits = child.visit_count if child else 0
            child_value = child.value() if child else 0.0
            ucb = child_value + c_puct * prior * exploration / (1 + child_visits)
            if ucb > best_score:
                best_score = ucb
                best_action = action_idx

        if best_action not in self.children:
            next_game = self.game.copy()
            next_game.execute(self.action_map[best_action])
            next_player = next_game.state.current_color()
            self.children[best_action] = MCTSNode(
                game=next_game,
                to_play=next_player,
                parent=self,
                prior=self.priors[best_action],
            )
        return best_action, self.children[best_action]


class NeuralMCTS:
    """PUCT search that leverages the shared policy/value network."""

    def __init__(self, trainer: AlphaZeroTrainer):
        self.trainer = trainer

    def search(
        self,
        game: Game,
        root_features: np.ndarray,
        temperature: float,
        add_noise: bool,
    ) -> Tuple[np.ndarray, Action]:
        root_game = game.copy()
        root = MCTSNode(
            game=root_game,
            to_play=root_game.state.current_color(),
            parent=None,
            prior=1.0,
        )
        priors, _ = self.trainer._policy_value(
            root.game,
            root.to_play,
            cached_features=root_features,
        )
        root.expand(priors)
        if add_noise:
            root.add_exploration_noise(self.trainer.config.dirichlet_alpha, self.trainer.config.dirichlet_frac)

        for _ in range(self.trainer.config.simulations):
            node = root
            search_path = [node]
            while node.expanded() and not node.is_terminal():
                _, node = node.select_child(self.trainer.config.c_puct)
                search_path.append(node)

            value = self._evaluate(node)
            self._backpropagate(search_path, value)

        policy = self._build_policy_vector(root, temperature)
        action_idx = self._sample_action(root, policy)
        action = root.action_map[action_idx]
        return policy, action

    def _evaluate(self, node: MCTSNode) -> float:
        winner = node.game.winning_color()
        if winner is not None:
            if winner == node.to_play:
                return 1.0
            return -1.0

        priors, value = self.trainer._policy_value(node.game, node.to_play)
        node.expand(priors)
        return value

    def _backpropagate(self, search_path: List[MCTSNode], value: float) -> None:
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _build_policy_vector(self, root: MCTSNode, temperature: float) -> np.ndarray:
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if not root.priors:
            return policy

        visits = []
        action_indices = []
        for action_idx in root.priors.keys():
            child_visits = root.children[action_idx].visit_count if action_idx in root.children else 0
            visits.append(child_visits)
            action_indices.append(action_idx)

        visits_arr = np.array(visits, dtype=np.float32)
        if temperature <= 1e-3:
            best = action_indices[int(visits_arr.argmax())]
            policy[best] = 1.0
            return policy

        adjusted = visits_arr ** (1.0 / temperature)
        if adjusted.sum() <= 0:
            adjusted = np.ones_like(adjusted) / len(adjusted)
        else:
            adjusted /= adjusted.sum()

        for idx, prob in zip(action_indices, adjusted):
            policy[idx] = prob
        return policy

    def _sample_action(
        self,
        root: MCTSNode,
        policy: np.ndarray,
    ) -> int:
        candidates = [idx for idx in root.priors.keys() if policy[idx] > 0]
        if not candidates:
            candidates = list(root.priors.keys())
            probs = np.array([root.priors[idx] for idx in candidates], dtype=np.float64)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        else:
            probs = np.array([policy[idx] for idx in candidates], dtype=np.float64)
            probs /= probs.sum()

        choice = int(np.random.choice(candidates, p=probs))
        return choice