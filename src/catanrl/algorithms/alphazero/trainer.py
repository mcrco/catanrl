"""
Placeholder AlphaZero trainer definitions.

The goal is to make it easy to plug in real AlphaZero-style training later by
defining a consistent interface today.
"""

from __future__ import annotations

from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.game import Game
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, to_action_space
from catanatron.models.enums import Action
from catanatron.models.map import build_map
from catanatron.models.player import Color, Player

from ...models.models import PolicyValueNetwork
from .mcts import NeuralMCTS


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero-style self-play + training."""

    num_players: int = 2
    map_type: str = "BASE"
    vps_to_win: int = 10
    simulations: int = 128
    c_puct: float = 1.5
    temperature: float = 1.0
    final_temperature: float = 0.1
    temperature_drop_move: int = 30
    noise_turns: int = 20
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    buffer_size: int = 50_000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    device: Optional[str] = None
    seed: Optional[int] = 42


@dataclass
class AlphaZeroExperience:
    """Single supervised target collected from self-play."""

    state: np.ndarray
    policy: np.ndarray
    value: float


class AlphaZeroTrainer:
    """AlphaZero-style trainer that orchestrates self-play, replay buffer, and SGD."""

    COLOR_ORDER: Sequence[Color] = (
        Color.RED,
        Color.BLUE,
        Color.ORANGE,
        Color.WHITE,
    )

    def __init__(
        self,
        config: Optional[AlphaZeroConfig] = None,
        model: Optional[PolicyValueNetwork] = None,
    ):
        self.config = config or AlphaZeroConfig()
        assert (
            2 <= self.config.num_players <= len(self.COLOR_ORDER)
        ), "AlphaZero currently supports between 2 and 4 players."

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(self.config.seed)

        self.feature_order = get_feature_ordering(
            num_players=self.config.num_players, map_type=self.config.map_type
        )
        input_dim = len(self.feature_order)

        self.model = model or PolicyValueNetwork(input_dim, ACTION_SPACE_SIZE)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        self.replay_buffer: Deque[AlphaZeroExperience] = deque(maxlen=self.config.buffer_size)
        self._current_game_samples: List[Tuple[Color, np.ndarray, np.ndarray]] = []
        self.colors = self.COLOR_ORDER[: self.config.num_players]
        self.mcts = NeuralMCTS(self)

    def self_play(self, num_games: int) -> Dict[str, float]:
        """Generate data via self-play games."""
        stats: Counter[str] = Counter()
        for game_idx in range(num_games):
            stats["games"] += 1
            result = self._play_game(seed=None if self.config.seed is None else self.config.seed + game_idx)
            winner = result.get("winner")
            if winner is not None:
                stats[f"wins_{winner.value}"] += 1
        return dict(stats)

    def update_weights(self) -> Optional[Dict[str, float]]:
        """Run one gradient step over a random mini-batch."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.config.batch_size)
        states = torch.from_numpy(np.stack([exp.state for exp in batch])).float().to(self.device)
        policy_targets = torch.from_numpy(np.stack([exp.policy for exp in batch])).float().to(self.device)
        value_targets = torch.from_numpy(np.array([exp.value for exp in batch], dtype=np.float32)).to(
            self.device
        )

        self.model.train()
        logits, values = self.model(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values.squeeze(-1), value_targets)
        loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def train(
        self,
        iterations: int,
        games_per_iteration: int,
        optimizer_steps: int,
    ) -> Dict[str, List[float]]:
        """Run alternating self-play and SGD loops."""
        history: Dict[str, List[float]] = {"loss": [], "policy_loss": [], "value_loss": []}
        for iteration in range(iterations):
            self.self_play(games_per_iteration)
            for _ in range(optimizer_steps):
                metrics = self.update_weights()
                if metrics:
                    for key, value in metrics.items():
                        history[key].append(value)
        return history

    def save(self, path: str) -> None:
        """Persist model weights to disk."""
        torch.save(self.model.state_dict(), path)

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _play_game(self, seed: Optional[int] = None) -> Dict[str, Optional[Color]]:
        players = [AlphaZeroSelfPlayPlayer(color, self) for color in self.colors]
        catan_map = build_map(self.config.map_type)
        game = Game(players=players, seed=seed, catan_map=catan_map, vps_to_win=self.config.vps_to_win)
        self._current_game_samples.clear()
        winner = None
        try:
            winner = game.play()
            return {"winner": winner}
        finally:
            self._finalize_game_samples(winner)

    def select_action(self, game: Game) -> Action:
        """Called by AlphaZeroSelfPlayPlayer to pick the next move."""
        color = game.state.current_color()
        state_vec = self._extract_features(game, color)
        move_number = len(game.state.actions)
        temperature = (
            self.config.temperature
            if move_number < self.config.temperature_drop_move
            else self.config.final_temperature
        )
        add_noise = move_number < self.config.noise_turns
        policy, action = self.mcts.search(
            game=game,
            root_features=state_vec,
            temperature=max(temperature, 1e-3),
            add_noise=add_noise,
        )
        self._record_sample(color, state_vec, policy)
        return action

    def _extract_features(self, game: Game, color: Color) -> np.ndarray:
        vec = np.asarray(create_sample_vector(game, color, self.feature_order), dtype=np.float32)
        return vec

    def _policy_value(
        self,
        game: Game,
        color: Color,
        cached_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run the neural network on the given state and mask invalid actions."""
        features = cached_features if cached_features is not None else self._extract_features(game, color)
        state_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
            value_scalar = float(value.squeeze(0).item())

        valid_actions = game.state.playable_actions
        if not valid_actions:
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.float32), value_scalar

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        indices = [to_action_space(action) for action in valid_actions]
        mask[indices] = 1.0
        masked = probs * mask
        total = masked.sum()
        if total <= 0:
            masked[indices] = 1.0 / len(indices)
        else:
            masked /= total
        return masked.astype(np.float32), value_scalar

    def _record_sample(self, color: Color, state: np.ndarray, policy: np.ndarray) -> None:
        self._current_game_samples.append((color, state.copy(), policy.copy()))

    def _finalize_game_samples(self, winner: Optional[Color]) -> None:
        if not self._current_game_samples:
            return
        for color, state, policy in self._current_game_samples:
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if color == winner else -1.0
            self.replay_buffer.append(AlphaZeroExperience(state=state, policy=policy, value=value))
        self._current_game_samples.clear()


class AlphaZeroSelfPlayPlayer(Player):
    """Thin wrapper that delegates decisions back to the trainer."""

    def __init__(self, color: Color, controller: AlphaZeroTrainer):
        super().__init__(color, is_bot=True)
        self.controller = controller

    def decide(self, game: Game, playable_actions: Sequence[Action]):
        del playable_actions  # unused
        return self.controller.select_action(game)

