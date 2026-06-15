"""AlphaZero trainer built on the unified NN MCTS engine.

Self-play data is generated with the across-games parallelism stack (the path
profiling showed scales best): independent game-worker processes run
``NNMCTSPlayer`` seats whose leaf evaluations are batched by a central
inference server. Training optimizes two networks - a policy net (cross-entropy
to the MCTS visit distribution) and a critic net (MSE to the game outcome).
"""

from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from catanatron.game import Game
from catanatron.models.player import Color, Player

from ...features.catanatron_utils import ActorObservationLevel, CriticObservationLevel
from ...models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from ...players import NNMCTSPlayer
from ...utils.catanatron_map import build_catan_map
from .parallel_self_play import SelfPlayExperience, generate_self_play_data


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero-style self-play + training."""

    num_players: int = 2
    map_type: str = "BASE"
    actor_observation_level: ActorObservationLevel = "private"
    critic_observation_level: CriticObservationLevel = "full"
    critic_mode: str = "full"
    model_type: str = "flat"
    vps_to_win: int = 10
    discard_limit: int = 7
    simulations: int = 128
    c_puct: float = 1.5
    prunning: bool = False
    temperature: float = 1.0
    final_temperature: float = 0.1
    temperature_drop_move: int = 30
    noise_turns: int = 20
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    # Across-games self-play parallelism.
    num_game_workers: int = 1
    inference_batch_size: int = 64
    inference_wait_ms: float = 2.0
    # Optimization.
    buffer_size: int = 50_000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    device: Optional[str] = None
    seed: Optional[int] = 42


class AlphaZeroTrainer:
    """Self-play data generation + dual-network SGD."""

    COLOR_ORDER = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)

    def __init__(
        self,
        config: Optional[AlphaZeroConfig],
        policy_model: PolicyNetworkWrapper,
        critic_model: ValueNetworkWrapper,
    ) -> None:
        self.config = config or AlphaZeroConfig()
        assert 2 <= self.config.num_players <= len(self.COLOR_ORDER), (
            "AlphaZero currently supports between 2 and 4 players."
        )

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(self.config.seed)

        self.policy_model = policy_model.to(self.device)
        self.critic_model = critic_model.to(self.device)
        self.colors = self.COLOR_ORDER[: self.config.num_players]

        self.replay_buffer: Deque[SelfPlayExperience] = deque(maxlen=self.config.buffer_size)
        params = list(self.policy_model.parameters()) + list(self.critic_model.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self._self_play_calls = 0

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    def self_play(self, num_games: int) -> Dict[str, float]:
        if num_games <= 0:
            return {}
        self.policy_model.eval()
        self.critic_model.eval()

        base_seed = (self.config.seed or 0) + self._self_play_calls * 1_000_003
        self._self_play_calls += 1

        experiences, stats = generate_self_play_data(
            policy_model=self.policy_model,
            critic_model=self.critic_model,
            model_type=self.config.model_type,
            map_type=self.config.map_type,
            num_players=self.config.num_players,
            num_games=num_games,
            num_game_workers=max(1, self.config.num_game_workers),
            num_simulations=self.config.simulations,
            c_puct=self.config.c_puct,
            prunning=self.config.prunning,
            critic_mode=self.config.critic_mode,
            actor_observation_level=self.config.actor_observation_level,
            critic_observation_level=self.config.critic_observation_level,
            inference_batch_size=self.config.inference_batch_size,
            inference_wait_ms=self.config.inference_wait_ms,
            temperature=self.config.temperature,
            final_temperature=self.config.final_temperature,
            temperature_drop_move=self.config.temperature_drop_move,
            noise_turns=self.config.noise_turns,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_frac=self.config.dirichlet_frac,
            vps_to_win=self.config.vps_to_win,
            discard_limit=self.config.discard_limit,
            seed=base_seed,
            device=self.device,
        )
        self.replay_buffer.extend(experiences)
        return {key: float(value) for key, value in stats.items()}

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _policy_logits(self, actor_states: torch.Tensor) -> torch.Tensor:
        if self.config.model_type == "flat":
            return self.policy_model(actor_states)
        if self.config.model_type == "hierarchical":
            action_type_logits, param_logits = self.policy_model(actor_states)
            return self.policy_model.get_flat_action_logits(action_type_logits, param_logits)
        raise ValueError(f"Unknown model_type '{self.config.model_type}'")

    def update_weights(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.config.batch_size)
        actor_states = torch.from_numpy(
            np.stack([exp.actor_state for exp in batch])
        ).float().to(self.device)
        critic_states = torch.from_numpy(
            np.stack([exp.critic_state for exp in batch])
        ).float().to(self.device)
        policy_targets = torch.from_numpy(
            np.stack([exp.policy for exp in batch])
        ).float().to(self.device)
        value_targets = torch.from_numpy(
            np.array([exp.value for exp in batch], dtype=np.float32)
        ).to(self.device)

        self.policy_model.train()
        self.critic_model.train()

        logits = self._policy_logits(actor_states)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()

        values = self.critic_model(critic_states).view(-1)
        value_loss = F.mse_loss(values, value_targets)

        loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_model.parameters()) + list(self.critic_model.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_against(
        self,
        opponent_factory: Callable[[Color], Player],
        num_games: int,
        desc: str,
    ) -> Dict[str, float]:
        if num_games <= 0:
            return {}
        self.policy_model.eval()
        self.critic_model.eval()

        stats: Counter[str] = Counter()
        seed_offset = 100_000
        for game_idx in range(num_games):
            agent_color = self.colors[game_idx % len(self.colors)]
            players: List[Player] = []
            for color in self.colors:
                if color == agent_color:
                    players.append(self._build_agent_player(color))
                else:
                    players.append(opponent_factory(color))

            seed = None if self.config.seed is None else self.config.seed + seed_offset + game_idx
            catan_map = build_catan_map(self.config.map_type, seed=seed, number_placement="random")
            game = Game(
                players=players,
                seed=seed,
                catan_map=catan_map,
                vps_to_win=self.config.vps_to_win,
            )
            winner = game.play()
            stats["games"] += 1
            if winner is None:
                stats["draws"] += 1
            else:
                stats[f"wins_{winner.value}"] += 1
                if winner == agent_color:
                    stats["agent_wins"] += 1
        return {key: float(value) for key, value in stats.items()}

    def _build_agent_player(self, color: Color) -> NNMCTSPlayer:
        return NNMCTSPlayer(
            color=color,
            model_type=self.config.model_type,
            policy_model=self.policy_model,
            critic_model=self.critic_model,
            map_type=self.config.map_type,
            num_simulations=self.config.simulations,
            c_puct=self.config.c_puct,
            prunning=self.config.prunning,
            critic_mode=self.config.critic_mode,
            actor_observation_level=self.config.actor_observation_level,
            critic_observation_level=self.config.critic_observation_level,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy_model.state_dict(),
                "critic": self.critic_model.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint["policy"])
        self.critic_model.load_state_dict(checkpoint["critic"])

    def close(self) -> None:
        return

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
