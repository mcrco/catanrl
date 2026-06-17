"""AlphaZero trainer built on the unified NN MCTS engine.

Self-play data is generated with the across-games parallelism stack (the path
profiling showed scales best): independent game-worker processes run
``NNMCTSPlayer`` seats whose leaf evaluations are batched by a central
inference server. Training optimizes policy (cross-entropy to the MCTS visit
distribution) and value (MSE to the game outcome), using either separate
networks or a shared policy-value backbone.
"""

from __future__ import annotations

import os
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
from ...models.inference_utils import forward_policy_value
from ...models.wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper
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
    critic_hidden_mode: str = "full"
    network_mode: str = "separate"
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
    """Self-play data generation + policy/value SGD."""

    COLOR_ORDER = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)

    def __init__(
        self,
        config: Optional[AlphaZeroConfig],
        policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
        critic_model: ValueNetworkWrapper | None = None,
    ) -> None:
        self.config = config or AlphaZeroConfig()
        assert 2 <= self.config.num_players <= len(self.COLOR_ORDER), (
            "AlphaZero currently supports between 2 and 4 players."
        )

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(self.config.seed)

        self.policy_model = policy_model.to(self.device)
        self.uses_shared_network = isinstance(self.policy_model, PolicyValueNetworkWrapper)
        if self.uses_shared_network:
            if critic_model is not None:
                raise ValueError(
                    "critic_model must be None when policy_model is a PolicyValueNetworkWrapper."
                )
            self.critic_model = None
        else:
            if critic_model is None:
                raise ValueError(
                    "critic_model is required when policy_model is not a PolicyValueNetworkWrapper."
                )
            self.critic_model = critic_model.to(self.device)

        self.colors = self.COLOR_ORDER[: self.config.num_players]

        self.replay_buffer: Deque[SelfPlayExperience] = deque(maxlen=self.config.buffer_size)
        trainable_params = (
            self.policy_model.parameters()
            if self.uses_shared_network
            else list(self.policy_model.parameters()) + list(self.critic_model.parameters())
        )
        self.optimizer = torch.optim.Adam(
            trainable_params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self._self_play_calls = 0

    def _set_eval_mode(self) -> None:
        self.policy_model.eval()
        if self.critic_model is not None:
            self.critic_model.eval()

    def _set_train_mode(self) -> None:
        self.policy_model.train()
        if self.critic_model is not None:
            self.critic_model.train()

    def _trainable_parameters(self):
        if self.uses_shared_network:
            return self.policy_model.parameters()
        return list(self.policy_model.parameters()) + list(self.critic_model.parameters())

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    def self_play(self, num_games: int) -> Dict[str, float]:
        if num_games <= 0:
            return {}
        self._set_eval_mode()

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
            critic_hidden_mode=self.config.critic_hidden_mode,
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

        self._set_train_mode()

        if self.uses_shared_network:
            logits, values = forward_policy_value(
                self.policy_model,
                None,
                actor_states,
                self.config.model_type,
            )
        else:
            logits, values = forward_policy_value(
                self.policy_model,
                self.critic_model,
                actor_states,
                self.config.model_type,
                critic_states=critic_states,
            )

        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)

        loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.config.max_grad_norm)
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
        self._set_eval_mode()

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
            critic_hidden_mode=self.config.critic_hidden_mode,
            actor_observation_level=self.config.actor_observation_level,
            critic_observation_level=self.config.critic_observation_level,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str, stem: str = "best") -> None:
        os.makedirs(save_dir, exist_ok=True)
        if self.uses_shared_network:
            torch.save(
                self.policy_model.state_dict(),
                os.path.join(save_dir, f"policy_value_{stem}.pt"),
            )
            return

        torch.save(
            self.policy_model.state_dict(),
            os.path.join(save_dir, f"policy_{stem}.pt"),
        )
        assert self.critic_model is not None
        torch.save(
            self.critic_model.state_dict(),
            os.path.join(save_dir, f"critic_{stem}.pt"),
        )

    def load(
        self,
        policy_path: str,
        critic_path: str | None = None,
    ) -> None:
        if self.uses_shared_network:
            self.policy_model.load_state_dict(
                torch.load(policy_path, map_location=self.device)
            )
            return

        self.policy_model.load_state_dict(
            torch.load(policy_path, map_location=self.device)
        )
        if critic_path is None:
            raise ValueError("critic_path is required when loading separate networks.")
        assert self.critic_model is not None
        self.critic_model.load_state_dict(
            torch.load(critic_path, map_location=self.device)
        )

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
