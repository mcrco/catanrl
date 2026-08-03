"""Search-guided policy training built on the unified NN MCTS engine.

The trainer deliberately separates the model that generates MCTS targets (the
teacher/champion) from the model being optimized (the student/candidate).  This
supports two useful workflows:

* frozen-teacher distillation, which tests whether a policy can absorb MCTS lift;
* iterative expert iteration, which promotes an accepted student into the next
  search teacher.

Game collection remains in :mod:`parallel_self_play`, where independent game
workers share a centrally batched neural inference server.
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from collections.abc import Iterator, Sequence
from typing import Any, Deque, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...features.catanatron_utils import ActorObservationLevel, COLOR_ORDER, CriticObservationLevel
from ...models.inference_utils import forward_policy_value
from ...models.wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper
from .native_self_play import generate_native_self_play_data
from .parallel_self_play import SelfPlayExperience, generate_self_play_data

TrainingMode = Literal["distill", "iterate"]
PolicyModel = PolicyNetworkWrapper | PolicyValueNetworkWrapper
CriticModel = ValueNetworkWrapper | None


@dataclass
class AlphaZeroConfig:
    """Configuration for search target generation and student optimization."""

    mode: TrainingMode = "distill"
    num_players: int = 2
    map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE"
    actor_observation_level: ActorObservationLevel = "private"
    critic_observation_level: CriticObservationLevel = "full"
    network_mode: str = "separate"
    model_type: str = "flat"
    vps_to_win: int = 15
    discard_limit: int = 9
    self_play_backend: Literal["python", "cppanatron"] = "python"

    # Search teacher.
    simulations: int = 64
    c_puct: float = 1.5
    value_scale: float = 1.0
    tree_reuse: bool = False
    canonical_pruning: bool = False
    full_search_probability: float = 1.0
    fast_simulations: int = 64
    search_value_weight_max: float = 0.0
    search_value_weight_ramp_iterations: int = 1
    prunning: bool = False
    ismcts_determinizations: int = 1
    temperature: float = 1.0
    final_temperature: float = 0.1
    target_temperature: float | None = None
    temperature_drop_move: int = 30
    noise_turns: int = 20
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    num_game_workers: int = 1
    inference_batch_size: int = 64
    inference_wait_ms: float = 2.0

    # Student optimization.
    buffer_size: int = 50_000
    batch_size: int = 256
    policy_lr: float = 5e-5
    critic_lr: float = 1e-4
    weight_decay: float = 0.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.0
    max_grad_norm: float = 1.0

    # Keeping only the active pair on the accelerator substantially reduces
    # memory use for the large xdim models.
    offload_inactive_models: bool = True
    device: Optional[str] = None
    seed: Optional[int] = 42


def _flat_policy_logits(model: PolicyModel, states: torch.Tensor, model_type: str) -> torch.Tensor:
    if isinstance(model, PolicyValueNetworkWrapper):
        logits, _ = forward_policy_value(model, None, states, model_type)
        return logits
    if model_type == "flat":
        return model(states)
    if model_type == "hierarchical":
        action_type_logits, param_logits = model(states)
        return model.get_flat_action_logits(action_type_logits, param_logits)
    raise ValueError(f"Unknown model_type '{model_type}'")


def _move_optimizer_state(optimizer: torch.optim.Optimizer | None, device: torch.device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _cpu_state_dict(model: torch.nn.Module | None) -> dict | None:
    if model is None:
        return None
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class AlphaZeroTrainer:
    """Manage a frozen/accepted search teacher and a trainable student."""

    def __init__(
        self,
        config: AlphaZeroConfig,
        student_policy_model: PolicyModel,
        student_critic_model: CriticModel,
        teacher_policy_model: PolicyModel,
        teacher_critic_model: CriticModel,
    ) -> None:
        self.config = config
        if not 2 <= config.num_players <= len(COLOR_ORDER):
            raise ValueError("Search-guided training supports between 2 and 4 players.")
        if config.mode not in ("distill", "iterate"):
            raise ValueError(f"Unknown training mode '{config.mode}'.")
        if config.self_play_backend not in ("python", "cppanatron"):
            raise ValueError(f"Unknown self-play backend '{config.self_play_backend}'.")
        if config.batch_size <= 0 or config.buffer_size < config.batch_size:
            raise ValueError("buffer_size must be at least batch_size, and both must be positive.")
        if config.policy_loss_weight <= 0:
            raise ValueError("policy_loss_weight must be positive.")
        if config.value_loss_weight < 0:
            raise ValueError("value_loss_weight cannot be negative.")
        if not 0.0 < config.full_search_probability <= 1.0:
            raise ValueError("full_search_probability must be in (0, 1].")
        if config.fast_simulations < 1 or (
            config.full_search_probability < 1.0 and config.fast_simulations > config.simulations
        ):
            raise ValueError("fast_simulations must be between 1 and simulations.")
        if not 0.0 <= config.search_value_weight_max <= 1.0:
            raise ValueError("search_value_weight_max must be between 0 and 1.")
        if config.search_value_weight_ramp_iterations < 0:
            raise ValueError("search_value_weight_ramp_iterations cannot be negative.")

        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cpu_device = torch.device("cpu")
        self.colors = COLOR_ORDER[: config.num_players]
        self._set_seed(config.seed)

        self.student_policy_model = student_policy_model
        self.student_critic_model = student_critic_model
        self.teacher_policy_model = teacher_policy_model
        self.teacher_critic_model = teacher_critic_model
        self.uses_shared_network = isinstance(student_policy_model, PolicyValueNetworkWrapper)
        teacher_is_shared = isinstance(teacher_policy_model, PolicyValueNetworkWrapper)
        if self.uses_shared_network != teacher_is_shared:
            raise ValueError("Teacher and student must use the same network mode.")
        if self.uses_shared_network:
            if student_critic_model is not None or teacher_critic_model is not None:
                raise ValueError("Shared policy-value networks must not receive separate critics.")
        elif student_critic_model is None or teacher_critic_model is None:
            raise ValueError("Separate-network training requires student and teacher critics.")

        self.replay_buffer: Deque[SelfPlayExperience] = deque(maxlen=config.buffer_size)
        self._self_play_calls = 0
        # Optimizers must be constructed after the first device placement so
        # their parameter references cannot point at pre-conversion tensors.
        self.student_policy_model.to(self.device)
        if self.student_critic_model is not None:
            self.student_critic_model.to(self.device)
        self._build_optimizers()
        self._activate_student()
        if config.offload_inactive_models:
            self._move_teacher(self.cpu_device)
        else:
            self._move_teacher(self.device)

    # Compatibility aliases for callers that only care about the candidate.
    @property
    def policy_model(self) -> PolicyModel:
        return self.student_policy_model

    @property
    def critic_model(self) -> CriticModel:
        return self.student_critic_model

    def _build_optimizers(self) -> None:
        if self.uses_shared_network:
            self.policy_optimizer = torch.optim.Adam(
                self.student_policy_model.parameters(),
                lr=self.config.policy_lr,
                weight_decay=self.config.weight_decay,
            )
            self.critic_optimizer = None
            return

        self.policy_optimizer = torch.optim.Adam(
            self.student_policy_model.parameters(),
            lr=self.config.policy_lr,
            weight_decay=self.config.weight_decay,
        )
        self.critic_optimizer = (
            torch.optim.Adam(
                self.student_critic_model.parameters(),
                lr=self.config.critic_lr,
                weight_decay=self.config.weight_decay,
            )
            if self.config.value_loss_weight > 0 and self.student_critic_model is not None
            else None
        )

    def reset_optimizers(self) -> None:
        """Discard candidate optimizer momentum after rolling back a rejection."""
        self._build_optimizers()
        self._activate_student()

    def _move_student(self, device: torch.device) -> None:
        parameter_ids = tuple(
            id(parameter)
            for parameter in (
                list(self.student_policy_model.parameters())
                + (
                    list(self.student_critic_model.parameters())
                    if self.student_critic_model is not None
                    else []
                )
            )
        )
        self.student_policy_model.to(device)
        if self.student_critic_model is not None:
            self.student_critic_model.to(device)
        moved_parameter_ids = tuple(
            id(parameter)
            for parameter in (
                list(self.student_policy_model.parameters())
                + (
                    list(self.student_critic_model.parameters())
                    if self.student_critic_model is not None
                    else []
                )
            )
        )
        if moved_parameter_ids != parameter_ids:
            raise RuntimeError(
                "PyTorch replaced Parameter objects during model offload; "
                "rerun with --no-offload-inactive-models."
            )
        _move_optimizer_state(self.policy_optimizer, device)
        _move_optimizer_state(self.critic_optimizer, device)

    def _move_teacher(self, device: torch.device) -> None:
        self.teacher_policy_model.to(device)
        if self.teacher_critic_model is not None:
            self.teacher_critic_model.to(device)

    def _activate_student(self) -> None:
        self._move_student(self.device)
        self.student_policy_model.train()
        if self.student_critic_model is not None:
            self.student_critic_model.train(self.config.value_loss_weight > 0)

    def _activate_teacher(self) -> None:
        if self.config.offload_inactive_models:
            self._move_student(self.cpu_device)
        self._move_teacher(self.device)
        self.teacher_policy_model.eval()
        if self.teacher_critic_model is not None:
            self.teacher_critic_model.eval()

    def prepare_student_evaluation(self) -> tuple[PolicyModel, CriticModel]:
        self._activate_student()
        self.student_policy_model.eval()
        if self.student_critic_model is not None:
            self.student_critic_model.eval()
        if self.config.offload_inactive_models:
            self._move_teacher(self.cpu_device)
        return self.student_policy_model, self.student_critic_model

    def teacher_evaluation_models(self) -> tuple[PolicyModel, CriticModel]:
        """Return the teacher on CPU when offloading is enabled.

        Policy-player evaluation follows each model's own device, so candidate
        versus teacher evaluation does not require both large pairs on CUDA.
        """
        target = self.cpu_device if self.config.offload_inactive_models else self.device
        self._move_teacher(target)
        self.teacher_policy_model.eval()
        if self.teacher_critic_model is not None:
            self.teacher_critic_model.eval()
        return self.teacher_policy_model, self.teacher_critic_model

    # ------------------------------------------------------------------
    # Search target collection
    # ------------------------------------------------------------------

    def collect_self_play(self, num_games: int) -> dict[str, float]:
        if num_games <= 0:
            return {}
        self._activate_teacher()
        self_play_iteration = self._self_play_calls
        base_seed = (self.config.seed or 0) + self_play_iteration * 1_000_003
        self._self_play_calls += 1
        if self.config.search_value_weight_ramp_iterations == 0:
            search_value_weight = self.config.search_value_weight_max
        else:
            search_value_weight = self.config.search_value_weight_max * min(
                1.0,
                self_play_iteration / self.config.search_value_weight_ramp_iterations,
            )

        self_play_generator = (
            generate_native_self_play_data
            if self.config.self_play_backend == "cppanatron"
            else generate_self_play_data
        )
        self_play_kwargs: dict[str, Any] = dict(
            policy_model=self.teacher_policy_model,
            critic_model=self.teacher_critic_model,
            model_type=self.config.model_type,
            map_type=self.config.map_type,
            num_players=self.config.num_players,
            num_games=num_games,
            num_game_workers=max(1, self.config.num_game_workers),
            num_simulations=self.config.simulations,
            c_puct=self.config.c_puct,
            prunning=self.config.prunning,
            actor_observation_level=self.config.actor_observation_level,
            critic_observation_level=self.config.critic_observation_level,
            ismcts_determinizations=self.config.ismcts_determinizations,
            inference_batch_size=self.config.inference_batch_size,
            inference_wait_ms=self.config.inference_wait_ms,
            temperature=self.config.temperature,
            final_temperature=self.config.final_temperature,
            target_temperature=self.config.target_temperature,
            temperature_drop_move=self.config.temperature_drop_move,
            noise_turns=self.config.noise_turns,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_frac=self.config.dirichlet_frac,
            vps_to_win=self.config.vps_to_win,
            discard_limit=self.config.discard_limit,
            seed=base_seed,
            device=self.device,
        )
        if self.config.self_play_backend == "cppanatron":
            self_play_kwargs.update(
                value_scale=self.config.value_scale,
                tree_reuse=self.config.tree_reuse,
                canonical_pruning=self.config.canonical_pruning,
                full_search_probability=self.config.full_search_probability,
                fast_simulations=self.config.fast_simulations,
                search_value_weight=search_value_weight,
            )
        experiences, stats = self_play_generator(**self_play_kwargs)
        self.replay_buffer.extend(experiences)
        if self.config.offload_inactive_models:
            self._move_teacher(self.cpu_device)
        self._activate_student()
        return {
            **{key: float(value) for key, value in stats.items()},
            "experiences": float(len(experiences)),
            "replay_size": float(len(self.replay_buffer)),
            "search_value_weight": float(search_value_weight),
        }

    # Retain the old method name for code using the trainer programmatically.
    def self_play(self, num_games: int) -> dict[str, float]:
        return self.collect_self_play(num_games)

    # ------------------------------------------------------------------
    # Student optimization
    # ------------------------------------------------------------------

    def iter_replay_epoch_batches(
        self, epochs: int
    ) -> Iterator[list[SelfPlayExperience]]:
        """Yield complete shuffled passes over a stable replay snapshot."""
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        replay_snapshot = list(self.replay_buffer)
        for _ in range(epochs):
            random.shuffle(replay_snapshot)
            for offset in range(0, len(replay_snapshot), self.config.batch_size):
                yield replay_snapshot[offset : offset + self.config.batch_size]

    def update_weights(
        self, batch: Sequence[SelfPlayExperience] | None = None
    ) -> Optional[dict[str, float]]:
        if batch is None:
            if len(self.replay_buffer) < self.config.batch_size:
                return None
            batch = random.sample(self.replay_buffer, self.config.batch_size)
        elif not batch:
            return None
        self._activate_student()

        actor_states = torch.from_numpy(np.stack([exp.actor_state for exp in batch])).float()
        policy_targets = torch.from_numpy(np.stack([exp.policy for exp in batch])).float()
        action_masks = torch.from_numpy(np.stack([exp.action_mask for exp in batch])).bool()
        policy_weights = torch.from_numpy(
            np.asarray([float(exp.full_search) for exp in batch], dtype=np.float32)
        )
        actor_states = actor_states.to(self.device)
        policy_targets = policy_targets.to(self.device)
        action_masks = action_masks.to(self.device)
        policy_weights = policy_weights.to(self.device)
        if not bool(action_masks.any(dim=1).all()):
            raise RuntimeError("Encountered a search target without any legal actions.")
        illegal_target_mass = policy_targets.masked_select(~action_masks).abs().sum()
        if float(illegal_target_mass.item()) > 1e-6:
            raise RuntimeError("Search policy assigns probability to an illegal action.")

        critic_states: torch.Tensor | None = None
        value_targets: torch.Tensor | None = None
        if self.config.value_loss_weight > 0:
            critic_states = torch.from_numpy(np.stack([exp.critic_state for exp in batch])).float()
            critic_states = critic_states.to(self.device)
            value_targets = torch.from_numpy(
                np.asarray([exp.value for exp in batch], dtype=np.float32)
            ).to(self.device)

        self.policy_optimizer.zero_grad(set_to_none=True)
        if self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad(set_to_none=True)

        values: torch.Tensor | None = None
        if self.uses_shared_network:
            logits, shared_values = forward_policy_value(
                self.student_policy_model,
                None,
                actor_states,
                self.config.model_type,
            )
            if self.config.value_loss_weight > 0:
                values = shared_values
        else:
            logits = _flat_policy_logits(
                self.student_policy_model,
                actor_states,
                self.config.model_type,
            )
            if self.config.value_loss_weight > 0:
                assert self.student_critic_model is not None
                assert critic_states is not None
                values = self.student_critic_model(critic_states).view(-1)

        masked_logits = logits.masked_fill(~action_masks, float("-inf"))
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        per_sample_policy_loss = -torch.where(
            action_masks,
            policy_targets * log_probs,
            torch.zeros_like(log_probs),
        ).sum(dim=1)
        policy_weight_sum = policy_weights.sum().clamp_min(1.0)
        policy_loss = (per_sample_policy_loss * policy_weights).sum() / policy_weight_sum
        if values is not None and value_targets is not None:
            value_loss = F.mse_loss(values, value_targets)
        else:
            value_loss = torch.zeros((), device=self.device)
        loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )
        loss.backward()

        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.student_policy_model.parameters(), self.config.max_grad_norm
        )
        critic_grad_norm = torch.zeros((), device=self.device)
        if self.critic_optimizer is not None and self.student_critic_model is not None:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student_critic_model.parameters(), self.config.max_grad_norm
            )
        self.policy_optimizer.step()
        if self.critic_optimizer is not None:
            self.critic_optimizer.step()

        with torch.no_grad():
            per_sample_target_entropy = -(
                policy_targets * torch.log(policy_targets.clamp_min(1e-12))
            ).sum(dim=1)
            target_entropy = (per_sample_target_entropy * policy_weights).sum() / policy_weight_sum
            probabilities = torch.softmax(masked_logits, dim=-1)
            per_sample_prediction_entropy = -torch.where(
                action_masks,
                probabilities * log_probs,
                torch.zeros_like(log_probs),
            ).sum(dim=1)
            prediction_entropy = (
                per_sample_prediction_entropy * policy_weights
            ).sum() / policy_weight_sum
            per_sample_top1_agreement = (
                masked_logits.argmax(dim=-1) == policy_targets.argmax(dim=-1)
            ).float()
            top1_agreement = (per_sample_top1_agreement * policy_weights).sum() / policy_weight_sum

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "target_entropy": float(target_entropy.item()),
            "prediction_entropy": float(prediction_entropy.item()),
            "top1_agreement": float(top1_agreement.item()),
            "full_search_fraction": float(policy_weights.mean().item()),
            "policy_grad_norm": float(policy_grad_norm.item()),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "replay_size": float(len(self.replay_buffer)),
        }

    # ------------------------------------------------------------------
    # Candidate/champion lifecycle
    # ------------------------------------------------------------------

    def promote_student(self) -> None:
        """Make the accepted student the teacher for subsequent collection."""
        self.teacher_policy_model.load_state_dict(self.student_policy_model.state_dict())
        if self.teacher_critic_model is not None and self.student_critic_model is not None:
            self.teacher_critic_model.load_state_dict(self.student_critic_model.state_dict())
        if self.config.offload_inactive_models:
            self._move_teacher(self.cpu_device)

    def restore_student_from_teacher(self) -> None:
        """Roll a rejected candidate back to the current champion."""
        self.student_policy_model.load_state_dict(self.teacher_policy_model.state_dict())
        if self.student_critic_model is not None and self.teacher_critic_model is not None:
            self.student_critic_model.load_state_dict(self.teacher_critic_model.state_dict())
        self.reset_optimizers()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str, stem: str = "best", *, source: str = "student") -> None:
        os.makedirs(save_dir, exist_ok=True)
        if source not in ("student", "teacher"):
            raise ValueError("source must be 'student' or 'teacher'.")
        policy = self.student_policy_model if source == "student" else self.teacher_policy_model
        critic = self.student_critic_model if source == "student" else self.teacher_critic_model
        if self.uses_shared_network:
            torch.save(policy.state_dict(), os.path.join(save_dir, f"policy_value_{stem}.pt"))
            return
        torch.save(policy.state_dict(), os.path.join(save_dir, f"policy_{stem}.pt"))
        assert critic is not None
        torch.save(critic.state_dict(), os.path.join(save_dir, f"critic_{stem}.pt"))

    def state_dict(self) -> dict:
        return {
            "student_policy_model": _cpu_state_dict(self.student_policy_model),
            "student_critic_model": _cpu_state_dict(self.student_critic_model),
            "teacher_policy_model": _cpu_state_dict(self.teacher_policy_model),
            "teacher_critic_model": _cpu_state_dict(self.teacher_critic_model),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": (
                self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None
            ),
            "self_play_calls": self._self_play_calls,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.student_policy_model.load_state_dict(state["student_policy_model"])
        if self.student_critic_model is not None and state.get("student_critic_model") is not None:
            self.student_critic_model.load_state_dict(state["student_critic_model"])
        self.teacher_policy_model.load_state_dict(state["teacher_policy_model"])
        if self.teacher_critic_model is not None and state.get("teacher_critic_model") is not None:
            self.teacher_critic_model.load_state_dict(state["teacher_critic_model"])
        if state.get("policy_optimizer") is not None:
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        if self.critic_optimizer is not None and state.get("critic_optimizer") is not None:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._self_play_calls = int(state.get("self_play_calls", 0))
        if state.get("python_random_state") is not None:
            random.setstate(state["python_random_state"])
        if state.get("numpy_random_state") is not None:
            np.random.set_state(state["numpy_random_state"])
        if state.get("torch_random_state") is not None:
            torch.set_rng_state(state["torch_random_state"])
        self._activate_student()
        if self.config.offload_inactive_models:
            self._move_teacher(self.cpu_device)

    def close(self) -> None:
        return

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
