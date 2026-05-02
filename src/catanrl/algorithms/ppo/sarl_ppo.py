#!/usr/bin/env python3
"""
Single Agent Reinforcement Learning training script for Catan Policy-Value Network.
Uses the Catanatron gym environment for training with PPO.
"""

import os
import random
import sys
from collections import deque
from typing import Any, List, Literal, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

import wandb

from ...envs import decode_puffer_batch
from ...envs.puffer.single_agent_env import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    compute_single_agent_dims,
    create_opponents,
    make_puffer_vectorized_envs,
)
from ...eval.training_eval import eval_policy_value_against_baselines
from ...features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
    get_observation_indices_from_full,
)
from ...models.models import (
    build_flat_policy_network,
    build_flat_policy_value_network,
    build_hierarchical_policy_network,
    build_hierarchical_policy_value_network,
    build_value_network,
)
from ...utils.catanatron_action_space import build_action_type_metadata, get_action_space_size
from .buffers import CentralCriticExperienceBuffer, ExperienceBuffer
from .agent import PolicyAgent
from .action_stats import build_action_type_time_series_chart, compute_action_distributions
from .backbone_builder import build_backbone_config
from .ppo_update import run_ppo_update


def train(
    input_dim: int | None = None,
    num_actions: int | None = None,
    model_type: str = "flat",
    backbone_type: str = "mlp",
    xdim_cnn_channels: Sequence[int] = (64, 128, 128),
    xdim_cnn_kernel_size: Tuple[int, int] = (3, 5),
    xdim_fusion_hidden_dim: int | None = None,
    xdim_critic_fusion_hidden_dim: int | None = None,
    hidden_dims: Sequence[int] = (512, 512),
    critic_hidden_dims: Sequence[int] | None = None,
    critic_mode: Literal["shared", "privileged"] = "shared",
    load_weights: str | None = None,
    load_critic_weights: str | None = None,
    total_timesteps: int = 1_000_000,
    rollout_steps: int = 4096,
    lr: float = 1e-4,
    critic_lr: float | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    activity_coef: float = 0.0,
    n_epochs: int = 4,
    batch_size: int = 64,
    save_path: str | None = "weights/sarl_ppo",
    save_every_updates: int = 1,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    wandb_config: dict | None = None,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    actor_observation_level: ActorObservationLevel = "private",
    critic_observation_level: CriticObservationLevel = "full",
    vps_to_win: int = 15,
    discard_limit: int = 9,
    opponent_configs: List[str] | None = None,
    reward_function: str = "shaped",
    num_envs: int = 4,
    use_lr_scheduler: bool = False,
    lr_scheduler_kwargs: dict | None = None,
    metric_window: int = 200,
    eval_games_per_opponent: int = 0,
    trend_eval_games_per_opponent: int | None = None,
    trend_eval_seed: int | None = 42,
    eval_every_updates: int = 0,
    deterministic_policy: bool = False,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
):
    """Train SARL PPO with optional privileged critic."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be > 0")
    if num_envs <= 0:
        raise ValueError("num_envs must be > 0")
    if critic_mode not in ("shared", "privileged"):
        raise ValueError(f"Unknown critic_mode '{critic_mode}'")
    uses_privileged_critic = critic_mode == "privileged"
    effective_critic_observation_level = (
        critic_observation_level if uses_privileged_critic else actor_observation_level
    )
    critic_lr = lr if critic_lr is None else critic_lr
    critic_hidden_dims = tuple(hidden_dims if critic_hidden_dims is None else critic_hidden_dims)

    print(f"\n{'=' * 60}")
    print("Training Policy-Value Network with Single Agent Reinforcement Learning (PPO)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type}")
    print(f"Game params: vps_to_win={vps_to_win}, discard_limit={discard_limit}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Parallel environments: {num_envs}")

    opponent_configs = opponent_configs or ["random"]
    opponents = create_opponents(opponent_configs)
    num_players = len(opponents) + 1  # add the learning agent
    action_space_size = num_actions or get_action_space_size(num_players, map_type)
    _, action_to_type_idx, action_type_names = build_action_type_metadata(num_players, map_type)
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
        critic_observation_level=effective_critic_observation_level,
    )
    full_dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
        critic_observation_level="full",
    )
    actor_input_dim = dims["actor_dim"]
    critic_input_dim = dims["critic_dim"]
    full_critic_input_dim = full_dims["critic_dim"]
    actor_numeric_dim = dims["actor_numeric_dim"]
    critic_numeric_dim = dims["critic_numeric_dim"]
    board_channels = dims["board_channels"]
    critic_observation_indices = get_observation_indices_from_full(
        num_players,
        map_type,
        level=effective_critic_observation_level,
    )
    if input_dim is not None and int(input_dim) != actor_input_dim:
        print(
            f"Warning: input_dim={input_dim} does not match computed actor dim {actor_input_dim}. "
            f"Using computed dim {actor_input_dim}."
        )

    print(f"Number of players: {num_players}")
    print(f"Opponents: {[repr(o) for o in opponents]}")
    print(
        f"Backbone: {backbone_type} | Model type: {model_type} | "
        f"Critic mode: {critic_mode} | Actor observation: {actor_observation_level} | "
        f"Critic observation: {effective_critic_observation_level}"
    )

    if wandb.run is None:
        if wandb_config:
            wandb.init(**wandb_config)
            print(
                f"Initialized wandb: {wandb_config['project']}/"
                f"{wandb_config.get('name', wandb.run.name if wandb.run is not None else 'run')}"
            )
        else:
            wandb.init(mode="disabled")

    default_save_dir = "weights/sarl_ppo"
    if save_path == default_save_dir and wandb.run is not None and getattr(wandb.run, "name", None):
        save_path = os.path.join("weights", wandb.run.name)

    xdim_cnn_channels = list(xdim_cnn_channels)
    if backbone_type not in ("mlp", "xdim", "xdim_res"):
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")
    if backbone_type != "mlp" and not xdim_cnn_channels:
        raise ValueError("xdim_cnn_channels cannot be empty")

    policy_backbone_config = build_backbone_config(
        backbone_type=backbone_type,
        hidden_dims=hidden_dims,
        input_dim=actor_input_dim,
        board_height=BOARD_HEIGHT,
        board_width=BOARD_WIDTH,
        board_channels=board_channels,
        numeric_dim=actor_numeric_dim,
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,
        xdim_fusion_hidden_dim=xdim_fusion_hidden_dim,
    )

    if uses_privileged_critic:
        if model_type == "flat":
            policy_model = build_flat_policy_network(
                backbone_config=policy_backbone_config, num_actions=action_space_size
            ).to(device)
        elif model_type == "hierarchical":
            policy_model = build_hierarchical_policy_network(
                backbone_config=policy_backbone_config,
                num_players=num_players,
                map_type=map_type,
            ).to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        critic_backbone_config = build_backbone_config(
            backbone_type=backbone_type,
            hidden_dims=critic_hidden_dims,
            input_dim=critic_input_dim,
            board_height=BOARD_HEIGHT,
            board_width=BOARD_WIDTH,
            board_channels=board_channels,
            numeric_dim=critic_numeric_dim,
            xdim_cnn_channels=xdim_cnn_channels,
            xdim_cnn_kernel_size=xdim_cnn_kernel_size,
            xdim_fusion_hidden_dim=xdim_critic_fusion_hidden_dim,
        )
        critic_model = build_value_network(backbone_config=critic_backbone_config).to(device)
    else:
        if model_type == "flat":
            policy_model = build_flat_policy_value_network(
                backbone_config=policy_backbone_config, num_actions=action_space_size
            ).to(device)
        elif model_type == "hierarchical":
            policy_model = build_hierarchical_policy_value_network(
                backbone_config=policy_backbone_config,
                num_players=num_players,
                map_type=map_type,
            ).to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        critic_model = None

    if load_weights and os.path.exists(load_weights):
        print(f"Loading policy weights from: {load_weights}")
        state_dict = torch.load(load_weights, map_location=device)
        if uses_privileged_critic:
            missing_keys, unexpected_keys = policy_model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  Missing keys while loading policy weights: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys while loading policy weights: {unexpected_keys}")
        else:
            policy_model.load_state_dict(state_dict)
        print("  Policy weights loaded successfully.")

    if uses_privileged_critic and load_critic_weights and os.path.exists(load_critic_weights):
        print(f"Loading critic weights from: {load_critic_weights}")
        assert critic_model is not None
        critic_state_dict = torch.load(load_critic_weights, map_location=device)
        critic_model.load_state_dict(critic_state_dict)
        print("  Critic weights loaded successfully.")

    agent = PolicyAgent(policy_model, model_type, device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=lr)
    critic_optimizer = (
        optim.Adam(critic_model.parameters(), lr=critic_lr) if critic_model is not None else None
    )

    def predict_values(
        actor_states: torch.Tensor,
        critic_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if uses_privileged_critic:
            if critic_states is None:
                raise ValueError("critic_states must be provided when using privileged critic.")
            assert critic_model is not None
            return critic_model(critic_states).squeeze(-1)

        if model_type == "flat":
            _, values = policy_model(actor_states)
        elif model_type == "hierarchical":
            _, _, values = policy_model(actor_states)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        return values.squeeze(-1)

    if uses_privileged_critic:
        assert critic_model is not None
        value_model: torch.nn.Module = critic_model
    else:
        value_model = policy_model

    # Create learning rate scheduler
    scheduler = None
    scheduler_total_iters = None
    scheduler_steps_taken = 0
    if use_lr_scheduler:
        scheduler_kwargs = lr_scheduler_kwargs or {}
        start_factor = scheduler_kwargs.get("start_factor", 1.0)
        end_factor = scheduler_kwargs.get("end_factor", 0.0)
        default_total_iters = max(1, total_timesteps // max(1, num_envs))
        total_iters = scheduler_kwargs.get("total_iters", default_total_iters)
        scheduler_total_iters = total_iters
        scheduler = lr_scheduler.LinearLR(
            policy_optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
        )
        print(
            f"LR Scheduler: LinearLR (start_factor={start_factor}, end_factor={end_factor}, total_iters={total_iters})"
        )
    else:
        print("LR Scheduler: None (constant learning rate)")

    policy_model.eval()
    if critic_model is not None:
        critic_model.eval()

    if uses_privileged_critic:
        buffer = CentralCriticExperienceBuffer(
            num_rollouts=rollout_steps,
            actor_state_dim=actor_input_dim,
            critic_state_dim=critic_input_dim,
            action_space_size=action_space_size,
            num_envs=num_envs,
        )
    else:
        buffer = ExperienceBuffer(
            num_rollouts=rollout_steps,
            state_dim=actor_input_dim,
            action_space_size=action_space_size,
            num_envs=num_envs,
        )
    pending_scheduler_steps = 0
    episode_rewards = deque(maxlen=metric_window)
    episode_lengths = deque(maxlen=metric_window)
    best_eval_win_rate = -float("inf")
    best_avg_reward = float("-inf")
    global_step = 0
    total_episodes = 0
    ppo_update_count = 0
    eval_every_updates = max(0, int(eval_every_updates))
    save_every_updates = max(1, int(save_every_updates))
    if eval_every_updates > 0:
        print(f"Eval cadence: every {eval_every_updates} update(s)")
    else:
        print("Eval cadence: disabled")
    print(f"Save cadence: every {save_every_updates} update(s)")
    do_eval = eval_every_updates > 0 and (
        eval_games_per_opponent > 0
        or (trend_eval_games_per_opponent is not None and trend_eval_games_per_opponent > 0)
    )
    trend_eval_games = (
        eval_games_per_opponent
        if trend_eval_games_per_opponent is None
        else max(1, trend_eval_games_per_opponent)
    )
    played_action_buffer: list[np.ndarray] = []
    played_action_type_steps: list[int] = []
    played_action_type_history: dict[str, list[float]] = {
        name: [] for name in action_type_names
    }

    envs = make_puffer_vectorized_envs(
        reward_function=reward_function,
        map_type=map_type,
        opponent_configs=opponent_configs,
        num_envs=num_envs,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        actor_observation_level=actor_observation_level,
    )
    driver_env = envs.driver_env
    if hasattr(driver_env, "env_single_observation_space"):
        obs_space = driver_env.env_single_observation_space
    else:
        obs_space = driver_env.observation_space
    obs_dtype = getattr(driver_env, "obs_dtype", np.float32)

    def decode_observations(
        flat_obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray | None, npt.NDArray[np.bool_]]:
        actor_batch, critic_batch, action_masks = decode_puffer_batch(
            flat_obs=flat_obs,
            obs_space=obs_space,
            obs_dtype=obs_dtype,
            actor_dim=actor_input_dim,
            critic_dim=full_critic_input_dim if uses_privileged_critic else None,
        )
        if critic_batch is not None:
            critic_batch = critic_batch[:, critic_observation_indices]
        return actor_batch, critic_batch, action_masks

    env_episode_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs, dtype=int)

    print("\nStarting training...")
    observations, infos = envs.reset()

    with tqdm(total=total_timesteps, desc="Steps", unit="step", unit_scale=True) as pbar:
        while global_step < total_timesteps:
            states, critic_states, valid_action_masks = decode_observations(observations)
            states_t = torch.from_numpy(states).float().to(device)
            critic_states_t = (
                torch.from_numpy(critic_states).float().to(device)
                if critic_states is not None
                else None
            )
            actions, log_probs, _ = agent.select_actions_batch(
                states_t,
                valid_action_masks,
                deterministic=deterministic_policy,
            )
            played_action_buffer.append(actions)
            with torch.no_grad():
                values = (
                    predict_values(states_t, critic_states_t)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            next_observations, rewards, terminations, truncations, infos = envs.step(actions)

            dones = np.logical_or(terminations, truncations)
            env_episode_rewards += rewards
            env_episode_lengths += 1

            if uses_privileged_critic:
                if critic_states is None:
                    raise RuntimeError("Expected critic observations in privileged critic mode.")
                cast(CentralCriticExperienceBuffer, buffer).add_batch(
                    states,
                    critic_states,
                    actions,
                    rewards,
                    values,
                    log_probs,
                    cast(npt.NDArray[np.bool_], valid_action_masks),
                    dones,
                )
            else:
                buffer.add_batch(
                    states,
                    actions,
                    rewards,
                    values,
                    log_probs,
                    cast(npt.NDArray[np.bool_], valid_action_masks),
                    dones,
                )
            global_step += num_envs
            pbar.update(num_envs)

            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                batch_completed_episodes_rewards = env_episode_rewards[done_indices]
                batch_completed_episodes_lengths = env_episode_lengths[done_indices]
                completed_episodes = len(done_indices)

                episode_rewards.extend(batch_completed_episodes_rewards.tolist())
                episode_lengths.extend(batch_completed_episodes_lengths.tolist())
                total_episodes += completed_episodes

                avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
                avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

                wandb.log(
                    {
                        "episode/batch_completed_episodes": completed_episodes,
                        "episode/batch_completed_episodes_reward_mean": float(
                            np.mean(batch_completed_episodes_rewards)
                        ),
                        "episode/batch_completed_episodes_reward_min": float(
                            np.min(batch_completed_episodes_rewards)
                        ),
                        "episode/batch_completed_episodes_reward_max": float(
                            np.max(batch_completed_episodes_rewards)
                        ),
                        "episode/batch_completed_episodes_length_mean": float(
                            np.mean(batch_completed_episodes_lengths)
                        ),
                        "episode/batch_completed_episodes_length_min": float(
                            np.min(batch_completed_episodes_lengths)
                        ),
                        "episode/batch_completed_episodes_length_max": float(
                            np.max(batch_completed_episodes_lengths)
                        ),
                        "episode/avg_episode_reward": avg_reward,
                        "episode/avg_episode_length": avg_length,
                        "episode": total_episodes,
                    },
                    step=global_step,
                )

                env_episode_rewards[done_indices] = 0.0
                env_episode_lengths[done_indices] = 0

                if scheduler is not None and (
                    scheduler_total_iters is None or scheduler_steps_taken < scheduler_total_iters
                ):
                    pending_scheduler_steps += completed_episodes

                if save_path and avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    os.makedirs(save_path, exist_ok=True)
                    best_train_path = os.path.join(save_path, "policy_best_train_reward.pt")
                    torch.save(policy_model.state_dict(), best_train_path)
                    if critic_model is not None:
                        critic_best_train_path = os.path.join(
                            save_path, "critic_best_train_reward.pt"
                        )
                        torch.save(critic_model.state_dict(), critic_best_train_path)
                    if wandb.run is not None:
                        wandb.run.summary["best_avg_reward"] = best_avg_reward

            observations = next_observations

            if len(buffer) >= rollout_steps:
                bootstrap_actor_states, bootstrap_critic_states, _ = decode_observations(
                    observations
                )

                played_action_log: dict[str, Any] = {
                    "train/rollout_steps_collected": float(buffer.steps_collected),
                    "train/rollout_samples_collected": float(len(buffer)),
                }
                if played_action_buffer:
                    played_actions = np.concatenate(played_action_buffer)
                    played_action_type_dist, _ = compute_action_distributions(
                        played_actions,
                        action_to_type_idx,
                        action_type_names,
                        action_space_size,
                    )
                    played_action_type_steps.append(global_step)
                    for name in action_type_names:
                        ratio = played_action_type_dist.get(name, 0.0)
                        played_action_type_history[name].append(ratio)
                        played_action_log[f"rollout_played_action_type/{name}"] = ratio
                    played_action_log["rollout/played_action_type_over_time"] = (
                        build_action_type_time_series_chart(
                            title="SARL PPO Played Action Type Over Time",
                            x_values=played_action_type_steps,
                            history_by_type=played_action_type_history,
                        )
                    )
                    played_action_buffer.clear()
                wandb.log(played_action_log, step=global_step)

                metrics = run_ppo_update(
                    agent=agent,
                    value_model=value_model,
                    predict_values=predict_values,
                    policy_optimizer=policy_optimizer,
                    buffer=buffer,
                    clip_epsilon=clip_epsilon,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    activity_coef=activity_coef,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    device=device,
                    last_actor_states=bootstrap_actor_states,
                    last_critic_states=bootstrap_critic_states,
                    critic_optimizer=critic_optimizer,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    max_grad_norm=max_grad_norm,
                    target_kl=target_kl,
                    include_single_action_fraction=True,
                    warn_on_empty=True,
                )
                buffer.clear()
                policy_model.eval()
                if critic_model is not None:
                    critic_model.eval()
                ppo_update_count += 1

                # Step learning rate scheduler based on completed episodes since last update
                updates_performed = metrics.get("num_updates", 0)
                lr_before_scheduler = policy_optimizer.param_groups[0]["lr"]
                lr_after_scheduler = lr_before_scheduler
                if updates_performed > 0 and scheduler is not None and pending_scheduler_steps > 0:
                    remaining_steps = (
                        (scheduler_total_iters - scheduler_steps_taken)
                        if scheduler_total_iters is not None
                        else pending_scheduler_steps
                    )
                    if remaining_steps <= 0:
                        pending_scheduler_steps = 0
                    else:
                        steps_to_apply = min(pending_scheduler_steps, remaining_steps)
                        for _ in range(steps_to_apply):
                            scheduler.step()
                        scheduler_steps_taken += steps_to_apply
                        pending_scheduler_steps -= steps_to_apply
                        lr_after_scheduler = policy_optimizer.param_groups[0]["lr"]
                        if lr_after_scheduler != lr_before_scheduler:
                            print(
                                f"  → LR updated: {lr_before_scheduler:.6f} → "
                                f"{lr_after_scheduler:.6f} ({steps_to_apply} episode step"
                                f"{'s' if steps_to_apply != 1 else ''})"
                            )
                current_lr = lr_after_scheduler

                log_dict = {
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy": -metrics["entropy_loss"],
                    "train/activity_loss": metrics["activity_loss"],
                    "train/total_loss": metrics["total_loss"],
                    "train/learning_rate": current_lr,
                    "train/approx_kl": metrics.get("approx_kl", 0.0),
                    "train/clipfrac": metrics.get("clipfrac", 0.0),
                    "train/ratio_mean": metrics.get("ratio_mean", 0.0),
                    "train/ratio_std": metrics.get("ratio_std", 0.0),
                    "train/early_stop_kl": metrics.get("early_stop", 0.0),
                    "train/single_action_fraction": metrics.get("single_action_fraction", 0.0),
                }
                wandb.log(log_dict, step=global_step)

                if save_path and ppo_update_count % save_every_updates == 0:
                    os.makedirs(save_path, exist_ok=True)
                    snapshot_path = os.path.join(save_path, f"policy_update_{ppo_update_count}.pt")
                    torch.save(policy_model.state_dict(), snapshot_path)
                    if critic_model is not None:
                        critic_snapshot_path = os.path.join(
                            save_path, f"critic_update_{ppo_update_count}.pt"
                        )
                        torch.save(critic_model.state_dict(), critic_snapshot_path)

                if do_eval and ppo_update_count % eval_every_updates == 0:
                    eval_value_model = critic_model if critic_model is not None else policy_model
                    fresh_eval_metrics = eval_policy_value_against_baselines(
                        policy_model=policy_model,
                        critic_model=eval_value_model,
                        model_type=model_type,
                        map_type=map_type,
                        eval_opponent_configs=opponent_configs,
                        num_games=eval_games_per_opponent,
                        gamma=gamma,
                        seed=random.randint(0, sys.maxsize),
                        vps_to_win=vps_to_win,
                        discard_limit=discard_limit,
                        actor_observation_level=actor_observation_level,
                        critic_observation_level=effective_critic_observation_level,
                        log_to_wandb=False,
                        global_step=global_step,
                        device=str(device),
                        num_envs=num_envs,
                    )
                    trend_eval_metrics = eval_policy_value_against_baselines(
                        policy_model=policy_model,
                        critic_model=eval_value_model,
                        model_type=model_type,
                        map_type=map_type,
                        eval_opponent_configs=opponent_configs,
                        num_games=trend_eval_games,
                        gamma=gamma,
                        seed=trend_eval_seed if trend_eval_seed is not None else 0,
                        vps_to_win=vps_to_win,
                        discard_limit=discard_limit,
                        actor_observation_level=actor_observation_level,
                        critic_observation_level=effective_critic_observation_level,
                        log_to_wandb=False,
                        global_step=global_step,
                        device=str(device),
                        num_envs=num_envs,
                    )
                    fresh_log = {}
                    for key, value in fresh_eval_metrics.items():
                        suffix = key.split("/", 1)[1] if "/" in key else key
                        fresh_log[f"eval_fresh/{suffix}"] = value
                    trend_log = {}
                    for key, value in trend_eval_metrics.items():
                        suffix = key.split("/", 1)[1] if "/" in key else key
                        trend_log[f"eval_trend/{suffix}"] = value
                    trend_log["eval_trend/seed"] = (
                        trend_eval_seed if trend_eval_seed is not None else 0
                    )
                    trend_log["eval_trend/games_per_opponent"] = trend_eval_games
                    wandb.log({**fresh_log, **trend_log}, step=global_step)

                    if save_path:
                        eval_win_rate = 0.0
                        if (
                            trend_eval_games_per_opponent is not None
                            and trend_eval_games_per_opponent > 0
                        ):
                            eval_win_rate = float(
                                trend_eval_metrics.get("eval/win_rate_vs_value", 0.0)
                            )
                        elif eval_games_per_opponent is not None and eval_games_per_opponent > 0:
                            eval_win_rate = float(
                                fresh_eval_metrics.get("eval/win_rate_vs_value", 0.0)
                            )

                        if eval_win_rate > best_eval_win_rate:
                            best_eval_win_rate = eval_win_rate
                            os.makedirs(save_path, exist_ok=True)
                            best_path = os.path.join(save_path, "policy_best.pt")
                            torch.save(policy_model.state_dict(), best_path)
                            if critic_model is not None:
                                best_critic_path = os.path.join(save_path, "critic_best.pt")
                                torch.save(critic_model.state_dict(), best_critic_path)
                            if wandb.run is not None:
                                wandb.run.summary["best_eval_win_rate_vs_value"] = (
                                    best_eval_win_rate
                                )
                            print(
                                f"  → Saved best policy (eval win rate vs value: {best_eval_win_rate:.3f})"
                            )

    envs.close()
    if do_eval and best_eval_win_rate > -float("inf"):
        print(
            f"\nTraining complete. {global_step:,} steps, {total_episodes} episodes. "
            f"Best eval win rate vs value: {best_eval_win_rate:.3f}"
        )
    else:
        print(
            f"\nTraining complete. {global_step:,} steps, {total_episodes} episodes. "
            f"Best avg training reward: {best_avg_reward:.3f}"
        )
    return policy_model
