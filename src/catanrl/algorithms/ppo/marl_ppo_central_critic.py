from __future__ import annotations

import os
import random
import sys
from collections import deque
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import wandb

from ...envs import compute_multiagent_input_dim, decode_puffer_batch
from ...envs.puffer.multi_agent_env import make_vectorized_envs as make_marl_vectorized_envs
from ...eval.training_eval import eval_policy_against_champion, eval_policy_value_against_baselines
from ...features.catanatron_utils import (
    get_full_numeric_feature_names,
    get_numeric_feature_names,
)
from ...models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from ...models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from ...utils.catanatron_action_space import build_action_type_metadata, get_action_space_size
from ...utils.seeding import set_global_seeds
from .action_stats import build_raw_policy_log_dict
from .agent import PolicyAgent
from .backbone_builder import build_backbone_config
from .buffers import CentralCriticExperienceBuffer
from .ppo_update import run_ppo_update


def train(
    num_players: int = 2,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    model_type: str = "flat",
    backbone_type: str = "mlp",
    xdim_cnn_channels: Sequence[int] = (64, 128, 128),
    xdim_cnn_kernel_size: Tuple[int, int] = (3, 5),
    xdim_policy_fusion_hidden_dim: Optional[int] = None,
    xdim_critic_fusion_hidden_dim: Optional[int] = None,
    total_timesteps: int = 1_000_000,
    rollout_steps: int = 4096,
    policy_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    activity_coef: float = 0.0,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    policy_hidden_dims: Sequence[int] = (512, 512),
    critic_hidden_dims: Sequence[int] = (512, 512),
    save_path: Optional[str] = "weights/marl_central_critic",
    load_policy_weights: Optional[str] = None,
    load_critic_weights: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    seed: Optional[int] = 42,
    max_grad_norm: float = 0.5,
    deterministic_policy: bool = False,
    eval_games_per_opponent: int = 250,
    trend_eval_games_per_opponent: Optional[int] = None,
    trend_eval_seed: Optional[int] = 42,
    h2h_eval_games: int = 0,
    h2h_eval_seed: Optional[int] = 123,
    eval_every_updates: int = 1,
    save_every_updates: int = 1,
    target_kl: Optional[float] = None,
    num_envs: int = 2,
    reward_function: Literal["shaped", "win"] = "shaped",
    vps_to_win: int = 10,
    discard_limit: int = 7,
    metric_window: int = 200,
) -> Tuple[PolicyNetworkWrapper, ValueNetworkWrapper]:
    """Train a shared policy with a centralized critic using PPO."""

    assert 2 <= num_players <= 4, "num_players must be between 2 and 4"
    assert num_envs >= 1, "num_envs must be >= 1"
    assert backbone_type in ("mlp", "xdim", "xdim_res"), f"Unknown backbone_type '{backbone_type}'"
    xdim_cnn_channels = list(xdim_cnn_channels)
    if not xdim_cnn_channels:
        raise ValueError("xdim_cnn_channels cannot be empty")

    set_global_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_space_size = get_action_space_size(num_players, map_type)
    _, action_to_type_idx, action_type_names = build_action_type_metadata(num_players, map_type)

    actor_input_dim, board_shape, numeric_dim = compute_multiagent_input_dim(num_players, map_type)
    numeric_dim = numeric_dim or len(get_numeric_feature_names(num_players, map_type))
    board_dim = actor_input_dim - numeric_dim
    full_numeric_len = len(get_full_numeric_feature_names(num_players, map_type))
    critic_input_dim = full_numeric_len + board_dim

    print(f"\n{'=' * 60}")
    print("Centralized Critic Multi-Agent PPO Training")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type} | Players: {num_players}")
    print(f"Game config: vps_to_win={vps_to_win} | discard_limit={discard_limit}")
    print(f"Backbone: {backbone_type} | Model type: {model_type}")
    print(f"Actor input dim: {actor_input_dim} | Critic input dim: {critic_input_dim}")
    if backbone_type in ("xdim", "xdim_res"):
        print(f"Board shape (C, W, H): {board_shape} | Numeric dim: {numeric_dim}")
        print(
            f"XDim config: cnn_channels={xdim_cnn_channels}, kernel={xdim_cnn_kernel_size}, "
            f"policy_fusion={xdim_policy_fusion_hidden_dim}, critic_fusion={xdim_critic_fusion_hidden_dim}"
        )
    print(f"Total timesteps: {total_timesteps:,} | Rollout steps: {rollout_steps}")
    print(f"Parallel environments: {num_envs}")

    if wandb.run is None:
        if wandb_config:
            wandb.init(**wandb_config)
        else:
            wandb.init(mode="disabled")

    default_save_dir = "weights/marl_central_critic"
    if save_path == default_save_dir and wandb.run is not None and getattr(wandb.run, "name", None):
        save_path = os.path.join("weights", wandb.run.name)

    assert board_shape is not None
    board_channels, board_width, board_height = board_shape
    policy_backbone_config = build_backbone_config(
        backbone_type=backbone_type,
        hidden_dims=policy_hidden_dims,
        input_dim=actor_input_dim,
        board_height=board_height,
        board_width=board_width,
        board_channels=board_channels,
        numeric_dim=numeric_dim,
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,
        xdim_fusion_hidden_dim=xdim_policy_fusion_hidden_dim,
    )

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
        raise ValueError(f"Unknown model_type '{model_type}'")

    critic_backbone_config = build_backbone_config(
        backbone_type=backbone_type,
        hidden_dims=critic_hidden_dims,
        input_dim=critic_input_dim,
        board_height=board_height,
        board_width=board_width,
        board_channels=board_channels,
        numeric_dim=full_numeric_len,
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,
        xdim_fusion_hidden_dim=xdim_critic_fusion_hidden_dim,
    )
    critic_model = build_value_network(backbone_config=critic_backbone_config).to(device)

    if load_policy_weights and os.path.exists(load_policy_weights):
        print(f"Loading policy weights from {load_policy_weights}")
        policy_model.load_state_dict(torch.load(load_policy_weights, map_location=device))
    if load_critic_weights and os.path.exists(load_critic_weights):
        print(f"Loading critic weights from {load_critic_weights}")
        critic_model.load_state_dict(torch.load(load_critic_weights, map_location=device))

    policy_agent = PolicyAgent(policy_model, model_type, device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=policy_lr)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=critic_lr)

    def predict_values(
        actor_states: torch.Tensor,
        critic_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if critic_states is None:
            raise ValueError("critic_states must be provided for centralized critic updates.")
        return critic_model(critic_states).squeeze(-1)

    buffer = CentralCriticExperienceBuffer(
        num_rollouts=rollout_steps,
        actor_state_dim=actor_input_dim,
        critic_state_dim=critic_input_dim,
        action_space_size=action_space_size,
        num_envs=num_envs * num_players,
    )

    envs = make_marl_vectorized_envs(
        num_players=num_players,
        map_type=map_type,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        shared_critic=True,
        reward_function=reward_function,
        num_envs=num_envs,
    )
    driver_env = envs.driver_env
    obs_space = driver_env.env_single_observation_space
    obs_dtype = driver_env.obs_dtype
    agents_per_env = getattr(envs, "agents_per_env", [driver_env.num_agents] * num_envs)
    env_offsets = []
    ptr = 0
    for count in agents_per_env:
        env_offsets.append(ptr)
        ptr += count

    def decode_observations(flat_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        actor_batch, critic_batch, action_masks = decode_puffer_batch(
            flat_obs=flat_obs,
            obs_space=obs_space,
            obs_dtype=obs_dtype,
            actor_dim=actor_input_dim,
            critic_dim=critic_input_dim,
        )
        if critic_batch is None:
            raise RuntimeError("Expected critic observations for MARL centralized critic training.")
        return actor_batch, critic_batch, action_masks

    metric_window = max(1, metric_window)
    eval_every_updates = max(1, int(eval_every_updates))
    save_every_updates = max(1, int(save_every_updates))
    update_every_updates = 1
    if save_every_updates % update_every_updates != 0:
        raise ValueError(
            f"save_every_updates must be a multiple of update frequency ({update_every_updates})"
        )
    if save_every_updates % eval_every_updates != 0:
        raise ValueError(
            f"save_every_updates must be a multiple of eval_every_updates ({eval_every_updates})"
        )
    print(f"Eval cadence: every {eval_every_updates} update(s)")
    print(f"Save cadence: every {save_every_updates} update(s)")
    trend_eval_games = (
        eval_games_per_opponent
        if trend_eval_games_per_opponent is None
        else max(1, trend_eval_games_per_opponent)
    )
    h2h_eval_games = max(0, int(h2h_eval_games))
    best_eval_win_rate = -float("inf")
    best_eval_critic_mse = float("inf")
    global_step = 0
    total_episodes = 0
    ppo_update_count = 0
    episode_lengths: list[int] = [0 for _ in range(num_envs)]
    length_window = deque(maxlen=metric_window)

    # Track raw policy preferences (argmax before masking) for diagnostics
    raw_policy_argmax_buffer: list[np.ndarray] = []

    def run_and_log_evals() -> None:
        nonlocal best_eval_win_rate, best_eval_critic_mse

        policy_agent.model.eval()
        critic_model.eval()
        with torch.no_grad():
            fresh_eval_metrics = eval_policy_value_against_baselines(
                policy_model=policy_model,
                critic_model=critic_model,
                model_type=model_type,
                map_type=map_type,
                eval_opponent_configs=["random"] * (num_players - 1),
                num_games=eval_games_per_opponent,
                gamma=gamma,
                seed=random.randint(0, sys.maxsize),
                vps_to_win=vps_to_win,
                discard_limit=discard_limit,
                log_to_wandb=False,
                global_step=global_step,
                device=device,
                num_envs=num_envs,
            )
            trend_eval_metrics = eval_policy_value_against_baselines(
                policy_model=policy_model,
                critic_model=critic_model,
                model_type=model_type,
                map_type=map_type,
                eval_opponent_configs=["random"] * (num_players - 1),
                num_games=trend_eval_games,
                gamma=gamma,
                seed=trend_eval_seed if trend_eval_seed is not None else 0,
                vps_to_win=vps_to_win,
                discard_limit=discard_limit,
                log_to_wandb=False,
                global_step=global_step,
                device=device,
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
        trend_log["eval_trend/seed"] = trend_eval_seed if trend_eval_seed is not None else 0
        trend_log["eval_trend/games_per_opponent"] = trend_eval_games
        h2h_log = {}
        champion_policy_path = None
        if save_path:
            champion_policy_path = os.path.join(save_path, "policy_best.pt")
        champion_available = bool(
            h2h_eval_games > 0 and champion_policy_path and os.path.exists(champion_policy_path)
        )
        h2h_log["eval_h2h/champion_available"] = float(champion_available)
        if h2h_eval_games > 0:
            h2h_log["eval_h2h/games"] = float(h2h_eval_games)
            h2h_log["eval_h2h/seed"] = h2h_eval_seed if h2h_eval_seed is not None else 0
        wandb.log({**fresh_log, **trend_log, **h2h_log}, step=global_step)

        if champion_available and champion_policy_path is not None:
            h2h_metrics = eval_policy_against_champion(
                policy_model=policy_model,
                model_type=model_type,
                map_type=map_type,
                num_players=num_players,
                champion_policy_path=champion_policy_path,
                num_games=h2h_eval_games,
                seed=h2h_eval_seed if h2h_eval_seed is not None else 0,
                vps_to_win=vps_to_win,
                discard_limit=discard_limit,
                log_to_wandb=False,
                global_step=global_step,
            )
            if h2h_metrics:
                wandb.log(h2h_metrics, step=global_step)
        elif h2h_eval_games > 0:
            print("  → Skipping H2H eval (no champion checkpoint available yet)")

        if save_path:
            eval_win_rate = None
            if trend_eval_games_per_opponent is not None and trend_eval_games_per_opponent > 0:
                eval_win_rate = float(trend_eval_metrics.get("eval/win_rate_vs_value", 0.0))
            elif eval_games_per_opponent is not None and eval_games_per_opponent > 0:
                eval_win_rate = float(fresh_eval_metrics.get("eval/win_rate_vs_value", 0.0))

            if eval_win_rate is not None and eval_win_rate > best_eval_win_rate:
                best_eval_win_rate = eval_win_rate
                save_dir = save_path
                os.makedirs(save_dir, exist_ok=True)
                best_policy_path = os.path.join(save_dir, "policy_best.pt")
                torch.save(policy_model.state_dict(), best_policy_path)
                if wandb.run is not None:
                    wandb.run.summary["best_eval_win_rate_vs_value"] = best_eval_win_rate
                print(
                    f"  → Saved best policy (eval win rate vs value: {best_eval_win_rate:.3f})"
                )

            eval_critic_mse = trend_eval_metrics.get("eval/value_mse", float("inf"))
            if eval_critic_mse < best_eval_critic_mse:
                best_eval_critic_mse = eval_critic_mse
                save_dir = save_path
                os.makedirs(save_dir, exist_ok=True)
                best_critic_path = os.path.join(save_dir, "critic_best.pt")
                torch.save(critic_model.state_dict(), best_critic_path)
                if wandb.run is not None:
                    wandb.run.summary["best_eval_critic_mse"] = best_eval_critic_mse
                print(f"  → Saved best critic (eval MSE: {best_eval_critic_mse:.4f})")

    print("Running step-0 baseline eval before PPO updates...")
    run_and_log_evals()
    observations, infos = envs.reset()

    try:
        with tqdm(total=total_timesteps, desc="Steps", unit="step", unit_scale=True) as pbar:
            while global_step < total_timesteps:
                actor_batch, critic_batch, valid_masks = decode_observations(observations)
                actor_tensor = torch.from_numpy(actor_batch).float().to(device)
                actions, log_probs, raw_argmax = policy_agent.select_actions_batch(
                    actor_tensor, valid_masks, deterministic=deterministic_policy
                )
                raw_policy_argmax_buffer.append(raw_argmax)

                critic_inputs = torch.from_numpy(critic_batch).float().to(device)
                critic_model.eval()
                with torch.no_grad():
                    values = critic_model(critic_inputs).squeeze(-1).detach().cpu().numpy()
                critic_model.train()

                next_observations, rewards, terminations, truncations, infos = envs.step(actions)
                rewards = rewards.astype(np.float32)
                dones = np.logical_or(terminations, truncations)

                buffer.add_batch(
                    actor_batch,
                    critic_batch,
                    actions,
                    rewards,
                    values,
                    log_probs,
                    valid_masks,
                    dones,
                )
                global_step += num_envs
                pbar.update(num_envs)

                for env_idx, start in enumerate(env_offsets):
                    end = start + agents_per_env[env_idx]
                    idx = int(env_idx)
                    episode_lengths[idx] += 1
                    if dones[start:end].all():
                        total_episodes += 1
                        length_window.append(episode_lengths[idx])
                        avg_length = float(np.mean(length_window))
                        wandb.log(
                            {
                                "episode/avg_length": avg_length,
                                "episode/count": total_episodes,
                            },
                            step=global_step,
                        )
                        episode_lengths[idx] = 0

                observations = next_observations

                if buffer.steps_collected >= rollout_steps and len(buffer) >= batch_size * 2:
                    # Log raw policy preference distribution (argmax before masking)
                    # This reveals what the network inherently prefers, regardless of action validity
                    raw_policy_log = build_raw_policy_log_dict(
                        raw_policy_argmax_buffer,
                        action_to_type_idx,
                        action_type_names,
                        action_space_size,
                        rollout_steps_collected=int(buffer.steps_collected),
                        rollout_samples_collected=int(len(buffer)),
                    )
                    wandb.log(raw_policy_log, step=global_step)
                    raw_policy_argmax_buffer.clear()

                    bootstrap_actor_batch, bootstrap_critic_batch, _ = decode_observations(
                        observations
                    )
                    policy_agent.model.train()
                    metrics = run_ppo_update(
                        agent=policy_agent,
                        value_model=critic_model,
                        predict_values=predict_values,
                        policy_optimizer=policy_optimizer,
                        critic_optimizer=critic_optimizer,
                        buffer=buffer,
                        clip_epsilon=clip_epsilon,
                        value_coef=value_coef,
                        entropy_coef=entropy_coef,
                        activity_coef=activity_coef,
                        n_epochs=ppo_epochs,
                        batch_size=batch_size,
                        device=device,
                        last_actor_states=bootstrap_actor_batch,
                        last_critic_states=bootstrap_critic_batch,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        max_grad_norm=max_grad_norm,
                        target_kl=target_kl,
                    )
                    buffer.clear()
                    policy_agent.model.eval()
                    critic_model.eval()
                    ppo_update_count += 1

                    wandb.log(
                        {
                            "train/policy_loss": metrics["policy_loss"],
                            "train/value_loss": metrics["value_loss"],
                            "train/entropy": -metrics["entropy_loss"],
                            "train/activity_loss": metrics["activity_loss"],
                            "train/total_loss": metrics["total_loss"],
                            "train/approx_kl": metrics["approx_kl"],
                            "train/clipfrac": metrics["clipfrac"],
                            "train/ratio_mean": metrics["ratio_mean"],
                            "train/ratio_std": metrics["ratio_std"],
                            "train/early_stop_kl": metrics["early_stop"],
                        },
                        step=global_step,
                    )

                    if save_path and ppo_update_count % save_every_updates == 0:
                        save_dir = save_path
                        os.makedirs(save_dir, exist_ok=True)
                        snapshot_name = f"policy_update_{ppo_update_count}.pt"
                        policy_path = os.path.join(save_dir, snapshot_name)
                        critic_path = os.path.join(
                            save_dir,
                            f"{os.path.splitext(snapshot_name)[0]}_critic.pt",
                        )
                        torch.save(policy_model.state_dict(), policy_path)
                        torch.save(critic_model.state_dict(), critic_path)

                    # Evaluate policy against catanatron bots and critic value predictions.
                    if ppo_update_count % eval_every_updates == 0:
                        run_and_log_evals()

    finally:
        envs.close()

    if best_eval_win_rate > -float("inf"):
        print(
            f"\nTraining complete. {global_step:,} steps, {total_episodes} episodes. "
            f"Best eval win rate vs value: {best_eval_win_rate:.3f}"
        )
    else:
        print(
            f"\nTraining complete. {global_step:,} steps, {total_episodes} episodes. "
            "Best eval win rate vs value: n/a"
        )
    return policy_model, critic_model
