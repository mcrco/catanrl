from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .agent import PolicyAgent
from .buffers import CentralCriticExperienceBuffer, ExperienceBuffer
from .gae import compute_gae_batched


def _has_nan_grad(parameters) -> bool:
    return any(param.grad is not None and torch.isnan(param.grad).any() for param in parameters)


def _empty_metrics(
    *,
    early_stop: bool = False,
    single_action_fraction: float | None = None,
    num_updates: int = 0,
) -> Dict[str, float]:
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_loss": 0.0,
        "activity_loss": 0.0,
        "total_loss": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "ratio_mean": 0.0,
        "ratio_std": 0.0,
        "num_updates": float(num_updates),
        "early_stop": 1.0 if early_stop else 0.0,
    }
    if single_action_fraction is not None:
        metrics["single_action_fraction"] = single_action_fraction
    return metrics


def run_ppo_update(
    agent: PolicyAgent,
    value_model: torch.nn.Module,
    predict_values: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    policy_optimizer: torch.optim.Optimizer,
    buffer: ExperienceBuffer | CentralCriticExperienceBuffer,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    activity_coef: float,
    n_epochs: int,
    batch_size: int,
    device: str | torch.device,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    last_actor_states: np.ndarray | None = None,
    last_critic_states: np.ndarray | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    include_single_action_fraction: bool = False,
    warn_on_empty: bool = False,
) -> Dict[str, float]:
    """Shared PPO update for SARL and MARL trainers."""
    if len(buffer) == 0:
        if warn_on_empty:
            print("No experiences found for PPO update. Skipping update.")
        return _empty_metrics(
            single_action_fraction=0.0 if include_single_action_fraction else None,
        )

    if isinstance(buffer, CentralCriticExperienceBuffer):
        (
            actor_states_tmj,
            critic_states_tmj,
            actions_tmj,
            rewards_tmj,
            old_values_tmj,
            old_log_probs_tmj,
            valid_action_masks_tmj,
            dones_tmj,
        ) = buffer.get()
    else:
        (
            actor_states_tmj,
            actions_tmj,
            rewards_tmj,
            old_values_tmj,
            old_log_probs_tmj,
            valid_action_masks_tmj,
            dones_tmj,
        ) = buffer.get()
        critic_states_tmj = None

    time_steps, num_envs = actions_tmj.shape

    uses_separate_critic = value_model is not agent.model
    if uses_separate_critic and critic_optimizer is None:
        raise ValueError("critic_optimizer is required when using a separate critic.")
    if not uses_separate_critic and critic_optimizer is not None:
        raise ValueError("critic_optimizer should be None when the value model is shared.")

    if last_actor_states is None:
        raise ValueError("last_actor_states is required for PPO bootstrap updates.")
    next_actor_state_t = torch.from_numpy(last_actor_states).float().to(device)
    next_critic_state_t = (
        torch.from_numpy(last_critic_states).float().to(device)
        if last_critic_states is not None
        else None
    )

    policy_was_training = agent.model.training
    value_was_training = value_model.training
    agent.model.eval()
    if value_model is not agent.model:
        value_model.eval()
    with torch.no_grad():
        next_values = (
            predict_values(next_actor_state_t, next_critic_state_t)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    if policy_was_training:
        agent.model.train()
    if value_model is not agent.model and value_was_training:
        value_model.train()

    advantages_tmj, returns_tmj = compute_gae_batched(
        rewards_tmj, old_values_tmj, dones_tmj, next_values, gamma, gae_lambda
    )

    actor_states = actor_states_tmj.reshape(time_steps * num_envs, -1)
    critic_states = (
        critic_states_tmj.reshape(time_steps * num_envs, -1)
        if critic_states_tmj is not None
        else None
    )
    actions = actions_tmj.reshape(-1)
    old_log_probs = old_log_probs_tmj.reshape(-1)
    advantages = advantages_tmj.reshape(-1)
    returns = returns_tmj.reshape(-1)
    valid_action_masks = valid_action_masks_tmj.reshape(time_steps * num_envs, -1)

    single_action_fraction = float((valid_action_masks.sum(axis=1) <= 1).mean())
    is_decision = valid_action_masks.sum(axis=-1) > 1
    if not np.any(is_decision):
        return _empty_metrics(
            single_action_fraction=single_action_fraction if include_single_action_fraction else None,
        )

    adv_decision = advantages[is_decision]
    if adv_decision.std() > 1e-8:
        adv_decision = (adv_decision - adv_decision.mean()) / (adv_decision.std() + 1e-8)
    advantages_norm = np.zeros_like(advantages, dtype=np.float32)
    advantages_norm[is_decision] = adv_decision.astype(np.float32, copy=False)

    actor_states_t = torch.from_numpy(actor_states).float()
    critic_states_t = torch.from_numpy(critic_states).float() if critic_states is not None else None
    actions_t = torch.from_numpy(actions).long()
    old_log_probs_t = torch.from_numpy(old_log_probs).float()
    advantages_t = torch.from_numpy(advantages_norm).float()
    returns_t = torch.from_numpy(returns).float()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_activity_loss = 0.0
    total_loss = 0.0
    total_approx_kl = 0.0
    total_clipfrac = 0.0
    total_ratio_mean = 0.0
    total_ratio_std = 0.0
    n_updates = 0
    early_stop = False

    for _ in range(n_epochs):
        indices = np.random.permutation(len(actor_states))
        for start in range(0, len(actor_states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            if len(batch_indices) <= 1:
                if warn_on_empty:
                    print(f"Skipping small batch: {len(batch_indices)}")
                continue

            batch_actor_states = actor_states_t[batch_indices].to(device)
            batch_critic_states = (
                critic_states_t[batch_indices].to(device) if critic_states_t is not None else None
            )
            batch_actions = actions_t[batch_indices].to(device)
            batch_old_log_probs = old_log_probs_t[batch_indices].to(device)
            batch_advantages = advantages_t[batch_indices].to(device)
            batch_returns = returns_t[batch_indices].to(device)
            batch_valid_action_masks = valid_action_masks[batch_indices]
            batch_is_decision = is_decision[batch_indices]

            log_probs, entropy, policy_logits = agent.evaluate_actions(
                batch_actor_states, batch_actions, batch_valid_action_masks
            )

            values = predict_values(batch_actor_states, batch_critic_states)

            decision_count = int(np.sum(batch_is_decision))
            if decision_count > 0:
                decision_mask_t = torch.as_tensor(batch_is_decision, device=device, dtype=torch.bool)
                log_probs_d = log_probs[decision_mask_t]
                entropy_d = entropy[decision_mask_t]
                logits_d = policy_logits[decision_mask_t]
                old_log_probs_d = batch_old_log_probs[decision_mask_t]
                advantages_d = batch_advantages[decision_mask_t]

                ratio = torch.exp(log_probs_d - old_log_probs_d)
                surr1 = ratio * advantages_d
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_d
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy_d.mean()
                activity_loss = (logits_d**2).mean()
                approx_kl = float((old_log_probs_d - log_probs_d).mean().item())
                clipfrac = float(((ratio - 1.0).abs() > clip_epsilon).float().mean().item())
                ratio_mean = float(ratio.mean().item())
                ratio_std = float(ratio.std(unbiased=False).item())
            else:
                policy_loss = torch.tensor(0.0, device=device)
                entropy_loss = torch.tensor(0.0, device=device)
                activity_loss = torch.tensor(0.0, device=device)
                approx_kl = 0.0
                clipfrac = 0.0
                ratio_mean = 0.0
                ratio_std = 0.0

            value_loss = F.mse_loss(values, batch_returns)
            if uses_separate_critic:
                assert critic_optimizer is not None
                if decision_count > 0:
                    policy_optimizer.zero_grad()
                    (
                        policy_loss + entropy_coef * entropy_loss + activity_coef * activity_loss
                    ).backward()
                    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_grad_norm)
                    if _has_nan_grad(agent.model.parameters()):
                        if warn_on_empty:
                            print("  WARNING: NaN policy gradients detected! Skipping batch update.")
                        policy_optimizer.zero_grad()
                        continue
                    policy_optimizer.step()

                critic_optimizer.zero_grad()
                (value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_grad_norm)
                if _has_nan_grad(value_model.parameters()):
                    if warn_on_empty:
                        print("  WARNING: NaN critic gradients detected! Skipping batch update.")
                    critic_optimizer.zero_grad()
                    continue
                critic_optimizer.step()

                loss = (
                    policy_loss
                    + value_coef * value_loss
                    + entropy_coef * entropy_loss
                    + activity_coef * activity_loss
                )
            else:
                loss = (
                    policy_loss
                    + value_coef * value_loss
                    + entropy_coef * entropy_loss
                    + activity_coef * activity_loss
                )
                policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_grad_norm)
                if _has_nan_grad(agent.model.parameters()):
                    if warn_on_empty:
                        print("  WARNING: NaN gradients detected! Skipping batch update.")
                    policy_optimizer.zero_grad()
                    continue
                policy_optimizer.step()

            total_approx_kl += approx_kl
            total_clipfrac += clipfrac
            total_ratio_mean += ratio_mean
            total_ratio_std += ratio_std
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())
            total_activity_loss += float(activity_loss.item())
            total_loss += float(loss.item())
            n_updates += 1

            if target_kl is not None and approx_kl > 1.5 * target_kl:
                early_stop = True
                break
        if early_stop:
            break

    if n_updates == 0:
        if warn_on_empty:
            print("  WARNING: n_updates was 0! Loop didn't run or all batches were too small.")
        return _empty_metrics(
            early_stop=early_stop,
            single_action_fraction=single_action_fraction if include_single_action_fraction else None,
        )

    metrics = {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy_loss": total_entropy_loss / n_updates,
        "activity_loss": total_activity_loss / n_updates,
        "total_loss": total_loss / n_updates,
        "approx_kl": total_approx_kl / n_updates,
        "clipfrac": total_clipfrac / n_updates,
        "ratio_mean": total_ratio_mean / n_updates,
        "ratio_std": total_ratio_std / n_updates,
        "num_updates": float(n_updates),
        "early_stop": 1.0 if early_stop else 0.0,
    }
    if include_single_action_fraction:
        metrics["single_action_fraction"] = single_action_fraction
    return metrics


__all__ = ["run_ppo_update"]
