from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ...agents.sarl_agent import SARLAgent
from .buffers import OnPolicyBuffer
from .utils import compute_gae


def ppo_update(
    agent: SARLAgent,
    optimizer: torch.optim.Optimizer,
    buffer: OnPolicyBuffer,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    n_epochs: int,
    batch_size: int,
    device: str,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """
    Run PPO updates over the collected on-policy buffer for SARL training.
    """
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

    states, actions, rewards, old_values, old_log_probs, valid_actions, dones = buffer.get()

    agent.model.eval()
    with torch.no_grad():
        next_state = torch.from_numpy(states[-1:]).float().to(device)
        if agent.model_type == 'flat':
            _, next_value = agent.model(next_state)
        elif agent.model_type == 'hierarchical':
            _, _, next_value = agent.model(next_state)
        next_value = next_value.item()
    agent.model.train()

    advantages, returns = compute_gae(rewards, old_values, dones, next_value, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states_t = torch.from_numpy(states).float().to(device)
    actions_t = torch.from_numpy(actions).to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_loss = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            if len(batch_indices) <= 1:
                continue

            batch_states = states_t[batch_indices]
            batch_actions = actions_t[batch_indices]
            batch_old_log_probs = old_log_probs_t[batch_indices]
            batch_advantages = advantages_t[batch_indices]
            batch_returns = returns_t[batch_indices]
            batch_valid_actions = [valid_actions[i] for i in batch_indices]

            log_probs, values, entropy = agent.evaluate_actions(
                batch_states, batch_actions, batch_valid_actions
            )

            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_grad_norm)

            has_nan_grad = any(
                param.grad is not None and torch.isnan(param.grad).any()
                for param in agent.model.parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad()
                continue

            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())
            total_loss += float(loss.item())
            n_updates += 1

    if n_updates == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy_loss": total_entropy_loss / n_updates,
        "total_loss": total_loss / n_updates,
    }

