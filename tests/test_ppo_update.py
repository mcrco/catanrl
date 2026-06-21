"""PPO math: action masking (the NaN guard) and a full run_ppo_update pass that
exercises GAE integration, advantage normalization, single-action exclusion, and
actual gradient steps with tiny stand-in networks."""

from __future__ import annotations

import copy

import numpy as np
import torch

from catanrl.algorithms.common import PolicyAgent, mask_action_logits
from catanrl.algorithms.ppo.buffers import ExperienceBuffer
from catanrl.algorithms.ppo.ppo_update import run_ppo_update


# ---------------------------------------------------------------------------
# mask_action_logits
# ---------------------------------------------------------------------------


def test_mask_sets_invalid_to_neg_inf_and_keeps_valid():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = np.array([[True, False, True, False]])
    masked, _ = mask_action_logits(logits, mask)
    assert masked[0, 0].item() == 1.0
    assert masked[0, 2].item() == 3.0
    assert masked[0, 1].item() == float("-inf")
    assert masked[0, 3].item() == float("-inf")


def test_all_invalid_row_stays_finite_to_avoid_nan_distribution():
    # If a row has no valid action the function must fall back to all-valid so the
    # downstream Categorical does not collapse to NaN (which poisons training).
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = np.array([[False, False, False]])
    masked, _ = mask_action_logits(logits, mask)
    assert torch.isfinite(masked).all()
    dist = torch.distributions.Categorical(logits=masked)
    assert torch.isfinite(dist.entropy()).all()


def test_mask_accepts_1d_mask_and_batches_it():
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    masked, mask = mask_action_logits(logits, np.array([True, False, True]))
    assert masked.shape == (1, 3)
    assert mask[0, 1].item() is False


# ---------------------------------------------------------------------------
# run_ppo_update
# ---------------------------------------------------------------------------


def _build_update_inputs(num_actions=4, actor_dim=3, time_steps=8, num_envs=4, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    policy_model = torch.nn.Linear(actor_dim, num_actions)
    value_model = torch.nn.Linear(actor_dim, 1)
    agent = PolicyAgent(policy_model, "flat", "cpu")

    def predict_values(actor_states, critic_states):
        return value_model(actor_states).squeeze(-1)

    buffer = ExperienceBuffer(
        num_rollouts=time_steps,
        state_dim=actor_dim,
        action_space_size=num_actions,
        num_envs=num_envs,
    )

    single_action_steps = 0
    total_steps = time_steps * num_envs
    for t in range(time_steps):
        masks = np.zeros((num_envs, num_actions), dtype=np.bool_)
        for e in range(num_envs):
            if (t + e) % 3 == 0:
                # single-action (forced) timestep: exactly one valid action
                masks[e, 0] = True
                single_action_steps += 1
            else:
                masks[e, : rng.integers(2, num_actions + 1)] = True
        actions = np.array(
            [int(np.flatnonzero(masks[e])[0]) for e in range(num_envs)], dtype=np.int64
        )
        buffer.add_batch(
            states=rng.standard_normal((num_envs, actor_dim)).astype(np.float32),
            actions=actions,
            rewards=rng.standard_normal(num_envs).astype(np.float32),
            values=rng.standard_normal(num_envs).astype(np.float32),
            log_probs=(-rng.random(num_envs)).astype(np.float32),
            valid_action_masks=masks,
            dones=(rng.random(num_envs) < 0.2),
        )

    last_actor_states = rng.standard_normal((num_envs, actor_dim)).astype(np.float32)
    expected_single_fraction = single_action_steps / total_steps
    return agent, value_model, predict_values, buffer, last_actor_states, expected_single_fraction


def test_run_ppo_update_produces_finite_metrics_and_updates_policy():
    (
        agent,
        value_model,
        predict_values,
        buffer,
        last_actor_states,
        expected_single_fraction,
    ) = _build_update_inputs()

    policy_before = copy.deepcopy(agent.model.state_dict())
    value_before = copy.deepcopy(value_model.state_dict())

    metrics = run_ppo_update(
        agent=agent,
        value_model=value_model,
        predict_values=predict_values,
        policy_optimizer=torch.optim.SGD(agent.model.parameters(), lr=0.1),
        buffer=buffer,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        activity_coef=0.0,
        train_epochs=3,
        batch_size=8,
        device="cpu",
        last_actor_states=last_actor_states,
        critic_optimizer=torch.optim.SGD(value_model.parameters(), lr=0.1),
        include_single_action_fraction=True,
    )

    assert metrics["num_updates"] >= 1
    for key in ("policy_loss", "value_loss", "entropy_loss", "total_loss", "approx_kl"):
        assert np.isfinite(metrics[key])

    assert abs(metrics["single_action_fraction"] - expected_single_fraction) < 1e-6

    # Both networks must actually move.
    assert any(
        not torch.equal(policy_before[k], agent.model.state_dict()[k])
        for k in policy_before
    )
    assert any(
        not torch.equal(value_before[k], value_model.state_dict()[k]) for k in value_before
    )


def test_run_ppo_update_on_empty_buffer_returns_zeroed_metrics():
    policy_model = torch.nn.Linear(3, 4)
    value_model = torch.nn.Linear(3, 1)
    agent = PolicyAgent(policy_model, "flat", "cpu")
    buffer = ExperienceBuffer(num_rollouts=4, state_dim=3, action_space_size=4, num_envs=2)

    metrics = run_ppo_update(
        agent=agent,
        value_model=value_model,
        predict_values=lambda a, c: value_model(a).squeeze(-1),
        policy_optimizer=torch.optim.SGD(agent.model.parameters(), lr=0.1),
        buffer=buffer,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        activity_coef=0.0,
        train_epochs=2,
        batch_size=8,
        device="cpu",
        critic_optimizer=torch.optim.SGD(value_model.parameters(), lr=0.1),
    )

    assert metrics["num_updates"] == 0.0
    assert metrics["total_loss"] == 0.0
