import copy

import numpy as np
import pytest
import torch

from catanrl.algorithms.common import PolicyAgent
from catanrl.algorithms.ppo.buffers import CentralCriticExperienceBuffer, TorchCentralCriticBuffer
from catanrl.algorithms.ppo.ppo_update import run_ppo_update
from catanrl.models.backbone_builder import build_backbone_config
from catanrl.models.models import build_flat_policy_network, build_value_network

STATE_DIM, NUM_ACTIONS, STEPS, ROWS = 24, 9, 16, 8


def _models():
    config = build_backbone_config(backbone_type="mlp", hidden_dims=(32,), input_dim=STATE_DIM)
    return build_flat_policy_network(config, NUM_ACTIONS), build_value_network(config)


def _fill(buffer, rng: np.random.Generator) -> np.ndarray:
    for _ in range(STEPS):
        states = rng.normal(size=(ROWS, STATE_DIM)).astype(np.float32)
        masks = rng.random((ROWS, NUM_ACTIONS)) < 0.5
        masks[:, 0] = True
        masks[::3] = False  # forced single-action rows, like the idle MARL seat
        masks[::3, 0] = True
        actions = np.array([rng.choice(np.flatnonzero(row)) for row in masks])
        buffer.add_batch(
            states,
            states,
            actions,
            rng.normal(size=ROWS).astype(np.float32),
            rng.normal(size=ROWS).astype(np.float32),
            np.log(1.0 / masks.sum(1)).astype(np.float32),
            masks,
            rng.random(ROWS) < 0.05,
        )
    return rng.normal(size=(ROWS, STATE_DIM)).astype(np.float32)


def _update(buffer, policy, critic, last_states, device):
    optimizers = (torch.optim.Adam(policy.parameters()), torch.optim.Adam(critic.parameters()))
    np.random.seed(0)
    return run_ppo_update(
        agent=PolicyAgent(policy, "flat", device),
        value_model=critic,
        predict_values=lambda actor, critic_states: critic(critic_states),
        policy_optimizer=optimizers[0],
        critic_optimizer=optimizers[1],
        buffer=buffer,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        activity_coef=0.0,
        train_epochs=2,
        batch_size=32,
        device=device,
        last_actor_states=last_states,
        last_critic_states=last_states,
        include_single_action_fraction=True,
    )


@pytest.mark.parametrize("share_states", [False, True])
def test_torch_buffer_update_matches_numpy_buffer(share_states):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    policy, critic = _models()
    policy, critic = policy.to(device), critic.to(device)
    reference = (copy.deepcopy(policy), copy.deepcopy(critic))

    numpy_buffer = CentralCriticExperienceBuffer(STEPS, STATE_DIM, STATE_DIM, NUM_ACTIONS, ROWS)
    last = _fill(numpy_buffer, np.random.default_rng(1))
    torch_buffer = TorchCentralCriticBuffer(
        STEPS, STATE_DIM, STATE_DIM, NUM_ACTIONS, ROWS, device=device, share_states=share_states
    )
    _fill(torch_buffer, np.random.default_rng(1))

    expected = _update(numpy_buffer, *reference, last, device)
    actual = _update(torch_buffer, policy, critic, torch.as_tensor(last), device)

    for key in ("policy_loss", "value_loss", "approx_kl", "single_action_fraction", "num_updates"):
        assert actual[key] == pytest.approx(expected[key], rel=1e-4, abs=1e-6)
    for model, ref in ((policy, reference[0]), (critic, reference[1])):
        for param, ref_param in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(param, ref_param, rtol=1e-4, atol=1e-6)


def test_storage_bytes_counts_shared_states_once():
    separate = TorchCentralCriticBuffer.storage_bytes(128, 3770, 3770, 313, 256)
    shared = TorchCentralCriticBuffer.storage_bytes(128, 3770, 3770, 313, 256, share_states=True)
    assert separate - shared == 128 * 256 * 3770 * 4
