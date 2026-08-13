from __future__ import annotations

import copy
import os
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from catanatron.models.player import Color
from torch import nn

from catanrl.algorithms.alphazero.parallel_self_play import SelfPlayExperience
from catanrl.algorithms.alphazero.replay_buffer import DiskReplayBuffer
from catanrl.algorithms.alphazero.native_self_play import (
    _NativeSelfPlaySample,
    _compute_auxiliary_value_targets,
)
from catanrl.algorithms.alphazero.trainer import AlphaZeroConfig, AlphaZeroTrainer
from catanrl.eval.search_training import decide_promotion
from catanrl.experiments.train_alphazero import (
    _load_initial_weights,
    _update_search_teacher,
    parse_args,
)
from catanrl.models.heads import FlatPolicyHead, ValueHead, WDLValueHead
from catanrl.models.wrappers import (
    PolicyNetworkWrapper,
    PolicyValueNetworkWrapper,
    ValueNetworkWrapper,
)
from catanrl.players import BeliefAveragedPolicyPlayer, NNPolicyPlayer


def _models():
    policy = PolicyNetworkWrapper(nn.Identity(), FlatPolicyHead(3, 4))
    critic = ValueNetworkWrapper(nn.Identity(), nn.Linear(2, 1))
    return policy, critic


def _trainer(
    *,
    value_loss_weight: float = 0.0,
    offload_inactive_models: bool = False,
    device: str = "cpu",
    self_play_iteration_offset: int = 0,
) -> AlphaZeroTrainer:
    torch.manual_seed(0)
    student_policy, student_critic = _models()
    teacher_policy = copy.deepcopy(student_policy)
    teacher_critic = copy.deepcopy(student_critic)
    config = AlphaZeroConfig(
        mode="distill",
        model_type="flat",
        num_players=2,
        buffer_size=8,
        batch_size=2,
        policy_lr=0.1,
        critic_lr=0.1,
        value_loss_weight=value_loss_weight,
        offload_inactive_models=offload_inactive_models,
        device=device,
        seed=7,
        self_play_iteration_offset=self_play_iteration_offset,
    )
    return AlphaZeroTrainer(
        config,
        student_policy,
        student_critic,
        teacher_policy,
        teacher_critic,
    )


def _shared_trainer(
    *,
    categorical_value: bool = False,
    soft_policy_temperature: float = 0.0,
    soft_policy_weight: float = 0.0,
    aux_value_horizons: tuple[int, ...] = (),
    aux_value_weight: float = 0.0,
) -> AlphaZeroTrainer:
    torch.manual_seed(0)
    student = PolicyValueNetworkWrapper(
        nn.Linear(3, 3),
        FlatPolicyHead(3, 4),
        WDLValueHead(3) if categorical_value else ValueHead(3),
    )
    teacher = copy.deepcopy(student)
    config = AlphaZeroConfig(
        mode="iterate",
        model_type="flat",
        num_players=2,
        buffer_size=8,
        batch_size=2,
        policy_lr=0.1,
        critic_lr=99.0,
        value_loss_weight=1.0,
        soft_policy_temperature=soft_policy_temperature,
        soft_policy_weight=soft_policy_weight,
        aux_value_horizons=aux_value_horizons,
        aux_value_weight=aux_value_weight,
        self_play_backend="cppanatron" if aux_value_horizons else "python",
        device="cpu",
        seed=7,
    )
    return AlphaZeroTrainer(config, student, None, teacher, None)


def _experience(
    action: int,
    value: float,
    *,
    legal_actions: tuple[int, ...] = (0, 1, 2, 3),
    full_search: bool = True,
    value_wdl: np.ndarray | None = None,
    aux_value_targets: np.ndarray | None = None,
) -> SelfPlayExperience:
    policy = np.zeros(4, dtype=np.float32)
    policy[action] = 1.0
    action_mask = np.zeros(4, dtype=np.bool_)
    action_mask[list(legal_actions)] = True
    return SelfPlayExperience(
        actor_state=np.asarray([1.0, -0.5, 0.25], dtype=np.float32),
        critic_state=np.asarray([0.5, -1.0], dtype=np.float32),
        policy=policy,
        action_mask=action_mask,
        value=value,
        full_search=full_search,
        value_wdl=value_wdl,
        aux_value_targets=aux_value_targets,
    )


def _parameters(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _same_parameters(left: nn.Module, right: nn.Module) -> bool:
    return all(
        torch.equal(left_value, right_value)
        for left_value, right_value in zip(left.parameters(), right.parameters())
    )


def test_policy_only_distillation_freezes_student_critic_and_teacher() -> None:
    trainer = _trainer(value_loss_weight=0.0)
    trainer.replay_buffer.extend([_experience(0, 1.0), _experience(1, -1.0)])
    policy_before = _parameters(trainer.student_policy_model)
    assert trainer.student_critic_model is not None
    assert trainer.teacher_critic_model is not None
    critic_before = _parameters(trainer.student_critic_model)
    teacher_policy_before = _parameters(trainer.teacher_policy_model)
    teacher_critic_before = _parameters(trainer.teacher_critic_model)

    metrics = trainer.update_weights()

    assert metrics is not None
    assert metrics["value_loss"] == 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(policy_before, trainer.student_policy_model.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(critic_before, trainer.student_critic_model.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(teacher_policy_before, trainer.teacher_policy_model.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(teacher_critic_before, trainer.teacher_critic_model.parameters())
    )


def test_value_training_updates_separate_critic() -> None:
    trainer = _trainer(value_loss_weight=1.0)
    trainer.replay_buffer.extend([_experience(0, 1.0), _experience(1, -1.0)])
    assert trainer.student_critic_model is not None
    critic_before = _parameters(trainer.student_critic_model)

    metrics = trainer.update_weights()

    assert metrics is not None
    assert metrics["value_loss"] > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, trainer.student_critic_model.parameters())
    )


def test_shared_policy_value_network_uses_one_combined_optimizer_step() -> None:
    trainer = _shared_trainer()
    trainer.replay_buffer.extend([_experience(0, 1.0), _experience(1, -1.0)])
    assert trainer.uses_shared_network
    assert trainer.student_critic_model is None
    assert trainer.critic_optimizer is None
    assert isinstance(trainer.policy_optimizer, torch.optim.AdamW)
    assert trainer.policy_optimizer.param_groups[0]["eps"] == 1e-8
    shared_student = trainer.student_policy_model
    assert isinstance(shared_student, PolicyValueNetworkWrapper)

    policy_head_before = _parameters(shared_student.policy_head)
    value_head_before = _parameters(shared_student.value_head)
    teacher_before = _parameters(trainer.teacher_policy_model)

    metrics = trainer.update_weights()

    assert metrics is not None
    assert metrics["policy_loss"] > 0.0
    assert metrics["value_loss"] > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(policy_head_before, shared_student.policy_head.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(value_head_before, shared_student.value_head.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(teacher_before, trainer.teacher_policy_model.parameters())
    )


def test_wdl_shared_value_head_trains_categorically_but_infers_scalar_q() -> None:
    trainer = _shared_trainer(categorical_value=True)
    trainer.replay_buffer.extend([_experience(0, 1.0), _experience(1, -1.0)])

    metrics = trainer.update_weights()

    assert metrics is not None
    assert metrics["value_loss"] > 0.0
    with torch.no_grad():
        _, values = trainer.student_policy_model(
            torch.tensor([[1.0, -0.5, 0.25], [1.0, -0.5, 0.25]])
        )
    assert values.shape == (2,)
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)


def test_wdl_shared_value_head_uses_search_refined_draw_mass() -> None:
    trainer = _shared_trainer(categorical_value=True)
    assert isinstance(trainer.student_policy_model, PolicyValueNetworkWrapper)
    assert isinstance(trainer.student_policy_model.value_head, WDLValueHead)
    with torch.no_grad():
        trainer.student_policy_model.value_head.value_head.weight.zero_()
        trainer.student_policy_model.value_head.value_head.bias.copy_(
            torch.tensor([2.0, 0.0, -2.0])
        )
    draw = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    trainer.replay_buffer.extend(
        [
            _experience(0, 0.0, value_wdl=draw),
            _experience(1, 0.0, value_wdl=draw),
        ]
    )

    metrics = trainer.update_weights()

    assert metrics is not None
    expected = -torch.log_softmax(torch.tensor([2.0, 0.0, -2.0]), dim=-1)[1]
    assert metrics["value_loss"] == pytest.approx(float(expected))


def test_auxiliary_soft_policy_head_is_fresh_masked_and_persisted() -> None:
    trainer = _shared_trainer(
        categorical_value=True,
        soft_policy_temperature=4.0,
        soft_policy_weight=8.0,
    )
    assert trainer.student_soft_policy_head is not None
    assert trainer.teacher_soft_policy_head is not None
    trainer.replay_buffer.extend(
        [
            _experience(0, 1.0, legal_actions=(0, 1)),
            _experience(1, -1.0, legal_actions=(0, 1)),
        ]
    )
    soft_before = _parameters(trainer.student_soft_policy_head)

    metrics = trainer.update_weights()

    assert metrics is not None
    assert metrics["soft_policy_loss"] > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(soft_before, trainer.student_soft_policy_head.parameters())
    )
    state = trainer.state_dict()
    assert state["student_soft_policy_head"] is not None
    trainer.promote_student()
    assert _same_parameters(
        trainer.student_soft_policy_head,
        trainer.teacher_soft_policy_head,
    )


def test_auxiliary_value_head_trains_shared_backbone_and_persists() -> None:
    trainer = _shared_trainer(
        categorical_value=True,
        aux_value_horizons=(10, 50, 150),
        aux_value_weight=0.5,
    )
    assert trainer.student_aux_value_head is not None
    assert trainer.teacher_aux_value_head is not None
    trainer.replay_buffer.extend(
        [
            _experience(
                0,
                1.0,
                aux_value_targets=np.asarray([0.8, 0.4, 0.1], dtype=np.float32),
            ),
            _experience(
                1,
                -1.0,
                aux_value_targets=np.asarray([-0.7, -0.3, 0.0], dtype=np.float32),
            ),
        ]
    )
    aux_before = _parameters(trainer.student_aux_value_head)
    backbone_before = _parameters(trainer.student_policy_model.backbone)

    metrics = trainer.update_weights()

    assert metrics is not None
    assert metrics["aux_value_loss"] > 0.0
    assert metrics["aux_value_loss_h10"] > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(aux_before, trainer.student_aux_value_head.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            backbone_before, trainer.student_policy_model.backbone.parameters()
        )
    )
    state = trainer.state_dict()
    assert state["student_aux_value_head"] is not None
    restored = _shared_trainer(
        categorical_value=True,
        aux_value_horizons=(10, 50, 150),
        aux_value_weight=0.5,
    )
    restored.load_state_dict(state)
    assert restored.student_aux_value_head is not None
    assert _same_parameters(trainer.student_aux_value_head, restored.student_aux_value_head)
    trainer.promote_student()
    assert _same_parameters(trainer.student_aux_value_head, trainer.teacher_aux_value_head)


def test_auxiliary_value_targets_match_canopy_backward_ema() -> None:
    def sample(player: int, q: float) -> _NativeSelfPlaySample:
        return _NativeSelfPlaySample(
            actor_state=np.zeros(1, dtype=np.float32),
            critic_state=np.zeros(1, dtype=np.float32),
            policy=np.ones(1, dtype=np.float32),
            action_mask=np.ones(1, dtype=np.bool_),
            player=player,
            search_value=q,
            search_wdl=np.asarray(
                [(1.0 + q) / 2.0, 0.0, (1.0 - q) / 2.0],
                dtype=np.float32,
            ),
            full_search=True,
        )

    samples = [sample(0, 0.8), sample(1, 0.4), sample(0, -0.2)]
    alpha = 1.0 - np.exp(-1.0)
    ema_2 = alpha * -0.2
    ema_1 = alpha * -0.4 + (1.0 - alpha) * ema_2
    ema_0 = alpha * 0.8 + (1.0 - alpha) * ema_1

    targets = _compute_auxiliary_value_targets(samples, (1, 10))

    assert targets.shape == (3, 2)
    np.testing.assert_allclose(targets[:, 0], [ema_0, -ema_1, ema_2], rtol=1e-6)
    assert np.isfinite(targets).all()
    assert np.all(np.abs(targets) <= 1.0)


def test_distillation_loss_masks_illegal_action_logits() -> None:
    trainer = _trainer(value_loss_weight=0.0)
    policy_head = trainer.student_policy_model.policy_head
    assert isinstance(policy_head, FlatPolicyHead)
    with torch.no_grad():
        head = policy_head.policy_head
        head.weight.zero_()
        head.bias.copy_(torch.tensor([1.0, 0.0, 100.0, -100.0]))
    trainer.replay_buffer.extend(
        [
            _experience(0, 1.0, legal_actions=(0, 1)),
            _experience(0, -1.0, legal_actions=(0, 1)),
        ]
    )

    metrics = trainer.update_weights()

    assert metrics is not None
    assert metrics["policy_loss"] == pytest.approx(
        -torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)[0].item(),
        rel=1e-5,
    )
    assert metrics["top1_agreement"] == 1.0


def test_fast_search_samples_only_contribute_to_value_loss() -> None:
    trainer = _trainer(value_loss_weight=1.0)
    policy_head = trainer.student_policy_model.policy_head
    assert isinstance(policy_head, FlatPolicyHead)
    with torch.no_grad():
        head = policy_head.policy_head
        head.weight.zero_()
        head.bias.copy_(torch.tensor([2.0, 0.0, -2.0, -4.0]))
    trainer.replay_buffer.extend(
        [
            _experience(0, 1.0, full_search=True),
            _experience(3, -1.0, full_search=False),
        ]
    )

    metrics = trainer.update_weights()

    assert metrics is not None
    expected = -torch.log_softmax(torch.tensor([2.0, 0.0, -2.0, -4.0]), dim=0)[0]
    assert metrics["policy_loss"] == pytest.approx(expected.item(), rel=1e-5)
    assert metrics["value_loss"] > 0.0
    assert metrics["full_search_fraction"] == 0.5


def test_replay_epochs_visit_every_sample_once_per_epoch() -> None:
    trainer = _trainer(value_loss_weight=1.0)
    trainer.config.batch_size = 2
    experiences = [_experience(index % 4, float(index % 2)) for index in range(5)]
    trainer.replay_buffer.extend(experiences)

    batches = list(trainer.iter_replay_epoch_batches(2))

    assert [len(batch) for batch in batches] == [2, 2, 1, 2, 2, 1]
    assert sorted(id(sample) for batch in batches for sample in batch) == sorted(
        id(sample) for sample in experiences for _ in range(2)
    )
    assert trainer.update_weights(batches[2]) is not None


def test_disk_replay_preserves_exact_targets_and_ring_order(tmp_path) -> None:
    replay = DiskReplayBuffer(3, str(tmp_path), shared_states=True)
    experiences = []
    for index in range(4):
        state = np.asarray([index, index + 0.5, -index], dtype=np.float32)
        policy = np.asarray([0.0, 0.25, 0.75, 0.0], dtype=np.float32)
        mask = np.asarray([False, True, True, False], dtype=np.bool_)
        experiences.append(
            SelfPlayExperience(
                actor_state=state,
                critic_state=state,
                policy=policy,
                action_mask=mask,
                value=float(index - 2),
                full_search=index % 2 == 0,
                value_wdl=(None if index == 2 else np.asarray([0.1, 0.2, 0.7], dtype=np.float32)),
            )
        )

    storage_path = replay.storage_path
    replay.extend(experiences)

    assert len(replay) == 3
    assert [sample.value for sample in replay] == [-1.0, 0.0, 1.0]
    for expected, actual in zip(experiences[1:], replay):
        assert actual.actor_state.dtype == np.float32
        assert actual.critic_state is actual.actor_state
        assert actual.policy.dtype == np.float32
        assert actual.action_mask.dtype == np.bool_
        np.testing.assert_array_equal(actual.actor_state, expected.actor_state)
        np.testing.assert_array_equal(actual.policy, expected.policy)
        np.testing.assert_array_equal(actual.action_mask, expected.action_mask)
        if expected.value_wdl is None:
            assert actual.value_wdl is None
        else:
            np.testing.assert_array_equal(actual.value_wdl, expected.value_wdl)

    replay.close()
    assert not os.path.exists(storage_path)


def test_promote_and_restore_keep_policy_critic_pairs_together() -> None:
    trainer = _trainer(value_loss_weight=1.0)
    assert trainer.student_critic_model is not None
    assert trainer.teacher_critic_model is not None
    with torch.no_grad():
        for parameter in trainer.student_policy_model.parameters():
            parameter.add_(1.0)
        for parameter in trainer.student_critic_model.parameters():
            parameter.sub_(1.0)

    trainer.promote_student()
    assert _same_parameters(trainer.student_policy_model, trainer.teacher_policy_model)
    assert _same_parameters(trainer.student_critic_model, trainer.teacher_critic_model)

    with torch.no_grad():
        for parameter in trainer.student_policy_model.parameters():
            parameter.mul_(0.0)
        for parameter in trainer.student_critic_model.parameters():
            parameter.mul_(0.0)
    trainer.restore_student_from_teacher()

    assert _same_parameters(trainer.student_policy_model, trainer.teacher_policy_model)
    assert _same_parameters(trainer.student_critic_model, trainer.teacher_critic_model)
    assert not trainer.policy_optimizer.state
    assert trainer.critic_optimizer is not None
    assert not trainer.critic_optimizer.state


def test_collect_self_play_uses_teacher_models(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer()
    called = {}

    def fake_generate_self_play_data(**kwargs):
        called.update(kwargs)
        return [_experience(2, 1.0)], {"games": 1, "wins_RED": 1}

    monkeypatch.setattr(
        "catanrl.algorithms.alphazero.trainer.generate_self_play_data",
        fake_generate_self_play_data,
    )
    stats = trainer.collect_self_play(1)

    assert called["policy_model"] is trainer.teacher_policy_model
    assert called["critic_model"] is trainer.teacher_critic_model
    assert stats["experiences"] == 1.0
    assert stats["replay_size"] == 1.0


def test_collect_self_play_can_dispatch_to_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    trainer.config.self_play_backend = "cppanatron"
    called = {}

    def fake_generate_native_self_play_data(**kwargs):
        called.update(kwargs)
        return [_experience(1, -1.0)], {"games": 1}

    monkeypatch.setattr(
        "catanrl.algorithms.alphazero.trainer.generate_native_self_play_data",
        fake_generate_native_self_play_data,
    )

    stats = trainer.collect_self_play(1)

    assert called["policy_model"] is trainer.teacher_policy_model
    assert called["full_search_probability"] == 1.0
    assert called["fast_simulations"] == 64
    assert called["search_value_weight"] == 0.0
    assert called["policy_target"] == "visits"
    assert called["c_visit"] == 50.0
    assert called["c_scale"] == 1.0
    assert called["search_selection"] == "puct"
    assert called["trajectory_action_selection"] == "visits"
    assert called["explore_actions"] == 24
    assert called["games_per_worker"] == 1
    assert called["max_actions"] == 0
    assert called["worker_stall_timeout_s"] == 600.0
    assert called["inference_response_timeout_s"] == 120.0
    assert called["result_chunk_size"] == 64
    assert called["aux_value_horizons"] == ()
    assert stats["experiences"] == 1.0
    assert stats["self_play_attempts"] == 1.0
    assert len(trainer.replay_buffer) == 1


def test_collect_self_play_retries_the_same_batch_without_partial_replay(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = _trainer()
    trainer.config.self_play_max_attempts = 3
    seeds = []

    def flaky_generate_self_play_data(**kwargs):
        seeds.append(kwargs["seed"])
        if len(seeds) < 3:
            raise RuntimeError("simulated worker stall")
        return [_experience(2, 1.0)], {"games": 1}

    monkeypatch.setattr(
        "catanrl.algorithms.alphazero.trainer.generate_self_play_data",
        flaky_generate_self_play_data,
    )

    stats = trainer.collect_self_play(1)
    output = capsys.readouterr().out

    assert seeds == [seeds[0], seeds[0], seeds[0]]
    assert stats["self_play_attempts"] == 3.0
    assert len(trainer.replay_buffer) == 1
    assert output.count("simulated worker stall") == 2
    assert "attempt 1/3 failed" in output
    assert "attempt 2/3 failed" in output


def test_native_search_value_weight_ramps_by_self_play_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    trainer.config.self_play_backend = "cppanatron"
    trainer.config.search_value_weight_max = 0.6
    trainer.config.search_value_weight_ramp_iterations = 2
    weights = []

    def fake_generate_native_self_play_data(**kwargs):
        weights.append(kwargs["search_value_weight"])
        return [_experience(1, -1.0)], {"games": 1}

    monkeypatch.setattr(
        "catanrl.algorithms.alphazero.trainer.generate_native_self_play_data",
        fake_generate_native_self_play_data,
    )

    trainer.collect_self_play(1)
    trainer.collect_self_play(1)
    trainer.collect_self_play(1)

    assert weights == pytest.approx([0.3, 0.6, 0.6])


def test_weight_only_warm_start_can_continue_self_play_iteration_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(self_play_iteration_offset=7)
    trainer.config.self_play_backend = "cppanatron"
    trainer.config.search_value_weight_max = 0.85
    trainer.config.search_value_weight_ramp_iterations = 60
    calls = []

    def fake_generate_native_self_play_data(**kwargs):
        calls.append((kwargs["seed"], kwargs["search_value_weight"]))
        return [_experience(1, -1.0)], {"games": 1}

    monkeypatch.setattr(
        "catanrl.algorithms.alphazero.trainer.generate_native_self_play_data",
        fake_generate_native_self_play_data,
    )

    trainer.collect_self_play(1)

    assert calls == [(7 + 7 * 1_000_003, pytest.approx(0.85 * 8 / 60))]
    assert trainer.state_dict()["self_play_calls"] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_offload_preserves_optimizer_parameter_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(offload_inactive_models=True, device="cuda")

    def fake_generate_self_play_data(**kwargs):
        assert next(kwargs["policy_model"].parameters()).device.type == "cuda"
        return [_experience(0, 1.0), _experience(1, -1.0)], {"games": 1}

    monkeypatch.setattr(
        "catanrl.algorithms.alphazero.trainer.generate_self_play_data",
        fake_generate_self_play_data,
    )
    trainer.collect_self_play(1)
    optimizer_parameters = {
        id(parameter)
        for group in trainer.policy_optimizer.param_groups
        for parameter in group["params"]
    }

    assert next(trainer.teacher_policy_model.parameters()).device.type == "cpu"
    assert next(trainer.student_policy_model.parameters()).device.type == "cuda"
    assert optimizer_parameters == {
        id(parameter) for parameter in trainer.student_policy_model.parameters()
    }
    assert trainer.update_weights() is not None


@pytest.mark.parametrize(
    ("h2h", "candidate", "champion", "expected"),
    [
        (0.53, 0.58, 0.59, True),
        (0.51, 0.60, 0.59, False),
        (0.54, 0.56, 0.59, False),
    ],
)
def test_promotion_requires_h2h_and_baseline_gates(
    h2h: float,
    candidate: float,
    champion: float,
    expected: bool,
) -> None:
    decision = decide_promotion(
        h2h_win_rate=h2h,
        candidate_baseline_win_rate=candidate,
        champion_baseline_win_rate=champion,
        h2h_threshold=0.52,
        max_baseline_regression=0.02,
    )
    assert decision.promote is expected


def test_cli_mode_defaults_freeze_value_only_for_distillation() -> None:
    distill = parse_args(["--mode", "distill", "--load-from-experiment", "teacher"])
    iterate = parse_args(["--mode", "iterate", "--config", "model.yaml"])

    assert distill.value_loss_weight == 0.0
    assert iterate.value_loss_weight == 1.0
    assert iterate.soft_policy_temperature == 0.0
    assert iterate.soft_policy_weight == 0.0


def test_cli_replay_epochs_override_is_opt_in() -> None:
    fixed_steps = parse_args(["--mode", "iterate", "--config", "model.yaml"])
    replay_epochs = parse_args(
        ["--mode", "iterate", "--config", "model.yaml", "--optimizer-epochs", "2"]
    )

    assert fixed_steps.optimizer_epochs == 0
    assert replay_epochs.optimizer_epochs == 2


def test_cli_accepts_bounded_native_self_play_controls() -> None:
    args = parse_args(
        [
            "--mode",
            "iterate",
            "--config",
            "model.yaml",
            "--max-actions",
            "2000",
            "--self-play-stall-timeout-seconds",
            "300",
            "--inference-response-timeout-seconds",
            "60",
            "--self-play-result-chunk-size",
            "32",
            "--self-play-max-attempts",
            "4",
            "--games-per-worker",
            "3",
            "--self-play-iteration-offset",
            "7",
        ]
    )

    assert args.max_actions == 2000
    assert args.self_play_stall_timeout_seconds == 300.0
    assert args.inference_response_timeout_seconds == 60.0
    assert args.self_play_result_chunk_size == 32
    assert args.self_play_max_attempts == 4
    assert args.games_per_worker == 3
    assert args.self_play_iteration_offset == 7


def test_warm_start_can_keep_policy_but_reset_shared_value_head(tmp_path) -> None:
    torch.manual_seed(11)
    source = PolicyValueNetworkWrapper(
        nn.Linear(3, 3),
        FlatPolicyHead(3, 4),
        WDLValueHead(3),
    )
    checkpoint = tmp_path / "policy_value.pt"
    torch.save(source.state_dict(), checkpoint)

    torch.manual_seed(17)
    student = PolicyValueNetworkWrapper(
        nn.Linear(3, 3),
        FlatPolicyHead(3, 4),
        WDLValueHead(3),
    )
    teacher = copy.deepcopy(student)
    fresh_value_state = copy.deepcopy(student.value_head.state_dict())
    warm_start = SimpleNamespace(
        checkpoints=SimpleNamespace(
            policy=str(checkpoint),
            critic=None,
            experiment_name="dagger",
            which="best",
        )
    )

    _load_initial_weights(
        student_policy=student,
        student_critic=None,
        teacher_policy=teacher,
        teacher_critic=None,
        warm_start=cast(Any, warm_start),
        reset_value_head=True,
    )

    assert _same_parameters(student.backbone, source.backbone)
    assert _same_parameters(student.policy_head, source.policy_head)
    assert all(
        torch.equal(value, fresh_value_state[key])
        for key, value in student.value_head.state_dict().items()
    )
    assert not _same_parameters(student.value_head, source.value_head)
    assert _same_parameters(student, teacher)


def test_reset_loaded_value_head_requires_new_warm_start() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "iterate",
                "--config",
                "model.yaml",
                "--reset-loaded-value-head",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "iterate",
                "--load-from-experiment",
                "dagger",
                "--resume",
                "--reset-loaded-value-head",
            ]
        )


def test_latest_teacher_update_promotes_without_waiting_for_evaluation() -> None:
    trainer = _trainer()
    with torch.no_grad():
        for parameter in trainer.student_policy_model.parameters():
            parameter.add_(1.0)
    assert not _same_parameters(
        trainer.student_policy_model,
        trainer.teacher_policy_model,
    )

    accepted, score, reason = _update_search_teacher(
        trainer=trainer,
        strategy="latest",
        candidate_win_rate=None,
        h2h_win_rate=None,
        champion_eval_score=0.2,
        promotion_threshold=0.52,
        max_baseline_regression=0.02,
    )

    assert accepted is True
    assert score == 0.2
    assert reason is not None
    assert _same_parameters(
        trainer.student_policy_model,
        trainer.teacher_policy_model,
    )


def test_gated_teacher_update_waits_for_evaluation() -> None:
    trainer = _trainer()
    with torch.no_grad():
        for parameter in trainer.student_policy_model.parameters():
            parameter.add_(1.0)
    teacher_before = _parameters(trainer.teacher_policy_model)

    accepted, score, reason = _update_search_teacher(
        trainer=trainer,
        strategy="gated",
        candidate_win_rate=None,
        h2h_win_rate=None,
        champion_eval_score=0.2,
        promotion_threshold=0.52,
        max_baseline_regression=0.02,
    )

    assert accepted is None
    assert score == 0.2
    assert reason is None
    assert all(
        torch.equal(before, after)
        for before, after in zip(teacher_before, trainer.teacher_policy_model.parameters())
    )


def test_cli_native_self_play_requires_plain_mcts() -> None:
    args = parse_args(
        [
            "--mode",
            "iterate",
            "--config",
            "model.yaml",
            "--self-play-backend",
            "cppanatron",
            "--ismcts-determinizations",
            "1",
        ]
    )
    assert args.self_play_backend == "cppanatron"

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "iterate",
                "--config",
                "model.yaml",
                "--self-play-backend",
                "cppanatron",
            ]
        )


def test_cli_completed_q_policy_target_requires_native_search() -> None:
    args = parse_args(
        [
            "--mode",
            "iterate",
            "--config",
            "model.yaml",
            "--self-play-backend",
            "cppanatron",
            "--ismcts-determinizations",
            "1",
            "--policy-target",
            "completed-q",
            "--c-visit",
            "50",
            "--c-scale",
            "1",
            "--soft-policy-temperature",
            "4",
            "--soft-policy-weight",
            "8",
        ]
    )
    assert args.policy_target == "completed-q"
    assert args.soft_policy_temperature == 4.0
    assert args.soft_policy_weight == 8.0

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "iterate",
                "--config",
                "model.yaml",
                "--policy-target",
                "completed-q",
            ]
        )


def test_cli_auxiliary_values_require_native_self_play_and_positive_weight() -> None:
    args = parse_args(
        [
            "--mode",
            "iterate",
            "--config",
            "model.yaml",
            "--self-play-backend",
            "cppanatron",
            "--ismcts-determinizations",
            "1",
            "--aux-value-horizons",
            "10",
            "50",
            "150",
            "--aux-value-weight",
            "0.5",
        ]
    )
    assert args.aux_value_horizons == (10, 50, 150)
    assert args.aux_value_weight == 0.5

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "iterate",
                "--config",
                "model.yaml",
                "--aux-value-horizons",
                "10",
                "--aux-value-weight",
                "0.5",
            ]
        )


def test_cli_completed_q_search_selection_requires_native_search() -> None:
    args = parse_args(
        [
            "--mode",
            "iterate",
            "--config",
            "model.yaml",
            "--self-play-backend",
            "cppanatron",
            "--ismcts-determinizations",
            "1",
            "--search-selection",
            "completed-q",
        ]
    )
    assert args.search_selection == "completed-q"

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "iterate",
                "--config",
                "model.yaml",
                "--search-selection",
                "completed-q",
            ]
        )


def test_cli_canopy_trajectory_requires_exact_search_controls() -> None:
    args = parse_args(
        [
            "--mode",
            "iterate",
            "--config",
            "model.yaml",
            "--self-play-backend",
            "cppanatron",
            "--ismcts-determinizations",
            "1",
            "--search-selection",
            "completed-q",
            "--policy-target",
            "completed-q",
            "--target-temperature",
            "1",
            "--trajectory-action-selection",
            "canopy",
            "--explore-actions",
            "24",
        ]
    )

    assert args.trajectory_action_selection == "canopy"
    assert args.explore_actions == 24

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "iterate",
                "--config",
                "model.yaml",
                "--trajectory-action-selection",
                "canopy",
            ]
        )


def test_policy_players_respect_existing_model_device() -> None:
    policy, _ = _models()
    policy.to("cpu")

    direct = NNPolicyPlayer(Color.RED, "flat", policy, device="cpu")
    belief = BeliefAveragedPolicyPlayer(Color.RED, "flat", policy, device="cpu")

    assert direct.device == "cpu"
    assert belief.device == "cpu"
