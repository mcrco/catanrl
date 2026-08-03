from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from catanatron.models.player import Color
from torch import nn

from catanrl.algorithms.alphazero.parallel_self_play import SelfPlayExperience
from catanrl.algorithms.alphazero.trainer import AlphaZeroConfig, AlphaZeroTrainer
from catanrl.eval.search_training import decide_promotion
from catanrl.experiments.train_alphazero import parse_args
from catanrl.models.heads import FlatPolicyHead
from catanrl.models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
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
    )
    return AlphaZeroTrainer(
        config,
        student_policy,
        student_critic,
        teacher_policy,
        teacher_critic,
    )


def _experience(
    action: int,
    value: float,
    *,
    legal_actions: tuple[int, ...] = (0, 1, 2, 3),
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
    assert stats["experiences"] == 1.0
    assert len(trainer.replay_buffer) == 1


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

    assert weights == pytest.approx([0.0, 0.3, 0.6])


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


def test_policy_players_respect_existing_model_device() -> None:
    policy, _ = _models()
    policy.to("cpu")

    direct = NNPolicyPlayer(Color.RED, "flat", policy, device="cpu")
    belief = BeliefAveragedPolicyPlayer(Color.RED, "flat", policy, device="cpu")

    assert direct.device == "cpu"
    assert belief.device == "cpu"
