import torch

from catanrl.experiment_store import (
    KIND_POLICY,
    KIND_VALUE,
    GameConfig,
    NetworkSpec,
    architecture_preset_from_experiment,
    build_network,
    load_experiment,
    network_spec_from_model,
    save_experiment,
)
from catanrl.experiments.architecture_config import (
    architecture_train_config_fields,
    load_architecture_preset,
)
from catanrl.models import build_critic_model, build_policy_model
from catanrl.models.heads import CatanGraphPolicyHead, CatanGraphValueHead

GRAPH_S_PRESET = "configs/models/catan-graph-s-flat-2p-full-sep.yaml"


def _graph_kwargs(arch):
    return dict(
        backbone_type=arch.backbone_type,
        num_players=arch.num_players,
        map_type=arch.map_type,
        device="cpu",
        graph_hidden_dim=arch.graph_hidden_dim,
        graph_global_hidden_dim=arch.graph_global_hidden_dim,
        graph_num_layers=arch.graph_num_layers,
        graph_head_hidden_dim=arch.graph_head_hidden_dim,
        graph_normalize_inputs=arch.graph_normalize_inputs,
        graph_semantic_inputs=arch.graph_semantic_inputs,
    )


def test_graph_preset_builds_structured_policy_and_critic():
    arch = load_architecture_preset(GRAPH_S_PRESET)
    policy = build_policy_model(
        model_type="flat",
        hidden_dims=arch.policy_hidden_dims,
        actor_observation_level="full",
        **_graph_kwargs(arch),
    )
    critic = build_critic_model(
        hidden_dims=arch.critic_hidden_dims,
        critic_observation_level="full",
        **_graph_kwargs(arch),
    )
    assert isinstance(policy.policy_head, CatanGraphPolicyHead)
    assert isinstance(critic.value_head, CatanGraphValueHead)
    x = torch.rand(3, policy.backbone.config.input_dim)
    assert policy(x).shape == (3, policy.action_space_size)
    assert critic(x).shape == (3,)


def test_graph_experiment_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_EXPERIMENTS_DIR", str(tmp_path))
    arch = load_architecture_preset(GRAPH_S_PRESET)
    policy = build_policy_model(
        model_type="flat",
        hidden_dims=arch.policy_hidden_dims,
        actor_observation_level="full",
        **_graph_kwargs(arch),
    )
    critic = build_critic_model(
        hidden_dims=arch.critic_hidden_dims,
        critic_observation_level="full",
        **_graph_kwargs(arch),
    )
    ckpt_dir = tmp_path / "graph-exp" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save(policy.state_dict(), ckpt_dir / "policy_best.pt")
    torch.save(critic.state_dict(), ckpt_dir / "critic_best.pt")
    game = GameConfig(num_players=2, map_type="BASE", vps_to_win=15, discard_limit=9)
    save_experiment(
        "graph-exp",
        str(ckpt_dir),
        algorithm="marl_cc",
        game=game,
        networks={
            "policy": network_spec_from_model(
                policy, kind=KIND_POLICY, model_type="flat", observation_level="full"
            ),
            "critic": network_spec_from_model(critic, kind=KIND_VALUE, observation_level="full"),
        },
        train_config=architecture_train_config_fields(arch),
    )

    exp = load_experiment("graph-exp")
    assert architecture_preset_from_experiment(exp) == arch
    rebuilt = exp.build_policy(which="best", device="cpu")
    for key, value in policy.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[key], value)
    spec = NetworkSpec.from_dict(exp.metadata.networks["critic"].to_dict())
    assert isinstance(build_network(spec, game).value_head, CatanGraphValueHead)
