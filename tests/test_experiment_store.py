from argparse import Namespace
from types import SimpleNamespace

from catanrl.experiment_store import WANDB_TAG_MAX_LENGTH, wandb_grouping_kwargs


def test_wandb_grouping_truncates_long_tags_with_stable_unique_suffixes():
    parent_name = "parent-" + "a" * 80
    parent = SimpleNamespace(
        metadata=SimpleNamespace(name=parent_name, wandb={"tags": []})
    )
    warm_start = SimpleNamespace(experiment=parent)
    args = Namespace(
        wandb_group=None,
        wandb_tags=["custom-" + "b" * 80, "custom-" + "c" * 80],
    )

    first = wandb_grouping_kwargs(
        args,
        group_default="marl-ppo",
        warm_start=warm_start,
    )
    second = wandb_grouping_kwargs(
        args,
        group_default="marl-ppo",
        warm_start=warm_start,
    )

    assert first == second
    assert first["group"] == "marl-ppo"
    assert len(first["tags"]) == 4
    assert len(set(first["tags"])) == 4
    assert all(1 <= len(tag) <= WANDB_TAG_MAX_LENGTH for tag in first["tags"])
