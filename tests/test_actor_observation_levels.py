from __future__ import annotations

import numpy as np
import pytest

from catanrl.features.catanatron_utils import (
    get_actor_indices_from_critic,
    get_actor_indices_from_full,
    get_actor_numeric_feature_names,
    get_full_numeric_feature_names,
)


def test_private_indices_preserve_existing_actor_mapping():
    assert np.array_equal(
        get_actor_indices_from_full(2, "BASE", "private"),
        get_actor_indices_from_critic(2, "BASE"),
    )


def test_full_indices_are_identity_over_full_observation():
    full_numeric_dim = len(get_full_numeric_feature_names(2, "BASE"))
    full_indices = get_actor_indices_from_full(2, "BASE", "full")

    assert np.array_equal(full_indices, np.arange(full_indices.shape[0]))
    assert full_indices.shape[0] > full_numeric_dim


def test_public_level_adds_only_opponent_resource_cards_for_1v1():
    names = get_actor_numeric_feature_names(2, "BASE", "public")

    for resource in ("BRICK", "ORE", "SHEEP", "WHEAT", "WOOD"):
        assert f"P1_{resource}_IN_HAND" in names

    assert "P1_KNIGHT_IN_HAND" not in names
    assert "P1_VICTORY_POINT_IN_HAND" not in names
    assert "P1_ACTUAL_VPS" not in names


def test_public_level_is_currently_1v1_only():
    with pytest.raises(ValueError, match="1v1"):
        get_actor_numeric_feature_names(3, "BASE", "public")
