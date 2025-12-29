import torch
import numpy as np
from typing import List

from catanatron.models.player import Player
from catanatron.cli.cli_players import register_cli_player
from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY, normalize_action

from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    game_to_features,
    get_numeric_feature_names,
)
from catanrl.models.models import HierarchicalPolicyValueNetwork


class NNHierarchicalPolicyPlayer(Player):
    """
    Player that loads a HierarchicalPolicyValueNetwork and selects the highest-probability action.
    """

    def __init__(
        self,
        color,
        model_path: str,
        hidden_dims: List[int],
        input_dim: int,
        epsilon: float | None = None,
        map_type: str = 'BASE',
        num_players: int = 2,
        numeric_features: List[str] | None = None,
        **kwargs,
    ):
        super().__init__(color, is_bot=True, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.model_path = model_path
        self.num_players = num_players
        self.input_dim = input_dim

        if numeric_features is None:
            numeric_features = list(get_numeric_feature_names(num_players, map_type))
        self.numeric_features = numeric_features

        self.policy_net = HierarchicalPolicyValueNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        if self.epsilon is not None and np.random.random() < self.epsilon:
            return np.random.choice(playable_actions)

        features = game_to_features(game, self.color, self.num_players, self.map_type)
        game_tensor = torch.from_numpy(features.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            action_type_logits, param_logits, _ = self.policy_net(game_tensor)
            flat_logits = self.policy_net.get_flat_action_logits(
                action_type_logits,
                param_logits,
            ).squeeze(0).cpu().numpy()

        normalized_actions = [normalize_action(a) for a in playable_actions]
        playable_indices = [
            ACTIONS_ARRAY.index((a.action_type, a.value)) for a in normalized_actions
        ]

        masked_logits = np.full_like(flat_logits, float('-inf'))
        masked_logits[playable_indices] = flat_logits[playable_indices]
        best_idx = int(np.argmax(masked_logits))

        best_action_signature = ACTIONS_ARRAY[best_idx]
        for original_action, normalized in zip(playable_actions, normalized_actions):
            if (normalized.action_type, normalized.value) == best_action_signature:
                return original_action

        return playable_actions[0]

    def __repr__(self) -> str:
        return super().__repr__().replace("Player", "NNHierarchicalPolicyPlayer")


def create_nn_hierarchical_policy_player(color, map_template='BASE', hidden_dims='2048,2048,1024', epsilon=None, **kwargs):
    model_paths = {
        'MINI': 'weights/sl-hiearchical-mini_policy_value.pt',
        'BASE': 'weights/sl-hiearchical-cont_policy_value.pt',
        'TOURNAMENT': 'weights/sl-hiearchical-tournament_policy_value.pt',
    }
    map_type_alias = {
        'm': 'MINI',
        'mini': 'MINI',
        't': 'TOURNAMENT',
        'tournament': 'TOURNAMENT',
        'b': 'BASE',
        'base': 'BASE',
    }
    normalized_map = map_type_alias.get(map_template, map_template)
    normalized_map = normalized_map.upper() if isinstance(normalized_map, str) else 'BASE'
    model_path = model_paths.get(normalized_map, model_paths['BASE'])
    hidden_dims_list = [int(dim) for dim in hidden_dims.split(',')] if isinstance(hidden_dims, str) else hidden_dims

    numeric_features = list(get_numeric_feature_names(2, normalized_map))
    input_dim = compute_feature_vector_dim(2, normalized_map)

    player = NNHierarchicalPolicyPlayer(
        color=color,
        model_path=model_path,
        hidden_dims=hidden_dims_list,
        input_dim=input_dim,
        epsilon=epsilon,
        map_type=normalized_map,
        num_players=2,
        numeric_features=numeric_features,
    )
    return player


register_cli_player("NNH", create_nn_hierarchical_policy_player)

