import torch
import numpy as np
from typing import List

from catanatron.models.player import Player, Color
from catanatron.cli.cli_players import register_cli_player
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, ACTIONS_ARRAY, normalize_action

from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    game_to_features,
    get_numeric_feature_names,
)
from catanrl.models.models import HierarchicalPolicyValueNetwork


class NNPolicyPlayer(Player):
    """
    A Player that uses a trained PolicyNetwork to select actions.
    """

    def __init__(
        self,
        color,
        model_path: str,
        input_dim: int,
        hidden_dims: List[int] = [2048, 2048, 1024],
        epsilon=None,
        map_type: str = 'BASE',
        num_players: int = 2,
        numeric_features: List[str] = None,
        **kwargs
    ):
        super().__init__(color, is_bot=True, **kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.model_path = model_path
        self.input_dim = input_dim
        self.map_type = map_type
        self.num_players = num_players
        
        # Load the trained policy network
        self.policy_net = HierarchicalPolicyValueNetwork(input_dim, hidden_dims).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()

    def decide(self, game, playable_actions):
        """
        Select action using the neural network policy.
        
        The policy network outputs logits for all actions in the action space.
        We select the action with the highest logit that is also in playable_actions.
        """
        if len(playable_actions) == 1:
            return playable_actions[0]

        if self.epsilon is not None and np.random.random() < self.epsilon:
            return np.random.choice(playable_actions)

        # Convert game state to features
        game_tensor = torch.from_numpy(
            game_to_features(game, self.color, self.num_players, self.map_type).reshape(1, -1)
        ).to(self.device)
        
        # Get policy logits
        with torch.no_grad():
            action_type_logits, param_logits, value = self.policy_net(game_tensor)
            
            logits = self.policy_net(game_tensor).squeeze(0).cpu().numpy()  # Shape: [ACTION_SPACE_SIZE]
        
        # Convert playable_actions to their indices in the action space
        # Need to normalize actions and find their index in ACTIONS_ARRAY
        normalized_playable = [normalize_action(a) for a in playable_actions]
        possibilities = [(a.action_type, a.value) for a in normalized_playable]
        playable_action_indices = [ACTIONS_ARRAY.index(x) for x in possibilities]
        
        # Mask out non-playable actions by setting their logits to -inf
        masked_logits = np.full_like(logits, float('-inf'))
        masked_logits[playable_action_indices] = logits[playable_action_indices]
        
        # Select action with highest logit among playable actions
        best_action_idx = np.argmax(masked_logits)
        
        # Find the action object corresponding to this index
        best_possibility = ACTIONS_ARRAY[best_action_idx]
        for i, possibility in enumerate(possibilities):
            if possibility == best_possibility:
                return playable_actions[i]
        
        # Fallback: should not happen, but return first action if something goes wrong
        return playable_actions[0]

    def __repr__(self) -> str:
        return super().__repr__().replace("Player", "NNPolicyPlayer")


def create_nn_policy_player(color, map_template='BASE'):
    """
    Factory function to create NNPolicyPlayer.
    
    Args:
        color: Player color
        map_template: Map template - 'BASE' (default), 'm' or 'MINI' for mini map, 
                      't' or 'TOURNAMENT' for tournament map
    """
    # Normalize map template parameter
    map_type_map = {
        'm': 'MINI',
        'MINI': 'MINI',
        't': 'TOURNAMENT',
        'TOURNAMENT': 'TOURNAMENT',
        'BASE': 'BASE',
        'b': 'BASE',
    }
    map_type = map_type_map.get(map_template if isinstance(map_template, str) else 'BASE', 'BASE')
    
    # Use different model weights based on map type
    model_paths = {
        'MINI': 'weights/ab2-ab2-100k-mini/best.pt',
        'BASE': 'weights/sl-hiearchical-cont_policy_value.pt',
        'TOURNAMENT': 'weights/ab2-ab2-100k-tournament/best.pt',
    }
    model_path = model_paths.get(map_type, model_paths['BASE'])
    
    # Calculate input dimension by creating a dummy game with dummy players
    num_players = 2
    numeric_features = list(get_numeric_feature_names(num_players, map_type))
    input_dim = compute_feature_vector_dim(num_players, map_type)
    
    return NNPolicyPlayer(
        color=color,
        model_path=model_path,
        input_dim=input_dim,
        map_type=map_type,
        num_players=num_players,
        numeric_features=numeric_features,
    )


register_cli_player("NNP", create_nn_policy_player)

