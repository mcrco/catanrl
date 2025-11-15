import torch
import numpy as np
from typing import Tuple

from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer, Color
from catanatron.cli.cli_players import register_cli_player
from catanatron.models.map import build_map
from catanatron.features import create_sample_vector
from catanatron.gym.board_tensor_features import create_board_tensor
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, to_action_space, ACTIONS_ARRAY, normalize_action
from rl.models import PolicyNetwork


def game_to_features(game: Game, color: int) -> np.ndarray:
    """
    Extracts features from the game state for the policy network.
    """
    # This function creates a fixed-order vector of features from the game state.
    feature_vector = create_sample_vector(game, color)
    board_tensor = create_board_tensor(game, color)
    return np.concatenate([feature_vector, board_tensor.flatten()], axis=0)


def _rebuild_nn_policy_player(color, model_path, input_dim, epsilon):
    """Helper function to rebuild NNPolicyPlayer from pickle."""
    return NNPolicyPlayer(color, model_path, input_dim, epsilon)


class NNPolicyPlayer(Player):
    """
    A Player that uses a trained PolicyNetwork to select actions.
    """

    def __init__(
        self,
        color,
        model_path: str,
        input_dim: int,
        epsilon=None,
        **kwargs
    ):
        super().__init__(color, is_bot=True, **kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.model_path = model_path
        self.input_dim = input_dim
        
        # Load the trained policy network
        self.policy_net = PolicyNetwork(input_dim, ACTION_SPACE_SIZE).to(self.device)
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
            game_to_features(game, self.color).reshape(1, -1).astype(np.float32)
        ).to(self.device)
        
        # Get policy logits
        with torch.no_grad():
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

    def __reduce__(self):
        """Custom pickle method to handle PyTorch model."""
        # Return a tuple (callable, args) to recreate the object
        # This avoids pickling the PyTorch model itself
        return (
            _rebuild_nn_policy_player,
            (self.color, self.model_path, self.input_dim, self.epsilon)
        )

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
        'MINI': 'weights/f-f-mini-10k-samp0.1/f-f-mini-10k-samp0.1_policy.pt',
        'BASE': 'weights/f-f-base-10k-samp0.1/f-f-base-10k-samp0.1_policy.pt',
        'TOURNAMENT': 'weights/f-f-tournament-10k-samp0.1/f-f-tournament-10k-samp0.1_policy.pt',
    }
    model_path = model_paths.get(map_type, model_paths['BASE'])
    
    # Calculate input dimension by creating a dummy game with dummy players
    dummy_players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    dummy_game = Game(dummy_players, catan_map=build_map(map_type))
    sample_vec = create_sample_vector(dummy_game, dummy_game.state.colors[0])
    board_tensor = create_board_tensor(dummy_game, dummy_game.state.colors[0])
    input_dim = len(sample_vec) + board_tensor.size
    
    return NNPolicyPlayer(
        color=color,
        model_path=model_path,
        input_dim=input_dim
    )


register_cli_player("NNP", create_nn_policy_player)

