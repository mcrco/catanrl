import torch
import numpy as np
from typing import Tuple

from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.cli.cli_players import register_cli_player
from catanatron.models.map import build_map
from catanatron.features import create_sample_vector
from catanatron.gym.board_tensor_features import create_board_tensor
from sl.models import ValueNetwork

def game_to_features(game: Game, color: int) -> np.ndarray:
    """
    Extracts features from the game state for the value network.
    """
    # This function creates a fixed-order vector of features from the game state.
    feature_vector = create_sample_vector(game, color)
    board_tensor = create_board_tensor(game, color)
    return np.concatenate([feature_vector, board_tensor.flatten()], axis=0)


class NNAlphaBetaPlayer(AlphaBetaPlayer):
    """
    An AlphaBetaPlayer that uses a trained ValueNetwork for leaf evaluation.
    """

    def __init__(
        self,
        color,
        model_path: str,
        input_dim: int,
        output_range: Tuple[float, float] = (-1, 1),
        **kwargs
    ):
        super().__init__(color, **kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_range = output_range
        
        # Load the trained value network
        self.value_net = ValueNetwork(input_dim).to(self.device)
        self.value_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.value_net.eval()

        self.use_value_function = True
        self.value_fn_builder_name = "nn_value_network"

    def value_function(self, game: Game, p0_color: int) -> float:
        """
        Overrides the heuristic value function to use the neural network.
        """
        game_tensor = torch.from_numpy(game_to_features(game, p0_color).reshape(1, -1).astype(np.float32)).to(self.device)
        with torch.no_grad():
            value = self.value_net(game_tensor).item()

        # Clip the value to the output range
        if value < self.output_range[0]:
            value = self.output_range[0]
        elif value > self.output_range[1]:
            value = self.output_range[1]
        return value

    def __repr__(self) -> str:
        return super().__repr__()

def create_nn_alpha_beta_player(color, map_template='BASE'):
    """
    Factory function to create NNAlphaBetaPlayer.
    
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
        'MINI': 'rl/weights/ab2-ab2-mini-100k_value.pt',
        'BASE': 'rl/weights/value_network_best.pt',  # Update with your base map model
        'TOURNAMENT': 'rl/weights/value_network_best.pt',  # Update with your tournament map model
    }
    model_path = model_paths.get(map_type, model_paths['BASE'])
    
    # Calculate input dimension by creating a dummy game with dummy players
    dummy_players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    dummy_game = Game(dummy_players, catan_map=build_map(map_type))
    sample_vec = create_sample_vector(dummy_game, dummy_game.state.colors[0])
    board_tensor = create_board_tensor(dummy_game, dummy_game.state.colors[0])
    input_dim = len(sample_vec) + board_tensor.size
    
    return NNAlphaBetaPlayer(
        color=color,
        model_path=model_path,
        input_dim=input_dim
    )

register_cli_player("NNAB", create_nn_alpha_beta_player)