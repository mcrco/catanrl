import torch
import numpy as np
from typing import Tuple

from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.cli.cli_players import register_cli_player
from catanatron.models.map import build_map
from catanatron.features import create_sample_vector
from catanatron.gym.board_tensor_features import create_board_tensor
from rl.models import ValueNetwork


def game_to_features(game: Game, color: int) -> np.ndarray:
    """
    Extracts features from the game state for the value network.
    """
    # This function creates a fixed-order vector of features from the game state.
    feature_vector = create_sample_vector(game, color)
    board_tensor = create_board_tensor(game, color)
    return np.concatenate([feature_vector, board_tensor.flatten()], axis=0)


class NNValuePlayer(ValueFunctionPlayer):
    """
    A ValueFunctionPlayer that uses a trained ValueNetwork for evaluation.
    """

    def __init__(
        self,
        color,
        model_path: str,
        input_dim: int,
        output_range: Tuple[float, float] = (-1, 1),
        epsilon=None,
        **kwargs
    ):
        # Initialize with a dummy value_fn_builder_name since we'll override decide
        super().__init__(color, value_fn_builder_name="base_fn", is_bot=True, epsilon=epsilon, **kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_range = output_range
        
        # Load the trained value network
        self.value_net = ValueNetwork(input_dim).to(self.device)
        self.value_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.value_net.eval()

    def nn_value_function(self, game: Game, p0_color: int) -> float:
        """
        Neural network value function that evaluates a game state.
        """
        game_tensor = torch.from_numpy(
            game_to_features(game, p0_color).reshape(1, -1).astype(np.float32)
        ).to(self.device)
        with torch.no_grad():
            value = self.value_net(game_tensor).item()

        # Clip the value to the output range
        if value < self.output_range[0]:
            value = self.output_range[0]
        elif value > self.output_range[1]:
            value = self.output_range[1]
        return value

    def decide(self, game, playable_actions):
        """
        Override decide to use the neural network value function.
        """
        if len(playable_actions) == 1:
            return playable_actions[0]

        if self.epsilon is not None and np.random.random() < self.epsilon:
            return np.random.choice(playable_actions)

        best_value = float("-inf")
        best_action = None
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            value = self.nn_value_function(game_copy, self.color)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def __repr__(self) -> str:
        return super().__repr__().replace("ValueFunctionPlayer", "NNValuePlayer")


def create_nn_value_player(color, map_template='BASE'):
    """
    Factory function to create NNValuePlayer.
    
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
        'MINI': 'weights/f-f-mini-10k-samp0.1/f-f-mini-10k-samp0.1-joint_policy_value.pt',
        'BASE': 'weights/f-f-base-10k-samp0.1/f-f-base-10k-samp0.1_value.pt',  # Update with your base map model
        'TOURNAMENT': 'weights/f-f-tournament-10k-samp0.1/f-f-tournament-10k-samp0.1_value.pt',  # Update with your tournament map model
    }
    model_path = model_paths.get(map_type, model_paths['BASE'])
    
    # Calculate input dimension by creating a dummy game with dummy players
    dummy_players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    dummy_game = Game(dummy_players, catan_map=build_map(map_type))
    sample_vec = create_sample_vector(dummy_game, dummy_game.state.colors[0])
    board_tensor = create_board_tensor(dummy_game, dummy_game.state.colors[0])
    input_dim = len(sample_vec) + board_tensor.size
    
    return NNValuePlayer(
        color=color,
        model_path=model_path,
        input_dim=input_dim
    )


register_cli_player("NNV", create_nn_value_player)

