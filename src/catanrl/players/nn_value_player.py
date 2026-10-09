import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from catanatron.game import Game
from catanatron.players.value import ValueFunctionPlayer
from catanatron.registry import REGISTRY

from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    game_to_features,
    get_numeric_feature_names,
)
from rl.models import ValueNetwork  # type: ignore


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
        map_type: str = "BASE",
        num_players: int = 2,
        numeric_features: List[str] | None = None,
        params: Optional[ValueFunctionPlayer.Params] = None,
    ):
        # decide() is overridden, so of the params only epsilon is used
        super().__init__(color, params)
        
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
        features = game_to_features(game, p0_color, self.num_players, self.map_type)
        game_tensor = torch.from_numpy(features.reshape(1, -1).astype(np.float32)).to(self.device)
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

        epsilon = self.params.epsilon
        if epsilon is not None and np.random.random() < epsilon:
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


class PretrainedNNValuePlayer(NNValuePlayer):
    """
    NNValuePlayer with the pretrained value network for a map template,
    registered as NNV (e.g. NNV:map_template=MINI).
    """

    @dataclass(frozen=True)
    class Params(ValueFunctionPlayer.Params):
        # 'BASE' (default), 'm' or 'MINI' for mini map, 't' or 'TOURNAMENT' for tournament map
        map_template: str = 'BASE'

    def __init__(self, color, params=None):
        params = params if params is not None else self.Params()
        # Normalize map template parameter
        map_type_map = {
            'm': 'MINI',
            'MINI': 'MINI',
            't': 'TOURNAMENT',
            'TOURNAMENT': 'TOURNAMENT',
            'BASE': 'BASE',
            'b': 'BASE',
        }
        map_type = map_type_map.get(params.map_template, 'BASE')

        # Use different model weights based on map type
        model_paths = {
            'MINI': 'weights/f-f-mini-10k-samp0.1/f-f-mini-10k-samp0.1-joint_policy_value.pt',
            'BASE': 'weights/f-f-base-10k-samp0.1/f-f-base-10k-samp0.1_value.pt',  # Update with your base map model
            'TOURNAMENT': 'weights/f-f-tournament-10k-samp0.1/f-f-tournament-10k-samp0.1_value.pt',  # Update with your tournament map model
        }
        model_path = model_paths.get(map_type, model_paths['BASE'])

        num_players = 2
        numeric_features = list(get_numeric_feature_names(num_players, map_type))
        input_dim = compute_feature_vector_dim(num_players, map_type)

        super().__init__(
            color=color,
            model_path=model_path,
            input_dim=input_dim,
            map_type=map_type,
            num_players=num_players,
            numeric_features=numeric_features,
            params=params,
        )


REGISTRY.register("NNV", PretrainedNNValuePlayer, replace=True)
