import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from catanatron.game import Game
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.registry import REGISTRY

from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    game_to_features,
    get_numeric_feature_names,
)
from rl.models import ValueNetwork  # type: ignore


class NNAlphaBetaPlayer(AlphaBetaPlayer):
    """
    An AlphaBetaPlayer that uses a trained ValueNetwork for leaf evaluation.
    """

    use_value_function = True
    # A class attribute, as AlphaBetaPlayer derives this read-only property from params.
    value_fn_builder_name = "nn_value_network"

    def __init__(
        self,
        color,
        model_path: str,
        input_dim: int,
        output_range: Tuple[float, float] = (-1, 1),
        map_type: str = "BASE",
        num_players: int = 2,
        numeric_features: List[str] | None = None,
        params: Optional[AlphaBetaPlayer.Params] = None,
    ):
        super().__init__(color, params)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_range = output_range
        
        # Load the trained value network
        self.value_net = ValueNetwork(input_dim).to(self.device)
        self.value_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.value_net.eval()

    def value_function(self, game: Game, p0_color: int) -> float:
        """
        Overrides the heuristic value function to use the neural network.
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

    def __repr__(self) -> str:
        return super().__repr__()


class PretrainedNNAlphaBetaPlayer(NNAlphaBetaPlayer):
    """
    NNAlphaBetaPlayer with the pretrained value network for a map template,
    registered as NNAB (e.g. NNAB:map_template=MINI).
    """

    @dataclass(frozen=True)
    class Params(AlphaBetaPlayer.Params):
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
            'MINI': 'weights/f-f-mini-10k-samp0.1/f-f-mini-10k-samp0.1_value.pt',
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


REGISTRY.register("NNAB", PretrainedNNAlphaBetaPlayer, replace=True)
