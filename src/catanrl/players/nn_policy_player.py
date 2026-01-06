import torch
import numpy as np
import torch.nn as nn
from typing import Literal, Dict

from catanatron.models.player import Player, Color, RandomPlayer
from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.cli.cli_players import register_cli_player
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, ACTIONS_ARRAY, normalize_action

from catanrl.features.catanatron_utils import (
    game_to_features,
    COLOR_ORDER,
)
from catanrl.models import (
    BackboneConfig,
    PolicyNetworkWrapper,
)


class NNPolicyPlayer(Player):
    """
    A Player that uses a trained PolicyNetwork to select actions.
    """

    def __init__(
        self,
        color: Color,
        model_type: str,
        model: PolicyNetworkWrapper,
        **kwargs,
    ):
        super().__init__(color, is_bot=True, **kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.model = model.to(self.device)
        self.model.eval()

    def decide(self, game, playable_actions):
        """
        Select action using the neural network policy.

        The policy network outputs logits for all actions in the action space.
        We select the action with the highest logit that is also in playable_actions.
        """
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Convert game state to features
        game_tensor = torch.from_numpy(
            game_to_features(game, self.color, len(game.state.colors), "BASE").reshape(1, -1)
        ).to(self.device)

        # Get policy logits
        with torch.no_grad():
            if self.model_type == "flat":
                policy_logits = self.model(game_tensor)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits = self.model(game_tensor)
                policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)
            logits = policy_logits.squeeze(0).cpu().numpy()

        # Convert playable_actions to their indices in the action space
        # Need to normalize actions and find their index in ACTIONS_ARRAY
        normalized_playable = [normalize_action(a) for a in playable_actions]
        possibilities = [(a.action_type, a.value) for a in normalized_playable]
        playable_action_indices = [ACTIONS_ARRAY.index(x) for x in possibilities]

        # Mask out non-playable actions by setting their logits to -inf
        masked_logits = np.full_like(logits, float("-inf"))
        masked_logits[playable_action_indices] = logits[playable_action_indices]

        # Select action with highest logit among playable actions
        best_action_idx = np.argmax(masked_logits)

        # Find the action object corresponding to this index
        best_possibility = ACTIONS_ARRAY[best_action_idx]
        for i, possibility in enumerate(possibilities):
            if possibility == best_possibility:
                return playable_actions[i]

        raise ValueError("NN Policy Player chose unavailable action.")

    def __repr__(self) -> str:
        return super().__repr__().replace("Player", "NNPolicyPlayer")


def create_nn_policy_player(
    color,
    model_type: str,
    backbone_config: BackboneConfig,
    model_path: str,
    map_template: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    num_players: int = 2,
):
    """
    Factory function to create NNPolicyPlayer.

    Args:
        color: Player color
        map_template: Map template - 'BASE' (default), 'm' or 'MINI' for mini map,
                      't' or 'TOURNAMENT' for tournament map
    """
    # Normalize map template parameter
    map_type_map: Dict[str, Literal["BASE", "TOURNAMENT", "MINI"]] = {
        "m": "MINI",
        "MINI": "MINI",
        "t": "TOURNAMENT",
        "TOURNAMENT": "TOURNAMENT",
        "BASE": "BASE",
        "b": "BASE",
    }
    map_type = map_type_map.get(map_template if isinstance(map_template, str) else "BASE", "BASE")

    num_players = 2
    dummy_game = Game(
        players=[RandomPlayer(c) for c in COLOR_ORDER[:num_players]],
    )
    input_dim = len(
        game_to_features(dummy_game, COLOR_ORDER[0], num_players=num_players, map_type=map_type)
    )

    return NNPolicyPlayer(
        color=color,
        model_type=model_type,
        backbone_config=backbone_config,
        model_path=model_path,
        input_dim=input_dim,
        map_type=map_type,
        num_players=num_players,
    )
