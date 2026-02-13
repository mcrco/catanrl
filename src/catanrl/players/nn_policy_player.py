from typing import Literal

import numpy as np
import torch
from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY, normalize_action
from catanatron.models.player import Color, Player

from catanrl.features.catanatron_utils import (
    game_to_features,
)
from catanrl.models import (
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
        map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE",
        **kwargs,
    ):
        super().__init__(color, is_bot=True, **kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.map_type = map_type
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
            game_to_features(game, self.color, len(game.state.colors), self.map_type).reshape(1, -1)
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
