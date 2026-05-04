from typing import Literal

import numpy as np
import torch
from catanatron.models.player import Color, Player

from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    full_game_to_features,
    game_to_features,
    get_observation_indices_from_full,
)
from catanrl.models import (
    PolicyNetworkWrapper,
)
from catanrl.utils.catanatron_action_space import to_action_space


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
        actor_observation_level: ActorObservationLevel = "private",
        **kwargs,
    ):
        super().__init__(color, is_bot=True, **kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.map_type = map_type
        self.actor_observation_level = actor_observation_level
        self.model = model.to(self.device)
        self.model.eval()
        self._observation_index_cache: dict[tuple[int, str], np.ndarray] = {}

    def decide(self, game, playable_actions):
        """
        Select action using the neural network policy.

        The policy network outputs logits for all actions in the action space.
        We select the action with the highest logit that is also in playable_actions.
        """
        if len(playable_actions) == 1:
            return playable_actions[0]

        game_tensor = torch.from_numpy(self._features(game).reshape(1, -1)).to(self.device)

        # Get policy logits
        with torch.no_grad():
            if self.model_type == "flat":
                policy_logits = self.model(game_tensor)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits = self.model(game_tensor)
                policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)
            logits = policy_logits.squeeze(0).cpu().numpy()

        num_players = len(game.state.colors)
        game_colors = tuple(game.state.colors)
        playable_action_indices = [
            to_action_space(action, num_players, self.map_type, game_colors) for action in playable_actions
        ]
        best_offset = int(np.argmax(logits[playable_action_indices]))
        return playable_actions[best_offset]

    def __repr__(self) -> str:
        return super().__repr__().replace("Player", "NNPolicyPlayer")

    def _features(self, game) -> np.ndarray:
        num_players = len(game.state.colors)
        if self.actor_observation_level == "private":
            return game_to_features(game, self.color, num_players, self.map_type)

        full_features = full_game_to_features(
            game,
            num_players,
            self.map_type,
            base_color=self.color,
        )
        cache_key = (num_players, self.actor_observation_level)
        indices = self._observation_index_cache.get(cache_key)
        if indices is None:
            indices = get_observation_indices_from_full(
                num_players,
                self.map_type,
                self.actor_observation_level,
            )
            self._observation_index_cache[cache_key] = indices
        return full_features[indices].astype(np.float32, copy=False)
