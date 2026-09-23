"""A cppanatron game seen through CatanRL's flat action space.

CatanRL's environments, self-play and evaluation address players by seat index
and actions by flat index. :class:`FlatGame` pairs a ``cppanatron.Game`` with
its ``FlatActionSpace`` so those call sites stay one line each; ``game`` is the
full structured engine API.
"""

from __future__ import annotations

import numpy as np
import cppanatron as cpp
from cppanatron.catanatron import observation_layout

from catanrl.envs.puffer.common import MapType
from catanrl.utils.catanatron_map import DEFAULT_NUMBER_PLACEMENT, NumberPlacement


class FlatGame:
    def __init__(
        self,
        num_players: int = 2,
        map_type: MapType = "BASE",
        *,
        seed: int = 0,
        map_seed: int | None = None,
        number_placement: NumberPlacement = DEFAULT_NUMBER_PLACEMENT,
        discard_limit: int = 7,
        friendly_robber: bool = False,
        vps_to_win: int = 10,
    ) -> None:
        self.map_type = map_type
        self.game = cpp.Game(
            num_players,
            map_type=cpp.MapType[map_type],
            seed=seed,
            map_seed=map_seed,
            number_placement=cpp.NumberPlacement[number_placement.upper()],
            discard_limit=discard_limit,
            friendly_robber=friendly_robber,
            vps_to_win=vps_to_win,
        )
        self.space = cpp.FlatActionSpace(num_players, cpp.MapType[map_type])

    @property
    def num_players(self) -> int:
        return self.game.num_players

    @property
    def action_space_size(self) -> int:
        return len(self.space)

    @property
    def current_player(self) -> int:
        return self.game.current_player_index

    @property
    def num_turns(self) -> int:
        return self.game.num_turns

    @property
    def winner(self) -> int | None:
        """Seat index of the winner, if any."""
        return self.game.winning_player_index

    def action_mask(self) -> np.ndarray:
        return self.space.mask(self.game)

    def step(
        self,
        action: int,
        *,
        dice: tuple[int, int] | None = None,
        development_card: cpp.DevelopmentCard | None = None,
        stolen_resource: cpp.Resource | None = None,
    ) -> None:
        """Play a flat action, replaying the given chance outcomes if any."""
        self.game.execute(
            self.space.decode(int(action), self.game),
            dice=dice,
            development_card=development_card,
            stolen_resource=stolen_resource,
        )

    def expert_action(self) -> int:
        """Flat index of Catanatron's ValueFunctionPlayer choice."""
        return self.space.index(cpp.value_action(self.game), self.game)

    def player(self, player: int) -> cpp.PlayerState:
        return self.game.player(player)

    def production(self, player: int) -> float:
        """Expected resource cards per roll for a seat, net of the robber."""
        return cpp.production_sum(self.game, self.game.colors[player])

    def observation(self, base_player: int) -> np.ndarray:
        """Full numeric features and board tensor from ``base_player``'s seat."""
        return cpp.full_observation(self.game, base_player, observation_layout())

    def search(
        self,
        *,
        c_puct: float = 1.5,
        seed: int = 0,
        canonical_pruning: bool = False,
    ) -> cpp.MCTSSearch:
        """A native search tree rooted at the current position."""
        return cpp.MCTSSearch(
            self.game,
            self.space,
            observation_layout(),
            c_puct=c_puct,
            seed=seed,
            canonical_pruning=canonical_pruning,
        )


__all__ = ["FlatGame"]
