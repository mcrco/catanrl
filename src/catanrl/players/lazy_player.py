"""Pickle-safe web player that delegates to a process-local cached agent."""

from __future__ import annotations

from catanatron.models.player import Color, Player

from catanrl.players.factory import get_cached_player


class LazyConfiguredPlayer(Player):
    """Stores only spec id + color; models live in factory module cache."""

    def __init__(self, spec_id: str, color: Color) -> None:
        super().__init__(color, is_bot=True)
        self.spec_id = spec_id

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        inner = get_cached_player(self.spec_id, self.color.value)
        return inner.decide(game, playable_actions)

    def __repr__(self) -> str:
        return f"LazyConfiguredPlayer:{self.spec_id}:{self.color.value}"

    def __getstate__(self):
        return {"spec_id": self.spec_id, "color": self.color}

    def __setstate__(self, state):
        Player.__init__(self, state["color"], is_bot=True)
        self.spec_id = state["spec_id"]
