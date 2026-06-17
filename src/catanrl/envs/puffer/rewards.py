from __future__ import annotations

from typing import Any, Dict

from catanatron.features import build_production_features
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points


class RewardFunction:
    """
    Base class for reward functions that attach to catanatron environments.
    Can be stateful, e.g. ShapedReward tracks previous victory points to compute the reward.
    """

    def reward(self, env: Any, game: Game, color: Color) -> float:
        raise NotImplementedError

    def after_step(self, env: Any, game: Game, color: Color) -> None:
        return None

    def after_reset(self, env: Any) -> None:
        return None


class WinReward(RewardFunction):
    def reward(self, env: Any, game: Game, color: Color) -> float:
        winning_color = game.winning_color()
        if color == winning_color:
            return 1.0
        if winning_color is None:
            return 0.0
        return -1.0


DEFAULT_SHAPED_REWARD_CONFIG = {
    "vp_diff_weight": 0.01,
    "production_diff_weight": 0.0025,
    "win_weight": 1.0,
}


def _vps_to_win(env: Any) -> int:
    return int(getattr(env, "vps_to_win", env.config.vps_to_win))


class ShapedReward(RewardFunction):
    def __init__(
        self,
        env: Any = None,
        config: Dict[str, float] = DEFAULT_SHAPED_REWARD_CONFIG,
    ) -> None:
        self.config = config
        self._multi_agent = env is not None and hasattr(env, "colors_order")
        if self._multi_agent:
            self.prev_vps: Dict[Color, int] = {color: 0 for color in env.colors_order}
            self.prev_production: Dict[Color, int] = {
                color: 0 for color in env.colors_order
            }
        else:
            self.prev_vps = 0
            self.prev_production = 0

    def reward(self, env: Any, game: Game, color: Color) -> float:
        """
        Reward is given for:
        - Winning the game (forces cumulative reward to 1)
        - Gaining/losing VP (VP gained / vps_to_win), although losing VP is not
          necessarily determined by current action.
        """
        winning_color = game.winning_color()
        if winning_color is not None and color == winning_color:
            return self.config["win_weight"]

        production = sum(build_production_features(True)(game, color).values())
        vps = _vps_to_win(env)
        if self._multi_agent:
            vp_diff = get_actual_victory_points(game.state, color) - self.prev_vps[color]
            production_diff = production - self.prev_production[color]
        else:
            vp_diff = get_actual_victory_points(game.state, color) - self.prev_vps
            production_diff = production - self.prev_production

        return (
            self.config["vp_diff_weight"] * (vp_diff / vps)
            + self.config["production_diff_weight"] * production_diff
        )

    def after_step(self, env: Any, game: Game, color: Color) -> None:
        production = sum(build_production_features(True)(game, color).values())
        vps = get_actual_victory_points(game.state, color)
        if self._multi_agent:
            self.prev_vps[color] = vps
            self.prev_production[color] = production
        else:
            self.prev_vps = vps
            self.prev_production = production

    def after_reset(self, env: Any) -> None:
        if self._multi_agent:
            for color in env.colors_order:
                self.prev_vps[color] = 0
                self.prev_production[color] = 0
        else:
            self.prev_vps = 0
            self.prev_production = 0
