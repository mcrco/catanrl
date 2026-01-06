from typing import TYPE_CHECKING, Dict

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points
from catanatron.features import build_production_features

if TYPE_CHECKING:
    from catanrl.envs.gym.single_env import SingleAgentCatanatronEnv


class RewardFunction:
    """
    Base class for reward functions that attach to catanatron environments.
    Can be stateful, e.g. ShapedReward tracks previous victory points to compute the reward.
    """

    def __init__(self) -> None:
        pass

    def reward(
        self,
        env: "SingleAgentCatanatronEnv",
        game: Game,
        color: Color,
    ) -> float:
        raise NotImplementedError

    def after_step(self, env: "SingleAgentCatanatronEnv", game: Game, color: Color) -> None:
        return None

    def after_reset(self, env: "SingleAgentCatanatronEnv") -> None:
        return None


class WinReward(RewardFunction):
    def reward(
        self,
        env: "SingleAgentCatanatronEnv",
        game: Game,
        color: Color,
    ) -> float:
        winning_color = game.winning_color()
        if color == winning_color:
            return 1.0
        elif winning_color is None:
            return 0.0
        else:
            return -1.0


DEFAULT_SHAPED_REWARD_CONFIG = {
    "vp_diff_weight": 0.01,
    "production_diff_weight": 0.0025,
    "win_weight": 1,
}


class ShapedReward(RewardFunction):
    def __init__(
        self,
        config: Dict[str, float] = DEFAULT_SHAPED_REWARD_CONFIG,
    ) -> None:
        self.config = config
        self.prev_vps = 0
        self.prev_production = 0

    def reward(
        self,
        env: "SingleAgentCatanatronEnv",
        game: Game,
        color: Color,
    ) -> float:
        """
        Reward is given for:
        - Winning the game (forces cumulative reward to 1)
        - Gaining/losing VP (VP gained / vps_to_win), although losing VP is not necessarily determined by current action.
        """
        winning_color = game.winning_color()
        if winning_color is not None and color == winning_color:
            return self.config["win_weight"]

        vp_diff = get_actual_victory_points(game.state, color) - self.prev_vps
        production_diff = (
            sum(build_production_features(True)(game, color).values()) - self.prev_production
        )
        reward = (
            self.config["vp_diff_weight"] * (vp_diff / env.vps_to_win)
            + self.config["production_diff_weight"] * production_diff
        )
        return reward

    def after_step(
        self,
        env: "SingleAgentCatanatronEnv",
        game: Game,
        color: Color,
    ) -> None:
        self.prev_vps = get_actual_victory_points(game.state, color)
        self.prev_production = sum(build_production_features(True)(game, color).values())

    def after_reset(self, env: "SingleAgentCatanatronEnv") -> None:
        self.prev_vps = 0
        self.prev_production = 0
