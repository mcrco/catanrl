"""CatanRL's construction of Catanatron's builtin bots and its player lifecycle."""

from __future__ import annotations

import pytest
from catanatron.models.player import Color
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.registry import SpecError

from catanrl.envs.puffer.common import create_expert, create_opponents
from catanrl.envs.puffer.rewards import WinReward
from catanrl.envs.puffer.single_agent_env import SingleAgentCatanatronPufferEnv
from catanrl.eval.parallel_mcts_eval import _build_opponents
from catanrl.players import NNMCTSPlayer
from catanrl.utils.catanatron_game import COLOR_ORDER


def test_opponent_configs_set_builtin_params():
    alphabeta, no_pruning, mcts, playouts = create_opponents(["AB:3", "ab:2:false", "M:7", "G:4"])
    assert isinstance(alphabeta, AlphaBetaPlayer)
    assert (alphabeta.params.depth, alphabeta.params.prunning) == (3, True)
    assert (no_pruning.params.depth, no_pruning.params.prunning) == (2, False)
    assert isinstance(mcts, MCTSPlayer) and mcts.params.num_simulations == 7
    assert isinstance(playouts, GreedyPlayoutsPlayer) and playouts.params.num_playouts == 4

    expert = create_expert("alphabeta")
    assert expert.color == Color.BLUE
    assert (expert.params.depth, expert.params.prunning) == (2, True)


def test_mcts_opponent_models_set_builtin_params():
    build = NNMCTSPlayer._build_opponent_factory
    alphabeta = build("alphabeta")(Color.RED)
    assert (alphabeta.params.depth, alphabeta.params.prunning) == (2, True)
    assert build("mcts")(Color.RED).params.num_simulations == 100
    assert build("playouts")(Color.RED).params.num_playouts == 25


def test_opponent_specs_go_through_catanatrons_registry():
    value, alphabeta = _build_opponents("F, AB:3:prunning=true")
    assert isinstance(value, ValueFunctionPlayer) and value.color == COLOR_ORDER[1]
    assert isinstance(alphabeta, AlphaBetaPlayer) and alphabeta.color == COLOR_ORDER[2]
    assert (alphabeta.params.depth, alphabeta.params.prunning) == (3, True)
    # Positional params follow declaration order (depth, value_fn, prunning).
    with pytest.raises(SpecError, match="value_fn"):
        _build_opponents("AB:2:True")


class _RecordingExpert(ValueFunctionPlayer):
    def __init__(self, color):
        super().__init__(color)
        self.games = []

    def before(self, game):
        self.games.append(game)


def test_single_agent_reset_starts_the_unseated_expert_on_the_new_game():
    expert = _RecordingExpert(Color.BLUE)
    env = SingleAgentCatanatronPufferEnv(
        config={
            "enemies": create_opponents(["R"]),
            "reward_function": WinReward(),
            "expert_player": expert,
        }
    )
    env.reset(seed=3)
    first_game = env.game
    env.reset(seed=4)
    assert expert.games == [first_game, env.game]
