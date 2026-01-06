from catanatron.players.value import base_fn, ValueFunctionPlayer
from catanatron.game import Game
from catanatron.models.player import Color

value_function = base_fn()
players = [ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)]
game = Game(players=players)

while game.winning_color() is None:
    game.play_tick()
    print(f"{game.state.current_color()}: {value_function(game, game.state.current_color())}")

print(game.winning_color())
