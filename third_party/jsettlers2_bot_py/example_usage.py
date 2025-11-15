#!/usr/bin/env python3
"""
Example usage of the JSettlers2 bot with Catanatron.

This script demonstrates:
1. Creating bots with different difficulty levels
2. Running a game
3. Testing bot against different opponents
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from jsettlers2_bot_py import JSettlersRobotBrain, RobotDifficulty, RobotParameters


def example_basic_game():
    """Run a simple game with JSettlers bots."""
    print("=" * 60)
    print("Example 1: Basic Game with JSettlers Bots")
    print("=" * 60)
    
    # Create bots with different difficulties
    smart_bot = JSettlersRobotBrain(Color.RED, difficulty=RobotDifficulty.SMART)
    hard_bot = JSettlersRobotBrain(Color.BLUE, difficulty=RobotDifficulty.HARD)
    moderate_bot = JSettlersRobotBrain(Color.ORANGE, difficulty=RobotDifficulty.MODERATE)
    easy_bot = JSettlersRobotBrain(Color.WHITE, difficulty=RobotDifficulty.EASY)
    
    print(f"Players:")
    print(f"  1. {smart_bot} (SMART)")
    print(f"  2. {hard_bot} (HARD)")
    print(f"  3. {moderate_bot} (MODERATE)")
    print(f"  4. {easy_bot} (EASY)")
    print()
    
    # Create and play game
    print("Starting game...")
    game = Game([smart_bot, hard_bot, moderate_bot, easy_bot], seed=42)
    
    try:
        game.play()
        winner = game.winning_color()
        print(f"\n✓ Game completed!")
        print(f"Winner: {winner.value if winner else 'No winner'}")
        print(f"Total turns: {game.state.num_turns}")
    except Exception as e:
        print(f"\n✗ Error during game: {e}")
        import traceback
        traceback.print_exc()


def example_custom_parameters():
    """Create a bot with custom parameters."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Bot Parameters")
    print("=" * 60)
    
    # Create custom aggressive bot
    aggressive_params = RobotParameters(
        strategy_type=1,  # SMART_STRATEGY
        trade_flag=2,  # Always trade
        adversarial_factor=2.5,  # Very aggressive blocking
        leader_adversarial_factor=4.0,  # Focus on leader
        threat_multiplier=1.5,  # High threat response
        robber_strategy_type="AGGRESSIVE"
    )
    
    aggressive_bot = JSettlersRobotBrain(Color.RED, robot_parameters=aggressive_params)
    print(f"Created custom aggressive bot: {aggressive_bot}")
    print(f"  Strategy type: {aggressive_params.strategy_type} (SMART)")
    print(f"  Trade flag: {aggressive_params.trade_flag} (Always)")
    print(f"  Adversarial factor: {aggressive_params.adversarial_factor}")
    print(f"  Robber strategy: {aggressive_params.robber_strategy_type}")
    
    # Play against random opponents
    opponents = [RandomPlayer(c) for c in [Color.BLUE, Color.ORANGE, Color.WHITE]]
    game = Game([aggressive_bot] + opponents, seed=123)
    
    print("\nPlaying game with custom bot...")
    try:
        game.play()
        print(f"✓ Game completed! Winner: {game.winning_color().value if game.winning_color() else 'None'}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_benchmark():
    """Benchmark bot performance against random players."""
    print("\n" + "=" * 60)
    print("Example 3: Benchmark vs Random Players (10 games)")
    print("=" * 60)
    
    num_games = 10
    wins = 0
    total_turns = 0
    
    print(f"Running {num_games} games...")
    
    for i in range(num_games):
        # Create new bot and opponents for each game
        bot = JSettlersRobotBrain(Color.RED, difficulty=RobotDifficulty.SMART)
        opponents = [RandomPlayer(c) for c in [Color.BLUE, Color.ORANGE, Color.WHITE]]
        
        game = Game([bot] + opponents, seed=i)
        
        try:
            game.play()
            if game.winning_color() == Color.RED:
                wins += 1
            total_turns += game.state.num_turns
            print(f"  Game {i+1}/{num_games}: {'WIN' if game.winning_color() == Color.RED else 'LOSS'} "
                  f"({game.state.num_turns} turns)")
        except Exception as e:
            print(f"  Game {i+1}/{num_games}: ERROR - {e}")
    
    print(f"\nResults:")
    print(f"  Wins: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"  Average turns: {total_turns/num_games:.1f}")


def example_difficulty_comparison():
    """Compare different difficulty levels."""
    print("\n" + "=" * 60)
    print("Example 4: Difficulty Level Showcase")
    print("=" * 60)
    
    difficulties = [
        (RobotDifficulty.EASY, "Conservative, simple strategy"),
        (RobotDifficulty.MODERATE, "Balanced approach with some trading"),
        (RobotDifficulty.HARD, "Smart strategy with full trading"),
        (RobotDifficulty.SMART, "Advanced strategy with aggressive tactics"),
    ]
    
    print("\nDifficulty Levels:")
    for difficulty, description in difficulties:
        params = RobotParameters.create_for_difficulty(difficulty)
        print(f"\n{difficulty.value}:")
        print(f"  {description}")
        print(f"  Strategy: {'SMART' if params.strategy_type == 1 else 'FAST'}")
        print(f"  Trading: {['Never', 'Sometimes', 'Always'][params.trade_flag]}")
        print(f"  Adversarial: {params.adversarial_factor}x")
        print(f"  Robber: {params.robber_strategy_type}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("JSettlers2 Bot for Catanatron - Example Usage")
    print("=" * 60)
    
    try:
        # Example 1: Basic game
        example_basic_game()
        
        # Example 2: Custom parameters
        example_custom_parameters()
        
        # Example 3: Benchmark
        example_benchmark()
        
        # Example 4: Difficulty comparison
        example_difficulty_comparison()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()





