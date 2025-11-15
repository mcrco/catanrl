"""
JSettlers2 Bot - Python Implementation for Catanatron

This package contains a Python port of the JSettlers2 robot AI for use with the Catanatron engine.
The original Java implementation is a sophisticated bot with strategic planning, opponent tracking,
and trading negotiation capabilities.

Main Components:
- JSettlersRobotBrain: Main bot class that implements the Player interface
- RobotParameters: Configuration for bot difficulty and behavior
- DecisionMaker: Core planning and decision logic
- Various strategy modules for different game phases and situations
- PlayerTracker: For tracking and predicting opponent behavior
- Negotiator: For trade offer creation and evaluation

Usage:
    from jsettlers2_bot_py import JSettlersRobotBrain, RobotDifficulty
    from catanatron.models.player import Color
    from catanatron.game import Game
    
    bot = JSettlersRobotBrain(Color.RED, difficulty=RobotDifficulty.SMART)
    game = Game([bot, ...])
    game.play()
"""

from .robot_brain import JSettlersRobotBrain
from .robot_parameters import RobotParameters, RobotDifficulty

__version__ = "1.0.0"
__all__ = ["JSettlersRobotBrain", "RobotParameters", "RobotDifficulty"]





