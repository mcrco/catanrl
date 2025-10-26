"""
Robot Parameters and Difficulty Settings

Port of soc.util.SOCRobotParameters from JSettlers2.
Defines bot behavior, difficulty levels, and strategic parameters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RobotDifficulty(Enum):
    """Bot difficulty levels"""
    EASY = "EASY"
    MODERATE = "MODERATE"
    HARD = "HARD"
    SMART = "SMART"


class StrategyType(Enum):
    """Strategy types for decision making"""
    FAST = 0  # Simple, fast decisions
    SMART = 1  # More sophisticated planning


@dataclass
class RobotParameters:
    """
    Configuration parameters for robot behavior.
    
    Port of SOCRobotParameters from JSettlers2.
    Controls trading behavior, timing, and strategic weights.
    """
    
    # Strategy
    strategy_type: int = 1  # 0=FAST_STRATEGY, 1=SMART_STRATEGY
    
    # Trading behavior
    trade_flag: int = 1  # 0=never trade, 1=trade sometimes, 2=always try trading
    
    # Game length affects urgency
    max_game_length: int = 300  # Expected maximum game length in turns
    
    # Building speed estimate factors
    building_speed_estimate_fast: float = 1.0
    building_speed_estimate_slow: float = 0.8
    
    # Strategic factors (from SOCRobotDM)
    eta_bonus_factor: float = 0.8  # ETA bonus multiplier
    adversarial_factor: float = 1.5  # Weight for blocking opponents
    leader_adversarial_factor: float = 3.0  # Extra weight for blocking leader
    dev_card_multiplier: float = 2.0  # Value multiplier for dev cards
    threat_multiplier: float = 1.1  # Weight for responding to threats
    
    # Largest Army / Longest Road pursuit
    dev_card_large_army: float = 1.0
    dev_card_longest_road: float = 1.0
    
    # Robber behavior
    robber_strategy_type: str = "SMART"  # SMART, DEFENSIVE, AGGRESSIVE
    
    # Opening strategy
    opening_strategy_type: str = "BALANCED"  # BALANCED, AGGRESSIVE, DEFENSIVE
    
    # Risk tolerance
    threat_response_factor: float = 1.0
    
    @classmethod
    def create_for_difficulty(cls, difficulty: RobotDifficulty) -> "RobotParameters":
        """
        Factory method to create parameters for different difficulties.
        
        Args:
            difficulty: The desired difficulty level
            
        Returns:
            RobotParameters configured for that difficulty
        """
        if difficulty == RobotDifficulty.EASY:
            return cls(
                strategy_type=0,  # FAST_STRATEGY
                trade_flag=0,  # Never trade
                building_speed_estimate_fast=0.5,
                adversarial_factor=1.0,
                leader_adversarial_factor=1.2,
                threat_multiplier=1.0,
                robber_strategy_type="DEFENSIVE",
                opening_strategy_type="DEFENSIVE"
            )
        elif difficulty == RobotDifficulty.MODERATE:
            return cls(
                strategy_type=0,  # FAST_STRATEGY
                trade_flag=1,  # Trade sometimes
                building_speed_estimate_fast=0.75,
                adversarial_factor=1.3,
                leader_adversarial_factor=2.0,
                threat_multiplier=1.05,
                robber_strategy_type="SMART",
                opening_strategy_type="BALANCED"
            )
        elif difficulty == RobotDifficulty.HARD:
            return cls(
                strategy_type=1,  # SMART_STRATEGY
                trade_flag=2,  # Always try trading
                building_speed_estimate_fast=1.0,
                adversarial_factor=1.5,
                leader_adversarial_factor=3.0,
                threat_multiplier=1.1,
                robber_strategy_type="SMART",
                opening_strategy_type="BALANCED"
            )
        else:  # SMART
            return cls(
                strategy_type=1,  # SMART_STRATEGY
                trade_flag=2,  # Always try trading
                building_speed_estimate_fast=1.2,
                adversarial_factor=1.8,
                leader_adversarial_factor=3.5,
                threat_multiplier=1.2,
                threat_response_factor=1.5,
                robber_strategy_type="AGGRESSIVE",
                opening_strategy_type="AGGRESSIVE"
            )
    
    def get_strategy_type(self) -> int:
        """Get the strategy type (0=FAST, 1=SMART)"""
        return self.strategy_type
    
    def get_trade_flag(self) -> int:
        """Get the trade flag (0=never, 1=sometimes, 2=always)"""
        return self.trade_flag





