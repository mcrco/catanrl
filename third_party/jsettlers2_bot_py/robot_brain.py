"""
JSettlers Robot Brain - Main AI Controller

Port of SOCRobotBrain from JSettlers2 to Python for Catanatron.
This is the main bot class that implements the Player interface.
"""

import sys
# Add parent directory to path if running as script
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Optional, Dict
from catanatron.models.player import Player, Color
from catanatron.models.enums import ActionType, Action

# Import after path fix
from jsettlers2_bot_py.robot_parameters import RobotParameters, RobotDifficulty
from jsettlers2_bot_py.decision.decision_maker import DecisionMaker
from jsettlers2_bot_py.strategies.opening_build import OpeningBuildStrategy
from jsettlers2_bot_py.strategies.robber_strategy import RobberStrategy
from jsettlers2_bot_py.strategies.monopoly_strategy import MonopolyStrategy
from jsettlers2_bot_py.strategies.discard_strategy import DiscardStrategy
from jsettlers2_bot_py.strategies.dev_card_strategy import DevCardStrategy
from jsettlers2_bot_py.tracking.player_tracker import PlayerTracker
from jsettlers2_bot_py.trading.negotiator import Negotiator


# Constants from Java version
MAX_DENIED_BUILDING_PER_TURN = 3
MAX_DENIED_BANK_TRADES_PER_TURN = 9
MAX_DENIED_PLAYER_TRADES_PER_TURN = 9


class JSettlersRobotBrain(Player):
    """
    Main AI controller - reimplementation of JSettlers SOCRobotBrain for Catanatron.
    
    Key differences from Java version:
    - Synchronous/turn-based instead of message-driven threading
    - No state machine flags needed (waitingFor*, expect*)
    - Simpler since Catanatron provides full game state directly
    
    This bot uses strategic planning, opponent tracking, and trade negotiation
    to make intelligent decisions throughout the game.
    """
    
    def __init__(self, color: Color, difficulty: Optional[RobotDifficulty] = None,
                 robot_parameters: Optional[RobotParameters] = None):
        """
        Initialize the robot brain.
        
        Args:
            color: Player color
            difficulty: Difficulty level (will create default parameters)
            robot_parameters: Custom parameters (overrides difficulty)
        """
        super().__init__(color, is_bot=True)
        
        # Configuration
        if robot_parameters:
            self.params = robot_parameters
        elif difficulty:
            self.params = RobotParameters.create_for_difficulty(difficulty)
        else:
            self.params = RobotParameters.create_for_difficulty(RobotDifficulty.SMART)
        
        # Core decision components (initialized on first decide() call)
        self.decision_maker: Optional[DecisionMaker] = None
        self.negotiator: Optional[Negotiator] = None
        
        # Strategy modules
        self.opening_strategy: Optional[OpeningBuildStrategy] = None
        self.robber_strategy: Optional[RobberStrategy] = None
        self.monopoly_strategy: Optional[MonopolyStrategy] = None
        self.discard_strategy: Optional[DiscardStrategy] = None
        self.dev_card_strategy: Optional[DevCardStrategy] = None
        
        # Player tracking
        self.player_trackers: Dict[Color, PlayerTracker] = {}
        
        # Current turn state (reset each turn)
        self.building_plan: List[Dict] = []
        self.failed_building_attempts = 0
        self.failed_bank_trades = 0
        self.declined_our_player_trades = 0
        self.done_trading = False
        
        # Track what we failed to build (to avoid retrying)
        self.what_we_failed_to_build = None
        
    def decide(self, game, playable_actions: List[Action]) -> Action:
        """
        Main decision method called by Catanatron engine.
        Replaces the run() message loop in Java version.
        
        Args:
            game: Complete game state
            playable_actions: List of valid actions right now
            
        Returns:
            The chosen action
        """
        # Lazy initialization (after game starts and we have game reference)
        if self.decision_maker is None:
            self._initialize_components(game)
        
        # Update player trackers with current game state
        self._update_player_trackers(game)
        
        # Route to appropriate decision method based on game phase
        if game.state.is_initial_build_phase:
            return self._decide_initial_placement(game, playable_actions)
        elif game.state.is_discarding:
            return self._decide_discard(game, playable_actions)
        elif game.state.is_moving_knight:
            return self._decide_move_robber(game, playable_actions)
        elif game.state.is_road_building:
            return self._decide_road_building(game, playable_actions)
        else:
            return self._decide_main_turn(game, playable_actions)
    
    def _initialize_components(self, game):
        """Initialize all decision components once game starts."""
        self.decision_maker = DecisionMaker(game, self.color, self.params, self.player_trackers)
        self.negotiator = Negotiator(game, self.color, self.params)
        
        self.opening_strategy = OpeningBuildStrategy(self.params)
        self.robber_strategy = RobberStrategy(self.params)
        self.monopoly_strategy = MonopolyStrategy(self.params)
        self.discard_strategy = DiscardStrategy(self.params)
        self.dev_card_strategy = DevCardStrategy(self.params)
        
        # Initialize trackers for all players
        for player_color in game.state.colors:
            self.player_trackers[player_color] = PlayerTracker(
                game, player_color, self.params
            )
    
    def _decide_initial_placement(self, game, playable_actions: List[Action]) -> Action:
        """
        Handle initial settlement and road placement.
        
        Equivalent to placeInitSettlement() and planAndPlaceInitRoad() in Java version.
        """
        # Check if we're placing a settlement or road
        settlement_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        
        if settlement_actions:
            return self.opening_strategy.choose_initial_settlement(
                game, settlement_actions, self.player_trackers
            )
        else:
            # Road placement
            road_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD]
            return self.opening_strategy.choose_initial_road(
                game, road_actions, self.player_trackers
            )
    
    def _decide_discard(self, game, playable_actions: List[Action]) -> Action:
        """
        Decide what resources to discard.
        
        Equivalent to discard() in Java version.
        """
        return self.discard_strategy.choose_discard(
            game, playable_actions, self.building_plan
        )
    
    def _decide_move_robber(self, game, playable_actions: List[Action]) -> Action:
        """
        Decide where to move robber and who to rob.
        
        Equivalent to moveRobber() in Java version.
        """
        return self.robber_strategy.choose_robber_placement(
            game, playable_actions, self.player_trackers
        )
    
    def _decide_road_building(self, game, playable_actions: List[Action]) -> Action:
        """
        Decide where to place free roads from Road Building dev card.
        """
        return self.decision_maker.choose_road_building_placement(
            game, playable_actions, self.building_plan
        )
    
    def _decide_main_turn(self, game, playable_actions: List[Action]) -> Action:
        """
        Main turn decision logic - the heart of the bot.
        
        Equivalent to planAndDoActionForPLAY1() and buildOrGetResourceByTradeOrCard()
        in Java version.
        """
        # Check if we need to roll first
        roll_action = next(
            (a for a in playable_actions if a.action_type == ActionType.ROLL),
            None
        )
        
        if roll_action:
            # Consider playing knight before rolling (equivalent to playKnightCardIfShould)
            knight_action = next(
                (a for a in playable_actions if a.action_type == ActionType.PLAY_KNIGHT_CARD),
                None
            )
            if knight_action and self.dev_card_strategy.should_play_knight_before_roll(game, self.player_trackers):
                return knight_action
            return roll_action
        
        # Post-roll actions
        # Check if we should end turn (failed too many times)
        if self._should_end_turn():
            return next(a for a in playable_actions if a.action_type == ActionType.END_TURN)
        
        # Update building plan if empty or outdated (equivalent to planBuilding)
        if not self.building_plan or self._plan_needs_update(game):
            self.building_plan = self.decision_maker.plan_building(game)
        
        # Try to execute plan (equivalent to buildOrGetResourceByTradeOrCard)
        return self._execute_building_plan(game, playable_actions)
    
    def _execute_building_plan(self, game, playable_actions: List[Action]) -> Action:
        """
        Execute the current building plan.
        
        Try to:
        1. Build the next piece in plan
        2. Use dev cards to get resources
        3. Trade with bank/ports
        4. Trade with other players
        5. Buy dev card if can't build
        6. End turn if nothing else possible
        """
        if not self.building_plan:
            # No plan, just end turn
            return next(a for a in playable_actions if a.action_type == ActionType.END_TURN)
        
        next_goal = self.building_plan[0]
        
        # Try to build the next piece in plan
        build_action = self._try_build_piece(playable_actions, next_goal)
        if build_action:
            self.building_plan.pop(0)  # Remove completed goal
            self.failed_building_attempts = 0
            self.what_we_failed_to_build = None
            return build_action
        
        # Can't build yet - try to get resources
        
        # 1. Try dev cards first (Year of Plenty, Monopoly)
        dev_card_action = self.dev_card_strategy.choose_dev_card_to_play(
            game, playable_actions, next_goal, self.building_plan, self.player_trackers
        )
        if dev_card_action:
            return dev_card_action
        
        # 2. Try bank/port trading
        if not self.done_trading and self.failed_bank_trades < MAX_DENIED_BANK_TRADES_PER_TURN:
            trade_action = self._try_bank_trade(game, playable_actions, next_goal)
            if trade_action:
                return trade_action
        
        # 3. Try player trading (if enabled in parameters)
        if (self.params.get_trade_flag() > 0 and not self.done_trading 
            and self.declined_our_player_trades < MAX_DENIED_PLAYER_TRADES_PER_TURN):
            trade_action = self._try_player_trade(game, playable_actions, next_goal)
            if trade_action:
                return trade_action
            # Mark done trading for this turn after trying
            self.done_trading = True
        
        # 4. Consider buying dev card if we can't make progress on plan
        buy_dev_action = next(
            (a for a in playable_actions if a.action_type == ActionType.BUY_DEVELOPMENT_CARD),
            None
        )
        if buy_dev_action and len(self.building_plan) == 1:
            # Buy dev card if it's our only option
            return buy_dev_action
        
        # 5. Give up on current plan and end turn
        self.failed_building_attempts += 1
        return next(a for a in playable_actions if a.action_type == ActionType.END_TURN)
    
    def _try_build_piece(self, playable_actions: List[Action], goal: Dict) -> Optional[Action]:
        """Try to build the piece specified in goal."""
        goal_action_type = goal['type']
        goal_location = goal.get('location')
        
        if goal_location is not None:
            # Specific location
            matching = next(
                (a for a in playable_actions 
                 if a.action_type == goal_action_type and a.value == goal_location),
                None
            )
            return matching
        else:
            # Any location of this type
            matching = [a for a in playable_actions if a.action_type == goal_action_type]
            if matching:
                # Use decision maker to pick best location
                return self.decision_maker.choose_best_build_action(matching, goal)
            return None
    
    def _try_bank_trade(self, game, playable_actions: List[Action], goal: Dict) -> Optional[Action]:
        """Try to trade with bank/ports to get resources for goal."""
        trade_actions = [
            a for a in playable_actions 
            if a.action_type == ActionType.MARITIME_TRADE
        ]
        if not trade_actions:
            return None
        
        best_trade = self.negotiator.evaluate_bank_trades(
            game, trade_actions, goal
        )
        if best_trade:
            return best_trade
        
        self.failed_bank_trades += 1
        return None
    
    def _try_player_trade(self, game, playable_actions: List[Action], goal: Dict) -> Optional[Action]:
        """Try to initiate trade with other players."""
        # Create trade offer
        trade_offer = self.negotiator.create_trade_offer(
            game, goal, self.player_trackers
        )
        if trade_offer:
            # Convert to Action and return
            return trade_offer
        
        self.declined_our_player_trades += 1
        return None
    
    def _should_end_turn(self) -> bool:
        """Check if we should give up and end turn."""
        return (
            self.failed_building_attempts >= MAX_DENIED_BUILDING_PER_TURN or
            (self.failed_bank_trades >= MAX_DENIED_BANK_TRADES_PER_TURN and self.done_trading)
        )
    
    def _plan_needs_update(self, game) -> bool:
        """
        Check if building plan needs recalculation.
        
        Replan if circumstances changed significantly (e.g., robber moved to our tile,
        opponent blocked us, game state changed significantly).
        """
        # Simple heuristic: replan every few turns or if plan is getting stale
        # TODO: Implement more sophisticated replanning logic
        return False
    
    def _update_player_trackers(self, game):
        """Update all player trackers with current game state."""
        for tracker in self.player_trackers.values():
            tracker.update(game)
    
    def reset_state(self):
        """
        Reset bot state between games.
        
        Called by Catanatron when starting a new game.
        """
        self.decision_maker = None
        self.negotiator = None
        self.player_trackers = {}
        self.building_plan = []
        self.failed_building_attempts = 0
        self.failed_bank_trades = 0
        self.declined_our_player_trades = 0
        self.done_trading = False
        self.what_we_failed_to_build = None
    
    def __repr__(self):
        """String representation for debugging."""
        return f"JSettlersRobotBrain({self.color.value}, difficulty={self.params.strategy_type})"


if __name__ == "__main__":
    # Simple test to check if imports work
    print("JSettlers Robot Brain - Python Implementation")
    print(f"Creating bot with SMART difficulty...")
    bot = JSettlersRobotBrain(Color.RED, RobotDifficulty.SMART)
    print(f"Bot created: {bot}")
    print("✓ Basic initialization successful")





