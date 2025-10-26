"""
Negotiator - Trading logic and offer evaluation.

Port of SOCRobotNegotiator from JSettlers2.
"""

from typing import List, Optional, Dict
from catanatron.models.player import Color
from catanatron.models.enums import Action, ActionType, RESOURCES

from jsettlers2_bot_py.utils.game_utils import get_player_resources


class Negotiator:
    """
    Handles trade negotiation and evaluation.
    
    Port of SOCRobotNegotiator from JSettlers2.
    Evaluates trade offers and creates offers to other players.
    """
    
    def __init__(self, game, color: Color, params):
        self.game = game
        self.color = color
        self.params = params
        
        # Track offers made
        self.offers_made_this_turn = 0
    
    def evaluate_bank_trades(self, game, trade_actions: List[Action],
                            goal: Dict) -> Optional[Action]:
        """
        Evaluate bank/port trades to work toward a goal.
        
        Args:
            game: Game state
            trade_actions: List of possible MARITIME_TRADE actions
            goal: The building goal we're working toward
            
        Returns:
            Best trade action, or None if no good trades
        """
        if not trade_actions:
            return None
        
        # Get what we have and what we need
        our_resources = get_player_resources(game, self.color)
        needed = self._calculate_needed_resources(goal, our_resources)
        
        # Score each trade
        scored_trades = []
        for action in trade_actions:
            score = self._score_trade(action, our_resources, needed)
            if score > 0:
                scored_trades.append((score, action))
        
        if not scored_trades:
            return None
        
        # Return best trade
        scored_trades.sort(key=lambda x: x[0], reverse=True)
        return scored_trades[0][1]
    
    def _score_trade(self, trade_action: Action, our_resources: Dict[str, int],
                    needed: Dict[str, int]) -> float:
        """
        Score a trade action.
        
        Positive score if trade gets us closer to goal.
        """
        # trade_action.value is a tuple: (give, give, give, give, receive)
        # or (give, give, give, None, receive) for port trades
        trade_value = trade_action.value
        
        if not trade_value or len(trade_value) < 5:
            return -999.0
        
        # Parse trade
        giving = [r for r in trade_value[:4] if r is not None]
        receiving = trade_value[4]
        
        score = 0.0
        
        # Check if we have resources to give
        for resource in giving:
            if resource not in our_resources or our_resources[resource] < 1:
                return -999.0  # Can't afford this trade
        
        # Check if receiving is useful
        if receiving in needed and needed[receiving] > 0:
            score += 100.0  # We need this resource
        
        # Penalty for giving away needed resources
        for resource in giving:
            if resource in needed and needed[resource] > 0:
                score -= 50.0  # Giving away something we need
        
        return score
    
    def create_trade_offer(self, game, goal: Dict,
                          player_trackers: Dict) -> Optional[Action]:
        """
        Create a trade offer to other players.
        
        Args:
            game: Game state
            goal: What we're trying to build
            player_trackers: Info about other players
            
        Returns:
            OFFER_TRADE action, or None if no good offer to make
        """
        if self.params.get_trade_flag() == 0:
            return None  # Trading disabled
        
        self.offers_made_this_turn += 1
        
        # Don't spam offers
        if self.offers_made_this_turn > 3:
            return None
        
        # Get what we have and need
        our_resources = get_player_resources(game, self.color)
        needed = self._calculate_needed_resources(goal, our_resources)
        
        # Find resource we need most
        most_needed = None
        max_need = 0
        for resource, amount in needed.items():
            if amount > max_need:
                max_need = amount
                most_needed = resource
        
        if not most_needed or max_need == 0:
            return None  # Don't need anything
        
        # Find resource we have extra of
        resource_to_give = None
        max_extra = 0
        for resource, amount in our_resources.items():
            need = needed.get(resource, 0)
            extra = amount - need
            if extra > max_extra:
                max_extra = extra
                resource_to_give = resource
        
        if not resource_to_give or max_extra < 1:
            return None  # Nothing to give
        
        # Create offer: give 1-2 of resource_to_give, receive 1 of most_needed
        # Format: (give1, give2, give3, give4, get1, get2, get3, get4, get5)
        # First 5 are offered, last 5 are requested
        
        offer_tuple = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Map resource names to indices
        resource_index = {res: i for i, res in enumerate(RESOURCES)}
        
        if resource_to_give in resource_index and most_needed in resource_index:
            # Give 1 (or 2 if we have plenty)
            give_amount = min(2, max_extra)
            offer_tuple[resource_index[resource_to_give]] = give_amount
            
            # Request 1
            offer_tuple[5 + resource_index[most_needed]] = 1
            
            return Action(self.color, ActionType.OFFER_TRADE, tuple(offer_tuple))
        
        return None
    
    def _calculate_needed_resources(self, goal: Dict,
                                   current_resources: Dict[str, int]) -> Dict[str, int]:
        """Calculate what resources we still need for a goal."""
        needed = {resource: 0 for resource in RESOURCES}
        
        goal_type = goal['type']
        
        # Resource costs
        costs = {
            ActionType.BUILD_SETTLEMENT: {'WOOD': 1, 'BRICK': 1, 'SHEEP': 1, 'WHEAT': 1},
            ActionType.BUILD_CITY: {'ORE': 3, 'WHEAT': 2},
            ActionType.BUILD_ROAD: {'WOOD': 1, 'BRICK': 1},
            ActionType.BUY_DEVELOPMENT_CARD: {'SHEEP': 1, 'WHEAT': 1, 'ORE': 1},
        }
        
        if goal_type in costs:
            cost = costs[goal_type]
            for resource, amount in cost.items():
                have = current_resources.get(resource, 0)
                need = max(0, amount - have)
                needed[resource] = need
        
        return needed
    
    def reset_offers_made(self):
        """Reset the offer counter for a new turn."""
        self.offers_made_this_turn = 0





