"""
Development Card Strategy - When to play dev cards.

Port of dev card playing logic from JSettlers2 SOCRobotBrain.
"""

from typing import List, Optional, Dict
from catanatron.models.enums import Action, ActionType, RESOURCES

from jsettlers2_bot_py.utils.game_utils import (
    get_player_dev_cards, has_largest_army, get_num_knights_played, get_player_resources
)


class DevCardStrategy:
    """
    Strategy for playing development cards.
    
    Port of dev card logic from JSettlers2 SOCRobotBrain.
    """
    
    def __init__(self, params):
        self.params = params
    
    def should_play_knight_before_roll(self, game, player_trackers: Dict) -> bool:
        """
        Decide if we should play a knight card before rolling.
        
        Strategy:
        - Play if we can get/maintain largest army
        - Play if we're close to winning and need the VP
        
        Args:
            game: Game state
            player_trackers: Player trackers
            
        Returns:
            True if we should play knight now
        """
        current_color = game.state.colors[game.state.current_turn_index]
        
        # Check if we already have largest army
        if has_largest_army(game, current_color):
            return False  # No need to play now
        
        # Check how many knights we've played
        our_knights = get_num_knights_played(game, current_color)
        
        # Find max knights by opponents
        max_opponent_knights = 0
        for color in game.state.colors:
            if color != current_color:
                knights = get_num_knights_played(game, color)
                max_opponent_knights = max(max_opponent_knights, knights)
        
        # Play if we can get largest army (need 3+ and more than anyone else)
        if our_knights >= 2 and our_knights >= max_opponent_knights:
            return True
        
        return False
    
    def choose_dev_card_to_play(self, game, playable_actions: List[Action],
                               next_goal: Dict, building_plan: list,
                               player_trackers: Dict) -> Optional[Action]:
        """
        Choose which development card to play (if any).
        
        Priority:
        1. Year of Plenty if we need just 1-2 resources
        2. Monopoly if we need lots of one resource
        3. Road Building if we need roads
        
        Args:
            game: Game state
            playable_actions: Available actions
            next_goal: Next goal in building plan
            building_plan: Full building plan
            player_trackers: Player trackers
            
        Returns:
            Dev card action to play, or None
        """
        current_color = game.state.colors[game.state.current_turn_index]
        our_resources = get_player_resources(game, current_color)
        dev_cards = get_player_dev_cards(game, current_color)
        
        # Determine what resources we need
        resources_needed = self._calculate_resources_needed(next_goal, our_resources)
        total_needed = sum(resources_needed.values())
        
        if total_needed == 0:
            return None  # We have everything we need
        
        # Check Year of Plenty
        if total_needed <= 2:
            yop_action = next(
                (a for a in playable_actions if a.action_type == ActionType.PLAY_YEAR_OF_PLENTY),
                None
            )
            if yop_action:
                # Choose which resources to take
                resources_to_take = []
                for resource, needed in resources_needed.items():
                    resources_to_take.extend([resource] * min(needed, 2))
                
                if len(resources_to_take) == 1:
                    resources_to_take.append(resources_to_take[0])  # Take 2 of same
                
                # Create action with resources
                return Action(current_color, ActionType.PLAY_YEAR_OF_PLENTY, tuple(resources_to_take[:2]))
        
        # Check Monopoly
        if total_needed >= 2:
            monopoly_action = next(
                (a for a in playable_actions if a.action_type == ActionType.PLAY_MONOPOLY),
                None
            )
            if monopoly_action:
                # Choose resource we need most
                most_needed_resource = max(resources_needed, key=resources_needed.get)
                if resources_needed[most_needed_resource] >= 2:
                    return Action(current_color, ActionType.PLAY_MONOPOLY, most_needed_resource)
        
        # Check Road Building
        if next_goal['type'] == ActionType.BUILD_ROAD:
            rb_action = next(
                (a for a in playable_actions if a.action_type == ActionType.PLAY_ROAD_BUILDING),
                None
            )
            if rb_action:
                return rb_action
        
        return None
    
    def _calculate_resources_needed(self, goal: Dict, current_resources: Dict) -> Dict[str, int]:
        """
        Calculate how many of each resource we need for a goal.
        
        Returns:
            Dictionary of resource: amount_needed
        """
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





