"""
Monopoly Strategy - Deciding when and what to monopolize.

Port of MonopolyStrategy from JSettlers2.
"""

from typing import Optional
from catanatron.models.player import Color
from catanatron.models.enums import RESOURCES

from jsettlers2_bot_py.utils.game_utils import get_player_resources


class MonopolyStrategy:
    """
    Strategy for playing Monopoly development cards.
    
    Port of MonopolyStrategy from JSettlers2.
    """
    
    def __init__(self, params):
        self.params = params
    
    def decide_play_monopoly(self) -> bool:
        """
        Decide if we should play a monopoly card now.
        
        Simple heuristic: Play if we need a specific resource badly
        and opponents likely have it.
        
        Returns:
            True if we should play monopoly
        """
        # TODO: Implement proper decision logic
        # For now, be conservative
        return False
    
    def choose_monopoly_resource(self, game, building_plan: list,
                                player_trackers: dict) -> Optional[str]:
        """
        Choose which resource to monopolize.
        
        Strategy:
        - Choose resource we need for our building plan
        - Prefer resource that opponents have more of
        - Avoid resource we already have plenty of
        
        Args:
            game: Game state
            building_plan: Current building plan
            player_trackers: Player trackers
            
        Returns:
            Resource name to monopolize, or None
        """
        if not building_plan:
            return None
        
        # Get resources we need for next goal
        next_goal = building_plan[0]
        goal_type = next_goal['type']
        
        # Determine what resources we need
        # TODO: Calculate based on goal type and what we have
        
        # Count what opponents have
        opponent_resources = {resource: 0 for resource in RESOURCES}
        
        current_player_color = game.state.colors[game.state.current_turn_index]
        for color in game.state.colors:
            if color != current_player_color:
                # Estimate based on tracker
                if color in player_trackers:
                    tracker = player_trackers[color]
                    # TODO: Use tracker estimates
                    pass
        
        # Choose resource we need that opponents have
        # Simplified: choose ORE or WHEAT (most valuable)
        our_resources = get_player_resources(game, current_player_color)
        
        if our_resources.get('ORE', 0) < 2:
            return 'ORE'
        if our_resources.get('WHEAT', 0) < 2:
            return 'WHEAT'
        
        return None





