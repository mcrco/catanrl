"""
Discard Strategy - Choosing what to discard when over limit.

Port of DiscardStrategy from JSettlers2.
"""

from typing import List
from catanatron.models.enums import Action, ActionType, RESOURCES

from jsettlers2_bot_py.utils.game_utils import get_player_resources


class DiscardStrategy:
    """
    Strategy for discarding resources when over the limit.
    
    Port of DiscardStrategy from JSettlers2.
    """
    
    def __init__(self, params):
        self.params = params
    
    def choose_discard(self, game, playable_actions: List[Action],
                      building_plan: list) -> Action:
        """
        Choose which resources to discard.
        
        Strategy:
        - Keep resources needed for building plan
        - Keep diverse resources
        - Discard excess of one type first
        - Prefer to keep WHEAT and ORE (more valuable)
        
        Args:
            game: Game state
            playable_actions: List of possible discard actions
            building_plan: Current building plan
            
        Returns:
            The chosen discard action
        """
        discard_actions = [a for a in playable_actions if a.action_type == ActionType.DISCARD]
        
        if not discard_actions:
            return playable_actions[0]
        
        # Get our current resources
        current_color = game.state.colors[game.state.current_turn_index]
        our_resources = get_player_resources(game, current_color)
        
        # Determine what we need for building plan
        needed_resources = self._get_needed_resources(building_plan)
        
        # Score each discard action
        scored_actions = []
        for action in discard_actions:
            # action.value is the list of resources to discard
            discard_list = action.value if action.value else []
            score = self._score_discard(our_resources, discard_list, needed_resources)
            scored_actions.append((score, action))
        
        # Choose best (highest score = keep most important resources)
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return scored_actions[0][1]
    
    def _get_needed_resources(self, building_plan: list) -> dict:
        """
        Determine what resources we need based on building plan.
        
        Returns:
            Dictionary of resource: amount_needed
        """
        needed = {resource: 0 for resource in RESOURCES}
        
        if not building_plan:
            return needed
        
        # Check first goal in plan
        next_goal = building_plan[0]
        goal_type = next_goal['type']
        
        # Map goal types to resource costs
        # Settlement: WOOD, BRICK, SHEEP, WHEAT
        # City: 3 ORE, 2 WHEAT
        # Road: WOOD, BRICK
        # Dev Card: SHEEP, WHEAT, ORE
        
        if goal_type == ActionType.BUILD_SETTLEMENT:
            needed['WOOD'] = 1
            needed['BRICK'] = 1
            needed['SHEEP'] = 1
            needed['WHEAT'] = 1
        elif goal_type == ActionType.BUILD_CITY:
            needed['ORE'] = 3
            needed['WHEAT'] = 2
        elif goal_type == ActionType.BUILD_ROAD:
            needed['WOOD'] = 1
            needed['BRICK'] = 1
        elif goal_type == ActionType.BUY_DEVELOPMENT_CARD:
            needed['SHEEP'] = 1
            needed['WHEAT'] = 1
            needed['ORE'] = 1
        
        return needed
    
    def _score_discard(self, our_resources: dict, discard_list: list,
                      needed_resources: dict) -> float:
        """
        Score a discard action (higher is better).
        
        Penalize discarding resources we need.
        """
        score = 100.0
        
        # Count what we're discarding
        discard_counts = {resource: 0 for resource in RESOURCES}
        for resource in discard_list:
            if resource in discard_counts:
                discard_counts[resource] += 1
        
        # Penalty for discarding needed resources
        for resource, amount in discard_counts.items():
            if resource in needed_resources:
                needed = needed_resources[resource]
                score -= amount * needed * 10  # Heavy penalty for needed resources
        
        # Penalty for reducing diversity
        resources_kept = set()
        for resource in RESOURCES:
            if our_resources.get(resource, 0) > discard_counts.get(resource, 0):
                resources_kept.add(resource)
        
        # Bonus for keeping diverse hand
        score += len(resources_kept) * 5
        
        return score





