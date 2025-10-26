"""
Opening Build Strategy - Initial settlement and road placement.

Port of OpeningBuildStrategy from JSettlers2.
"""

from typing import List
from catanatron.models.player import Color
from catanatron.models.enums import Action, ActionType, SETTLEMENT

from jsettlers2_bot_py.utils.board_utils import evaluate_settlement_spot


class OpeningBuildStrategy:
    """
    Strategy for initial game setup - placing first settlements and roads.
    
    Port of OpeningBuildStrategy from JSettlers2.
    """
    
    def __init__(self, params):
        self.params = params
        self.last_settlement_node = None
    
    def choose_initial_settlement(self, game, settlement_actions: List[Action], 
                                  player_trackers: dict) -> Action:
        """
        Choose where to place initial settlement.
        
        Strategy:
        - First settlement: Maximize production value
        - Second settlement: Balance production and resource diversity
        
        Args:
            game: Game state
            settlement_actions: List of possible settlement placement actions
            player_trackers: Player trackers for opponent analysis
            
        Returns:
            The chosen settlement action
        """
        # Evaluate each possible settlement spot
        scored_actions = []
        
        for action in settlement_actions:
            node_id = action.value
            score = evaluate_settlement_spot(game, node_id, action.color)
            
            # Bonus for resource diversity
            # TODO: Check which resources we already have settlements on
            
            # Penalty if too close to opponents (for second placement)
            # TODO: Check opponent settlements
            
            scored_actions.append((score, action))
        
        # Return the best scoring action
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        
        # Save for road placement
        best_action = scored_actions[0][1]
        self.last_settlement_node = best_action.value
        
        return best_action
    
    def choose_initial_road(self, game, road_actions: List[Action],
                           player_trackers: dict) -> Action:
        """
        Choose where to place initial road after settlement.
        
        Strategy:
        - Extend towards good expansion spots
        - Block opponents if necessary
        - Point towards center of board
        
        Args:
            game: Game state
            road_actions: List of possible road placement actions
            player_trackers: Player trackers
            
        Returns:
            The chosen road action
        """
        if not road_actions:
            return None
        
        # If we know where we just placed settlement, prefer roads from there
        if self.last_settlement_node is not None:
            # Find roads adjacent to our last settlement
            adjacent_roads = [
                action for action in road_actions
                if self.last_settlement_node in action.value  # edge is a tuple
            ]
            
            if adjacent_roads:
                # Score each road by where it leads
                scored_roads = []
                for action in adjacent_roads:
                    edge = action.value
                    # Get the other node
                    other_node = edge[0] if edge[1] == self.last_settlement_node else edge[1]
                    
                    # Score based on expansion potential
                    # TODO: Evaluate where this road leads
                    score = 50.0  # Placeholder
                    
                    scored_roads.append((score, action))
                
                scored_roads.sort(key=lambda x: x[0], reverse=True)
                return scored_roads[0][1]
        
        # Fallback: just take first available road
        return road_actions[0]





