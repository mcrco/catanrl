"""
Player Tracker - Track and predict opponent behavior.

Port of SOCPlayerTracker from JSettlers2.
"""

from typing import Dict
from catanatron.models.player import Color
from catanatron.models.enums import RESOURCES

from jsettlers2_bot_py.utils.game_utils import (
    get_player_production, get_victory_points, get_settlements_for_player,
    get_cities_for_player, get_roads_for_player, get_player_resources
)


class PlayerTracker:
    """
    Tracks and predicts a single player's behavior and resources.
    
    Port of SOCPlayerTracker from JSettlers2.
    Uses production estimates and game state to predict when opponents
    can build various pieces.
    """
    
    def __init__(self, game, color: Color, params):
        self.color = color
        self.params = params
        self.game = game
        
        # Predictions
        self.win_game_eta = 999  # Estimated turns to win
        self.estimated_turns_to_settlement = 999
        self.estimated_turns_to_city = 999
        self.estimated_turns_to_dev_card = 999
        self.estimated_turns_to_road = 999
        
        # Production estimate (resources per turn)
        self.production_estimate: Dict[str, float] = {r: 0.0 for r in RESOURCES}
        
        # Resource estimation (approximate)
        self.estimated_resources: Dict[str, int] = {r: 0 for r in RESOURCES}
        
        # Victory point tracking
        self.victory_points = 0
        
    def update(self, game):
        """
        Update all predictions based on current game state.
        
        Args:
            game: Current game state
        """
        self.game = game
        
        # Update production estimate
        self.production_estimate = get_player_production(game, self.color)
        
        # Update victory points
        self.victory_points = get_victory_points(game, self.color)
        
        # Update build time estimates
        self._update_build_time_estimates()
        
        # Update win game ETA
        self._update_win_game_eta()
    
    def _update_build_time_estimates(self):
        """
        Estimate how many turns until player can build each piece type.
        
        Uses production estimates to predict resource accumulation.
        """
        # Get current resources (if it's us)
        current_color = self.game.state.colors[self.game.state.current_turn_index]
        if self.color == current_color:
            current_resources = get_player_resources(self.game, self.color)
        else:
            # Estimate based on production
            current_resources = self.estimated_resources
        
        # Calculate turns needed for each piece type
        
        # Settlement: WOOD, BRICK, SHEEP, WHEAT
        self.estimated_turns_to_settlement = self._calculate_turns_needed(
            current_resources,
            {'WOOD': 1, 'BRICK': 1, 'SHEEP': 1, 'WHEAT': 1}
        )
        
        # City: 3 ORE, 2 WHEAT
        self.estimated_turns_to_city = self._calculate_turns_needed(
            current_resources,
            {'ORE': 3, 'WHEAT': 2}
        )
        
        # Road: WOOD, BRICK
        self.estimated_turns_to_road = self._calculate_turns_needed(
            current_resources,
            {'WOOD': 1, 'BRICK': 1}
        )
        
        # Dev Card: SHEEP, WHEAT, ORE
        self.estimated_turns_to_dev_card = self._calculate_turns_needed(
            current_resources,
            {'SHEEP': 1, 'WHEAT': 1, 'ORE': 1}
        )
    
    def _calculate_turns_needed(self, current_resources: Dict[str, int],
                               cost: Dict[str, int]) -> int:
        """
        Calculate turns needed to accumulate resources.
        
        Args:
            current_resources: Resources player currently has
            cost: Resources needed
            
        Returns:
            Estimated number of turns needed
        """
        max_turns = 0
        
        for resource, needed in cost.items():
            have = current_resources.get(resource, 0)
            production = self.production_estimate.get(resource, 0.1)  # Avoid division by zero
            
            if have >= needed:
                continue  # Already have enough
            
            still_needed = needed - have
            turns = int(still_needed / production) + 1 if production > 0 else 99
            max_turns = max(max_turns, turns)
        
        return max_turns
    
    def _update_win_game_eta(self):
        """
        Estimate turns until this player wins.
        
        Based on:
        - Current victory points
        - How fast they can build settlements/cities
        - Potential for longest road/largest army
        """
        vp_needed = 10 - self.victory_points
        
        if vp_needed <= 0:
            self.win_game_eta = 0
            return
        
        # Estimate based on building settlements and cities
        # Simplified: assume they alternate between settlements (1 VP) and cities (1 additional VP)
        
        # Each settlement + city combo = 2 VP, takes ~settlement_eta + city_eta turns
        cycle_vp = 2
        cycle_turns = self.estimated_turns_to_settlement + self.estimated_turns_to_city
        
        if cycle_turns > 0:
            cycles_needed = (vp_needed + cycle_vp - 1) // cycle_vp  # Round up
            self.win_game_eta = cycles_needed * cycle_turns
        else:
            self.win_game_eta = 99
        
        # Cap at reasonable maximum
        self.win_game_eta = min(self.win_game_eta, 99)
    
    def get_win_game_eta(self) -> int:
        """Get the win game ETA."""
        return self.win_game_eta
    
    def get_estimated_turns_to_settlement(self) -> int:
        """Get estimated turns to build a settlement."""
        return self.estimated_turns_to_settlement
    
    def get_estimated_turns_to_city(self) -> int:
        """Get estimated turns to build a city."""
        return self.estimated_turns_to_city
    
    def __repr__(self):
        return f"PlayerTracker({self.color.value}, VP={self.victory_points}, WinETA={self.win_game_eta})"





