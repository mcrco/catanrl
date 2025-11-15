"""
Building Speed Estimate - Estimate how fast pieces can be built.

Port of SOCBuildingSpeedEstimate from JSettlers2.
"""

from typing import Dict
from catanatron.models.enums import RESOURCES


class BuildingSpeedEstimate:
    """
    Estimates how many turns needed to build each piece type.
    
    Port of SOCBuildingSpeedEstimate from JSettlers2.
    Uses dice roll probabilities and current resources/ports.
    """
    
    def __init__(self, production: Dict[str, float]):
        """
        Initialize with production estimates.
        
        Args:
            production: Dictionary of resource: expected_per_turn
        """
        self.production = production
    
    def get_estimates_from_now(self, current_resources: Dict[str, int],
                               ports: list) -> Dict[str, int]:
        """
        Estimate turns to build each piece type from current state.
        
        Args:
            current_resources: Resources we currently have
            ports: List of ports we control
            
        Returns:
            Dictionary mapping piece_type to estimated_turns
        """
        estimates = {}
        
        # Settlement: WOOD, BRICK, SHEEP, WHEAT
        estimates['SETTLEMENT'] = self._estimate_turns(
            current_resources,
            {'WOOD': 1, 'BRICK': 1, 'SHEEP': 1, 'WHEAT': 1}
        )
        
        # City: 3 ORE, 2 WHEAT
        estimates['CITY'] = self._estimate_turns(
            current_resources,
            {'ORE': 3, 'WHEAT': 2}
        )
        
        # Road: WOOD, BRICK
        estimates['ROAD'] = self._estimate_turns(
            current_resources,
            {'WOOD': 1, 'BRICK': 1}
        )
        
        # Dev Card: SHEEP, WHEAT, ORE
        estimates['DEV_CARD'] = self._estimate_turns(
            current_resources,
            {'SHEEP': 1, 'WHEAT': 1, 'ORE': 1}
        )
        
        return estimates
    
    def _estimate_turns(self, current_resources: Dict[str, int],
                       cost: Dict[str, int]) -> int:
        """
        Estimate turns needed to accumulate resources for a cost.
        
        Args:
            current_resources: What we have now
            cost: What we need
            
        Returns:
            Estimated turns needed
        """
        max_turns = 0
        
        for resource, needed in cost.items():
            have = current_resources.get(resource, 0)
            production = self.production.get(resource, 0.1)  # Small default to avoid /0
            
            if have >= needed:
                continue
            
            still_needed = needed - have
            turns = int(still_needed / production) + 1 if production > 0 else 99
            max_turns = max(max_turns, turns)
        
        return max_turns





