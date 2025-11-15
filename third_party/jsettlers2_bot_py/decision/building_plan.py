"""
Building Plan and Possible Piece classes.

Port of SOCBuildingSpeedEstimate and related classes from JSettlers2.
"""

from typing import Optional, Dict
from enum import Enum
from catanatron.models.enums import ActionType


class PieceType(Enum):
    """Types of pieces that can be built."""
    SETTLEMENT = ActionType.BUILD_SETTLEMENT
    CITY = ActionType.BUILD_CITY
    ROAD = ActionType.BUILD_ROAD
    DEVELOPMENT_CARD = ActionType.BUY_DEVELOPMENT_CARD


class PossiblePiece:
    """
    Represents a possible piece to build with its score and ETA.
    
    Port of SOCPossiblePiece from JSettlers2.
    """
    
    def __init__(self, piece_type: ActionType, location: Optional[int] = None):
        self.piece_type = piece_type
        self.location = location
        self.score = 0.0
        self.eta = 99  # Estimated Turns to Arrival
        self.priority = 0
        
        # Resources needed
        self.resources_needed: Dict[str, int] = {}
        
    def reset_score(self):
        """Reset the score to 0."""
        self.score = 0.0
    
    def set_score(self, score: float):
        """Set the score for this piece."""
        self.score = score
    
    def get_score(self) -> float:
        """Get the current score."""
        return self.score
    
    def set_eta(self, eta: int):
        """Set the ETA (estimated turns to build)."""
        self.eta = eta
    
    def get_eta(self) -> int:
        """Get the ETA."""
        return self.eta
    
    def __repr__(self):
        loc_str = f" at {self.location}" if self.location else ""
        return f"PossiblePiece({self.piece_type.value}{loc_str}, score={self.score:.1f}, ETA={self.eta})"


class PossibleSettlement(PossiblePiece):
    """A possible settlement to build."""
    
    def __init__(self, location: int):
        super().__init__(ActionType.BUILD_SETTLEMENT, location)
        self.necessary_roads = []  # Roads needed to reach this settlement
        self.production_value = 0.0
        self.diversity_bonus = 0.0
        
    def set_necessary_roads(self, roads):
        """Set the roads needed to build this settlement."""
        self.necessary_roads = roads


class PossibleCity(PossiblePiece):
    """A possible city upgrade."""
    
    def __init__(self, location: int):
        super().__init__(ActionType.BUILD_CITY, location)
        self.speedup_total = 0.0  # How much this city speeds up future builds
        

class PossibleRoad(PossiblePiece):
    """A possible road to build."""
    
    def __init__(self, location: tuple):
        super().__init__(ActionType.BUILD_ROAD, location)
        self.longest_road_value = 0.0
        self.opens_settlements = []  # Settlements this road enables


class BuildingPlan:
    """
    A plan of what to build.
    
    Port of SOCBuildPlanStack from JSettlers2.
    Represents a prioritized list of building goals.
    """
    
    def __init__(self):
        self.plan: list = []
    
    def push(self, piece: Dict):
        """Add a piece to the plan."""
        self.plan.append(piece)
    
    def pop(self) -> Optional[Dict]:
        """Remove and return the first piece from the plan."""
        if self.plan:
            return self.plan.pop(0)
        return None
    
    def peek(self) -> Optional[Dict]:
        """Get the first piece without removing it."""
        if self.plan:
            return self.plan[0]
        return None
    
    def is_empty(self) -> bool:
        """Check if the plan is empty."""
        return len(self.plan) == 0
    
    def clear(self):
        """Clear the entire plan."""
        self.plan.clear()
    
    def get_depth(self) -> int:
        """Get the number of items in the plan."""
        return len(self.plan)
    
    def __len__(self):
        return len(self.plan)
    
    def __repr__(self):
        return f"BuildingPlan({len(self.plan)} items)"





