"""
Decision Maker - Core planning and strategy logic.

Port of SOCRobotDM from JSettlers2.
This class determines what to build next based on game state and player trackers.
"""

from typing import List, Dict, Optional
from catanatron.models.player import Color
from catanatron.models.enums import ActionType, Action, SETTLEMENT, CITY, ROAD
from catanatron.models.actions import settlement_possibilities, city_possibilities, road_building_possibilities

from jsettlers2_bot_py.decision.building_plan import (
    PossiblePiece, PossibleSettlement, PossibleCity, PossibleRoad
)
from jsettlers2_bot_py.utils.game_utils import (
    get_player_resources, get_victory_points, get_settlements_for_player,
    get_cities_for_player, get_longest_road_length
)
from jsettlers2_bot_py.utils.board_utils import (
    evaluate_settlement_spot, find_buildable_node_ids, find_buildable_edges
)


# Strategy constants (from Java SOCRobotDM)
SMART_STRATEGY = 1
FAST_STRATEGY = 0

# Choice constants
LA_CHOICE = 0  # Largest Army
LR_CHOICE = 1  # Longest Road
CITY_CHOICE = 2
SETTLEMENT_CHOICE = 3


class DecisionMaker:
    """
    Main planning and decision logic.
    
    Port of SOCRobotDM from JSettlers2.
    Determines what pieces to build based on game state, building speed estimates,
    and player tracking information.
    """
    
    def __init__(self, game, color: Color, params, player_trackers: Dict):
        self.game = game
        self.color = color
        self.params = params
        self.player_trackers = player_trackers
        
        # Planning results
        self.favorite_settlement: Optional[PossibleSettlement] = None
        self.favorite_city: Optional[PossibleCity] = None
        self.favorite_road: Optional[PossibleRoad] = None
        
        # Lists of threatened and good pieces
        self.threatened_settlements: List[PossibleSettlement] = []
        self.good_settlements: List[PossibleSettlement] = []
        self.threatened_roads: List[PossibleRoad] = []
        self.good_roads: List[PossibleRoad] = []
        
    def plan_building(self, game) -> List[Dict]:
        """
        Create a prioritized build plan.
        
        Port of planStuff() from Java SOCRobotDM.
        This is the main entry point for building planning.
        
        Returns:
            List of build goals like:
            [
                {'type': ActionType.BUILD_CITY, 'location': 42, 'priority': 100},
                {'type': ActionType.BUILD_SETTLEMENT, 'location': 15, 'priority': 90},
                ...
            ]
        """
        self.game = game  # Update game reference
        
        # Reset planning state
        self.threatened_settlements.clear()
        self.good_settlements.clear()
        self.threatened_roads.clear()
        self.good_roads.clear()
        self.favorite_settlement = None
        self.favorite_city = None
        self.favorite_road = None
        
        # Choose strategy based on parameters
        if self.params.strategy_type == SMART_STRATEGY:
            return self._smart_game_strategy()
        else:
            return self._fast_game_strategy()
    
    def _smart_game_strategy(self) -> List[Dict]:
        """
        Smart game strategy - uses player trackers and win game ETAs.
        
        Port of smartGameStrategy() from Java SOCRobotDM.
        More sophisticated than fast strategy.
        """
        goals = []
        
        # Get our current victory points
        our_vp = get_victory_points(self.game, self.color)
        
        # Update player trackers to get win game ETAs
        for tracker in self.player_trackers.values():
            tracker.update(self.game)
        
        # Find leader's win game ETA
        leader_wgeta = 999
        for tracker in self.player_trackers.values():
            wgeta = tracker.get_win_game_eta()
            if wgeta < leader_wgeta:
                leader_wgeta = wgeta
        
        # Evaluate all possible pieces
        self._evaluate_possible_settlements()
        self._evaluate_possible_cities()
        self._evaluate_possible_roads()
        
        # Choose favorites based on scores
        self._choose_favorite_settlement()
        self._choose_favorite_city()
        self._choose_favorite_road()
        
        # Build goals list based on what has best score
        best_score = -999999
        best_choice = None
        
        # Compare city
        if self.favorite_city:
            city_score = self.favorite_city.get_score()
            if city_score > best_score:
                best_score = city_score
                best_choice = CITY_CHOICE
        
        # Compare settlement
        if self.favorite_settlement:
            settlement_score = self.favorite_settlement.get_score()
            if settlement_score > best_score:
                best_score = settlement_score
                best_choice = SETTLEMENT_CHOICE
        
        # Compare road (for longest road)
        if self.favorite_road and our_vp >= 5:
            road_score = self.favorite_road.get_score()
            if road_score > best_score:
                best_score = road_score
                best_choice = LR_CHOICE
        
        # Build the plan based on best choice
        if best_choice == CITY_CHOICE and self.favorite_city:
            goals.append({
                'type': ActionType.BUILD_CITY,
                'location': self.favorite_city.location,
                'priority': 100
            })
        elif best_choice == SETTLEMENT_CHOICE and self.favorite_settlement:
            # Add necessary roads first
            for road_edge in self.favorite_settlement.necessary_roads:
                goals.append({
                    'type': ActionType.BUILD_ROAD,
                    'location': road_edge,
                    'priority': 85
                })
            goals.append({
                'type': ActionType.BUILD_SETTLEMENT,
                'location': self.favorite_settlement.location,
                'priority': 90
            })
        elif best_choice == LR_CHOICE and self.favorite_road:
            goals.append({
                'type': ActionType.BUILD_ROAD,
                'location': self.favorite_road.location,
                'priority': 80
            })
        
        # Consider dev card if score is high enough
        # TODO: Implement dev card scoring
        
        return goals
    
    def _fast_game_strategy(self) -> List[Dict]:
        """
        Fast game strategy - uses simple rules.
        
        Port of dumbFastGameStrategy() from Java SOCRobotDM.
        Simpler and faster than smart strategy.
        """
        goals = []
        
        # Simple priority: Cities > Settlements > Dev Cards > Roads
        
        # 1. Check for city upgrades
        possible_cities = self._get_possible_cities()
        if possible_cities:
            best_city = max(possible_cities, key=lambda c: c.speedup_total)
            goals.append({
                'type': ActionType.BUILD_CITY,
                'location': best_city.location,
                'priority': 100
            })
        
        # 2. Check for new settlements
        possible_settlements = self._get_possible_settlements()
        if possible_settlements:
            # Score settlements by production
            best_settlement = max(possible_settlements, key=lambda s: s.production_value)
            
            # Add necessary roads
            for road_edge in best_settlement.necessary_roads:
                goals.append({
                    'type': ActionType.BUILD_ROAD,
                    'location': road_edge,
                    'priority': 85
                })
            
            goals.append({
                'type': ActionType.BUILD_SETTLEMENT,
                'location': best_settlement.location,
                'priority': 90
            })
        
        # 3. Consider development card
        # Simple rule: buy if we have 5+ VP
        our_vp = get_victory_points(self.game, self.color)
        if our_vp >= 5:
            goals.append({
                'type': ActionType.BUY_DEVELOPMENT_CARD,
                'location': None,
                'priority': 70
            })
        
        return goals
    
    def _evaluate_possible_settlements(self):
        """Evaluate all possible settlement locations."""
        buildable_nodes = find_buildable_node_ids(self.game, self.color)
        
        for node_id in buildable_nodes:
            settlement = PossibleSettlement(node_id)
            
            # Evaluate production value
            score = evaluate_settlement_spot(self.game, node_id, self.color)
            settlement.production_value = score
            settlement.set_score(score)
            
            # TODO: Calculate necessary roads
            settlement.necessary_roads = []
            
            # Add to appropriate list
            # TODO: Determine if threatened
            self.good_settlements.append(settlement)
    
    def _evaluate_possible_cities(self):
        """Evaluate all possible city upgrades."""
        settlements = get_settlements_for_player(self.game, self.color)
        
        for node_id in settlements:
            city = PossibleCity(node_id)
            
            # Cities double production - calculate speedup
            # TODO: Calculate actual speedup value
            city.speedup_total = 50.0  # Placeholder
            city.set_score(city.speedup_total)
    
    def _evaluate_possible_roads(self):
        """Evaluate possible road placements."""
        buildable_edges = find_buildable_edges(self.game, self.color)
        
        for edge in buildable_edges:
            road = PossibleRoad(edge)
            
            # Score based on longest road potential
            current_lr = get_longest_road_length(self.game, self.color)
            # TODO: Estimate if this road extends longest road
            road.longest_road_value = current_lr * 10
            road.set_score(road.longest_road_value)
            
            self.good_roads.append(road)
    
    def _get_possible_settlements(self) -> List[PossibleSettlement]:
        """Get list of possible settlements."""
        if not self.good_settlements:
            self._evaluate_possible_settlements()
        return self.good_settlements
    
    def _get_possible_cities(self) -> List[PossibleCity]:
        """Get list of possible cities."""
        settlements = get_settlements_for_player(self.game, self.color)
        cities = []
        for node_id in settlements:
            city = PossibleCity(node_id)
            city.speedup_total = 50.0  # Simplified scoring
            cities.append(city)
        return cities
    
    def _choose_favorite_settlement(self):
        """Choose the best settlement from good/threatened settlements."""
        all_settlements = self.good_settlements + self.threatened_settlements
        if all_settlements:
            self.favorite_settlement = max(all_settlements, key=lambda s: s.get_score())
    
    def _choose_favorite_city(self):
        """Choose the best city upgrade."""
        cities = self._get_possible_cities()
        if cities:
            self.favorite_city = max(cities, key=lambda c: c.get_score())
    
    def _choose_favorite_road(self):
        """Choose the best road."""
        all_roads = self.good_roads + self.threatened_roads
        if all_roads:
            self.favorite_road = max(all_roads, key=lambda r: r.get_score())
    
    def choose_best_build_action(self, matching_actions: List[Action], goal: Dict) -> Action:
        """
        Choose the best specific location from matching actions.
        
        Args:
            matching_actions: List of actions with the same type
            goal: The build goal we're trying to achieve
            
        Returns:
            The best action to take
        """
        # Simple heuristic: just take the first one
        # TODO: Implement proper scoring
        return matching_actions[0]
    
    def choose_road_building_placement(self, game, playable_actions: List[Action], 
                                      building_plan: List[Dict]) -> Action:
        """
        Choose where to place roads for Road Building dev card.
        
        Args:
            game: Game state
            playable_actions: Available actions
            building_plan: Current building plan
            
        Returns:
            The road action to take
        """
        # If we have roads in the plan, use those
        road_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD]
        
        if not road_actions:
            # No roads available, this shouldn't happen
            return playable_actions[0]
        
        # Check if we have a specific road goal in the plan
        for goal in building_plan:
            if goal['type'] == ActionType.BUILD_ROAD and goal.get('location'):
                matching = next(
                    (a for a in road_actions if a.value == goal['location']),
                    None
                )
                if matching:
                    return matching
        
        # Otherwise, pick the road that extends longest road
        # Simplified: just take first available
        return road_actions[0]





