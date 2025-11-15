"""
Robber Strategy - Deciding where to place the robber.

Port of RobberStrategy from JSettlers2.
"""

from typing import List, Dict
import random
from catanatron.models.player import Color
from catanatron.models.enums import Action, ActionType

from jsettlers2_bot_py.utils.game_utils import get_victory_points


class RobberStrategy:
    """
    Strategy for robber placement.
    
    Port of RobberStrategy from JSettlers2.
    """
    
    def __init__(self, params):
        self.params = params
    
    def choose_robber_placement(self, game, playable_actions: List[Action],
                               player_trackers: Dict) -> Action:
        """
        Choose where to move the robber.
        
        Strategy depends on parameters:
        - SMART: Block leading player, target resource-rich tiles
        - DEFENSIVE: Avoid own tiles, minimize harm
        - AGGRESSIVE: Block leading player more aggressively
        
        Args:
            game: Game state
            playable_actions: List of MOVE_ROBBER actions
            player_trackers: Player trackers for opponent info
            
        Returns:
            The chosen robber action
        """
        robber_actions = [a for a in playable_actions if a.action_type == ActionType.MOVE_ROBBER]
        
        if not robber_actions:
            return playable_actions[0]
        
        strategy_type = self.params.robber_strategy_type
        
        if strategy_type == "DEFENSIVE":
            return self._defensive_placement(game, robber_actions)
        elif strategy_type == "AGGRESSIVE":
            return self._aggressive_placement(game, robber_actions, player_trackers)
        else:  # SMART
            return self._smart_placement(game, robber_actions, player_trackers)
    
    def _defensive_placement(self, game, robber_actions: List[Action]) -> Action:
        """
        Defensive: Just move robber away from our tiles.
        """
        # TODO: Filter out tiles adjacent to our settlements
        # For now, just pick randomly
        return random.choice(robber_actions)
    
    def _aggressive_placement(self, game, robber_actions: List[Action],
                            player_trackers: Dict) -> Action:
        """
        Aggressive: Block the leading player.
        """
        # Find leader
        leader_color = self._find_leader(game)
        
        if leader_color and leader_color != game.state.colors[game.state.current_turn_index]:
            # Find tiles where leader has settlements
            # TODO: Implement proper tile filtering
            pass
        
        # Fallback: smart placement
        return self._smart_placement(game, robber_actions, player_trackers)
    
    def _smart_placement(self, game, robber_actions: List[Action],
                        player_trackers: Dict) -> Action:
        """
        Smart: Balance between blocking leader and getting resources.
        """
        leader_color = self._find_leader(game)
        
        scored_actions = []
        for action in robber_actions:
            tile_coord = action.value[0]  # First element is tile coordinate
            score = 0.0
            
            # TODO: Score based on:
            # - Tile production value (dice number)
            # - How many opponent settlements are on this tile
            # - Whether leader has settlement here
            # - Avoid our own tiles
            
            # Placeholder scoring
            score = random.random() * 100
            
            scored_actions.append((score, action))
        
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return scored_actions[0][1]
    
    def _find_leader(self, game) -> Color:
        """Find the player with most victory points."""
        max_vp = -1
        leader = None
        
        for color in game.state.colors:
            vp = get_victory_points(game, color)
            if vp > max_vp:
                max_vp = vp
                leader = color
        
        return leader





