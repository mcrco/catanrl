"""
Utility functions for accessing game state information.

These functions provide a convenient API for extracting player and game information
from Catanatron's state object.
"""

from typing import Dict, List
from catanatron.models.player import Color
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS, SETTLEMENT, CITY, ROAD
from catanatron.state_functions import player_key, player_num_resource_cards, player_num_dev_cards


def get_player_resources(game, color: Color) -> Dict[str, int]:
    """
    Get all resources for a player.
    
    Args:
        game: The Game object
        color: Player color
        
    Returns:
        Dictionary mapping resource name to count
    """
    state = game.state
    key = player_key(state, color)
    
    resources = {}
    for resource in RESOURCES:
        resources[resource] = state.player_state[f"{key}_{resource}_IN_HAND"]
    
    return resources


def get_player_dev_cards(game, color: Color) -> Dict[str, int]:
    """
    Get development cards for a player.
    
    Args:
        game: The Game object
        color: Player color
        
    Returns:
        Dictionary mapping dev card name to count
    """
    state = game.state
    key = player_key(state, color)
    
    dev_cards = {}
    for card in DEVELOPMENT_CARDS:
        dev_cards[card] = state.player_state[f"{key}_{card}_IN_HAND"]
    
    return dev_cards


def get_player_production(game, color: Color) -> Dict[str, float]:
    """
    Estimate production (expected resources per turn) for a player.
    
    Args:
        game: The Game object
        color: Player color
        
    Returns:
        Dictionary mapping resource to expected production per turn
    """
    state = game.state
    board = state.board
    
    production = {resource: 0.0 for resource in RESOURCES}
    
    # Get all settlements and cities for this player
    settlements = state.buildings_by_color[color].get(SETTLEMENT, [])
    cities = state.buildings_by_color[color].get(CITY, [])
    
    # Calculate production from settlements (1x)
    for node_id in settlements:
        tiles = board.map.adjacent_tiles.get(node_id, [])
        for tile in tiles:
            if tile in board.map.tiles:
                placed_tile = board.map.tiles[tile]
                if hasattr(placed_tile, 'resource') and hasattr(placed_tile, 'number'):
                    resource = placed_tile.resource
                    number = placed_tile.number
                    if resource and resource in RESOURCES:
                        # Probability of rolling this number (out of 36 combinations)
                        prob = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}.get(number, 0) / 36.0
                        production[resource] += prob
    
    # Calculate production from cities (2x)
    for node_id in cities:
        tiles = board.map.adjacent_tiles.get(node_id, [])
        for tile in tiles:
            if tile in board.map.tiles:
                placed_tile = board.map.tiles[tile]
                if hasattr(placed_tile, 'resource') and hasattr(placed_tile, 'number'):
                    resource = placed_tile.resource
                    number = placed_tile.number
                    if resource and resource in RESOURCES:
                        prob = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}.get(number, 0) / 36.0
                        production[resource] += 2 * prob
    
    return production


def get_settlements_for_player(game, color: Color) -> List[int]:
    """Get list of settlement node IDs for a player."""
    return game.state.buildings_by_color[color].get(SETTLEMENT, [])


def get_cities_for_player(game, color: Color) -> List[int]:
    """Get list of city node IDs for a player."""
    return game.state.buildings_by_color[color].get(CITY, [])


def get_roads_for_player(game, color: Color) -> List[tuple]:
    """Get list of road edge IDs for a player."""
    return game.state.buildings_by_color[color].get(ROAD, [])


def get_victory_points(game, color: Color) -> int:
    """
    Get public victory points for a player.
    
    Args:
        game: The Game object
        color: Player color
        
    Returns:
        Number of public victory points
    """
    state = game.state
    key = player_key(state, color)
    return state.player_state[f"{key}_VICTORY_POINTS"]


def get_actual_victory_points(game, color: Color) -> int:
    """
    Get actual victory points (including hidden VP cards) for a player.
    
    Args:
        game: The Game object
        color: Player color
        
    Returns:
        Number of actual victory points
    """
    state = game.state
    key = player_key(state, color)
    return state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]


def has_rolled_dice(game, color: Color) -> bool:
    """Check if player has rolled dice this turn."""
    state = game.state
    key = player_key(state, color)
    return state.player_state[f"{key}_HAS_ROLLED"]


def has_played_dev_card(game, color: Color) -> bool:
    """Check if player has played a dev card this turn."""
    state = game.state
    key = player_key(state, color)
    return state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]


def get_longest_road_length(game, color: Color) -> int:
    """Get longest road length for a player."""
    state = game.state
    key = player_key(state, color)
    return state.player_state[f"{key}_LONGEST_ROAD_LENGTH"]


def has_longest_road(game, color: Color) -> bool:
    """Check if player has the longest road."""
    state = game.state
    key = player_key(state, color)
    return state.player_state[f"{key}_HAS_ROAD"]


def has_largest_army(game, color: Color) -> bool:
    """Check if player has the largest army."""
    state = game.state
    key = player_key(state, color)
    return state.player_state[f"{key}_HAS_ARMY"]


def get_num_knights_played(game, color: Color) -> int:
    """Get number of knights played by a player."""
    state = game.state
    key = player_key(state, color)
    from catanatron.models.enums import KNIGHT
    return state.player_state[f"{key}_PLAYED_{KNIGHT}"]





