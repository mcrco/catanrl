"""
Utility functions for board analysis and evaluation.

Provides functions for evaluating settlement locations, finding buildable positions, etc.
"""

from typing import List, Tuple, Dict
from catanatron.models.player import Color
from catanatron.models.enums import RESOURCES


def evaluate_settlement_spot(game, node_id: int, color: Color) -> float:
    """
    Evaluate how good a settlement spot is.
    
    Considers:
    - Resource production (based on dice numbers)
    - Resource diversity
    - Port access
    - Expansion potential
    
    Args:
        game: The Game object
        node_id: The node to evaluate
        color: Player color
        
    Returns:
        Score for the settlement spot (higher is better)
    """
    board = game.state.board
    
    score = 0.0
    resource_counts = {r: 0 for r in RESOURCES}
    
    # Get adjacent tiles
    adjacent_tiles = board.map.adjacent_tiles.get(node_id, [])
    
    for tile_coord in adjacent_tiles:
        if tile_coord not in board.map.tiles:
            continue
        
        tile = board.map.tiles[tile_coord]
        
        # Check if tile has resources
        if hasattr(tile, 'resource') and hasattr(tile, 'number'):
            resource = tile.resource
            number = tile.number
            
            if resource and resource in RESOURCES and number:
                # Dice probability (out of 36)
                prob = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}.get(number, 0)
                
                # Add to score (6 and 8 are best)
                score += prob * 10
                resource_counts[resource] += prob
    
    # Bonus for resource diversity
    num_different_resources = sum(1 for count in resource_counts.values() if count > 0)
    score += num_different_resources * 20
    
    # Check for port access
    # TODO: Implement port checking when we understand Catanatron's port structure
    
    # Penalty if on the robber (for initial placement)
    # TODO: Check robber location
    
    return score


def find_buildable_node_ids(game, color: Color) -> List[int]:
    """
    Find all node IDs where the player can build settlements.
    
    Args:
        game: The Game object
        color: Player color
        
    Returns:
        List of buildable node IDs
    """
    return list(game.state.board.buildable_node_ids(color))


def find_buildable_edges(game, color: Color) -> List[Tuple[int, int]]:
    """
    Find all edges where the player can build roads.
    
    Args:
        game: The Game object
        color: Player color
        
    Returns:
        List of buildable edges
    """
    return list(game.state.board.buildable_edges(color))


def find_path_between_nodes(game, color: Color, start_node: int, end_node: int) -> List[Tuple[int, int]]:
    """
    Find a path of roads needed to connect two nodes.
    
    Args:
        game: The Game object
        color: Player color
        start_node: Starting node ID
        end_node: Target node ID
        
    Returns:
        List of edges forming the path, or empty list if no path found
    """
    # Simple BFS to find path
    from collections import deque
    
    board = game.state.board
    visited = set()
    queue = deque([(start_node, [])])
    
    while queue:
        current_node, path = queue.popleft()
        
        if current_node == end_node:
            return path
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        # Get neighboring nodes
        # This is a simplified version - you may need to adapt based on Catanatron's board structure
        for neighbor in board.map.adjacent_nodes.get(current_node, []):
            edge = tuple(sorted([current_node, neighbor]))
            new_path = path + [edge]
            queue.append((neighbor, new_path))
    
    return []


def get_adjacent_nodes_to_tiles(game, tile_coords: List[Tuple]) -> List[int]:
    """
    Get all node IDs adjacent to given tiles.
    
    Args:
        game: The Game object
        tile_coords: List of tile coordinates
        
    Returns:
        List of adjacent node IDs
    """
    board = game.state.board
    nodes = set()
    
    for tile_coord in tile_coords:
        # Find nodes touching this tile
        for node_id, adjacent_tiles in board.map.adjacent_tiles.items():
            if tile_coord in adjacent_tiles:
                nodes.add(node_id)
    
    return list(nodes)


def estimate_expansion_potential(game, node_id: int, color: Color) -> int:
    """
    Estimate how many additional settlements could be reached from this node.
    
    Args:
        game: The Game object
        node_id: Node to evaluate
        color: Player color
        
    Returns:
        Number of potential expansion spots
    """
    # This is a simplified heuristic
    # Count buildable nodes within 2-3 road connections
    
    count = 0
    buildable = find_buildable_node_ids(game, color)
    
    # For now, just return count of nearby buildable nodes
    # TODO: Implement proper distance calculation
    
    return len(buildable)





