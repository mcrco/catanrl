import torch
import torch.nn as nn
from typing import Tuple

from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY
from catanatron.models.enums import ActionType


class ValueHead(nn.Module):
    """Value head for value network."""

    def __init__(self, input_dim: int, output_sigmoid: bool = False):
        super().__init__()
        if output_sigmoid:
            self.value_head = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        else:
            self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.value_head(x).squeeze(-1)


class FlatPolicyHead(nn.Module):
    """Simple linear head for policy network that outputs flat action logits."""

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.policy_head = nn.Linear(input_dim, num_actions)

    def forward(self, x):
        return self.policy_head(x)


class HierarchicalPolicyHead(nn.Module):
    """
    Hierarchical policy head that predicts action type first, then action parameters.

    Action types in Catanatron:
    0. ROLL (no params)
    1. MOVE_ROBBER (tiles: 19 for BASE, 7 for MINI)
    2. DISCARD (5 resources)
    3. BUILD_ROAD (edges: 72 for BASE, 30 for MINI)
    4. BUILD_SETTLEMENT (nodes: 54 for BASE, 24 for MINI)
    5. BUILD_CITY (nodes: 54 for BASE, 24 for MINI)
    6. BUY_DEVELOPMENT_CARD (no params)
    7. PLAY_KNIGHT_CARD (no params)
    8. PLAY_YEAR_OF_PLENTY (20 resource combinations)
    9. PLAY_ROAD_BUILDING (no params)
    10. PLAY_MONOPOLY (5 resources)
    11. MARITIME_TRADE (60 trade combinations)
    12. END_TURN (no params)
    """

    # Action type indices (order matches ACTIONS_ARRAY in catanatron_env.py)
    ROLL = 0
    MOVE_ROBBER = 1
    DISCARD = 2
    BUILD_ROAD = 3
    BUILD_SETTLEMENT = 4
    BUILD_CITY = 5
    BUY_DEVELOPMENT_CARD = 6
    PLAY_KNIGHT_CARD = 7
    PLAY_YEAR_OF_PLENTY = 8
    PLAY_ROAD_BUILDING = 9
    PLAY_MONOPOLY = 10
    MARITIME_TRADE = 11
    END_TURN = 12

    NUM_ACTION_TYPES = 13
    NUM_RESOURCES = 5

    def __init__(self, input_dim: int):
        super().__init__()

        # Action type head (13 action types)
        self.action_type_head = nn.Linear(input_dim, self.NUM_ACTION_TYPES)

        self._analyze_action_space()

        # Parameter heads for each action type that needs them
        self.tile_head = nn.Linear(input_dim, self.num_tiles)  # MOVE_ROBBER
        self.edge_head = nn.Linear(input_dim, self.num_edges)  # BUILD_ROAD
        self.settlement_node_head = nn.Linear(input_dim, self.num_nodes)  # BUILD_SETTLEMENT
        self.city_node_head = nn.Linear(input_dim, self.num_nodes)  # BUILD_CITY

        # These are constant across map types
        self.year_of_plenty_head = nn.Linear(input_dim, self.num_year_of_plenty_combos)
        self.monopoly_head = nn.Linear(input_dim, self.NUM_RESOURCES)
        self.maritime_trade_head = nn.Linear(input_dim, self.num_maritime_trades)
        self.discard_head = nn.Linear(input_dim, self.num_discard_combinations)

        # Build action space mappings
        self._build_action_mappings()

    def _analyze_action_space(self):
        # Count unique parameter values for each action type
        tiles = set()
        edges = set()
        nodes_settlement = set()
        nodes_city = set()
        year_of_plenty_combos = set()
        monopoly_resources = set()
        maritime_trades = set()
        discard_combinations = set()

        for action_type, value in ACTIONS_ARRAY:
            if action_type == ActionType.MOVE_ROBBER:
                tiles.add(value)
            elif action_type == ActionType.BUILD_ROAD:
                edges.add(value)
            elif action_type == ActionType.BUILD_SETTLEMENT:
                nodes_settlement.add(value)
            elif action_type == ActionType.BUILD_CITY:
                nodes_city.add(value)
            elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
                year_of_plenty_combos.add(value)
            elif action_type == ActionType.PLAY_MONOPOLY:
                monopoly_resources.add(value)
            elif action_type == ActionType.MARITIME_TRADE:
                maritime_trades.add(value)
            elif action_type == ActionType.DISCARD:
                discard_combinations.add(value)

        # Store dimensions
        self.num_tiles = len(tiles)
        self.num_edges = len(edges)
        self.num_nodes = max(len(nodes_settlement), len(nodes_city))  # Should be the same
        self.num_year_of_plenty_combos = len(year_of_plenty_combos)
        self.num_maritime_trades = len(maritime_trades)
        self.num_discard_combinations = len(discard_combinations)

    def _build_action_mappings(self):
        """Build mappings between flat action indices and (action_type, action_value)."""

        self.actions_array = ACTIONS_ARRAY
        self.action_space_size = len(ACTIONS_ARRAY)

        # Map ActionType enum to our integer indices
        self.action_type_to_idx = {
            ActionType.ROLL: self.ROLL,
            ActionType.MOVE_ROBBER: self.MOVE_ROBBER,
            ActionType.DISCARD: self.DISCARD,
            ActionType.BUILD_ROAD: self.BUILD_ROAD,
            ActionType.BUILD_SETTLEMENT: self.BUILD_SETTLEMENT,
            ActionType.BUILD_CITY: self.BUILD_CITY,
            ActionType.BUY_DEVELOPMENT_CARD: self.BUY_DEVELOPMENT_CARD,
            ActionType.PLAY_KNIGHT_CARD: self.PLAY_KNIGHT_CARD,
            ActionType.PLAY_YEAR_OF_PLENTY: self.PLAY_YEAR_OF_PLENTY,
            ActionType.PLAY_ROAD_BUILDING: self.PLAY_ROAD_BUILDING,
            ActionType.PLAY_MONOPOLY: self.PLAY_MONOPOLY,
            ActionType.MARITIME_TRADE: self.MARITIME_TRADE,
            ActionType.END_TURN: self.END_TURN,
        }

        # Build index to (action_type_idx, param_idx) mapping
        self.flat_to_hierarchical = {}
        self.hierarchical_to_flat = {}

        # Build per-action-type parameter lists
        self.action_type_params = {i: [] for i in range(self.NUM_ACTION_TYPES)}

        for flat_idx, (action_type, value) in enumerate(ACTIONS_ARRAY):
            action_type_idx = self.action_type_to_idx[action_type]

            # Get parameter index within action type
            param_idx = len(self.action_type_params[action_type_idx])
            self.action_type_params[action_type_idx].append(value)

            # Store bidirectional mappings
            self.flat_to_hierarchical[flat_idx] = (action_type_idx, param_idx)
            self.hierarchical_to_flat[(action_type_idx, param_idx)] = flat_idx

    def forward(self, features):
        """
        Forward pass returns action type logits and parameter logits.

        Args:
            features: Output of the shared backbone.

        Returns:
            action_type_logits: [batch_size, 13]
            param_logits: Dict[action_type_idx, logits]
        """

        # Predict action type
        action_type_logits = self.action_type_head(features)

        # Predict parameters for all action types
        param_logits = {
            self.MOVE_ROBBER: self.tile_head(features),
            self.BUILD_ROAD: self.edge_head(features),
            self.BUILD_SETTLEMENT: self.settlement_node_head(features),
            self.BUILD_CITY: self.city_node_head(features),
            self.PLAY_YEAR_OF_PLENTY: self.year_of_plenty_head(features),
            self.PLAY_MONOPOLY: self.monopoly_head(features),
            self.MARITIME_TRADE: self.maritime_trade_head(features),
            self.DISCARD: self.discard_head(features),
        }

        return action_type_logits, param_logits

    def get_flat_action_logits(self, action_type_logits, param_logits):
        """
        Convert hierarchical logits to flat action space logits.

        Args:
            action_type_logits: [batch_size, 13]
            param_logits: Dict of parameter logits

        Returns:
            flat_logits: [batch_size, action_space_size]
        """
        batch_size = action_type_logits.shape[0]
        device = action_type_logits.device
        flat_logits = torch.zeros(batch_size, self.action_space_size, device=device)

        # For each flat action, compute its log probability as:
        # log P(flat_action) = log P(action_type) + log P(param | action_type)
        action_type_log_probs = torch.log_softmax(action_type_logits, dim=-1)

        for flat_idx in range(self.action_space_size):
            action_type_idx, param_idx = self.flat_to_hierarchical[flat_idx]

            # Start with action type log prob
            logit = action_type_log_probs[:, action_type_idx]

            # Add parameter log prob if this action type has parameters
            if action_type_idx in param_logits:
                param_log_probs = torch.log_softmax(param_logits[action_type_idx], dim=-1)
                logit = logit + param_log_probs[:, param_idx]

            flat_logits[:, flat_idx] = logit

        return flat_logits

    def flat_to_hierarchical_action(self, flat_action_idx: int) -> Tuple[int, int]:
        """Convert flat action index to (action_type_idx, param_idx)."""
        return self.flat_to_hierarchical[flat_action_idx]

    def hierarchical_to_flat_action(self, action_type_idx: int, param_idx: int) -> int:
        """Convert (action_type_idx, param_idx) to flat action index."""
        return self.hierarchical_to_flat[(action_type_idx, param_idx)]
