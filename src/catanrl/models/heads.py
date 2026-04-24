import torch
import torch.nn as nn
from typing import Sequence, Tuple

from catanatron.gym.envs.action_space import ACTION_TYPES
from catanatron.models.enums import ActionType

from .utils import orthogonal_init


class ValueHead(nn.Module):
    """Value head for value network."""

    # Smaller gain for value output layer (common PPO practice)
    VALUE_OUTPUT_GAIN = 1.0

    def __init__(self, input_dim: int, output_sigmoid: bool = False):
        super().__init__()
        if output_sigmoid:
            self.value_head = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        else:
            self.value_head = nn.Linear(input_dim, 1)
        # For PPO
        self._init_weights()

    def _init_weights(self) -> None:
        if isinstance(self.value_head, nn.Sequential):
            # Only init the Linear layer (first element)
            orthogonal_init(self.value_head[0], gain=self.VALUE_OUTPUT_GAIN)
        else:
            orthogonal_init(self.value_head, gain=self.VALUE_OUTPUT_GAIN)

    def forward(self, x):
        return self.value_head(x).squeeze(-1)


class FlatPolicyHead(nn.Module):
    """Simple linear head for policy network that outputs flat action logits."""

    # Small gain for policy output layer to start with near-uniform action probs
    POLICY_OUTPUT_GAIN = 0.01

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.policy_head = nn.Linear(input_dim, num_actions)
        # For PPO
        self._init_weights()

    def _init_weights(self) -> None:
        orthogonal_init(self.policy_head, gain=self.POLICY_OUTPUT_GAIN)

    def forward(self, x):
        return self.policy_head(x)


class HierarchicalPolicyHead(nn.Module):
    """
    Hierarchical policy head that predicts action type first, then action parameters.

    Action-type indices are derived from Catanatron's live ACTION_TYPES ordering
    so this head stays aligned if upstream reorders or expands the action space.
    """

    ACTION_TYPE_ATTRS = {
        ActionType.ROLL: "ROLL",
        ActionType.MOVE_ROBBER: "MOVE_ROBBER",
        ActionType.DISCARD_RESOURCE: "DISCARD",
        ActionType.BUILD_ROAD: "BUILD_ROAD",
        ActionType.BUILD_SETTLEMENT: "BUILD_SETTLEMENT",
        ActionType.BUILD_CITY: "BUILD_CITY",
        ActionType.BUY_DEVELOPMENT_CARD: "BUY_DEVELOPMENT_CARD",
        ActionType.PLAY_KNIGHT_CARD: "PLAY_KNIGHT_CARD",
        ActionType.PLAY_YEAR_OF_PLENTY: "PLAY_YEAR_OF_PLENTY",
        ActionType.PLAY_ROAD_BUILDING: "PLAY_ROAD_BUILDING",
        ActionType.PLAY_MONOPOLY: "PLAY_MONOPOLY",
        ActionType.MARITIME_TRADE: "MARITIME_TRADE",
        ActionType.END_TURN: "END_TURN",
    }

    NUM_RESOURCES = 5

    # Small gain for policy output layers to start with near-uniform action probs
    POLICY_OUTPUT_GAIN = 0.01

    def __init__(
        self,
        input_dim: int,
        actions_array: Sequence[tuple[ActionType, object | None]],
    ):
        super().__init__()
        self.actions_array = tuple(actions_array)
        self.action_types = tuple(ACTION_TYPES)
        self.NUM_ACTION_TYPES = len(self.action_types)
        self.action_type_to_idx = {
            action_type: idx for idx, action_type in enumerate(self.action_types)
        }
        for action_type, attr_name in self.ACTION_TYPE_ATTRS.items():
            if action_type not in self.action_type_to_idx:
                raise ValueError(f"Missing required action type in ACTION_TYPES: {action_type}")
            setattr(self, attr_name, self.action_type_to_idx[action_type])

        # Action type head aligned with the upstream action-type ordering.
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

        # For PPO
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=self.POLICY_OUTPUT_GAIN)

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

        for action_type, value in self.actions_array:
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
            elif action_type == ActionType.DISCARD_RESOURCE:
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

        self.action_space_size = len(self.actions_array)

        # Build index to (action_type_idx, param_idx) mapping
        self.flat_to_hierarchical = {}
        self.hierarchical_to_flat = {}

        # Build per-action-type parameter lists
        self.action_type_params = {i: [] for i in range(self.NUM_ACTION_TYPES)}

        for flat_idx, (action_type, value) in enumerate(self.actions_array):
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
            action_type_logits: [batch_size, NUM_ACTION_TYPES]
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
            action_type_logits: [batch_size, NUM_ACTION_TYPES]
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
