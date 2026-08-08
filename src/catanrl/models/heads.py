from typing import NamedTuple, Sequence, Tuple

import torch
import torch.nn as nn
from catanatron.gym.envs.action_space import ACTION_TYPES
from catanatron.models.enums import ActionType

from catanrl.utils.catanatron_action_space import get_action_array

from .backbones import CatanGraphBackbone
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


class WDLValueHead(nn.Module):
    """Categorical win/draw/loss head with scalar-Q inference compatibility."""

    VALUE_OUTPUT_GAIN = 1.0

    def __init__(self, input_dim: int):
        super().__init__()
        self.value_head = nn.Linear(input_dim, 3)
        orthogonal_init(self.value_head, gain=self.VALUE_OUTPUT_GAIN)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x)

    @staticmethod
    def values_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities[..., 0] - probabilities[..., 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values_from_logits(self.logits(x))


class CatanGraphValueHead(ValueHead):
    """Deep scalar value integration head over pooled graph features."""

    def __init__(self, backbone: CatanGraphBackbone):
        nn.Module.__init__(self)
        hidden_dim = backbone.head_hidden_dim
        self.pooled_dim = backbone.pooled_dim
        self.value_head = nn.Sequential(
            nn.Linear(self.pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=self.VALUE_OUTPUT_GAIN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x[:, : self.pooled_dim]).squeeze(-1)


class CatanGraphWDLValueHead(WDLValueHead):
    """Canopy-style two-hidden-layer WDL head over pooled graph features."""

    def __init__(self, backbone: CatanGraphBackbone):
        nn.Module.__init__(self)
        hidden_dim = backbone.head_hidden_dim
        self.pooled_dim = backbone.pooled_dim
        self.value_head = nn.Sequential(
            nn.Linear(self.pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=self.VALUE_OUTPUT_GAIN)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x[:, : self.pooled_dim])


class AuxiliaryValueHead(nn.Module):
    """Short-horizon value predictions used only as a training objective."""

    VALUE_OUTPUT_GAIN = 1.0

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Tanh(),
        )
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=self.VALUE_OUTPUT_GAIN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x)


class CatanGraphAuxiliaryValueHead(AuxiliaryValueHead):
    """Canopy-style EMA value heads over the graph backbone's pooled state."""

    def __init__(self, backbone: CatanGraphBackbone, num_heads: int) -> None:
        self.pooled_dim = backbone.pooled_dim
        super().__init__(self.pooled_dim, backbone.head_hidden_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x[:, : self.pooled_dim])


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


class CatanGraphPolicyHidden(NamedTuple):
    settlement: torch.Tensor
    road: torch.Tensor
    city: torch.Tensor
    robber: torch.Tensor
    other: torch.Tensor


class CatanGraphPolicyHead(nn.Module):
    """Topology-aware flat head mapped onto CatanRL's unchanged action array."""

    POLICY_OUTPUT_GAIN = 0.01

    settlement_action_indices: torch.Tensor
    settlement_node_indices: torch.Tensor
    road_action_indices: torch.Tensor
    road_edge_indices: torch.Tensor
    city_action_indices: torch.Tensor
    city_node_indices: torch.Tensor
    robber_action_indices: torch.Tensor
    robber_tile_indices: torch.Tensor
    other_action_indices: torch.Tensor
    edge_endpoints: torch.Tensor

    def __init__(
        self,
        backbone: CatanGraphBackbone,
        num_players: int,
        map_type: str,
    ) -> None:
        super().__init__()
        actions = get_action_array(num_players, map_type)  # type: ignore[arg-type]
        node_to_local = {
            node_id: local_index for local_index, node_id in enumerate(backbone.node_ids)
        }
        tile_to_local = {
            coordinate: local_index
            for local_index, coordinate in enumerate(backbone.tile_coordinates)
        }

        edge_to_local = {
            tuple(sorted((backbone.node_ids[first], backbone.node_ids[second]))): index
            for index, (first, second) in enumerate(backbone.edge_pairs)
        }
        settlement_actions: list[int] = []
        settlement_nodes: list[int] = []
        road_actions: list[int] = []
        road_edges: list[int] = []
        city_actions: list[int] = []
        city_nodes: list[int] = []
        robber_actions: list[int] = []
        robber_tiles: list[int] = []
        other_actions: list[int] = []
        for action_index, (action_type, value) in enumerate(actions):
            if action_type == ActionType.BUILD_SETTLEMENT:
                settlement_actions.append(action_index)
                settlement_nodes.append(node_to_local[int(value)])  # type: ignore[arg-type]
            elif action_type == ActionType.BUILD_ROAD:
                road_actions.append(action_index)
                road_edges.append(edge_to_local[tuple(sorted(value))])  # type: ignore[arg-type]
            elif action_type == ActionType.BUILD_CITY:
                city_actions.append(action_index)
                city_nodes.append(node_to_local[int(value)])  # type: ignore[arg-type]
            elif action_type == ActionType.MOVE_ROBBER:
                coordinate, _ = value  # type: ignore[misc]
                robber_actions.append(action_index)
                robber_tiles.append(tile_to_local[coordinate])
            else:
                other_actions.append(action_index)

        classified = (
            len(settlement_actions)
            + len(road_actions)
            + len(city_actions)
            + len(robber_actions)
            + len(other_actions)
        )
        if classified != len(actions):
            raise RuntimeError("Catan graph policy mapping did not cover the flat action space")

        self.input_dim = backbone.output_dim
        self.action_space_size = len(actions)
        self.pooled_dim = backbone.pooled_dim
        self.hidden_dim = backbone.hidden_dim
        self.num_nodes = backbone.num_nodes
        self.num_tiles = backbone.num_tiles
        self.num_edges = backbone.num_edges
        hidden_dim = backbone.head_hidden_dim
        self.head_hidden_dim = hidden_dim

        self.settlement_hidden = nn.Sequential(
            nn.Linear(backbone.hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.settlement_output = nn.Linear(hidden_dim, 1)
        self.road_hidden = nn.Sequential(
            nn.Linear(2 * backbone.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.road_output = nn.Linear(hidden_dim, 1)
        self.city_hidden = nn.Sequential(
            nn.Linear(backbone.hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.city_output = nn.Linear(hidden_dim, 1)
        self.robber_hidden = nn.Sequential(
            nn.Linear(backbone.hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.robber_output = nn.Linear(hidden_dim, 1)
        self.other_hidden = nn.Sequential(
            nn.Linear(backbone.pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.other_output = nn.Linear(hidden_dim, len(other_actions))

        self.register_buffer(
            "settlement_action_indices", torch.tensor(settlement_actions, dtype=torch.long)
        )
        self.register_buffer(
            "settlement_node_indices", torch.tensor(settlement_nodes, dtype=torch.long)
        )
        self.register_buffer("road_action_indices", torch.tensor(road_actions, dtype=torch.long))
        self.register_buffer("road_edge_indices", torch.tensor(road_edges, dtype=torch.long))
        self.register_buffer("city_action_indices", torch.tensor(city_actions, dtype=torch.long))
        self.register_buffer("city_node_indices", torch.tensor(city_nodes, dtype=torch.long))
        self.register_buffer(
            "robber_action_indices", torch.tensor(robber_actions, dtype=torch.long)
        )
        self.register_buffer("robber_tile_indices", torch.tensor(robber_tiles, dtype=torch.long))
        self.register_buffer("other_action_indices", torch.tensor(other_actions, dtype=torch.long))
        self.register_buffer(
            "edge_endpoints",
            torch.tensor(backbone.edge_pairs, dtype=torch.long),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for hidden, output in (
            (self.settlement_hidden, self.settlement_output),
            (self.road_hidden, self.road_output),
            (self.city_hidden, self.city_output),
            (self.robber_hidden, self.robber_output),
            (self.other_hidden, self.other_output),
        ):
            for module in hidden:
                if not isinstance(module, nn.Linear):
                    continue
                orthogonal_init(module)
            orthogonal_init(output, gain=self.POLICY_OUTPUT_GAIN)

    def hidden_features(self, features: torch.Tensor) -> CatanGraphPolicyHidden:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected Catan graph features [batch, {self.input_dim}], "
                f"got {tuple(features.shape)}"
            )
        batch_size = features.shape[0]
        node_start = self.pooled_dim
        tile_start = node_start + self.num_nodes * self.hidden_dim
        pooled = features[:, : self.pooled_dim]
        nodes = features[:, node_start:tile_start].reshape(
            batch_size,
            self.num_nodes,
            self.hidden_dim,
        )
        tiles = features[:, tile_start:].reshape(
            batch_size,
            self.num_tiles,
            self.hidden_dim,
        )

        endpoints = self.edge_endpoints[self.road_edge_indices]
        road_features = torch.cat(
            (nodes[:, endpoints[:, 0], :], nodes[:, endpoints[:, 1], :]),
            dim=-1,
        )
        return CatanGraphPolicyHidden(
            settlement=self.settlement_hidden(nodes)[:, self.settlement_node_indices],
            road=self.road_hidden(road_features),
            city=self.city_hidden(nodes)[:, self.city_node_indices],
            robber=self.robber_hidden(tiles)[:, self.robber_tile_indices],
            other=self.other_hidden(pooled),
        )

    def _scatter_action_logits(
        self,
        settlement_logits: torch.Tensor,
        road_logits: torch.Tensor,
        city_logits: torch.Tensor,
        robber_logits: torch.Tensor,
        other_logits: torch.Tensor,
    ) -> torch.Tensor:
        logits = settlement_logits.new_zeros((settlement_logits.shape[0], self.action_space_size))
        logits = logits.index_copy(1, self.settlement_action_indices, settlement_logits)
        logits = logits.index_copy(1, self.road_action_indices, road_logits)
        logits = logits.index_copy(1, self.city_action_indices, city_logits)
        logits = logits.index_copy(1, self.robber_action_indices, robber_logits)
        return logits.index_copy(1, self.other_action_indices, other_logits)

    def logits_from_hidden(self, hidden: CatanGraphPolicyHidden) -> torch.Tensor:
        return self._scatter_action_logits(
            self.settlement_output(hidden.settlement).squeeze(-1),
            self.road_output(hidden.road).squeeze(-1),
            self.city_output(hidden.city).squeeze(-1),
            self.robber_output(hidden.robber).squeeze(-1),
            self.other_output(hidden.other),
        )

    def forward_with_hidden(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, CatanGraphPolicyHidden]:
        hidden = self.hidden_features(features)
        return self.logits_from_hidden(hidden), hidden

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_hidden(features)
        return logits


class CatanGraphSoftPolicyHead(nn.Module):
    """Separate Nexus-v3 final projections over shared hard-policy features."""

    POLICY_OUTPUT_GAIN = CatanGraphPolicyHead.POLICY_OUTPUT_GAIN

    def __init__(self, policy_head: CatanGraphPolicyHead) -> None:
        super().__init__()
        hidden_dim = policy_head.head_hidden_dim
        self.settlement_output = nn.Linear(hidden_dim, 1)
        self.road_output = nn.Linear(hidden_dim, 1)
        self.city_output = nn.Linear(hidden_dim, 1)
        self.robber_output = nn.Linear(hidden_dim, 1)
        self.other_output = nn.Linear(hidden_dim, policy_head.other_action_indices.numel())
        for output in (
            self.settlement_output,
            self.road_output,
            self.city_output,
            self.robber_output,
            self.other_output,
        ):
            orthogonal_init(output, gain=self.POLICY_OUTPUT_GAIN)

    def forward(
        self,
        hidden: CatanGraphPolicyHidden,
        policy_head: CatanGraphPolicyHead,
    ) -> torch.Tensor:
        return policy_head._scatter_action_logits(
            self.settlement_output(hidden.settlement).squeeze(-1),
            self.road_output(hidden.road).squeeze(-1),
            self.city_output(hidden.city).squeeze(-1),
            self.robber_output(hidden.robber).squeeze(-1),
            self.other_output(hidden.other),
        )


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
