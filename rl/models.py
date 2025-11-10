import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import numpy as np


class PolicyValueNetwork(nn.Module):
    """MLP with shared backbone and separate heads for policy and value prediction."""
    
    def __init__(self, input_dim: int, num_actions: int, hidden_dims: List[int] = [512, 512]):
        super().__init__()
        
        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Policy head (action prediction)
        self.policy_head = nn.Linear(prev_dim, num_actions)
        
        # Value head (return prediction)
        self.value_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        """Returns both policy logits and value prediction."""
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


class HierarchicalPolicyValueNetwork(nn.Module):
    """
    Hierarchical policy-value network that predicts action type first, 
    then action parameters, plus value estimation.
    
    Supports both BASE and MINI map types by dynamically determining
    parameter space sizes from ACTIONS_ARRAY.
    
    Action types in Catanatron:
    0. ROLL (no params)
    1. MOVE_ROBBER (tiles: 19 for BASE, 7 for MINI)
    2. DISCARD (no params)
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
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 512]):
        super().__init__()
        
        # Analyze ACTIONS_ARRAY first to determine parameter space sizes
        self._analyze_action_space()
        
        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Action type head (13 action types)
        self.action_type_head = nn.Linear(prev_dim, self.NUM_ACTION_TYPES)
        
        # Parameter heads for each action type that needs them
        # Sizes are determined dynamically from ACTIONS_ARRAY
        self.tile_head = nn.Linear(prev_dim, self.num_tiles)  # MOVE_ROBBER
        self.edge_head = nn.Linear(prev_dim, self.num_edges)  # BUILD_ROAD
        self.settlement_node_head = nn.Linear(prev_dim, self.num_nodes)  # BUILD_SETTLEMENT
        self.city_node_head = nn.Linear(prev_dim, self.num_nodes)  # BUILD_CITY
        
        # These are constant across map types
        self.year_of_plenty_head = nn.Linear(prev_dim, self.num_year_of_plenty_combos)
        self.monopoly_head = nn.Linear(prev_dim, self.NUM_RESOURCES)
        self.maritime_trade_head = nn.Linear(prev_dim, self.num_maritime_trades)
        
        # Value head (return prediction)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Build action space mappings
        self._build_action_mappings()
    
    def _analyze_action_space(self):
        """Analyze ACTIONS_ARRAY to determine parameter space sizes for current map type."""
        from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY
        from catanatron.models.enums import ActionType
        
        # Count unique parameter values for each action type
        tiles = set()
        edges = set()
        nodes_settlement = set()
        nodes_city = set()
        year_of_plenty_combos = set()
        monopoly_resources = set()
        maritime_trades = set()
        
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
        
        # Store dimensions
        self.num_tiles = len(tiles)
        self.num_edges = len(edges)
        self.num_nodes = max(len(nodes_settlement), len(nodes_city))  # Should be the same
        self.num_year_of_plenty_combos = len(year_of_plenty_combos)
        self.num_maritime_trades = len(maritime_trades)
        
        # Sanity check
        assert len(nodes_settlement) == len(nodes_city), \
            f"Settlement and city nodes should be the same, got {len(nodes_settlement)} vs {len(nodes_city)}"
        
        # Store for reference
        self.map_type = 'BASE' if self.num_nodes == 54 else 'MINI' if self.num_nodes == 24 else 'UNKNOWN'
        
        print(f"Detected map type: {self.map_type}")
        print(f"  - Tiles: {self.num_tiles}")
        print(f"  - Edges: {self.num_edges}")
        print(f"  - Nodes: {self.num_nodes}")
        print(f"  - Year of Plenty combos: {self.num_year_of_plenty_combos}")
        print(f"  - Maritime trades: {self.num_maritime_trades}")
    
    def _build_action_mappings(self):
        """Build mappings between flat action indices and (action_type, action_value)."""
        from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY
        from catanatron.models.enums import ActionType
        
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
    
    def forward(self, x):
        """
        Forward pass returns action type logits, parameter logits for all heads, and value.
        
        Returns:
            action_type_logits: [batch_size, 13]
            param_logits: Dict with keys matching action types
            value: [batch_size]
        """
        features = self.backbone(x)
        
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
        }
        
        # Predict value
        value = self.value_head(features).squeeze(-1)
        
        return action_type_logits, param_logits, value
    
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
    
    def compute_loss(self, 
                     action_type_logits, 
                     param_logits, 
                     value_pred,
                     target_actions, 
                     target_values,
                     action_type_weight: float = 1.0,
                     param_weight: float = 1.0,
                     value_weight: float = 1.0):
        """
        Compute hierarchical loss.
        
        Args:
            action_type_logits: [batch_size, 13]
            param_logits: Dict of parameter logits
            value_pred: [batch_size]
            target_actions: [batch_size] flat action indices
            target_values: [batch_size] target values
            action_type_weight: Weight for action type loss
            param_weight: Weight for parameter loss
            value_weight: Weight for value loss
            
        Returns:
            total_loss, action_type_loss, param_loss, value_loss
        """
        batch_size = target_actions.shape[0]
        device = action_type_logits.device
        
        # Convert flat actions to hierarchical (action_type, param_idx)
        target_action_types = torch.zeros(batch_size, dtype=torch.long, device=device)
        target_param_indices = {}
        
        for i in range(batch_size):
            flat_idx = target_actions[i].item()
            action_type_idx, param_idx = self.flat_to_hierarchical[flat_idx]
            target_action_types[i] = action_type_idx
            
            # Group by action type for efficient loss computation
            if action_type_idx not in target_param_indices:
                target_param_indices[action_type_idx] = []
            target_param_indices[action_type_idx].append((i, param_idx))
        
        # Action type loss
        action_type_loss = nn.functional.cross_entropy(action_type_logits, target_action_types)
        
        # Parameter loss (only for samples with parameterized actions)
        param_loss = 0.0
        num_param_samples = 0
        
        for action_type_idx, samples in target_param_indices.items():
            if action_type_idx in param_logits:
                batch_indices = [s[0] for s in samples]
                param_targets = torch.tensor([s[1] for s in samples], dtype=torch.long, device=device)
                
                # Extract logits for these samples
                logits_for_type = param_logits[action_type_idx][batch_indices]
                param_loss += nn.functional.cross_entropy(logits_for_type, param_targets, reduction='sum')
                num_param_samples += len(samples)
        
        if num_param_samples > 0:
            param_loss = param_loss / num_param_samples
        
        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(value_pred, target_values)
        
        # Total loss
        total_loss = (action_type_weight * action_type_loss + 
                     param_weight * param_loss + 
                     value_weight * value_loss)
        
        return total_loss, action_type_loss, param_loss, value_loss
    
    def flat_to_hierarchical_action(self, flat_action_idx: int) -> Tuple[int, int]:
        """Convert flat action index to (action_type_idx, param_idx)."""
        return self.flat_to_hierarchical[flat_action_idx]
    
    def hierarchical_to_flat_action(self, action_type_idx: int, param_idx: int) -> int:
        """Convert (action_type_idx, param_idx) to flat action index."""
        return self.hierarchical_to_flat[(action_type_idx, param_idx)]

