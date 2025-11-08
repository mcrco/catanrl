import torch.nn as nn
from typing import List


class ValueNetwork(nn.Module):
    """MLP for predicting game value (expected return)."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 512]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class PolicyNetwork(nn.Module):
    """MLP for predicting best action (behavior cloning)."""
    
    def __init__(self, input_dim: int, num_actions: int, hidden_dims: List[int] = [512, 512]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


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

