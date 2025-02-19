import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class RandomFourierFeatures(nn.Module):
    # Keep original RFF implementation unchanged
    def __init__(
        self, 
        input_dim: int,
        num_grids: int, 
        dropout: float = 0.0,
        activation_expectation: float = 1.64
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_grids = num_grids
        self.dropout = nn.Dropout(dropout)

        var_w = 1.0 / (input_dim * activation_expectation)
        self.weight = nn.Parameter(torch.randn(input_dim, num_grids) * math.sqrt(var_w))
        
        self.bias = nn.Parameter(torch.empty(num_grids))
        nn.init.uniform_(self.bias, 0, 2 * math.pi)

        self.combination = nn.Linear(2 * num_grids, input_dim)
        nn.init.xavier_uniform_(self.combination.weight)
        if self.combination.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.combination.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.combination.bias, -bound, bound)

    def forward(self, x):
        projection = torch.matmul(x, self.weight) + self.bias
        fourier_features = torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)
        fourier_features = self.dropout(fourier_features)
        return self.combination(fourier_features)

class RFFActivation(nn.Module):
    """Pluggable activation function with CUDA support
    Args:
        num_grids: Number of Fourier feature grids,default is 9
        dropout: Dropout probability (default: 0.0)
        activation_expectation: Expected activation scale (default: 1.64)
        use_layernorm: Enable layer normalization (default: False)
        base_activation: Base activation function (default: F.gelu)
    """
    def __init__(self, num_grids=9, dropout=0.0, activation_expectation=1.64, use_layernorm=False, base_activation=F.gelu):
        super().__init__()
        # Lazy initialization parameters
        self.input_dim = None
        self.num_grids = num_grids
        self.dropout = dropout
        self.activation_expectation = activation_expectation
        self.use_layernorm = use_layernorm
        self.base_activation = base_activation
        
        # Components to be lazily created
        self.layernorm = None
        self.rff = None
        self.base_scale = None
        self.spline_scale = None

    def _init_parameters(self, input_dim):
        # Automatically detect current device
        device = next(self.parameters()).device if any(self.parameters()) else torch.device('cpu')
        
        self.input_dim = input_dim
        # Initialize layer norm with proper device placement
        self.layernorm = nn.LayerNorm(input_dim, device=device) if self.use_layernorm and input_dim > 1 else None
        
        # Initialize RFF module on detected device
        self.rff = RandomFourierFeatures(
            input_dim=input_dim,
            num_grids=self.num_grids,
            dropout=self.dropout,
            activation_expectation=self.activation_expectation
        ).to(device)  # Explicit device placement

        # Initialize scale parameters on correct device
        self.base_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.spline_scale = nn.Parameter(torch.tensor(1e-2, device=device))

    def forward(self, x):
        if self.input_dim is None:
            self._init_parameters(x.size(-1))
            
        # Ensure device consistency between input and components
        if self.layernorm is not None:
            self.layernorm = self.layernorm.to(x.device)
        if self.rff is not None:
            self.rff = self.rff.to(x.device)
            
        # Mixed precision friendly computation
        x_norm = self.layernorm(x) if self.layernorm is not None else x
        return self.base_scale * self.base_activation(x) + self.spline_scale * self.rff(x_norm)

    def _apply(self, fn):
        """Override _apply to ensure child modules stay on correct device"""
        super()._apply(fn)
        # Recursively apply device placement to submodules
        if self.layernorm is not None:
            self.layernorm = fn(self.layernorm)
        if self.rff is not None:
            self.rff = fn(self.rff)
        return self

