import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class RandomFourierFeatures(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        num_grids: int, 
        dropout: float = 0.0,  # Dropout probability for Fourier transform
        activation_expectation: float = 1.64  # Expected value of SiLU activation function
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_grids = num_grids
        self.dropout = nn.Dropout(dropout)

        # Calculate the variance of weights
        var_w = 1.0 / (input_dim * activation_expectation)

        # Initialize frequency matrix as learnable parameters using normal distribution
        self.weight = nn.Parameter(torch.randn(input_dim, num_grids) * math.sqrt(var_w))
        
        # Initialize bias with uniform distribution [0, 2π]
        self.bias = nn.Parameter(torch.empty(num_grids))
        nn.init.uniform_(self.bias, 0, 2 * math.pi)

        # Map to input_dim
        self.combination = nn.Linear(2 * num_grids, input_dim)
        
        # Initialize the combination layer weights using Xavier uniform initialization
        # For the bias term, calculate proper bounds based on fan_in and initialize 
        # uniformly within [-1/sqrt(fan_in), 1/sqrt(fan_in)] to maintain variance
        nn.init.xavier_uniform_(self.combination.weight)
        if self.combination.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.combination.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.combination.bias, -bound, bound)

    def forward(self, x):
        projection = torch.matmul(x, self.weight) + self.bias  # (B, num_grids)

        # Fourier transform
        fourier_features = torch.cat(
            [torch.cos(projection), torch.sin(projection)], dim=-1
        )  # (B, 2 * num_grids)
        fourier_features = self.dropout(fourier_features)

        # Map to (B, input_dim)
        output = self.combination(fourier_features)  # (B, input_dim)
        return output

class FastKAFLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_grids: int = 8,
        use_layernorm: bool = True,
        spline_dropout: float = 0,
        base_activation = F.gelu,
        activation_expectation: float = 1.64
    ) -> None:
        super().__init__()
        self.base_activation = base_activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = nn.LayerNorm(input_dim) if use_layernorm and input_dim > 1 else None

        # 修改特征变换，使其输出维度与 input_dim 匹配
        print("use rff")
        self.feature_transform = RandomFourierFeatures(
            input_dim=input_dim, 
            num_grids=num_grids, 
            dropout=spline_dropout,
            activation_expectation=activation_expectation
        )

                # Layer 中
        # 初始化可学习的缩放参数
        self.base_scale = nn.Parameter(torch.tensor(1.0))  # 初始化为1
        self.spline_scale = nn.Parameter(torch.tensor(1e-2))  # 初始化为小量

        # 不再使用单独的 spline_linear 和 base_linear，而是统一使用一个 final_linear
        self.final_linear = nn.Linear(input_dim, output_dim)

        nn.init.xavier_uniform_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            nn.init.zeros_(self.final_linear.bias)

    def forward(self, x, use_layernorm=True):
        # with torch.amp.autocast('cuda'):
            if self.layernorm is not None and use_layernorm:
                x_norm = self.layernorm(x)
            else:
                x_norm = x

            # b(x)
            b = self.base_activation(x)

            # spline(x)
            s = self.feature_transform(x_norm)
            # ϕ(x) = W(ab(x) + cs(x))
            combined = self.base_scale * b + self.spline_scale * s
            ret = self.final_linear(combined)
            return ret
        
        
class KAF(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        num_grids: int = 8,
        spline_dropout: float = 0,
        use_layernorm: bool = True,
        activation_expectation: float = 1.64
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKAFLayer(
                in_dim, out_dim,
                num_grids=num_grids,
                spline_dropout=spline_dropout,
                use_layernorm=use_layernorm,
                activation_expectation=activation_expectation
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        x_range: Tuple[float, float] = (-5.0, 5.0)  # 默认绘制范围
    ):
        '''绘制特定输入和输出神经元之间的曲线'''
        ng = self.layers[0].feature_transform.num_grids
        assert input_index < self.layers[0].input_dim
        assert output_index < self.layers[0].output_dim

        # 创建输入张量
        x_vals = torch.linspace(x_range[0], x_range[1], num_pts)
        
        # 准备输入数据
        input_data = torch.zeros((num_pts, self.layers[0].input_dim))
        input_data[:, input_index] = x_vals
        
        # 计算输出
        with torch.no_grad():
            output = self.forward(input_data)
            y = output[:, output_index]
        
        return x_vals, y
    
class AttentionWithFastKAFTransform(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_grids: int = 8, spline_dropout: float = 0, use_layernorm: bool = True, activation_expectation: float = 1.64):
        super().__init__()
        self.fastKAF = KAF(input_dim, output_dim, num_grids, spline_dropout, use_layernorm, activation_expectation)
        self.attention = nn.MultiheadAttention(input_dim, num_heads=1)

    def forward(self, x):
        x = self.fastKAF(x)
        x = self.attention(x, x, x)
        return x