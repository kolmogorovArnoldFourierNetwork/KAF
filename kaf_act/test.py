import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
from kaf_act import RFFActivation

def test_rff_activation():
    # Test 1: Basic functionality test
    print("=== Test 1: Basic Functionality ===")
    input_dim = 128
    x = torch.randn(32, input_dim)  # batch size 32, feature dim 128
    
    # Version with explicit dimension
    activation = RFFActivation()
    out = activation(x)
    print(f"Output shape: {out.shape} (should be [32, 128])")
    print(f"Gradient check - base_scale requires_grad: {activation.base_scale.requires_grad}")
    print(f"Gradient check - spline_scale requires_grad: {activation.spline_scale.requires_grad}\n")

    # Test 2: Auto dimension inference
    print("=== Test 2: Auto Dimension Inference ===")
    auto_activation = RFFActivation()  # input_dim not specified
    out_auto = auto_activation(x)      # First forward pass triggers initialization
    print(f"Auto-inferred input dimension: {auto_activation.input_dim} (should be 128)")
    print(f"Output shape: {out_auto.shape} (should be [32, 128])\n")

    # Test 3: Gradient backpropagation
    print("=== Test 3: Gradient Backpropagation ===")
    y = torch.randn_like(out)
    loss = F.mse_loss(out, y)
    loss.backward()
    
    print(f"base_scale gradient: {activation.base_scale.grad is not None}")
    print(f"spline_scale gradient: {activation.spline_scale.grad is not None}")
    print(f"RFF weights gradient: {activation.rff.weight.grad is not None}\n")

    # Test 4: Integration into simple model
    print("=== Test 4: Model Integration Test ===")
    model = nn.Sequential(
        nn.Linear(128, 256),
        RFFActivation(num_grids=16),
        nn.Linear(256, 10)
    )
    
    test_input = torch.randn(5, 128)  # batch size 5
    output = model(test_input)
    print(f"Final output shape: {output.shape} (should be [5, 10])\n")

    # Test 5: Activation function compatibility
    print("=== Test 5: Activation Function Compatibility ===")
    for act in [F.relu, F.gelu, torch.sigmoid]:
        model = nn.Sequential(
            nn.Linear(128, 128),
            RFFActivation(base_activation=act)
        )
        out = model(torch.randn(3, 128))
        print(f"Output range with {act.__name__}: [{out.min().item():.4f}, {out.max().item():.4f}]")

if __name__ == "__main__":
    test_rff_activation()