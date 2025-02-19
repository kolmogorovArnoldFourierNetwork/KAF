import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import scipy.special
from kaf_act import RFFActivation
import json
import torch.multiprocessing as mp
import sys
from FANLayer import FANLayer
from kat_rational import KAT_Group
from efficient_kan import KAN  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestEquations:
    @staticmethod
    def polynomial(x: torch.Tensor) -> torch.Tensor:
        """High-dimensional polynomial: f(x) = Σ(x_i^4)"""
        return torch.sum(x**4, dim=1, keepdim=True)
    
    @staticmethod
    def periodic(x: torch.Tensor) -> torch.Tensor:
        """Periodic function: f(x) = sin(2πx) + cos(4πx)"""
        return torch.sin(2*np.pi*x) + torch.cos(4*np.pi*x)
    
    @staticmethod
    def piecewise(x: torch.Tensor) -> torch.Tensor:
        """Piecewise function: f(x) = x^2 if x<=0 else e^(-x)"""
        y = torch.zeros_like(x)
        mask = x <= 0
        y[mask] = x[mask]**2
        y[~mask] = torch.exp(-x[~mask])
        return y
    
    @staticmethod
    def multimodal(x: torch.Tensor) -> torch.Tensor:
        """Multimodal function: f(x) = x*sin(5x)"""
        return x * torch.sin(5*x)

class KAFModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                RFFActivation() if i < len(dims)-2 else nn.Identity()
            ])
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class FANModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        layers.append(
            nn.Linear(dims[0], dims[1])
        )
        
        for i in range(1, len(dims)-1):
            layers.append(
                FANLayer(
                    dims[i],
                    dims[i+1],
                    p_ratio=0.25,
                    activation='gelu'
                )
            )
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class GPKANModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        # hidden_dims = [16, 16, 16]
        dims = [input_dim] + hidden_dims  # -> [in, 16, 16, 16]
        
        self.fc_layers = nn.ModuleList()
        self.kat_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Only use KAT for middle layers, not for the last layer (output dim=1)
        for i in range(len(dims) - 1):
            self.fc_layers.append(nn.Linear(dims[i], dims[i+1], bias=True))
            mode = "identity" if i == 0 else "gelu"
            self.kat_layers.append(KAT_Group(mode=mode))
            self.dropouts.append(nn.Dropout(0.1))
        
        # Final layer: from 16 -> output_dim(=1), no KAT
        self.final_fc = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        # Ensure x: [batch, 1, feature_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Process through middle layers
        for fc_layer, kat_layer, dropout in zip(self.fc_layers, self.kat_layers, self.dropouts):
            x = x.view(x.size(0), -1)  # flatten to 2D
            x = fc_layer(x)
            x = x.unsqueeze(1)        # expand back to 3D for KAT
            x = kat_layer(x)
            x = dropout(x)
        
        # Final layer (no KAT) output
        x = x.view(x.size(0), -1)   # [batch, hidden_dims[-1]]
        x = self.final_fc(x)        # get [batch, output_dim]
        return x

class KANModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # First layer uses linear layer
        layers.append(
            nn.Linear(dims[0], dims[1])
        )
        
        # Subsequent layers use KAN
        for i in range(1, len(dims)-1):
            layers.append(
                KAN(  # Use correct parameter format
                    input_dim=dims[i],
                    output_dim=dims[i+1]
                )
            )
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def generate_dataset(equation: Callable, n_samples: int, input_dim: int, x_range: Tuple[float, float] = (-1, 1)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dataset, supporting multi-dimensional input"""
    # Generate input data
    x = torch.rand(n_samples, input_dim) * (x_range[1] - x_range[0]) + x_range[0]
    x = x.to(device)
    
    # For single input functions
    if input_dim == 1:
        y = equation(x)
    # For multi-input functions
    else:
        y = equation(*[x[:, i] for i in range(input_dim)])
    
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    
    return x, y.float().reshape(-1, 1).to(device)

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU() if i < len(dims)-2 else nn.Identity()
            ])
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def count_parameters(model: nn.Module) -> int:
    """Calculate model parameter count"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-3
) -> dict:
    """Train model and return training history"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'best_test_loss': float('inf'),
        'best_epoch': 0
    }
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        train_losses = []
        
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_x)
            test_loss = criterion(test_outputs, test_y).item()
            
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(test_loss)
        
        # Record best results
        if test_loss < history['best_test_loss']:
            history['best_test_loss'] = test_loss
            history['best_epoch'] = epoch
    
    print(f"Best test loss: {history['best_test_loss']:.6f} at epoch {history['best_epoch']}")
    return history

def run_comparison_experiment(
    equation_name: str,
    equation_func: Callable,
    input_dim: int,
    n_train: int = 1000,
    n_test: int = 200,
    n_runs: int = 1
):
    """Run experiments with multiple network depths"""
    results = {
        'equation': equation_name,
        'mlp_results': [],
        'kaf_results': [],
        'fan_results': [],
        'gpkan_results': [],
        'kan_results': [],  # Add KAN results
        'mlp_configs': [],
        'kaf_configs': [],
        'fan_configs': [],
        'gpkan_configs': [],
        'kan_configs': []   # Add KAN configs
    }
    
    # Reduce configuration count
    depths = [3]  # Remove depth 5
    widths = [16,8,32, 64,128,256,512]  # Remove smallest width
    
    for depth in depths:
        for width in widths:
            hidden_dims = [width] * (depth - 1)
            
            mlp_losses = []
            kaf_losses = []
            fan_losses = []
            gpkan_losses = []
            kan_losses = []
            
            for _ in range(n_runs):
                train_x, train_y = generate_dataset(equation_func, n_train, input_dim)
                test_x, test_y = generate_dataset(equation_func, n_test, input_dim)
                
                # Train MLP
                mlp = MLP(input_dim, hidden_dims, 1).to(device)
                mlp_params = count_parameters(mlp)
                mlp_history = train_model(mlp, train_x, train_y, test_x, test_y)
                mlp_losses.append(mlp_history['best_test_loss'])
                
                # Train KAF
                kaf = KAFModel(input_dim, hidden_dims, 1).to(device)
                kaf_params = count_parameters(kaf)
                kaf_history = train_model(kaf, train_x, train_y, test_x, test_y)
                kaf_losses.append(kaf_history['best_test_loss'])
                
                # Train FAN
                fan = FANModel(input_dim, hidden_dims, 1).to(device)
                fan_params = count_parameters(fan)
                fan_history = train_model(fan, train_x, train_y, test_x, test_y)
                fan_losses.append(fan_history['best_test_loss'])
                
                # Train GPKAN
                gpkan = GPKANModel(input_dim, hidden_dims, 1).to(device)
                gpkan_params = count_parameters(gpkan)
                gpkan_history = train_model(gpkan, train_x, train_y, test_x, test_y)
                gpkan_losses.append(gpkan_history['best_test_loss'])
                
                # Train KAN
                kan = KANModel(input_dim, hidden_dims, 1).to(device)
                kan_params = count_parameters(kan)
                kan_history = train_model(kan, train_x, train_y, test_x, test_y)
                kan_losses.append(kan_history['best_test_loss'])
            
            # Record MLP results
            results['mlp_results'].append({
                'params': mlp_params,
                'best_test_loss': np.mean(mlp_losses),
                'loss_std': np.std(mlp_losses)
            })
            results['mlp_configs'].append({
                'depth': depth,
                'width': width
            })
            
            # Record KAF results
            results['kaf_results'].append({
                'params': kaf_params,
                'best_test_loss': np.mean(kaf_losses),
                'loss_std': np.std(kaf_losses)
            })
            results['kaf_configs'].append({
                'depth': depth,
                'width': width
            })
            
            # Record FAN results
            results['fan_results'].append({
                'params': fan_params,
                'best_test_loss': np.mean(fan_losses),
                'loss_std': np.std(fan_losses)
            })
            results['fan_configs'].append({
                'depth': depth,
                'width': width
            })
            
            # Record GPKAN results
            results['gpkan_results'].append({
                'params': gpkan_params,
                'best_test_loss': np.mean(gpkan_losses),
                'loss_std': np.std(gpkan_losses)
            })
            results['gpkan_configs'].append({
                'depth': depth,
                'width': width
            })
            
            # Record KAN results
            results['kan_results'].append({
                'params': kan_params,
                'best_test_loss': np.mean(kan_losses),
                'loss_std': np.std(kan_losses)
            })
            results['kan_configs'].append({
                'depth': depth,
                'width': width
            })
    
    return results

def bessel_func(x):
    return torch.from_numpy(scipy.special.j0(20*x.cpu().numpy())).float().to(x.device)

def exp_sine_func(x, y):
    return torch.exp(torch.sin(np.pi*x) + y**2)

def simple_product_func(x, y):
    return x*y

def high_freq_sum_func(x):
    linspace = torch.linspace(1, 100, 100, device=x.device).reshape(1, -1)
    return torch.sum(torch.sin(linspace * x.reshape(-1, 1) / 100), dim=1, keepdim=True)

def multi_scale_func(x1, x2, x3, x4):
    return torch.exp(torch.sin(x1**2 + x2**2) + torch.sin(x3**2 + x4**2))

def discontinuous_func(x):
    """Discontinuous step function"""
    y = torch.zeros_like(x)
    mask1 = x < -0.5
    mask2 = (x >= -0.5) & (x < 0)
    mask3 = (x >= 0) & (x < 0.5)
    mask4 = x >= 0.5
    
    y[mask1] = -1
    y[mask2] = x[mask2]**2
    y[mask3] = torch.sin(4*np.pi*x[mask3])
    y[mask4] = 1
    return y

def oscillating_decay_func(x):
    """Decaying oscillation function"""
    return torch.exp(-x**2) * torch.sin(10*np.pi*x)

def rational_func(x1, x2):
    """Rational function"""
    return (x1**2 + x2**2) / (1 + x1**2 + x2**2)

def highly_nonlinear_func(x1, x2, x3):
    """Highly nonlinear function"""
    return torch.tanh(x1*x2*x3) + torch.sin(np.pi*x1) * torch.cos(np.pi*x2) * torch.exp(-x3**2)

def chaotic_func(x1, x2):
    """Chaotic function"""
    return torch.sin(50*x1) * torch.cos(50*x2) + torch.exp(-((x1-0.5)**2 + (x2-0.5)**2)/0.1)

def process_equation(rank, eq_name, eq_func, input_dim):
    # Set GPU for current process
    torch.cuda.set_device(rank)
    global device  # Important: update global device variable
    device = torch.device(f'cuda:{rank}')
    print(f"\nTesting equation on GPU {rank}: {eq_name}")
    
    results = run_comparison_experiment(eq_name, eq_func, input_dim)
    
    # Save results
    with open(f"results_{eq_name}.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn')
    
    # Define test equations
    equations = {
        # 'Bessel': (bessel_func, 1),
        # 'Exp-Sine': (exp_sine_func, 2),
        # 'Simple-Product': (simple_product_func, 2),
        # 'High-Freq-Sum': (high_freq_sum_func, 1),
        # 'Multi-Scale': (multi_scale_func, 4),
        'Discontinuous': (discontinuous_func, 1),
        'Oscillating-Decay': (oscillating_decay_func, 1),
        'Rational': (rational_func, 2),
        'Highly-Nonlinear': (highly_nonlinear_func, 3),
        'Chaotic': (chaotic_func, 2)
    }
    
    # Check available GPU count
    n_gpus = torch.cuda.device_count()
    if n_gpus < 5:
        print(f"Warning: Available GPUs ({n_gpus}) less than number of equations (5)")
        sys.exit(1)
            
    # Create processes
    processes = []
    for i, (eq_name, (eq_func, input_dim)) in enumerate(equations.items()):
        p = mp.Process(
            target=process_equation,
            args=(i, eq_name, eq_func, input_dim)
        )
        p.start()
        processes.append(p)
            
    # Wait for all processes to complete
    for p in processes:
        p.join()
