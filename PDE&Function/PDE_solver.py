import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import torch.multiprocessing as mp
from FANLayer import FANLayer
from kat_rational import KAT_Group
from efficient_kan import KAN 
from kaf_act import RFFActivation

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class KAN_Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        layers.append(
            nn.Linear(dims[0], dims[1])
        )

        for i in range(1, len(dims)-1):
            layers.append(
                KAN(  
                    layers_hidden=[dims[i], dims[i+1]]
                )
            )
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class KAF_Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(RFFActivation())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class FAN_Model(nn.Module):
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

class GPKAN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        # hidden_dims = [16, 16, 16]
        dims = [input_dim] + hidden_dims  # -> [in, 16, 16, 16]
        
        self.fc_layers = nn.ModuleList()
        self.kat_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.fc_layers.append(nn.Linear(dims[i], dims[i+1], bias=True))
            mode = "identity" if i == 0 else "gelu"
            self.kat_layers.append(KAT_Group(mode=mode))
            self.dropouts.append(nn.Dropout(0.1))
        
        self.final_fc = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        for fc_layer, kat_layer, dropout in zip(self.fc_layers, self.kat_layers, self.dropouts):
            x = x.view(x.size(0), -1)  # flatten to 2D
            x = fc_layer(x)
            x = x.unsqueeze(1)      
            x = dropout(x)
        
        x = x.view(x.size(0), -1)  
        x = self.final_fc(x)     
        return x

# -------------------- PDE Solver Definition --------------------
class PINN(nn.Module):
    def __init__(self, model: nn.Module, pde_params: dict):
        super().__init__()
        self.model = model
        self.pde_params = pde_params
    
    def forward(self, x):
        return self.model(x)

# -------------------- PDE Solver Definition (Modified) --------------------
class PoissonPINN(PINN):
    """2D Poisson Equation"""
    def compute_pde_residual(self, x):
        x.requires_grad_(True)
        u = self.forward(x)
        
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        du_x = du[:, 0]
        du_y = du[:, 1]
        
        du_xx_grad = torch.autograd.grad(du_x, x, grad_outputs=torch.ones_like(du_x), 
                                      create_graph=True, allow_unused=True)[0]
        du_xx = du_xx_grad[:, 0] if du_xx_grad is not None else torch.zeros_like(du_x)
        
        du_yy_grad = torch.autograd.grad(du_y, x, grad_outputs=torch.ones_like(du_y), 
                                      create_graph=True, allow_unused=True)[0]
        du_yy = du_yy_grad[:, 1] if du_yy_grad is not None else torch.zeros_like(du_y)
        
        source = -torch.sin(2*np.pi*x[:, 0]) * torch.sin(2*np.pi*x[:, 1])
        return du_xx + du_yy - source.unsqueeze(-1)

class NavierStokes1D(PINN):
    """1D Navier-Stokes Equation (Laminar Flow)"""
    def compute_pde_residual(self, x):
        x.requires_grad_(True)
        u = self.forward(x)
        
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_t = du[:, 0]  
        du_x = du[:, 1] 
        
        # Handle second order derivatives
        du_xx_grad = torch.autograd.grad(du_x, x, 
                                       grad_outputs=torch.ones_like(du_x), 
                                       create_graph=True, 
                                       allow_unused=True)[0]
        du_xx = du_xx_grad[:, 1] if du_xx_grad is not None else torch.zeros_like(du_x)
        
        nu = self.pde_params['nu']
        return (du_t + u.squeeze()*du_x - nu*du_xx).unsqueeze(-1)

class WaveEquation1D(PINN):
    """1D Wave Equation"""
    def compute_pde_residual(self, x):
        x.requires_grad_(True)
        u = self.forward(x)
        
        # First order derivatives
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_t = du[:, 0]
        du_x = du[:, 1]
        
        # Second order time derivative
        du_tt_grad = torch.autograd.grad(du_t, x, 
                                       grad_outputs=torch.ones_like(du_t), 
                                       create_graph=True, 
                                       allow_unused=True)[0]
        du_tt = du_tt_grad[:, 0] if du_tt_grad is not None else torch.zeros_like(du_t)
        
        # Second order spatial derivative
        du_xx_grad = torch.autograd.grad(du_x, x, 
                                       grad_outputs=torch.ones_like(du_x), 
                                       create_graph=True, 
                                       allow_unused=True)[0]
        du_xx = du_xx_grad[:, 1] if du_xx_grad is not None else torch.zeros_like(du_x)
        
        c = self.pde_params['c']
        return (du_tt - c**2 * du_xx).unsqueeze(-1)

class HeatPINN(PINN):
    """2D Heat Conduction Equation"""
    def compute_pde_residual(self, x):
        x.requires_grad_(True)
        u = self.forward(x)
        
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_t = du[:, 0]
        du_x = du[:, 1]
        du_y = du[:, 2]
        
        # Handle second order derivatives
        du_xx_grad = torch.autograd.grad(du_x, x, grad_outputs=torch.ones_like(du_x),
                                      create_graph=True, allow_unused=True)[0]
        du_xx = du_xx_grad[:, 1] if du_xx_grad is not None else torch.zeros_like(du_x)
        
        du_yy_grad = torch.autograd.grad(du_y, x, grad_outputs=torch.ones_like(du_y),
                                      create_graph=True, allow_unused=True)[0]
        du_yy = du_yy_grad[:, 2] if du_yy_grad is not None else torch.zeros_like(du_y)
        
        Q = 2*torch.exp(-x[:, 0]*(x[:, 1]**2 + x[:, 2]**2))
        alpha = self.pde_params['alpha']
        return (du_t - alpha*(du_xx + du_yy) - Q.unsqueeze(-1))

class HelmholtzPINN(PINN):
    """Helmholtz Equation"""
    def compute_pde_residual(self, x):
        x.requires_grad_(True)
        u = self.forward(x)
        
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Handle second order derivatives
        du_x = du[:, 0]
        du_y = du[:, 1]
        
        du_xx_grad = torch.autograd.grad(du_x, x, grad_outputs=torch.ones_like(du_x),
                                      create_graph=True, allow_unused=True)[0]
        du_xx = du_xx_grad[:, 0] if du_xx_grad is not None else torch.zeros_like(du_x)
        
        du_yy_grad = torch.autograd.grad(du_y, x, grad_outputs=torch.ones_like(du_y),
                                      create_graph=True, allow_unused=True)[0]
        du_yy = du_yy_grad[:, 1] if du_yy_grad is not None else torch.zeros_like(du_y)
        
        f = 2*(np.pi**2)*torch.sin(np.pi*x[:, 0])*torch.sin(np.pi*x[:, 1])
        k = self.pde_params['k']
        return (du_xx + du_yy + (k**2)*u.squeeze() - f.unsqueeze(-1))

class Laplace2D(PINN):
    """2D Laplace Equation"""
    def compute_pde_residual(self, x):
        x.requires_grad_(True)
        u = self.forward(x)
        
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_x = du[:, 0]
        du_y = du[:, 1]
        
        # Calculate second order derivatives
        du_xx_grad = torch.autograd.grad(du_x, x, 
                                       grad_outputs=torch.ones_like(du_x), 
                                       create_graph=True, 
                                       allow_unused=True)[0]
        du_xx = du_xx_grad[:, 0] if du_xx_grad is not None else torch.zeros_like(du_x)
        
        du_yy_grad = torch.autograd.grad(du_y, x, 
                                       grad_outputs=torch.ones_like(du_y), 
                                       create_graph=True, 
                                       allow_unused=True)[0]
        du_yy = du_yy_grad[:, 1] if du_yy_grad is not None else torch.zeros_like(du_y)
        
        return (du_xx + du_yy).unsqueeze(-1)

class KdV1D(PINN):
    """1D KdV Equation"""
    def compute_pde_residual(self, x):
        x.requires_grad_(True)
        u = self.forward(x)
        
        # Calculate first order derivatives
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_t = du[:, 0]  # Time derivative
        du_x = du[:, 1]  # Spatial derivative
        
        # Calculate third order spatial derivative
        du_xx_grad = torch.autograd.grad(du_x, x, 
                                       grad_outputs=torch.ones_like(du_x), 
                                       create_graph=True, 
                                       allow_unused=True)[0]
        du_xx = du_xx_grad[:, 1] if du_xx_grad is not None else torch.zeros_like(du_x)
        
        du_xxx_grad = torch.autograd.grad(du_xx, x, 
                                        grad_outputs=torch.ones_like(du_xx), 
                                        create_graph=True, 
                                        allow_unused=True)[0]
        du_xxx = du_xxx_grad[:, 1] if du_xxx_grad is not None else torch.zeros_like(du_xx)
        
        return (du_t + 6*u.squeeze()*du_x + du_xxx).unsqueeze(-1)

# -------------------- Data Generation --------------------
def generate_training_data(pde_type: str, n_points: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate training data (explicit device control)"""
    if pde_type == 'poisson':
        x = torch.rand(n_points, 2, device=device)
        n_boundary = n_points // 4
        boundary_x = torch.cat([
            torch.cat([torch.zeros(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([torch.ones(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([torch.rand(n_boundary, 1, device=device), torch.zeros(n_boundary, 1, device=device)], 1),
            torch.cat([torch.rand(n_boundary, 1, device=device), torch.ones(n_boundary, 1, device=device)], 1)
        ], 0)
        boundary_values = torch.zeros(boundary_x.shape[0], 1, device=device)
        return x, boundary_x, boundary_values
    
    elif pde_type == 'navier_stokes_1d':
        t = torch.rand(n_points, 1, device=device)
        x = torch.rand(n_points, 1, device=device)
        points = torch.cat([t, x], 1)
        
        n_boundary = n_points//4
        t_boundary = torch.rand(n_boundary, 1, device=device)
        boundary_x = torch.cat([
            torch.cat([torch.zeros(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([t_boundary, torch.zeros(n_boundary, 1, device=device)], 1),
            torch.cat([t_boundary, torch.ones(n_boundary, 1, device=device)], 1)
        ], 0)
        boundary_values = torch.zeros(boundary_x.shape[0], 1, device=device)
        return points, boundary_x, boundary_values
    
    elif pde_type == 'wave_1d':
        t = torch.rand(n_points, 1, device=device)
        x = torch.rand(n_points, 1, device=device)
        points = torch.cat([t, x], 1)
        
        n_boundary = n_points//4
        t_boundary = torch.rand(n_boundary, 1, device=device)
        boundary_x = torch.cat([
            torch.cat([torch.zeros(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([t_boundary, torch.zeros(n_boundary, 1, device=device)], 1),
            torch.cat([t_boundary, torch.ones(n_boundary, 1, device=device)], 1)
        ], 0)
        boundary_values = torch.zeros(boundary_x.shape[0], 1, device=device)
        return points, boundary_x, boundary_values
    
    elif pde_type == 'heat':
        t = torch.rand(n_points, 1, device=device)
        x = torch.rand(n_points, 2, device=device)
        points = torch.cat([t, x], 1)
        
        n_boundary = n_points//5
        boundary_x = torch.cat([
            torch.cat([torch.zeros(n_boundary, 1, device=device), torch.rand(n_boundary, 2, device=device)], 1),
            torch.cat([torch.rand(n_boundary, 1, device=device), torch.zeros(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([torch.rand(n_boundary, 1, device=device), torch.ones(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([torch.rand(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device), torch.zeros(n_boundary, 1, device=device)], 1),
            torch.cat([torch.rand(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device), torch.ones(n_boundary, 1, device=device)], 1)
        ], 0)
        boundary_values = torch.zeros(boundary_x.shape[0], 1, device=device)
        return points, boundary_x, boundary_values
    
    elif pde_type == 'helmholtz':
        x = torch.rand(n_points, 2, device=device)
        n_boundary = n_points//4
        boundary_x = torch.cat([
            torch.cat([torch.zeros(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([torch.ones(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([torch.rand(n_boundary, 1, device=device), torch.zeros(n_boundary, 1, device=device)], 1),
            torch.cat([torch.rand(n_boundary, 1, device=device), torch.ones(n_boundary, 1, device=device)], 1)
        ], 0)
        boundary_values = torch.zeros(boundary_x.shape[0], 1, device=device)
        return x, boundary_x, boundary_values
    
    elif pde_type == 'laplace_2d':
        x = torch.rand(n_points, 2, device=device)
        
        # Generate boundary points
        n_boundary = n_points // 4
        x_boundary = torch.linspace(0, 1, n_boundary, device=device)
        
        boundary_x = torch.cat([
            torch.cat([x_boundary.unsqueeze(1), torch.zeros(n_boundary, 1, device=device)], 1),
            torch.cat([x_boundary.unsqueeze(1), torch.ones(n_boundary, 1, device=device)], 1),
            torch.cat([torch.zeros(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1),
            torch.cat([torch.ones(n_boundary, 1, device=device), torch.rand(n_boundary, 1, device=device)], 1)
        ], 0)
        
        # Calculate boundary values
        boundary_values = torch.zeros_like(boundary_x[:, 0:1])
        bottom_mask = (boundary_x[:, 1] == 0)
        boundary_values[bottom_mask] = torch.sin(np.pi * boundary_x[bottom_mask, 0]).unsqueeze(1)
        
        return x, boundary_x, boundary_values
    
    elif pde_type == 'kdv_1d':
        t = torch.rand(n_points, 1, device=device)
        x = torch.linspace(-5, 5, int(np.sqrt(n_points)), device=device).unsqueeze(1)
        x = x.repeat(int(np.sqrt(n_points)), 1)
        points = torch.cat([t, x], 1)
        
        # Initial condition points
        x_init = torch.linspace(-5, 5, n_points//4, device=device).unsqueeze(1)
        t_init = torch.zeros_like(x_init)
        boundary_x = torch.cat([t_init, x_init], 1)
        
        # Calculate initial values (sech^2(x))
        boundary_values = 1 / torch.cosh(x_init) ** 2
        
        return points, boundary_x, boundary_values

# -------------------- Training Process --------------------
def train_pinn(
    pinn: PINN,
    train_points: torch.Tensor,
    boundary_points: torch.Tensor,
    boundary_values: torch.Tensor,
    epochs: int = 2000,
    lr: float = 1e-3
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    history = {'total_loss': []}
    
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        # PDE residual
        pde_res = pinn.compute_pde_residual(train_points)
        pde_loss = torch.mean(pde_res**2)
        
        # Boundary conditions
        b_pred = pinn(boundary_points)
        b_loss = torch.mean((b_pred - boundary_values)**2)
        
        total_loss = pde_loss + b_loss
        total_loss.backward()
        optimizer.step()
        
        history['total_loss'].append(total_loss.item())
    
    return history

def run_experiment(
    model_type: str,
    pde_type: str,
    model_params: dict,
    pde_params: dict,
    device: torch.device,
    n_points: int = 5000,
    n_runs: int = 3
) -> dict:
    """Run a single experiment"""
    # Model selection
    model_classes = {
        'mlp': MLP,
        'kan': KAN_Model,
        'kaf': KAF_Model,
        'fan': FAN_Model,
        'gpkan': GPKAN_Model
    }
    pde_classes = {
        'poisson': PoissonPINN,
        'navier_stokes_1d': NavierStokes1D,
        'wave_1d': WaveEquation1D,
        'heat': HeatPINN,
        'helmholtz': HelmholtzPINN,
        'laplace_2d': Laplace2D,
        'kdv_1d': KdV1D
    }
    
    results = {'runs': []}
    for _ in range(n_runs):
        # Data generation
        train_points, boundary_points, boundary_values = generate_training_data(
            pde_type, n_points, device
        )
        
        # Model initialization
        model = model_classes[model_type](**model_params).to(device)
        pinn = pde_classes[pde_type](model, pde_params).to(device)
        
        # Training
        history = train_pinn(pinn, train_points, boundary_points, boundary_values)
        
        # Record results
        results['runs'].append({
            'final_loss': history['total_loss'][-1],
            'n_params': sum(p.numel() for p in pinn.parameters()),
            'loss_history': history['total_loss']
        })
    
    # Statistical results
    losses = [r['final_loss'] for r in results['runs']]
    results['stats'] = {
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'min_loss': np.min(losses),
        'max_loss': np.max(losses)
    }
    return results

def process_pde(rank: int, pde_type: str, config: dict):
    """Process multi-model experiments for a single PDE type"""
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Model configuration
    input_dims = {
        'poisson': 2,
        'navier_stokes_1d': 2,
        'wave_1d': 2,
        'heat': 3,
        'laplace_2d': 2,
        'kdv_1d': 2
    }
    model_configs = [
        {'hidden_dims': [32, 32, 32]},
        {'hidden_dims': [64, 64, 64]},
        {'hidden_dims': [128, 128, 64]}
    ]
    
    all_results = []
    for model_type in ['mlp','kan','kaf','fan','gpkan']:
        for cfg in model_configs:
            model_params = {
                'input_dim': input_dims[pde_type],
                'output_dim': 1,
                **cfg
            }
            
            print(f"Testing {model_type.upper()}@{pde_type} on GPU {rank}")
            
            result = run_experiment(
                model_type=model_type,
                pde_type=pde_type,
                model_params=model_params,
                pde_params=config[pde_type],
                device=device,
                n_points=5000,
                n_runs=3
            )
            
            all_results.append({
                'model_type': model_type,
                'config': cfg,
                'results': result
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("1_pde_results", exist_ok=True)
    with open(f"pde_results/{pde_type}_gpu{rank}_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    # PDE parameter configuration
    pde_config = {
        'poisson': {},
        'navier_stokes_1d': {'nu': 0.01},
        'wave_1d': {'c': 1.0},
        'heat': {'alpha': 0.1},
        'laplace_2d': {},
        'kdv_1d': {}
    }
    
    # Launch processes to handle PDEs
    processes = []
    for i, pde_type in enumerate(['laplace_2d', 'kdv_1d']):
        p = mp.Process(
            target=process_pde,
            args=(i%torch.cuda.device_count(), pde_type, pde_config)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("All experiments completed! Results saved in pde_results directory")