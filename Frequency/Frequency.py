import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from kaf_act import RFFActivation
from scipy.fft import fft
import os
from efficient_kan import KAN

# Data Generator
class SineDataGenerator:
    def __init__(self, train_range=(-20, 20), test_range=(-60, 60), samples_per_range=1000):
        self.train_range = train_range
        self.test_range = test_range
        self.samples_per_range = samples_per_range
        
    def generate_data(self):
        # Generate training data
        x_train = np.linspace(self.train_range[0], self.train_range[1], self.samples_per_range)
        y_train = np.sin(x_train)
        # Can also be changed to cos(x)
        
        # Generate test data
        x_test = np.linspace(self.test_range[0], self.test_range[1], self.samples_per_range)
        y_test = np.sin(x_test)
        
        # Save original data
        data = {
            'train': {'x': x_train.tolist(), 'y': y_train.tolist()},
            'test': {'x': x_test.tolist(), 'y': y_test.tolist()}
        }
        
        with open('sine_data.json', 'w') as f:
            json.dump(data, f)
        
        return (torch.FloatTensor(x_train).reshape(-1, 1), 
                torch.FloatTensor(y_train).reshape(-1, 1),
                torch.FloatTensor(x_test).reshape(-1, 1), 
                torch.FloatTensor(y_test).reshape(-1, 1))

# Model Definitions
class MLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class KAFModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            RFFActivation(),
            nn.Linear(hidden_size, hidden_size),
            RFFActivation(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class KANModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64], output_dim=1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        
        # Use KAN for subsequent layers
        for i in range(0, len(dims)-1):
            layers.append(
                KAN(  # Use correct parameter format
                    layers_hidden=[dims[i], dims[i+1]]
                )
            )
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
class MLPGELU(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Training and Evaluation
class ExperimentRunner:
    def __init__(self, models, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.models = models
        self.device = device
        self.results = {}
        
    def train_model(self, model_name, model, x_train, y_train, epochs=1000):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'{model_name} - Epoch {epoch+1}, Loss: {loss.item():.6f}')
    
    def evaluate_models(self, x_test, y_test):
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        predictions = {}
        mse_scores = {}
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                y_pred = model(x_test)
                mse = nn.MSELoss()(y_pred, y_test).item()
                predictions[name] = y_pred.cpu().numpy()
                mse_scores[name] = mse
        
        return predictions, mse_scores
    
    def analyze_frequency(self, predictions, y_test):
        frequency_analysis = {}
        for name, pred in predictions.items():
            # Calculate FFT
            fft_pred = np.abs(fft(pred.flatten()))
            fft_true = np.abs(fft(y_test.cpu().numpy().flatten()))
            frequency_analysis[name] = {
                'pred_fft': fft_pred[:len(fft_pred)//2].tolist(),
                'true_fft': fft_true[:len(fft_true)//2].tolist()
            }
        return frequency_analysis

def main():
    # Set random seed
    seed = 1314  # Use a fixed seed value
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Generate data
    data_gen = SineDataGenerator()
    x_train, y_train, x_test, y_test = data_gen.generate_data()
    
    # Save experiment data
    experiment_data = {
        'train': {
            'x': x_train.numpy().flatten().tolist(),
            'y': y_train.numpy().flatten().tolist()
        },
        'test': {
            'x': x_test.numpy().flatten().tolist(),
            'y': y_test.numpy().flatten().tolist()
        }
    }
    
    with open('results/experiment_data.json', 'w') as f:
        json.dump(experiment_data, f)
    
    # Initialize models
    models = {
        'MLP': MLP(),
        'KAN': KANModel(),
        'KAF': KAFModel(),
        'MLPGELU': MLPGELU()
    }
    
    # Create separate results dictionary for each model
    model_results = {}
    
    # Run experiments
    runner = ExperimentRunner(models)
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        runner.train_model(name, model, x_train, y_train)
        
        # Get predictions for this model
        with torch.no_grad():
            model.eval()
            predictions = model(x_test.to(runner.device))
            predictions = predictions.cpu().numpy().flatten().tolist()
            
            # Calculate MSE
            mse = nn.MSELoss()(torch.tensor(predictions), y_test.flatten()).item()
            
            # Calculate spectrum
            fft_pred = np.abs(fft(predictions))
            fft_true = np.abs(fft(y_test.numpy().flatten()))
            
            # Save all results for this model
            model_results[name] = {
                'predictions': predictions,
                'mse': mse,
                'frequency_analysis': {
                    'pred_fft': fft_pred[:len(fft_pred)//2].tolist(),
                    'true_fft': fft_true[:len(fft_true)//2].tolist()
                }
            }
    
    # Save results for all models
    with open('results/model_results.json', 'w') as f:
        json.dump(model_results, f)

if __name__ == '__main__':
    main() 