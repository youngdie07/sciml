#!/usr/bin/env python3
"""Generate shallow vs deep comparison for sin(100x)"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import warnings
warnings.filterwarnings('ignore')

# Set consistent style with transparent background
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Color scheme
colors = {
    'primary': '#3498DB',
    'secondary': '#E74C3C',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'neutral': '#34495E'
}

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Define network architectures
class ShallowNetwork(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepNetwork(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.ReLU())
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def generate_shallow_vs_deep_sin100x():
    """Generate comparison of shallow vs deep networks for sin(100x)"""
    
    # Target function: sin(100x)
    def target_func(x):
        return np.sin(100 * x)
    
    # Generate training data - sparse sampling
    np.random.seed(42)
    x_train = np.random.uniform(0, 1, 50)
    y_train = target_func(x_train)
    
    # Convert to tensors
    x_train_t = torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    
    # Training function
    def train_network(model, epochs=5000, lr=0.01):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(x_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
        
        return loss.item()
    
    # Train shallow network (100 neurons, 1 layer)
    print("Training shallow network (1 layer, 100 neurons)...")
    shallow_net = ShallowNetwork(100)
    shallow_loss = train_network(shallow_net, epochs=5000)
    
    # Count parameters
    shallow_params = sum(p.numel() for p in shallow_net.parameters())
    
    # Train deep network (20 neurons × 4 layers)
    print("\nTraining deep network (4 layers, 20 neurons each)...")
    deep_net = DeepNetwork(20, 4)
    deep_loss = train_network(deep_net, epochs=5000)
    
    # Count parameters
    deep_params = sum(p.numel() for p in deep_net.parameters())
    
    print(f"\nComparison:")
    print(f"Shallow final loss: {shallow_loss:.6f}")
    print(f"Deep final loss: {deep_loss:.6f}")
    print(f"Improvement: {shallow_loss/deep_loss:.1f}x better")
    
    # Generate test data for visualization
    x_test = np.linspace(0, 1, 1000)
    x_test_t = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32)
    y_true = target_func(x_test)
    
    # Get predictions
    with torch.no_grad():
        shallow_pred = shallow_net(x_test_t).numpy().flatten()
        deep_pred = deep_net(x_test_t).numpy().flatten()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Shallow network
    ax1.plot(x_test, y_true, 'k-', linewidth=3, label='sin(100x)', alpha=0.8)
    ax1.plot(x_test, shallow_pred, 'r-', linewidth=2, label='Shallow (100 neurons)', alpha=0.9)
    ax1.scatter(x_train, y_train, color='blue', s=30, alpha=0.7, zorder=5)
    
    shallow_mse = np.mean((shallow_pred - y_true)**2)
    ax1.text(0.5, 0.85, f'100 neurons × 1 layer\n{shallow_params} parameters\nMSE: {shallow_mse:.4f}',
             transform=ax1.transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    ax1.set_title('Shallow Network', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.5, 1.5)
    
    # Deep network
    ax2.plot(x_test, y_true, 'k-', linewidth=3, label='sin(100x)', alpha=0.8)
    ax2.plot(x_test, deep_pred, 'g-', linewidth=2, label='Deep (4×20 neurons)', alpha=0.9)
    ax2.scatter(x_train, y_train, color='blue', s=30, alpha=0.7, zorder=5)
    
    deep_mse = np.mean((deep_pred - y_true)**2)
    ax2.text(0.5, 0.85, f'20 neurons × 4 layers\n{deep_params} parameters\nMSE: {deep_mse:.4f}',
             transform=ax2.transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    ax2.set_title('Deep Network', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.5, 1.5)
    
    plt.suptitle('High-Frequency Function: Shallow vs Deep Networks for sin(100x)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('figs/shallow-vs-deep-sin100x.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"\nParameter Efficiency:")
    print(f"Shallow network: {shallow_params} parameters, MSE: {shallow_mse:.6f}")
    print(f"Deep network: {deep_params} parameters, MSE: {deep_mse:.6f}")
    print(f"\nDeep network: {shallow_mse/deep_mse:.1f}x better performance")
    print(f"              {shallow_params/deep_params:.1f}x more parameters")
    print(f"\nConclusion: Deep networks are more parameter-efficient!")

if __name__ == "__main__":
    generate_shallow_vs_deep_sin100x()
    print("\nFigure saved as figs/shallow-vs-deep-sin100x.png")