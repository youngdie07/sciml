#!/usr/bin/env python3
"""
Generate shallow vs deep network comparison with transparent background
Based on the last example in mlp.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Set style for publication-quality figures with transparent background
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# Define the high-frequency target function
def target_function(x):
    return np.sin(100 * x)

# Create shallow network (wide, single hidden layer)
class ShallowNet(nn.Module):
    def __init__(self, width=100):
        super().__init__()
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, 1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

# Create deep network (narrow, multiple layers)
class DeepNet(nn.Module):
    def __init__(self, width=20, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.Tanh())
        
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_network(model, X_train, y_train, epochs=5000, lr=0.01):
    """Train a network"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return model

def generate_comparison_plot():
    """Generate the shallow vs deep comparison plot"""
    
    # Generate training data
    np.random.seed(42)
    torch.manual_seed(42)
    
    x_train = np.random.uniform(0, 1, 100)
    y_train = target_function(x_train)
    
    X_train = torch.FloatTensor(x_train.reshape(-1, 1))
    Y_train = torch.FloatTensor(y_train.reshape(-1, 1))
    
    # Test points for evaluation
    x_test = np.linspace(0, 1, 1000)
    X_test = torch.FloatTensor(x_test.reshape(-1, 1))
    y_true = target_function(x_test)
    
    # Train shallow network (100 neurons, 1 hidden layer)
    print("Training shallow network (100 neurons, 1 layer)...")
    shallow_net = ShallowNet(width=100)
    shallow_net = train_network(shallow_net, X_train, Y_train, epochs=5000)
    
    # Train deep network (20 neurons per layer, 4 layers)
    print("\nTraining deep network (20 neurons per layer, 4 layers)...")
    deep_net = DeepNet(width=20, depth=4)
    deep_net = train_network(deep_net, X_train, Y_train, epochs=5000)
    
    # Evaluate
    with torch.no_grad():
        y_shallow = shallow_net(X_test).numpy().flatten()
        y_deep = deep_net(X_test).numpy().flatten()
    
    # Calculate errors
    error_shallow = np.mean((y_shallow - y_true)**2)**0.5
    error_deep = np.mean((y_deep - y_true)**2)**0.5
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Target function
    axes[0, 0].plot(x_test, y_true, 'b-', linewidth=2, label='Target: $\\sin(100x)$')
    axes[0, 0].scatter(x_train, y_train, c='red', s=10, alpha=0.5, label='Training data')
    axes[0, 0].set_title('Target Function: $\\sin(100x)$', fontsize=14)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Shallow network approximation
    axes[0, 1].plot(x_test, y_true, 'b-', linewidth=1, alpha=0.5, label='True')
    axes[0, 1].plot(x_test, y_shallow, 'r-', linewidth=2, label='Shallow (100 neurons)')
    axes[0, 1].set_title(f'Shallow Network (1 layer, 100 neurons)\nRMSE: {error_shallow:.4f}', fontsize=14)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Deep network approximation
    axes[1, 0].plot(x_test, y_true, 'b-', linewidth=1, alpha=0.5, label='True')
    axes[1, 0].plot(x_test, y_deep, 'g-', linewidth=2, label='Deep (4×20 neurons)')
    axes[1, 0].set_title(f'Deep Network (4 layers, 20 neurons each)\nRMSE: {error_deep:.4f}', fontsize=14)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error comparison
    axes[1, 1].plot(x_test, np.abs(y_shallow - y_true), 'r-', linewidth=1.5, 
                    label=f'Shallow error (RMSE: {error_shallow:.4f})', alpha=0.7)
    axes[1, 1].plot(x_test, np.abs(y_deep - y_true), 'g-', linewidth=1.5, 
                    label=f'Deep error (RMSE: {error_deep:.4f})', alpha=0.7)
    axes[1, 1].set_title('Absolute Error Comparison', fontsize=14)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('|Error|')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 0.5])
    
    plt.suptitle('Shallow vs Deep Networks: Approximating High-Frequency Functions', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../shallow-vs-deep-comparison-transparent.png', 
                dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults:")
    print(f"Shallow Network (100 neurons, 1 layer): RMSE = {error_shallow:.4f}")
    print(f"Deep Network (20 neurons × 4 layers): RMSE = {error_deep:.4f}")
    print(f"Deep network has {(error_shallow/error_deep - 1)*100:.1f}% better accuracy")
    print("\nFigure saved as: shallow-vs-deep-comparison-transparent.png")

def generate_architecture_comparison():
    """Generate a visual comparison of the architectures"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Shallow network visualization
    ax = axes[0]
    ax.set_title('Shallow Network\n(100 neurons, 1 hidden layer)', fontsize=14)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')
    
    # Input layer
    ax.scatter([0], [5], s=200, c='blue', zorder=3)
    ax.text(0, 4.2, 'Input', ha='center', fontsize=12)
    
    # Hidden layer (show subset of neurons)
    for i in range(10):
        y = i
        ax.scatter([1.5], [y], s=100, c='orange', zorder=3)
        ax.plot([0, 1.5], [5, y], 'gray', alpha=0.3, linewidth=0.5)
        ax.plot([1.5, 3], [y, 5], 'gray', alpha=0.3, linewidth=0.5)
    ax.text(1.5, -0.8, '100 neurons', ha='center', fontsize=12)
    ax.text(1.5, 10.2, '...', ha='center', fontsize=16)
    
    # Output layer
    ax.scatter([3], [5], s=200, c='green', zorder=3)
    ax.text(3, 4.2, 'Output', ha='center', fontsize=12)
    
    # Deep network visualization
    ax = axes[1]
    ax.set_title('Deep Network\n(20 neurons × 4 layers)', fontsize=14)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')
    
    # Input layer
    ax.scatter([0], [5], s=200, c='blue', zorder=3)
    ax.text(0, 4.2, 'Input', ha='center', fontsize=12)
    
    # Hidden layers
    for layer in range(1, 5):
        for i in range(5):
            y = i * 2 + 0.5
            ax.scatter([layer * 1.2], [y], s=100, c='orange', zorder=3)
            
            # Connect to previous layer
            if layer == 1:
                ax.plot([0, layer * 1.2], [5, y], 'gray', alpha=0.3, linewidth=0.5)
            else:
                for j in range(5):
                    y_prev = j * 2 + 0.5
                    ax.plot([(layer-1) * 1.2, layer * 1.2], [y_prev, y], 
                           'gray', alpha=0.2, linewidth=0.5)
            
            # Connect to output
            if layer == 4:
                ax.plot([layer * 1.2, 5], [y, 5], 'gray', alpha=0.3, linewidth=0.5)
        
        ax.text(layer * 1.2, -0.8, '20', ha='center', fontsize=10)
    
    # Output layer
    ax.scatter([5], [5], s=200, c='green', zorder=3)
    ax.text(5, 4.2, 'Output', ha='center', fontsize=12)
    
    plt.suptitle('Network Architecture Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../network-architectures-transparent.png', 
                dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    
    print("Architecture comparison saved as: network-architectures-transparent.png")

if __name__ == "__main__":
    print("Generating shallow vs deep network comparison plots...")
    generate_comparison_plot()
    generate_architecture_comparison()
    print("All plots generated successfully!")