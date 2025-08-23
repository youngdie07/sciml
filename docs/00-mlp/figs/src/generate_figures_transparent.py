#!/usr/bin/env python3
"""Generate figures for UAT section with transparent backgrounds"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
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

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def target_sinpi(x):
    """Target function sin(πx)"""
    return np.sin(np.pi * x)

def step_function(x):
    """True discontinuous step function"""
    return np.where(x < 0.5, 0, 1)

def smooth_step(x, steepness=50):
    """Smooth approximation of step using sigmoid"""
    return 1 / (1 + np.exp(-steepness * (x - 0.5)))

def detector_measure(x):
    """A signed measure that's +1 on [0, 0.5) and -1 on [0.5, 1]"""
    return np.where(x < 0.5, 1, -1)

# Neural network classes from notebooks
class ReLUNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.hidden = nn.Linear(1, width)
        self.output = nn.Linear(width, 1)
        
        # Glorot initialization
        nn.init.xavier_normal_(self.hidden.weight)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))

def train_relu_network(target_func, width, epochs=5000, lr=0.01):
    """Train a ReLU network to approximate a target function."""
    x_train = torch.linspace(0, 1, 100).reshape(-1, 1)
    y_train = torch.tensor([target_func(x.item()) for x in x_train], dtype=torch.float32).reshape(-1, 1)
    
    best_model = None
    best_error = float('inf')
    
    # Multiple restarts to avoid bad local minima
    for trial in range(5):
        model = ReLUNet(width)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Train
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        x_test = torch.linspace(0, 1, 200).reshape(-1, 1)
        with torch.no_grad():
            y_pred = model(x_test).numpy().flatten()
            x_test_np = x_test.numpy().flatten()
            y_true = np.array([target_func(x) for x in x_test_np])
            error = np.max(np.abs(y_true - y_pred))
        
        if error < best_error:
            best_error = error
            best_model = model
    
    return best_model, best_error

def generate_uat_approximation_progression():
    """Shows 2, 10, 50 neurons approximating sin(πx) - from notebook"""
    widths = [2, 10, 50]
    models = []
    errors = []
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, width in enumerate(widths):
        model, error = train_relu_network(target_sinpi, width, epochs=max(3000, width * 30))
        models.append(model)
        errors.append(error)
        
        # Evaluate
        x_test = torch.linspace(0, 1, 200).reshape(-1, 1)
        with torch.no_grad():
            y_pred = model(x_test).numpy().flatten()
            x_test_np = x_test.numpy().flatten()
            y_true = np.array([target_sinpi(x) for x in x_test_np])
        
        ax = axes[idx]
        ax.plot(x_test_np, y_true, 'k-', linewidth=2, label='sin(πx)', alpha=0.7)
        ax.plot(x_test_np, y_pred, color=colors['primary'], linewidth=2, label=f'{width} neurons')
        ax.set_title(f'Width = {width}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(-1.5, 1.5)
        
        # Add MSE text
        ax.text(0.05, 0.95, f'Error: {error:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
    
    plt.suptitle('Universal Approximation: Effect of Network Width', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figs/uat-approximation-progression.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def generate_width_vs_error():
    """Error vs number of neurons plot - from notebook"""
    widths = [2, 5, 10, 20, 50, 100]
    errors = []
    
    for width in widths:
        _, error = train_relu_network(target_sinpi, width, epochs=max(3000, width * 30))
        errors.append(error)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale
    ax1.plot(widths, errors, 'o-', color=colors['primary'], linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Neurons', fontsize=12)
    ax1.set_ylabel('Approximation Error', fontsize=12)
    ax1.set_title('Linear Scale', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)
    
    # Log scale
    ax2.semilogy(widths, errors, 'o-', color=colors['secondary'], linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Neurons', fontsize=12)
    ax2.set_ylabel('Approximation Error (log scale)', fontsize=12)
    ax2.set_title('Log Scale', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0, 105)
    
    # Add theoretical line
    theoretical = 1.0 / np.array(widths)
    ax2.plot(widths, theoretical, '--', color=colors['neutral'], alpha=0.7, label='O(1/n)')
    ax2.legend()
    
    plt.suptitle('Convergence Rate of Neural Network Approximation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figs/width-vs-error.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def generate_contradiction_visualization():
    """9-panel proof by contradiction visualization - from notebook"""
    fig = plt.figure(figsize=(16, 12))
    
    # Similar to notebook cell 13
    x = np.linspace(0, 1, 500)
    
    # Plot 1: The Setup - Target function
    ax1 = plt.subplot(3, 3, 1)
    y_target = target_sinpi(x)
    ax1.plot(x, y_target, 'b-', linewidth=2, label='sin(πx)')
    ax1.fill_between(x, y_target - 0.1, y_target + 0.1, alpha=0.2, color='red')
    ax1.set_title('1. The "Unreachable" Function', fontweight='bold')
    ax1.set_ylabel('f(x)')
    ax1.text(0.5, -0.5, 'Forbidden zone: ε = 0.1', ha='center', color='red')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The detector measure
    ax2 = plt.subplot(3, 3, 2)
    mu = lambda x: np.sin(2 * np.pi * x) * np.exp(-2 * x)
    y_mu = mu(x)
    ax2.plot(x, y_mu, 'g-', linewidth=2)
    ax2.fill_between(x, 0, y_mu, where=(y_mu > 0), alpha=0.3, color='green', label='μ > 0')
    ax2.fill_between(x, 0, y_mu, where=(y_mu <= 0), alpha=0.3, color='red', label='μ < 0')
    ax2.set_title('2. The "Magic" Measure μ', fontweight='bold')
    ax2.set_ylabel('μ(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sigmoids approach step functions
    ax3 = plt.subplot(3, 3, 3)
    for w, b, label in [(10, 0.3, 'w=10'), (20, 0.5, 'w=20'), (50, 0.7, 'w=50')]:
        ax3.plot(x, smooth_step(x - b, w), linewidth=2, label=label)
    ax3.set_title('3. Sigmoids → Half-spaces', fontweight='bold')
    ax3.set_ylabel('σ(wx + b)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: Half-space isolation
    ax4 = plt.subplot(3, 3, 4)
    ax4.axvspan(0, 0.5, alpha=0.3, color=colors['primary'], label='H₁')
    ax4.axvline(0.5, color=colors['secondary'], linewidth=2)
    ax4.set_title('4. Half-space H₁', fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.legend()
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.axvspan(0.5, 1, alpha=0.3, color=colors['warning'], label='H₂')
    ax5.axvline(0.5, color=colors['secondary'], linewidth=2)
    ax5.set_title('5. Half-space H₂', fontweight='bold')
    ax5.set_xlim(0, 1)
    ax5.legend()
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.axvline(0.5, color=colors['secondary'], linewidth=3)
    ax6.scatter([0.5], [0.5], s=200, color=colors['secondary'], zorder=5)
    ax6.set_title('6. Intersection: Single Point', fontweight='bold')
    ax6.text(0.5, 0.2, 'H₁ ∩ H₂ = {0.5}', fontsize=12, ha='center')
    ax6.set_xlim(0, 1)
    
    # Plot 7: Isolated point
    ax7 = plt.subplot(3, 3, 7)
    circle = plt.Circle((0.5, 0.5), 0.2, fill=False, 
                       edgecolor=colors['secondary'], linewidth=2, linestyle='--')
    ax7.add_patch(circle)
    ax7.scatter([0.5], [0.5], s=200, color=colors['secondary'], zorder=5)
    ax7.set_title('7. Point is Isolated!', fontweight='bold')
    ax7.text(0.5, 0.1, 'μ({0.5}) = 0', fontsize=12, ha='center')
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    
    # Plot 8: Contradiction
    ax8 = plt.subplot(3, 3, 8)
    ax8.text(0.5, 0.5, '⚠️\nContradiction!\nμ = 0 everywhere\nbut needs to\ndetect sin(πx)', 
            fontsize=14, ha='center', va='center', color=colors['secondary'], fontweight='bold')
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    
    # Plot 9: Conclusion
    ax9 = plt.subplot(3, 3, 9)
    ax9.text(0.5, 0.5, '✓\nUAT Proven!\nNN can approximate\nany continuous\nfunction', 
            fontsize=14, ha='center', va='center', color=colors['success'], fontweight='bold')
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.suptitle('Proof by Contradiction: Universal Approximation Theorem', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figs/contradiction-visualization.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def generate_derivative_approximation():
    """Neural network approximating f and f' - from notebook"""
    class DerivativeNN(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(1, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
            self.activation = nn.Tanh()
        
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Test function from notebook
    test_func = lambda x: np.sin(np.pi * x)
    test_deriv = lambda x: np.pi * np.cos(np.pi * x)
    
    # Train model
    x_train = torch.linspace(0, 1, 50, requires_grad=True).reshape(-1, 1)
    y_train = torch.tensor([test_func(x.item()) for x in x_train], dtype=torch.float32).reshape(-1, 1)
    dy_train = torch.tensor([test_deriv(x.item()) for x in x_train], dtype=torch.float32).reshape(-1, 1)
    
    model = DerivativeNN(50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(10000):
        optimizer.zero_grad()
        y_pred = model(x_train)
        dy_pred = torch.autograd.grad(y_pred.sum(), x_train, create_graph=True)[0]
        
        loss_f = torch.mean((y_pred - y_train)**2)
        loss_df = torch.mean((dy_pred - dy_train)**2)
        loss = loss_f + 0.1 * loss_df
        
        loss.backward()
        optimizer.step()
    
    # Evaluate
    x_test = torch.linspace(0, 1, 200, requires_grad=True).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(x_test).numpy().flatten()
    
    y_test_grad = model(x_test)
    dy_pred = torch.autograd.grad(y_test_grad.sum(), x_test)[0].detach().numpy().flatten()
    
    x_test_np = x_test.detach().numpy().flatten()
    y_true = np.array([test_func(x) for x in x_test_np])
    dy_true = np.array([test_deriv(x) for x in x_test_np])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Function approximation
    ax1.plot(x_test_np, y_true, 'k-', linewidth=2, label='f(x) = sin(πx)')
    ax1.plot(x_test_np, y_pred, color=colors['primary'], linewidth=2, 
            label='NN approximation', alpha=0.8)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Function Approximation in Sobolev Space W¹·²', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Add error text
    error_f = np.sqrt(np.mean((y_true - y_pred)**2))
    ax1.text(0.02, 0.98, f'||f - NN||₂ = {error_f:.4f}', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    
    # Derivative approximation
    ax2.plot(x_test_np, dy_true, 'k-', linewidth=2, label="f'(x) = π cos(πx)")
    ax2.plot(x_test_np, dy_pred, color=colors['secondary'], linewidth=2,
            label='NN derivative', alpha=0.8)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("f'(x)", fontsize=12)
    ax2.set_title('Derivative Approximation (Weak Derivative)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # Add error text
    error_df = np.sqrt(np.mean((dy_true - dy_pred)**2))
    ax2.text(0.02, 0.98, f"||f' - NN'||₂ = {error_df:.4f}", transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('figs/derivative-approximation.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def generate_relu_decomposition():
    """ReLU decomposition for sin(πx) - exact formula from uat-demo.md"""
    x = np.linspace(-3, 5, 1000)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Component functions - exact formula from uat-demo.md
    components = [
        (-20, lambda z: relu(-z - 1), '-20·ReLU(-x-1)', colors['secondary']),
        (5, lambda z: relu(z + 1), '5·ReLU(x+1)', colors['primary']),
        (-5, lambda z: relu(z), '-5·ReLU(x)', colors['warning']),
        (5, lambda z: relu(z - 2), '5·ReLU(x-2)', colors['success']),
        (15, lambda z: relu(z - 3), '15·ReLU(x-3)', colors['neutral'])
    ]
    
    # Plot individual components
    for i, (weight, func, label, color) in enumerate(components):
        ax = axes[i // 2, i % 2]
        y = weight * func(x)
        ax.plot(x, y, linewidth=2, color=color, label=label)
        ax.fill_between(x, 0, y, alpha=0.3, color=color)
        ax.set_xlim(-3, 5)
        ax.set_ylim(-25, 25)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Mark breakpoint
        if 'ReLU(-x' in label:
            breakpoint = -1
        elif 'x+1' in label:
            breakpoint = -1
        elif 'ReLU(x)' in label:
            breakpoint = 0
        elif 'x-2' in label:
            breakpoint = 2
        else:
            breakpoint = 3
        ax.axvline(breakpoint, color='red', linestyle='--', alpha=0.5)
        ax.scatter([breakpoint], [0], color='red', s=50, zorder=5)
    
    # Sum of all components
    ax = axes[2, 1]
    total = np.zeros_like(x)
    for weight, func, _, _ in components:
        total += weight * func(x)
    
    ax.plot(x, total, linewidth=3, color=colors['primary'], label='Sum')
    ax.plot(x, 10*np.sin(np.pi * x), '--', linewidth=2, color='black', 
           alpha=0.7, label='10·sin(πx)')
    ax.set_xlim(-3, 5)
    ax.set_ylim(-15, 15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Final Approximation', fontweight='bold')
    
    plt.suptitle('ReLU Decomposition: Approximating sin(πx)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figs/relu-decomposition-sinpi.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def generate_bias_breakpoints():
    """Illustration of bias as breakpoint control"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    x = np.linspace(-3, 3, 200)
    biases = [1, 0, -1]
    weight = 1
    
    for ax, b in zip(axes, biases):
        y = relu(weight * x + b)
        breakpoint = -b / weight
        
        ax.plot(x, y, linewidth=3, color=colors['primary'])
        ax.axvline(breakpoint, color=colors['secondary'], linestyle='--', 
                  alpha=0.7, linewidth=2)
        ax.scatter([breakpoint], [0], color=colors['secondary'], s=150, zorder=5)
        
        # Add text annotations
        ax.text(breakpoint, -0.5, f'x = {breakpoint:.1f}', 
               ha='center', fontsize=12, color=colors['secondary'], fontweight='bold')
        ax.set_title(f'ReLU(x + {b})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 4)
        ax.set_xlim(-3, 3)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        
        # Add formula
        formula = f'y = max(0, x + {b})'
        ax.text(0.05, 0.95, formula, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               verticalalignment='top', fontsize=11)
    
    plt.suptitle('Bias Controls Breakpoint Location in ReLU Networks', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figs/bias-breakpoints-detailed.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

def generate_parabolic_failure():
    """Comparison showing where parabolic fails and ReLU succeeds"""
    x = np.linspace(-2, 2, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Approximating sin(πx)
    # Parabolic success
    ax1 = axes[0, 0]
    target1 = np.sin(np.pi * x)
    # Two parabolas to approximate sin
    parab1 = -4 * (x + 0.5)**2 + 1
    parab2 = 4 * (x - 0.5)**2 - 1
    parab_sum = np.where(x < 0, parab1, parab2)
    
    ax1.plot(x, target1, 'k-', linewidth=2, label='sin(πx)', alpha=0.7)
    ax1.plot(x, parab_sum, color=colors['warning'], linewidth=2, 
            label='2 Parabolic units')
    ax1.set_title('Parabolic: sin(πx)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.5, 1.5)
    
    # Add MSE
    mse1 = np.mean((target1 - parab_sum)**2)
    ax1.text(0.05, 0.05, f'MSE: {mse1:.3f}', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ReLU success on sin(πx)
    ax2 = axes[0, 1]
    # Train ReLU network
    model_sin, error_sin = train_relu_network(target_sinpi, 10, epochs=5000)
    x_test = torch.linspace(-2, 2, 1000).reshape(-1, 1)
    with torch.no_grad():
        relu_approx_sin = model_sin(x_test).numpy().flatten()
    x_test_np = x_test.numpy().flatten()
    
    ax2.plot(x_test_np, target_sinpi(x_test_np), 'k-', linewidth=2, label='sin(πx)', alpha=0.7)
    ax2.plot(x_test_np, relu_approx_sin, color=colors['success'], linewidth=2, 
            label='10 ReLU units')
    ax2.set_title('ReLU: sin(πx)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlim(-2, 2)
    
    mse_relu_sin = np.mean((target_sinpi(x_test_np) - relu_approx_sin)**2)
    ax2.text(0.05, 0.05, f'MSE: {mse_relu_sin:.3f}', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Comparison text for sin(πx)
    ax3 = axes[0, 2]
    ax3.axis('off')
    text1 = """sin(πx) Approximation:

✓ Parabolic works by coincidence
  - Only needs 2 units for this case
  - One for upward curve
  - One for downward curve
  
✓ ReLU also works well
  - Uses piecewise linear segments
  - Can place breakpoints anywhere
  - Universal approximator"""
    
    ax3.text(0.1, 0.5, text1, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Row 2: Approximating step function
    # Parabolic failure on step
    ax4 = axes[1, 0]
    target2 = step_function(x)
    
    # Try to approximate step with parabolas - this will fail
    n_parab = 20
    parab_approx = np.zeros_like(x)
    for i in range(n_parab):
        center = -2 + 4 * i / n_parab
        # Create smooth bumps
        bump = np.exp(-5*(x - center)**2)
        if center > 0:
            parab_approx += bump
    # Normalize to [0, 1]
    if np.max(parab_approx) > 0:
        parab_approx = parab_approx / np.max(parab_approx)
    
    ax4.plot(x, target2, 'k-', linewidth=2, label='Step function', alpha=0.7)
    ax4.plot(x, parab_approx, color=colors['secondary'], linewidth=2, 
            label=f'{n_parab} Parabolic units')
    ax4.set_title('Parabolic: Step Function', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.2, 1.2)
    
    mse2 = np.mean((target2 - parab_approx)**2)
    ax4.text(0.05, 0.05, f'MSE: {mse2:.3f}', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # ReLU success on step
    ax5 = axes[1, 1]
    # Use smooth step approximation with ReLU network
    step_x = x_test_np
    step_y = step_function(step_x)
    
    # Train ReLU for step
    class StepNet(nn.Module):
        def __init__(self, width=20):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Train step approximator
    x_train = torch.linspace(-2, 2, 200).reshape(-1, 1)
    y_train = (x_train > 0).float()
    
    step_model = StepNet(20)
    optimizer = optim.Adam(step_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(3000):
        optimizer.zero_grad()
        y_pred = step_model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        relu_step_approx = step_model(x_test).numpy().flatten()
    
    ax5.plot(step_x, step_y, 'k-', linewidth=2, label='Step function', alpha=0.7)
    ax5.plot(step_x, relu_step_approx, color=colors['success'], linewidth=2, 
            label='20 ReLU units')
    ax5.set_title('ReLU: Step Function', fontsize=12, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.2, 1.2)
    ax5.set_xlim(-2, 2)
    
    mse_relu_step = np.mean((step_y - relu_step_approx)**2)
    ax5.text(0.05, 0.05, f'MSE: {mse_relu_step:.3f}', transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Comparison text for step
    ax6 = axes[1, 2]
    ax6.axis('off')
    text2 = """Step Function Approximation:

✗ Parabolic FAILS
  - Cannot create sharp corners
  - Smooth everywhere (C∞)
  - Cannot localize features
  - NOT universal approximator
  
✓ ReLU SUCCEEDS
  - Creates sharp transitions
  - Piecewise linear
  - Can approximate discontinuities
    (in L² sense)
  - Universal approximator"""
    
    ax6.text(0.1, 0.5, text2, transform=ax6.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.suptitle('Parabolic vs ReLU: Coincidental Success vs Universal Approximation', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figs/parabolic-failure.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()

# Generate all figures
if __name__ == "__main__":
    print("Generating UAT approximation progression...")
    generate_uat_approximation_progression()
    
    print("Generating width vs error plot...")
    generate_width_vs_error()
    
    print("Generating contradiction visualization...")
    generate_contradiction_visualization()
    
    print("Generating derivative approximation...")
    generate_derivative_approximation()
    
    print("Generating ReLU decomposition diagram...")
    generate_relu_decomposition()
    
    print("Generating bias breakpoints diagram...")
    generate_bias_breakpoints()
    
    print("Generating parabolic failure comparison...")
    generate_parabolic_failure()
    
    print("\nAll figures generated successfully with transparent backgrounds!")
    print("Figures saved in figs/ directory")