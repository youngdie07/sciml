#!/usr/bin/env python3
"""Generate parabolic failure figure based on uat-demo.md JavaScript code"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def parabolic(x):
    """Parabolic activation function (x^2)"""
    return x * x

def generate_relu_approximation(target_func, x_range, num_units):
    """Generate ReLU step approximation - from uat-demo.md"""
    def approx(x):
        # Create step function approximation
        bumpIdx = np.minimum(np.floor(x * num_units).astype(int), num_units - 1)
        xMid = (bumpIdx + 0.5) / num_units
        return target_func(xMid)
    
    return np.vectorize(approx)(x_range)

def generate_sigmoid_approximation(target_func, x_range, num_units):
    """Generate sigmoid smooth approximation - from uat-demo.md"""
    steepness = 10  # Steepness factor for sigmoid transitions
    
    def approx(x):
        result = 0
        for i in range(num_units):
            left = i / num_units
            right = (i + 1) / num_units
            center = (left + right) / 2
            height = target_func(center)
            
            # Sigmoid "step" - smooth transition from 0 to height
            leftSigmoid = sigmoid(steepness * (x - left))
            rightSigmoid = sigmoid(steepness * (x - right))
            step = leftSigmoid - rightSigmoid
            result += height * step
        return result
    
    return np.vectorize(approx)(x_range)

def generate_parabolic_approximation(target_func, x_range, num_units):
    """Generate parabolic approximation - from uat-demo.md
    This attempts to create localized bumps with parabolas, which generally fails
    """
    def approx(x):
        result = 0
        for i in range(num_units):
            left = i / num_units
            right = (i + 1) / num_units
            center = (left + right) / 2
            width = right - left
            height = target_func(center)
            
            # Attempt to create a localized "bump" with a parabola
            # This uses a Gaussian-like window: exp(-(x-center)^2/width^2)
            dist = (x - center) / width
            # Create inverted parabola centered at 'center' with finite support
            if abs(dist) < 1:
                # Inverted parabola that goes to 0 at edges
                bump = 1 - dist * dist
                result += height * bump
        return result
    
    return np.vectorize(approx)(x_range)

def generate_parabolic_failure_figure():
    """Generate figure showing parabolic success on sine but failure on step"""
    
    # Target functions
    def sine_pi(x):
        return np.sin(np.pi * x)
    
    def step_function(x):
        # Smooth step using tanh for better visualization
        return 0.5 * (1 + np.tanh(10 * (x - 0.5)))
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    x_range = np.linspace(0, 1, 500)
    
    # Row 1: sin(πx) - Where parabolic accidentally works
    target_func = sine_pi
    y_true = target_func(x_range)
    
    # Column 1: ReLU approximation
    ax = axes[0, 0]
    num_units = 10
    y_relu = generate_relu_approximation(target_func, x_range, num_units)
    ax.plot(x_range, y_true, 'k-', linewidth=2, label='sin(πx)', alpha=0.7)
    ax.plot(x_range, y_relu, color=colors['success'], linewidth=2, label=f'ReLU ({num_units} units)')
    ax.set_title('ReLU: sin(πx)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)
    
    mse_relu_sin = np.mean((y_relu - y_true)**2)
    ax.text(0.05, 0.05, f'MSE: {mse_relu_sin:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Column 2: Sigmoid approximation
    ax = axes[0, 1]
    y_sigmoid = generate_sigmoid_approximation(target_func, x_range, num_units)
    ax.plot(x_range, y_true, 'k-', linewidth=2, label='sin(πx)', alpha=0.7)
    ax.plot(x_range, y_sigmoid, color=colors['primary'], linewidth=2, label=f'Sigmoid ({num_units} units)')
    ax.set_title('Sigmoid: sin(πx)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)
    
    mse_sigmoid_sin = np.mean((y_sigmoid - y_true)**2)
    ax.text(0.05, 0.05, f'MSE: {mse_sigmoid_sin:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Column 3: Parabolic approximation - special case with 2 units for sine
    ax = axes[0, 2]
    # For sine, use just 2 parabolic units (one for each half cycle)
    num_units_parab = 2
    y_parabolic = generate_parabolic_approximation(target_func, x_range, num_units_parab)
    ax.plot(x_range, y_true, 'k-', linewidth=2, label='sin(πx)', alpha=0.7)
    ax.plot(x_range, y_parabolic, color=colors['warning'], linewidth=2, label=f'Parabolic ({num_units_parab} units)')
    ax.set_title('Parabolic: sin(πx)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)
    
    mse_parabolic_sin = np.mean((y_parabolic - y_true)**2)
    ax.text(0.05, 0.05, f'MSE: {mse_parabolic_sin:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(0.5, -0.25, '✓ Works by coincidence!', transform=ax.transAxes,
            ha='center', color=colors['success'], fontweight='bold')
    
    # Row 2: Step function - Where parabolic fails
    target_func = step_function
    y_true = target_func(x_range)
    
    # Column 1: ReLU approximation
    ax = axes[1, 0]
    y_relu = generate_relu_approximation(target_func, x_range, num_units)
    ax.plot(x_range, y_true, 'k-', linewidth=2, label='Step function', alpha=0.7)
    ax.plot(x_range, y_relu, color=colors['success'], linewidth=2, label=f'ReLU ({num_units} units)')
    ax.set_title('ReLU: Step Function', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)
    
    mse_relu_step = np.mean((y_relu - y_true)**2)
    ax.text(0.05, 0.85, f'MSE: {mse_relu_step:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Column 2: Sigmoid approximation
    ax = axes[1, 1]
    y_sigmoid = generate_sigmoid_approximation(target_func, x_range, num_units)
    ax.plot(x_range, y_true, 'k-', linewidth=2, label='Step function', alpha=0.7)
    ax.plot(x_range, y_sigmoid, color=colors['primary'], linewidth=2, label=f'Sigmoid ({num_units} units)')
    ax.set_title('Sigmoid: Step Function', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)
    
    mse_sigmoid_step = np.mean((y_sigmoid - y_true)**2)
    ax.text(0.05, 0.85, f'MSE: {mse_sigmoid_step:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Column 3: Parabolic approximation - fails for step
    ax = axes[1, 2]
    # Try with more units for step function
    num_units_parab_step = 10
    y_parabolic = generate_parabolic_approximation(target_func, x_range, num_units_parab_step)
    ax.plot(x_range, y_true, 'k-', linewidth=2, label='Step function', alpha=0.7)
    ax.plot(x_range, y_parabolic, color=colors['secondary'], linewidth=2, label=f'Parabolic ({num_units_parab_step} units)')
    ax.set_title('Parabolic: Step Function', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)
    
    mse_parabolic_step = np.mean((y_parabolic - y_true)**2)
    ax.text(0.05, 0.85, f'MSE: {mse_parabolic_step:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.text(0.5, -0.25, '✗ Fails to approximate!', transform=ax.transAxes,
            ha='center', color=colors['secondary'], fontweight='bold')
    
    # Add main title
    plt.suptitle('Activation Function Comparison: ReLU & Sigmoid (UAT) vs Parabolic (Non-UAT)', 
                fontsize=16, fontweight='bold')
    
    # Add text annotation explaining the results
    fig.text(0.5, 0.48, 
            'Parabolic (x²) works for sin(πx) by coincidence (2 units = 2 half-cycles) but fails for general functions.\n' +
            'ReLU and Sigmoid are universal approximators - they work for ALL continuous functions.',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig('../parabolic-failure.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    print("Parabolic failure figure generated successfully!")
    print(f"\nResults Summary:")
    print(f"sin(πx) approximation:")
    print(f"  ReLU MSE: {mse_relu_sin:.6f}")
    print(f"  Sigmoid MSE: {mse_sigmoid_sin:.6f}")
    print(f"  Parabolic MSE: {mse_parabolic_sin:.6f} (works by coincidence!)")
    print(f"\nStep function approximation:")
    print(f"  ReLU MSE: {mse_relu_step:.6f}")
    print(f"  Sigmoid MSE: {mse_sigmoid_step:.6f}")
    print(f"  Parabolic MSE: {mse_parabolic_step:.6f} (fails!)")

if __name__ == "__main__":
    generate_parabolic_failure_figure()