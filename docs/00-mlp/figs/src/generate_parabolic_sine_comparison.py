#!/usr/bin/env python3
"""Generate focused comparison of parabolic activation for sin(πx) - matching uat.md exactly"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set consistent style with transparent background
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

def generate_parabolic_approximation(target_func, x_range, num_units):
    """Generate parabolic approximation using 4t(1-t) formula from uat.md"""
    def approx(x):
        result = 0
        for i in range(num_units):
            left = i / num_units
            right = (i + 1) / num_units
            center = (left + right) / 2
            width = right - left
            height = target_func(center)
            
            # Parabolic bump: 4t(1-t) where t is normalized position in [0,1]
            if x >= left and x <= right:
                t = (x - left) / width
                parabolaBump = 4 * t * (1 - t)
                result += height * parabolaBump
        return result
    
    return np.vectorize(approx)(x_range)

def generate_relu_approximation(target_func, x_range, num_units):
    """Generate ReLU step approximation"""
    def approx(x):
        bumpIdx = np.minimum(np.floor(x * num_units).astype(int), num_units - 1)
        xMid = (bumpIdx + 0.5) / num_units
        return target_func(xMid)
    
    return np.vectorize(approx)(x_range)

def main():
    # Create figure showing how parabolic works for sin(πx)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use x range [0, 1] as in uat.md
    x = np.linspace(0, 1, 500)
    y_true = np.sin(np.pi * x)
    
    # Test with different numbers of units
    unit_counts = [2, 5, 10]
    
    for idx, num_units in enumerate(unit_counts):
        ax = axes[idx]
        
        # Generate approximations
        y_parabolic = generate_parabolic_approximation(lambda x: np.sin(np.pi * x), x, num_units)
        y_relu = generate_relu_approximation(lambda x: np.sin(np.pi * x), x, num_units)
        
        # Plot
        ax.plot(x, y_true, 'k-', linewidth=2.5, label='sin(πx)', alpha=0.8)
        ax.plot(x, y_parabolic, 'orange', linewidth=2, label=f'Parabolic ({num_units} units)')
        ax.plot(x, y_relu, 'green', linewidth=2, label=f'ReLU ({num_units} units)', linestyle='--', alpha=0.7)
        
        # Calculate errors
        mse_parabolic = np.mean((y_parabolic - y_true)**2)
        mse_relu = np.mean((y_relu - y_true)**2)
        
        # Styling
        ax.set_title(f'{num_units} Units Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.2, 1.2)
        
        # Add MSE text
        ax.text(0.05, 0.85, f'Parabolic MSE: {mse_parabolic:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.text(0.05, 0.75, f'ReLU MSE: {mse_relu:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Show individual parabolic bumps for clarity
        if num_units <= 5:
            for i in range(num_units):
                left = i / num_units
                right = (i + 1) / num_units
                center = (left + right) / 2
                height = np.sin(np.pi * center)
                
                # Draw individual bump
                x_bump = np.linspace(left, right, 50)
                t_bump = (x_bump - left) / (right - left)
                y_bump = height * 4 * t_bump * (1 - t_bump)
                ax.plot(x_bump, y_bump, 'orange', alpha=0.3, linewidth=1)
    
    plt.suptitle('Parabolic Activation for sin(πx): Range [0,1] matching uat.md', 
                 fontsize=16, fontweight='bold')
    
    # Add explanation
    fig.text(0.5, 0.02, 
            'Parabolic activation using 4t(1-t) formula creates localized bumps. Works reasonably for smooth periodic functions.',
            ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig('../parabolic-sine-comparison.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    print("Parabolic sine comparison figure generated successfully!")
    
    # Generate additional detailed analysis
    print("\nDetailed Analysis for sin(πx) on [0,1]:")
    print("="*50)
    
    for num_units in [2, 3, 4, 5, 10, 20]:
        x = np.linspace(0, 1, 1000)
        y_true = np.sin(np.pi * x)
        y_parabolic = generate_parabolic_approximation(lambda x: np.sin(np.pi * x), x, num_units)
        y_relu = generate_relu_approximation(lambda x: np.sin(np.pi * x), x, num_units)
        
        mse_parabolic = np.mean((y_parabolic - y_true)**2)
        mse_relu = np.mean((y_relu - y_true)**2)
        max_error_parabolic = np.max(np.abs(y_parabolic - y_true))
        max_error_relu = np.max(np.abs(y_relu - y_true))
        
        print(f"\n{num_units} units:")
        print(f"  Parabolic - MSE: {mse_parabolic:.6f}, Max Error: {max_error_parabolic:.6f}")
        print(f"  ReLU      - MSE: {mse_relu:.6f}, Max Error: {max_error_relu:.6f}")
        print(f"  Ratio (Parabolic/ReLU): {mse_parabolic/mse_relu:.2f}x")

if __name__ == "__main__":
    main()