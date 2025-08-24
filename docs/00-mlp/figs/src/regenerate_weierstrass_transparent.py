#!/usr/bin/env python3
"""
Regenerate Weierstrass approximation plots with transparent backgrounds
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import matplotlib.patches as mpatches

# Set style for publication-quality figures
plt.rcParams['figure.facecolor'] = 'none'  # Transparent background
plt.rcParams['axes.facecolor'] = 'none'     # Transparent axes
plt.rcParams['savefig.facecolor'] = 'none'  # Transparent save
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

def bernstein_poly(n, k, x):
    """Bernstein polynomial basis function"""
    return comb(n, k) * x**k * (1-x)**(n-k)

def bernstein_approximation(f, n, x):
    """Approximate function f using Bernstein polynomials of degree n"""
    return sum(f(k/n) * bernstein_poly(n, k, x) for k in range(n+1))

def target_function(x):
    """Target function: sin(pi*x)"""
    return np.sin(np.pi * x)

def create_bernstein_convergence_plot():
    """Create Bernstein polynomial convergence plot"""
    x = np.linspace(0, 1, 1000)
    y_true = target_function(x)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    degrees = [5, 10, 20, 50, 100, 200]
    
    for ax, n in zip(axes.flat, degrees):
        y_approx = bernstein_approximation(target_function, n, x)
        
        ax.plot(x, y_true, 'b-', linewidth=2, label='$\sin(\pi x)$', alpha=0.7)
        ax.plot(x, y_approx, 'r--', linewidth=2, label=f'$B_{{{n}}}[f](x)$')
        
        # Calculate error
        error = np.max(np.abs(y_true - y_approx))
        ax.set_title(f'Degree n={n}, Max Error={error:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(-0.1, 1.1)
    
    plt.suptitle('Bernstein Polynomial Approximation Convergence', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../bernstein-convergence-transparent.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print("Created: bernstein-convergence-transparent.png")

def create_polynomial_vs_nn_comparison():
    """Create polynomial vs neural network comparison plot"""
    x = np.linspace(0, 1, 1000)
    y_true = target_function(x)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Polynomial approximation
    n_poly = 20
    y_poly = bernstein_approximation(target_function, n_poly, x)
    
    axes[0].plot(x, y_true, 'b-', linewidth=2, label='True: $\sin(\pi x)$')
    axes[0].plot(x, y_poly, 'r--', linewidth=2, label=f'Polynomial (n={n_poly})')
    axes[0].set_title('Polynomial Approximation')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Simulate neural network with ReLU (piecewise linear)
    n_neurons = 20
    breakpoints = np.linspace(0, 1, n_neurons)
    y_nn = np.interp(x, breakpoints, target_function(breakpoints))
    
    axes[1].plot(x, y_true, 'b-', linewidth=2, label='True: $\sin(\pi x)$')
    axes[1].plot(x, y_nn, 'g--', linewidth=2, label=f'NN ReLU ({n_neurons} neurons)')
    axes[1].set_title('Neural Network Approximation')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Error comparison
    error_poly = np.abs(y_true - y_poly)
    error_nn = np.abs(y_true - y_nn)
    
    axes[2].semilogy(x, error_poly, 'r-', linewidth=2, label='Polynomial Error')
    axes[2].semilogy(x, error_nn, 'g-', linewidth=2, label='NN Error')
    axes[2].set_title('Approximation Errors (log scale)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('|Error|')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Polynomial vs Neural Network Approximation', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../polynomial-vs-nn-transparent.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print("Created: polynomial-vs-nn-transparent.png")

def create_weierstrass_theorem_visualization():
    """Create visualization of Weierstrass approximation theorem"""
    x = np.linspace(0, 1, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Different target functions
    functions = [
        (lambda x: np.sin(2*np.pi*x), '$\sin(2\pi x)$'),
        (lambda x: np.exp(-5*(x-0.5)**2), '$e^{-5(x-0.5)^2}$'),
        (lambda x: np.abs(x - 0.5), '$|x - 0.5|$'),
        (lambda x: x**3 - x**2 + 0.5*x, '$x^3 - x^2 + 0.5x$')
    ]
    
    for ax, (f, label) in zip(axes.flat, functions):
        y_true = f(x)
        
        # Plot true function
        ax.plot(x, y_true, 'b-', linewidth=3, label=f'True: {label}', alpha=0.8)
        
        # Plot approximations with different degrees
        for n, color in [(10, 'orange'), (30, 'green'), (50, 'red')]:
            y_approx = bernstein_approximation(f, n, x)
            ax.plot(x, y_approx, '--', color=color, linewidth=1.5, 
                   label=f'n={n}', alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Approximating {label}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weierstrass Approximation Theorem: Universal Approximation by Polynomials', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('../weierstrass-theorem-transparent.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print("Created: weierstrass-theorem-transparent.png")

def create_convergence_rate_plot():
    """Create convergence rate plot for different degrees"""
    degrees = np.arange(5, 101, 5)
    x = np.linspace(0, 1, 1000)
    y_true = target_function(x)
    
    max_errors = []
    mean_errors = []
    
    for n in degrees:
        y_approx = bernstein_approximation(target_function, n, x)
        error = np.abs(y_true - y_approx)
        max_errors.append(np.max(error))
        mean_errors.append(np.mean(error))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(degrees, max_errors, 'r-o', linewidth=2, markersize=6, 
                label='Max Error ($L_\infty$)')
    ax.semilogy(degrees, mean_errors, 'b-s', linewidth=2, markersize=6, 
                label='Mean Error ($L_1$)')
    
    # Add theoretical convergence rate
    theoretical = 1.0 / degrees
    ax.semilogy(degrees, theoretical, 'k--', linewidth=1.5, alpha=0.5, 
                label='$O(1/n)$ theoretical')
    
    ax.set_xlabel('Polynomial Degree (n)', fontsize=14)
    ax.set_ylabel('Approximation Error', fontsize=14)
    ax.set_title('Convergence Rate of Bernstein Polynomial Approximation', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../bernstein-convergence-rate-transparent.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print("Created: bernstein-convergence-rate-transparent.png")

if __name__ == "__main__":
    print("Generating Weierstrass approximation plots with transparent backgrounds...")
    create_bernstein_convergence_plot()
    create_polynomial_vs_nn_comparison()
    create_weierstrass_theorem_visualization()
    create_convergence_rate_plot()
    print("All plots generated successfully!")