#!/usr/bin/env python3
"""Generate Weierstrass Approximation content for MLP notebook"""

# This will be inserted into the notebook
weierstrass_content = {
    "markdown_intro": """## From Polynomials to Neural Networks: The Weierstrass Approximation Theorem

Before diving into the Universal Approximation Theorem for neural networks, let's explore its historical predecessor: the **Weierstrass Approximation Theorem** (1885). This foundational result shows that polynomials can approximate any continuous function, providing the mathematical intuition for why neural networks work.

### The Weierstrass Approximation Theorem

**Theorem (Weierstrass, 1885):** Every continuous function on a closed interval $[a, b]$ can be uniformly approximated as closely as desired by a polynomial function.

Formally: For any continuous function $f: [a, b] \\to \\mathbb{R}$ and any $\\epsilon > 0$, there exists a polynomial $p(x)$ such that:

$$\\sup_{x \\in [a,b]} |f(x) - p(x)| < \\epsilon$$

This theorem tells us that the set of polynomials is **dense** in the space of continuous functions under the uniform norm.""",

    "markdown_why": """### Why This Matters for Neural Networks

The Weierstrass theorem establishes a crucial principle: **simple building blocks (monomials $x^n$) can approximate arbitrarily complex continuous functions**. Neural networks follow the same principle but with different building blocks:

- **Polynomials**: Build from monomials $(1, x, x^2, x^3, ...)$
- **Neural Networks**: Build from activation functions (ReLU, sigmoid, tanh, ...)

Both achieve universal approximation, but neural networks often do it more efficiently!""",

    "code_polynomial_approx": '''
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def approximate_with_polynomial(func, x_range, degree, title="Polynomial Approximation"):
    """Approximate a function using polynomial regression (least squares)"""
    x = np.linspace(x_range[0], x_range[1], 100)
    y = func(x)
    
    # Fit polynomial using numpy
    poly = Polynomial.fit(x, y, degree)
    y_approx = poly(x)
    
    # Calculate approximation error
    error = np.max(np.abs(y - y_approx))
    
    # Visualization
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2, label='Target: $\\sin(\\pi x)$')
    plt.plot(x, y_approx, 'r--', linewidth=2, label=f'Polynomial (degree {degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, np.abs(y - y_approx), 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title(f'Approximation Error (max: {error:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return poly, error

# Target function: sin(πx)
target_func = lambda x: np.sin(np.pi * x)

print("Weierstrass Approximation of sin(πx) with Polynomials")
print("=" * 60)

# Show increasing polynomial degrees
degrees = [3, 5, 7, 9]
errors = []

for degree in degrees:
    poly, error = approximate_with_polynomial(
        target_func, [0, 1], degree, 
        f"Polynomial Approximation (Degree {degree})"
    )
    errors.append(error)
    print(f"Degree {degree:2d}: Max error = {error:.6f}")
    
    # Print polynomial coefficients
    coeffs = poly.coef
    print(f"  Polynomial: p(x) = ", end="")
    terms = []
    for i, coef in enumerate(coeffs):
        if abs(coef) > 1e-10:  # Skip near-zero coefficients
            if i == 0:
                terms.append(f"{coef:.3f}")
            elif i == 1:
                terms.append(f"{coef:+.3f}x")
            else:
                terms.append(f"{coef:+.3f}x^{i}")
    print(" ".join(terms[:4]) + " ...")  # Show first few terms
    print()
''',

    "markdown_convergence": """### Convergence Analysis

As we increase the polynomial degree, the approximation improves:""",

    "code_convergence": '''
# Plot convergence
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(degrees, errors, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('Maximum Approximation Error')
plt.title('Weierstrass Convergence')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
# Compare different degrees visually
x = np.linspace(0, 1, 200)
y_true = target_func(x)
plt.plot(x, y_true, 'k-', linewidth=3, label='sin(πx)', alpha=0.8)

for i, degree in enumerate([3, 5, 9]):
    poly = Polynomial.fit(np.linspace(0, 1, 100), 
                          target_func(np.linspace(0, 1, 100)), degree)
    plt.plot(x, poly(x), '--', linewidth=2, label=f'Degree {degree}', alpha=0.7)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Multiple Polynomial Approximations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

print("\\nKey Observations:")
print("1. Higher degree polynomials provide better approximation")
print("2. Error decreases exponentially with polynomial degree")
print("3. But high-degree polynomials can be unstable (Runge's phenomenon)")
''',

    "markdown_bernstein": """### Bernstein Polynomials: A Constructive Proof

One elegant proof of the Weierstrass theorem uses **Bernstein polynomials**, which provide an explicit construction:

$$B_n(f; x) = \\sum_{k=0}^{n} f\\left(\\frac{k}{n}\\right) \\binom{n}{k} x^k (1-x)^{n-k}$$

These polynomials converge uniformly to $f$ as $n \\to \\infty$.""",

    "code_bernstein": '''
def bernstein_polynomial(f, n, x):
    """Compute n-th Bernstein polynomial approximation"""
    from scipy.special import comb
    
    result = np.zeros_like(x)
    for k in range(n + 1):
        # Bernstein basis polynomial
        basis = comb(n, k) * (x ** k) * ((1 - x) ** (n - k))
        # Weight by function value at k/n
        result += f(k / n) * basis
    return result

# Demonstrate Bernstein approximation
plt.figure(figsize=(12, 4))

x = np.linspace(0, 1, 200)
y_true = target_func(x)

degrees = [5, 10, 20, 40]
for i, n in enumerate(degrees):
    plt.subplot(1, 4, i + 1)
    y_bernstein = bernstein_polynomial(target_func, n, x)
    
    plt.plot(x, y_true, 'b-', linewidth=2, label='sin(πx)', alpha=0.7)
    plt.plot(x, y_bernstein, 'r--', linewidth=2, label=f'Bernstein n={n}')
    
    error = np.max(np.abs(y_true - y_bernstein))
    plt.title(f'n = {n}\\nError: {error:.4f}')
    plt.xlabel('x')
    if i == 0:
        plt.ylabel('y')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.2, 1.2)

plt.suptitle('Bernstein Polynomial Approximation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\\nBernstein Polynomials:")
print("• Provide a constructive proof of Weierstrass theorem")
print("• Converge uniformly but slowly")
print("• Always stay within the range of the function")
print("• Form the basis for Bézier curves in computer graphics!")
''',

    "markdown_comparison": """### Polynomials vs Neural Networks: A Direct Comparison

Let's compare polynomial approximation with neural network approximation for the same function:""",

    "code_comparison": '''
import torch
import torch.nn as nn
import torch.optim as optim

# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Train neural network
def train_nn(hidden_size=10, epochs=5000):
    x_train = torch.linspace(0, 1, 200).reshape(-1, 1)
    y_train = torch.tensor([target_func(x.item()) for x in x_train], 
                           dtype=torch.float32).reshape(-1, 1)
    
    model = SimpleNN(hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # Train with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > 500 or loss.item() < 1e-6:
            break
    
    return model

# Compare polynomial vs neural network
plt.figure(figsize=(12, 5))

x_test = np.linspace(0, 1, 200)
y_true = target_func(x_test)

# Polynomial approximation
poly_degree = 9
poly = Polynomial.fit(np.linspace(0, 1, 100), 
                      target_func(np.linspace(0, 1, 100)), poly_degree)
y_poly = poly(x_test)
poly_params = poly_degree + 1

# Neural network approximation
nn_hidden = 10
model = train_nn(nn_hidden, epochs=10000)
x_test_torch = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32)
with torch.no_grad():
    y_nn = model(x_test_torch).numpy().flatten()
nn_params = 1 * nn_hidden + nn_hidden + nn_hidden * 1 + 1  # weights + biases

# Plot comparison
plt.subplot(1, 3, 1)
plt.plot(x_test, y_true, 'k-', linewidth=3, label='Target', alpha=0.8)
plt.plot(x_test, y_poly, 'b--', linewidth=2, label=f'Polynomial (deg {poly_degree})')
plt.plot(x_test, y_nn, 'r:', linewidth=2, label=f'Neural Net ({nn_hidden} hidden)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Approximation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x_test, np.abs(y_true - y_poly), 'b-', linewidth=2, 
         label=f'Polynomial ({poly_params} params)')
plt.plot(x_test, np.abs(y_true - y_nn), 'r-', linewidth=2, 
         label=f'Neural Net ({nn_params} params)')
plt.xlabel('x')
plt.ylabel('|Error|')
plt.title('Approximation Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 3, 3)
errors_comparison = {
    'Polynomial\\n(10 params)': np.max(np.abs(y_true - y_poly)),
    'Neural Net\\n(21 params)': np.max(np.abs(y_true - y_nn))
}
bars = plt.bar(errors_comparison.keys(), errors_comparison.values(), 
               color=['blue', 'red'], alpha=0.7)
plt.ylabel('Maximum Error')
plt.title('Error Comparison')
for bar, error in zip(bars, errors_comparison.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{error:.4f}', ha='center', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Weierstrass (Polynomial) vs Universal Approximation (Neural Network)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\\nComparison Summary:")
print(f"Polynomial (degree {poly_degree}):")
print(f"  Parameters: {poly_params}")
print(f"  Max Error: {np.max(np.abs(y_true - y_poly)):.6f}")
print(f"\\nNeural Network ({nn_hidden} hidden units):")
print(f"  Parameters: {nn_params}")
print(f"  Max Error: {np.max(np.abs(y_true - y_nn)):.6f}")
print(f"\\nNote: Neural network uses Sigmoid activation for smooth approximation")
''',

    "markdown_limitations": """### Limitations of Polynomial Approximation

While polynomials can theoretically approximate any continuous function, they have practical limitations:

1. **Runge's Phenomenon**: High-degree polynomials oscillate wildly at boundaries
2. **Global Support**: Changing one coefficient affects the entire function
3. **Computational Instability**: High-degree polynomials suffer from numerical issues
4. **Poor Extrapolation**: Polynomials diverge rapidly outside the training interval

Neural networks address these limitations:
- **Local Support**: ReLU networks create piecewise linear approximations
- **Stability**: Bounded activations (sigmoid, tanh) prevent divergence
- **Compositionality**: Deep networks build complex functions from simple pieces
- **Adaptivity**: Networks learn where to place their "basis functions"

### From Weierstrass to Universal Approximation

The progression from Weierstrass to neural networks represents a evolution in approximation theory:

1. **1885 - Weierstrass**: Polynomials are universal approximators
2. **1989 - Cybenko**: Single-layer neural networks are universal approximators
3. **Modern Deep Learning**: Deep networks are exponentially more efficient

This historical perspective shows that neural networks are not magical – they're the latest chapter in a long mathematical story about approximating complex functions with simple building blocks!"""
}

# Save as Python script that can be converted to notebook cells
print("Weierstrass section content generated. Use this to update mlp.ipynb")
print("\nContent includes:")
for key in weierstrass_content.keys():
    print(f"  - {key}")