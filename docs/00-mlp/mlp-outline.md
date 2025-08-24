# MLP Slides Outline (Following mlp.ipynb Structure)

## 1. Neural Networks and Function Approximation
- Introduction and motivation
- Overview of function approximation problem

## 2. The 1D Poisson Equation: Our Benchmark Problem
- Problem formulation
- Why use this as benchmark

## 3. The Function Approximation Challenge
- Traditional approaches
- Why This Proves the Weierstrass Theorem

## 4. Traditional Methods: Finite Difference
- Discrete approximation approach
- Limitations

## 5. From Polynomials to Neural Networks: The Weierstrass Approximation Theorem
- The Weierstrass Approximation Theorem
- Why This Matters for Neural Networks
- Convergence Analysis
- Bernstein Polynomials: A Constructive Proof
- Polynomials vs Neural Networks: A Direct Comparison
- Limitations of Polynomial Approximation
- From Weierstrass to Universal Approximation

## 6. The Neural Network Approach: Function Approximation
- Traditional Numerical Method vs Neural Network: Discrete vs Continuous

## 7. The Perceptron: Building Block of Neural Networks
- Linear Perceptron in NumPy
- Training: Gradient Descent from Scratch
- Example: Linear Classification
- Limitation: Non-linear Patterns
- Solution: Adding Non-linear Activation
- Interactive Demo: Visualizing Nonlinear Transformation
- Backpropagation with Activation Functions

## 8. The Critical Role of Nonlinearity
- Why nonlinearity is essential

## 9. Nonlinear Activation Functions
- Common activation functions
- Properties and uses

## 10. Building Capacity: The Single Hidden Layer Neural Network
- Architecture
- Expressiveness

## 11. Training a Neural Network
- Loss functions
- Optimization objectives

## 12. Computing Gradients with Automatic Differentiation
- Overview of AD

## 13. Forward Mode Automatic Differentiation
- Computing partial derivatives forward
- Examples

## 14. Reverse Mode Automatic Differentiation
- The Backward Pass Algorithm
- Computing All Partial Derivatives in One Pass
- AD: The Mathematical Foundation
- Automatic Differentiation in Practice: PyTorch
- A More Complex Example: Neural Network
- When to Use Forward vs Reverse Mode
- Computational Considerations
- Memory vs Computation Trade-offs
- Modern Optimizations

## 15. Gradient Descent
- Algorithm
- Variants
- Limitations
- Effect of learning rate
- Finding the Right Balance

## 16. Universal Approximation Theorem: The Mathematical Foundation
- What the Theorem States
- Why This Matters for Scientific Machine Learning