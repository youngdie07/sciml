# Reorganized: Neural Networks and Function Approximation

## Part I: Motivation and Problem Setup (3-4 slides)
**Goal: Establish the "why" and concrete target**

1. **Introduction and Motivation**
   - The central challenge: learning continuous functions from data
   - Applications in scientific ML
   - Preview of what we'll cover

2. **The Benchmark Problem: 1D Poisson Equation**
   - Problem formulation
   - The function approximation challenge (sparse, noisy data)

3. **Traditional Methods and Their Limitations**
   - Finite difference approach
   - Key limitations: discrete points, curse of dimensionality

## Part II: Mathematical Foundations (4-5 slides)
**Goal: Establish theoretical guarantees before building anything**

4. **From Weierstrass to Universal Approximation**
   - Weierstrass Approximation Theorem (1885)
   - Why this matters for neural networks
   - Historical progression to UAT

5. **Universal Approximation Theorem**
   - Statement and meaning
   - What it guarantees (and what it doesn't)
   - Why this matters for SciML

6. **Function Spaces and Mathematical Setting**
   - Banach and Hilbert spaces
   - Density and approximation theory

7. **What Neural Networks Cannot Approximate**
   - The continuity requirement
   - Discontinuous functions and practical implications

## Part III: Building Neural Networks (5-6 slides)  
**Goal: Build up from simple to complex, addressing limitations as we go**

8. **The Perceptron: Starting Simple**
   - Mathematical model and geometric interpretation
   - Linear perceptron implementation
   - Training with gradient descent

9. **The Critical Limitation: XOR Problem**
   - Why linear models fail
   - Mathematical proof of impossibility
   - Historical context (Minsky-Papert crisis)

10. **The Role of Nonlinearity**
    - Why activation functions are essential  
    - Common activation functions and properties
    - How nonlinearity enables complex boundaries

11. **Multi-Layer Networks: Solving XOR**
    - Single hidden layer architecture
    - How XOR is solved with hidden layers
    - Building network capacity

## Part IV: Training Neural Networks (4-5 slides)
**Goal: How to actually learn these functions**

12. **Automatic Differentiation Fundamentals**
    - Forward vs reverse mode
    - Computational graphs
    - Why AD beats numerical/symbolic

13. **Backpropagation Algorithm**
    - The backward pass in detail
    - Implementation from scratch
    - Chain rule mechanics

14. **Optimization and Training**
    - Gradient descent variants
    - Learning rate effects
    - Modern optimizers (Adam, etc.)
    - Loss functions for different problems

## Part V: Architecture Design (3-4 slides)
**Goal: Practical design choices**

15. **Width vs Depth Trade-offs**
    - Shallow vs deep network comparison
    - High-frequency function example
    - When to choose each approach

16. **Practical Implementation Guidelines**
    - Width vs approximation quality
    - Overfitting and regularization
    - Training and validation strategies

## Part VI: Wrap-up (1-2 slides)

17. **Key Takeaways and Summary**
    - Mathematical foundations → building blocks → training → practice
    - Key insights for SciML applications
    - The path from Weierstrass to modern neural networks