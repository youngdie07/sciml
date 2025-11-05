# SINDy Course Materials

This directory contains comprehensive materials on Sparse Identification of Nonlinear Dynamics (SINDy).

## Files

### Slides
- **`sindy-slides.tex`** - LaTeX beamer slides covering SINDy methodology
- **`sindy-slides.pdf`** - Compiled slides PDF (40 pages)

### Notebooks
- **`07-sindy.ipynb`** - Main SINDy tutorial with PySINDy implementation
  - Linear system example
  - Lorenz system discovery
  - Library comparison (polynomial vs Fourier)
  - Noisy data handling

- **`07a-sindy-experiment.ipynb`** - Extended SINDy experiments

- **`07b-sindy-pendulum.ipynb`** - Pendulum discovery example
  - Nonlinear pendulum simulation
  - Combined polynomial + trigonometric library
  - Model validation with different initial conditions
  - Comparison: correct library vs polynomial-only
  - Error analysis and phase portraits

- **`07-sindy-exercise.ipynb`** - Practice exercises for students

## Content Overview

### Slides Topics
1. **Introduction**
   - The discovery problem
   - What is SINDy?
   - Lorenz system example

2. **How SINDy Works: An Explanation**
   - Core assumption: physics is sparse
   - Step-by-step guide:
     - Collect data and derivatives
     - Build candidate function library
     - Set up coefficient-finding problem
     - Sparse regression (STLSQ algorithm)
   - Pendulum example walkthrough

3. **Mathematical Framework**
   - The SINDy equation: $\dot{\mathbf{U}} \approx \boldsymbol{\Theta}(\mathbf{U})\boldsymbol{\Xi}$
   - Data matrix $\mathbf{U}$
   - Library matrix $\boldsymbol{\Theta}$
   - Coefficient matrix $\boldsymbol{\Xi}$

4. **The SINDy Algorithm**
   - Sparse regression problem formulation
   - STLSQ (Sequentially Thresholded Least Squares)
   - Concrete walkthrough with simple example

5. **Examples**
   - Linear system ($\dot{x} = -2x$, $\dot{y} = y$)
   - Lorenz system (chaotic dynamics)
   - Library comparison (polynomial vs Fourier)

6. **Implementation**
   - PySINDy workflow
   - Code examples for:
     - Creating libraries
     - Fitting models
     - Simulating discovered systems
     - Evaluating accuracy

7. **Advanced Topics**
   - Handling noisy data
   - Partial differential equations (PDE-FIND)
   - Weak formulation for PDEs
   - Neural network libraries
   - Constrained SINDy

8. **Applications and Limitations**
   - Use cases (system identification, control, discovery)
   - When SINDy works best
   - Comparison with Neural ODEs
   - Best practices

## Usage

### Compiling LaTeX Files

```bash
pdflatex sindy-slides.tex
```

### Running Notebooks

```bash
# Activate virtual environment (if available)
source ../../env/bin/activate

# Launch Jupyter
jupyter notebook 07-sindy.ipynb
```

## Key Pedagogical Approach

The materials follow a clear progression:
1. **Intuitive Explanation First**: "How SINDy Works" section explains the method before diving into mathematics
2. **Concrete Examples**: Pendulum, linear system, and Lorenz system throughout
3. **Mathematical Rigor**: Formal framework after intuition is established
4. **Implementation Details**: PySINDy code examples with practical workflow

### Writing Principles
- **Clarity First**: Intuition before formalism
- **Sparsity in Explanation**: Only essential information (matching SINDy's philosophy)
- **Show, Don't Tell**: Examples and step-by-step derivations
- **Connection to Physics**: Emphasize interpretability and physical insight

## Key Concepts

- **Sparsity**: Most physical laws use only a few terms
- **Library Functions**: Candidate basis functions (polynomials, trig, etc.)
- **STLSQ Algorithm**: Iterative thresholding for sparse solutions
- **Data-Driven Discovery**: Extract governing equations from measurements
- **Interpretability**: Unlike neural networks, SINDy produces human-readable equations

## Comparison with Other Methods

| Method | Interpretable | Data Efficiency | Handles Chaos | Extrapolation |
|--------|--------------|----------------|---------------|---------------|
| SINDy | Yes (symbolic equations) | High (if library is correct) | Yes | Good (if sparse) |
| Neural ODE | No (black box) | Medium | Yes | Limited |
| PINNs | Partial (known physics) | Low | Yes | Good |

## References

1. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. PNAS, 113(15), 3932-3937.

2. Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L. (2019). Data-driven discovery of coordinates and governing equations. PNAS, 116(45), 22445-22451.

3. Rudy, S. H., Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2017). Data-driven discovery of partial differential equations. Science Advances, 3(4), e1602614.

4. de Silva, B. M., Champion, K., Quade, M., Loiseau, J.-C., Kutz, J. N., & Brunton, S. L. (2020). PySINDy: A Python package for the sparse identification of nonlinear dynamics from data. arXiv:2004.08424.

## License

CC BY-NC-SA 4.0 (http://creativecommons.org/licenses/by-nc-sa/4.0/)
