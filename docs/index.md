# Scientific Machine Learning (SciML)

Scientific Machine Learning (SciML) represents a multi-disciplinary approach that fuses the physical laws governing a system (such as equations from physics or engineering) with data-driven machine learning methodologies. SciML uses domain knowledge to design appropriate machine-learning models for different scientific challenges. This domain expertise helps select relevant features, appropriate model architectures, useful validation metrics, etc. SciML exploits structure in scientific data like symmetries and conservation laws to develop more suitable machine learning techniques. For example, physics-informed neural networks incorporate physical principles into their design of loss functions. SciML research also includes extracting fundamental laws or PDEs from neural networks by observing system behavior.

## Course Description

This course provides a rigorous introduction to Scientific Machine Learning (SciML), focusing on the development, analysis, and application of machine learning techniques for solving complex problems governed by ordinary and partial differential equations (ODEs/PDEs). Bridging numerical analysis, scientific computing, and deep learning, SciML offers novel computational paradigms for challenges where traditional methods face limitations, such as high-dimensional problems, inverse problems, and systems with incomplete physical knowledge.

We will delve into the mathematical foundations underpinning modern SciML solvers. Key topics include Physics-Informed Neural Networks (PINNs), Neural Ordinary Differential Equations (NODEs), and Operator Learning frameworks (e.g., DeepONets, Fourier Neural Operators), which learn mappings between infinite-dimensional function spaces. The course will explore the theoretical basis for these methods, including function approximation theory in relevant spaces (e.g., Sobolev spaces), the role of automatic differentiation for computing derivatives and residuals, and the specific optimization challenges encountered when training physics-informed models.

Emphasis will be placed on formulating differential equations as learning problems, analyzing the properties of different SciML architectures and loss functions, understanding techniques for enforcing boundary conditions, and evaluating the convergence and accuracy of the resulting solutions. We will also cover methods for uncertainty quantification, transfer learning approaches, and explore the discovery of governing equations from data. Practical implementation will be demonstrated using modern frameworks like PyTorch and JAX, enabling students to apply these advanced computational techniques to challenging scientific and engineering problems.

## Learning Outcomes

Upon successful completion of this course, students will be able to:

- Formulate scientific problems governed by ODEs/PDEs within machine learning frameworks; implement, train, and evaluate core methods like Physics-Informed Neural Networks (PINNs), Neural ODEs, and Operator Learning techniques, including appropriate handling of boundary conditions and sampling strategies.

- Analyze the mathematical underpinnings of SciML methods, including relevant function approximation theory (e.g., Universal Approximation Theorems in Sobolev spaces) and the specific optimization challenges (loss landscapes, convergence properties) associated with physics-informed training.

- Implement transfer learning approaches for domain adaptation in physical systems and apply few-shot learning techniques for rapid adaptation to new physical regimes with limited data.

- Implement and interpret methods for uncertainty quantification (e.g., Bayesian Neural Networks, Gaussian Processes) and data-driven discovery of governing differential equations (e.g., SINDy) within scientific applications.

- Critically assess the applicability, performance characteristics (accuracy, convergence, cost), and limitations of various SciML techniques in comparison to traditional numerical methods for solving differential equations.

- Effectively employ deep learning libraries (PyTorch and JAX) leveraging automatic differentiation and best practices for reproducible SciML research and development, culminating in substantial project implementations.

# Course Modules

The following 12 modules provide a comprehensive technical foundation in SciML:

### Module 1: Foundations of Machine Learning for Scientific Computing
Introduction to neural networks, optimization algorithms, and automatic differentiation for scientific applications. Covers Universal Approximation Theorem, gradient descent variants, and regularization techniques with emphasis on physics-based applications.

### Module 2: Mathematical Foundations for SciML
Essential mathematical concepts including function spaces (Hilbert, Banach, Sobolev), differential equations theory, numerical analysis fundamentals, and inverse problems formulation.

### Module 3: Physics-Informed Neural Networks (PINNs)
Comprehensive study of PINNs including mathematical formulation, boundary condition enforcement, training strategies, and advanced variants. Applications to forward and inverse ODE/PDE problems.

### Module 4: Neural Ordinary Differential Equations (Neural ODEs)
Continuous-depth neural networks through ODE parameterization, adjoint method for memory-efficient backpropagation, and applications to time-series modeling and latent dynamics learning.

### Module 5: Differentiable Programming and Physics Simulation
End-to-end differentiation through physics simulators, implementation of differentiable time-stepping schemes, and applications to inverse problems and parameter estimation.

### Module 6: Operator Learning I: DeepONet and Extensions
DeepONet architectures for learning function-to-function mappings, Universal Approximation Theorem for operators, and Physics-Informed DeepONet with multi-fidelity learning approaches.

### Module 7: Operator Learning II: Fourier Neural Operators and Advanced Methods
Fourier Neural Operators (FNOs) with spectral methods and advanced neural operators (Basis-to-Basis operator learning based on function encoders).

### Module 8: Graph Neural Networks (GNNs) for Scientific Simulation
Physical systems as graphs, message passing neural networks, symmetries and equivariance, with applications to particle-based systems and fluid dynamics.

### Module 9: Transfer Learning and Few-Shot Learning in SciML
Domain adaptation across physical regimes, meta-learning for new physical systems, multi-task learning, and active learning for optimal experimental design.

### Module 10: Transformers in Scientific Machine Learning
Adaptation of transformer architectures for scientific data, spatio-temporal modeling with attention mechanisms and applications.

### Module 11: Equation Discovery and Symbolic Methods
Sparse Identification of Nonlinear Dynamics (SINDy), symbolic regression, physics-guided discovery, and implementation workflows for discovering governing equations from data.

### Module 12: Advanced Generative Models and Uncertainty Quantification
Normalizing flows, Bayesian Neural Networks, Gaussian Processes, and physics-constrained VAEs for uncertainty quantification in scientific applications.

## Instructor

**Krishna Kumar**  
Dr. Krishna Kumar is an Assistant Professor at the University of Texas at Austin. His research is
at the intersection of AI/ML, geotechnical engineering, and robotics. He directs a $7M
NSF-funded national ecosystem for AI integration in civil engineering and received an NSF
CAREER Award in 2024. His research involves developing differentiable simulations, graph
neural networks and numerical methods for understanding natural hazards. As an educator, he
received the Dean's Award for Outstanding Teaching at UT Austin and runs coding clubs at
Austin Public Libraries teaching AI/ML robotics to children ages 7-12.

## License

This work is licensed under the MIT License.