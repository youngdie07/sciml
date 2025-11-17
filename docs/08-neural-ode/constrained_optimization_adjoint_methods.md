# Constrained Optimization and Adjoint Methods

## Geometric Interpretation: Finding the Closest Point

Geometrically, you are looking for the point (x, y) on the line x+y=1 that is closest to the origin (0, 0).

### The Function: f(x,y) = x² + y²

The function f(x,y) = x² + y² represents the square of the distance from any point (x, y) to the origin (0, 0).

If we set this function equal to a constant k, we get x² + y² = k. These are the level curves of the function, which are a family of concentric circles centered at the origin with a radius of √k.

Minimizing f(x,y) is the same as finding the circle with the smallest possible radius √k.

### The Constraint: x + y = 1

This equation represents a straight line in the 2D plane. It passes through the points (1, 0) and (0, 1).

### The Solution: Finding the Tangent Point

The problem is to find a point that is:

- On the line x + y = 1
- On the smallest possible circle x² + y² = k

Imagine "growing" a circle from the origin. The very first point it "touches" on the line x + y = 1 will be the minimum. At this exact point, the circle and the line will be tangent.

This point of tangency is also the point on the line that is closest to the origin. This happens where the line from the origin to the point is perpendicular to the constraint line.

**Finding the perpendicular:**

- **Slope of the line**: The line x + y = 1 (or y = -x + 1) has a slope of -1
- **Slope of the perpendicular**: A line perpendicular to it must have the negative reciprocal slope, which is m = -1/(-1) = 1
- **Find the point**: We need the point that is on both the constraint line (x + y = 1) and the perpendicular line passing through the origin (y = x)

Substitute y = x into the constraint:

```
x + x = 1
2x = 1
x = 1/2
```

Since y = x, we have y = 1/2.

**The minimum occurs at the point (1/2, 1/2), which is the point on the line x + y = 1 closest to the origin.**

---

## 1. Naïve (Substitution) Method

This method involves using the constraint to reduce the problem from two variables to one.

### Steps:

**Solve the constraint**: The constraint is x + y = 1. We can easily solve for y:

```
y = 1 - x
```

**Substitute into the function**: Now, substitute this expression for y into the function f(x,y) = x² + y²:

```
f(x) = x² + (1-x)²
```

**Find the minimum of the new function**: Expand the function and find the derivative with respect to x.

```
f(x) = x² + (1 - 2x + x²)
f(x) = 2x² - 2x + 1
```

To find the minimum, set the derivative f'(x) to zero:

```
f'(x) = 4x - 2
4x - 2 = 0
4x = 2
x = 1/2
```

(To confirm it's a minimum, the second derivative f''(x) = 4, which is positive.)

**Find the corresponding y value**: Use the constraint equation y = 1 - x:

```
y = 1 - (1/2)
y = 1/2
```

**The minimum occurs at the point (1/2, 1/2).**

---

## 2. Lagrange Multiplier Method

This method introduces a new variable, λ (lambda), to set up a new system of equations.

### Steps:

**Define the functions:**

- Function to minimize: f(x,y) = x² + y²
- Constraint (set to zero): g(x,y) = x + y - 1 = 0

**Set up the Lagrangian function**: The Lagrangian L is defined as L(x,y,λ) = f(x,y) - λ·g(x,y).

```
L(x,y,λ) = x² + y² - λ(x + y - 1)
```

**Find the gradient of L and set it to zero**: We take the partial derivative with respect to x, y, and λ and set each one equal to zero.

1. ∂L/∂x = 2x - λ = 0  ⟹  2x = λ
2. ∂L/∂y = 2y - λ = 0  ⟹  2y = λ
3. ∂L/∂λ = -(x + y - 1) = 0  ⟹  x + y = 1

**Solve the system of equations:**

- From (1) and (2), we see that 2x = λ and 2y = λ
- This immediately tells us that 2x = 2y, which simplifies to **x = y**
- Now, substitute this result (x = y) into equation (3):

```
x + x = 1
2x = 1
x = 1/2
```

Since x = y, it follows that y = 1/2.

**Both methods yield the same solution: the function f(x,y) is minimized at the point (1/2, 1/2), subject to the constraint x + y = 1.**

---

## PDE-Constrained Optimization: The 1D Heat Rod Problem

This is a problem in **PDE-Constrained Optimization**, which is the formal name for minimizing a cost function subject to a Partial Differential Equation (or, in this 1D case, an Ordinary Differential Equation). The Lagrange Multiplier method, generalized to function spaces, is the most efficient way to solve it, and its "multiplier" is the **Adjoint variable**.

### 1. The 1D Heat Rod and the Inverse Problem

#### The Forward Problem: State Equation

The equation

```
-d²u/dx² = s(x)  for x ∈ [0,1]
```

with boundary conditions

```
u(0) = u(1) = 0
```

describes the steady-state temperature distribution u(x) in a 1D rod of unit length.

- The term -d²u/dx² represents the negative of the rate of heat flow change (proportional to heat conduction)
- s(x) is the **source function** (or heat generation rate) at each point x
- The boundary conditions u(0) = u(1) = 0 mean the ends of the rod are held at zero temperature (Dirichlet boundary conditions)

#### The Inverse Problem

The goal is to find the source function s(x) that produces a resulting temperature profile u(x) closest to a desired target temperature u_target(x).

#### The Cost Function (Objective Function)

The cost function to minimize is:

```
J(u) = (1/2) ∫₀¹ (u - u_target)² dx
```

The factor of 1/2 is conventional to simplify the derivative. Since u depends on s (through the PDE), we are effectively minimizing J(s). **The control variable is s(x).**

---

### 2. Constrained Optimization using the Lagrangian

#### The Lagrangian L and the Lagrange Multiplier λ

In continuous optimization, the Lagrangian functional L is constructed by adding the constraint (the PDE) multiplied by a function called the **Lagrange Multiplier**, denoted here as λ(x). This function λ(x) is the **Adjoint Variable**.

The Lagrangian is:

```
L(u,s,λ) = J(u) + ∫₀¹ λ(x)·(-d²u/dx² - s) dx
```

#### The Optimality Conditions

To find the minimum, we set the Gâteaux derivative (the functional equivalent of the gradient) of L with respect to each variable (u, s, and λ) equal to zero. This gives a system of three equations:

**1. State Equation (from ∂L/∂λ = 0)**

```
∂L/∂λ = -d²u/dx² - s = 0  ⟹  -d²u/dx² = s
```

This recovers the original PDE constraint.

**2. Adjoint Equation (from ∂L/∂u = 0)**

```
∂L/∂u = 0  ⟹  u - u_target - d²λ/dx² = 0
```

We need to integrate by parts on the λ·(-d²u/dx²) term to move the derivatives from u to λ.

```
∫₀¹ λ(-d²u/dx²) dx = ∫₀¹ (-d²λ/dx²)u dx + Boundary Terms
                     = [-λ(du/dx) + (dλ/dx)u]₀¹
```

Since u(0) = u(1) = 0, the boundary terms disappear. Setting the variation ∂L/∂u to zero yields:

```
(u - u_target) - d²λ/dx² = 0  ⟹  -d²λ/dx² = u - u_target
```

The boundary conditions for λ come from the remaining boundary terms, which must also be zero (since we are free to choose u values for a valid boundary condition). If u(0) = u(1) = 0 is a Dirichlet (essential) boundary condition, the adjoint variable must satisfy the homogeneous boundary condition:

```
λ(0) = λ(1) = 0
```

**3. Gradient Equation (from ∂L/∂s = 0)**

```
∂L/∂s = -λ = 0
```

This is the gradient of the Lagrangian with respect to the control variable s(x).

```
∂L/∂s = dJ/ds - λ = 0  ⟹  dJ/ds = λ
```

The functional derivative is:

```
δL/δs = -λ(x)
```

---

### 3. Gradients for Inverse Problems (Adjoint Method)

#### What is the Adjoint λ(x)?

The **Adjoint variable** λ(x) is the Lagrange multiplier for a PDE constraint. Physically, it can be interpreted as the sensitivity of the cost function J to perturbations in the State Equation.

#### What are we doing when we say we want to get gradients for inverse?

We are finding the **sensitivity of the cost function J with respect to the control variable s(x)**. This gradient δJ/δs tells us the direction to change s(x) to most rapidly decrease the cost J. This gradient is required by iterative optimization algorithms (like gradient descent or L-BFGS) to find the optimal s(x).

The result from the Lagrange Multiplier method (Equation 3) is the crucial identity:

```
δJ/δs = -λ(x)
```

#### Why the Adjoint Method is Faster (Solving for the Inverse)

The adjoint method is exceptionally efficient for inverse problems where the control variable s is a function, meaning it has a very large number of discrete parameters (e.g., N grid points).

**Naïve Method (Direct Differentiation)**: Requires N+1 simulations.

- One Forward Solve to get u
- N additional Forward Solves, one for each small perturbation to a single parameter sᵢ, to estimate the gradient component ∂J/∂sᵢ using finite differences

**Adjoint Method**: Requires only two simulations, regardless of the number of parameters N.

- **Forward Solve**: Solve the State Equation for u(x) using the current guess s(x)
- **Adjoint Solve**: Solve the Adjoint Equation for λ(x), using u(x) as the "source" term
- **Gradient Calculation**: The gradient is then simply δJ/δs = -λ(x)

Since the cost of solving the Adjoint Equation is comparable to the cost of solving the State Equation, the total computational complexity is reduced from **O(N)** Forward Solves to **O(1)** Forward Solves (or two total solves), making it dramatically faster for large N.

---

### Summary of Steps to Get the Gradient δJ/δs

**Step 1: Forward Step**

Given the current guess s(x), solve the State Equation for u(x):

```
-d²u/dx² = s(x)  with  u(0) = u(1) = 0
```

**Step 2: Adjoint Step**

Solve the Adjoint Equation for λ(x):

```
-d²λ/dx² = u(x) - u_target(x)  with  λ(0) = λ(1) = 0
```

The source term is the misfit.

**Step 3: Gradient Step**

The gradient of the cost function with respect to the source is simply the negative of the adjoint variable:

```
δJ/δs(x) = -λ(x)
```

This gradient is then used to update the source s(x) in an optimization scheme:

```
s_new = s_old - α·(δJ/δs)
```

where α is the step size.

---

## The Adjoint Equation

The adjoint equation for λ(x) is derived from the optimality condition where the derivative of the Lagrangian (L) with respect to the state variable u is set to zero, ∂L/∂u = 0.

### The Complete Adjoint Equation

```
-d²λ/dx² = u(x) - u_target(x)
```

### Breakdown of the Adjoint Equation

**Left Hand Side (LHS)**: -d²λ/dx²

This represents the differential operator acting on the adjoint variable λ. For self-adjoint problems (like this one involving d²/dx²), the adjoint operator is often the same as the original operator.

**Right Hand Side (RHS)**: u(x) - u_target(x)

This is the source term for the adjoint equation. It represents the **misfit** or the difference between the resulting temperature profile u(x) (from the current guess of the source s(x)) and the desired target profile u_target(x). This misfit drives the adjoint solution.

### Adjoint Boundary Conditions

Since the original state variable u(x) had homogeneous Dirichlet (fixed) boundary conditions:

```
u(0) = 0  and  u(1) = 0
```

the adjoint variable λ(x) must also have the corresponding homogeneous boundary conditions:

```
λ(0) = 0  and  λ(1) = 0
```

### Purpose in the Inverse Problem

Solving this adjoint equation gives you the function λ(x), which is immediately the gradient of your cost function J with respect to the control variable s(x):

```
δJ/δs(x) = -λ(x)
```

This relationship is what makes the adjoint method so efficient for inverse problems.
