# Figures Needed for UAT Section of MLP Slides

## Figures to Extract from Notebooks

### From uat-theory.ipynb
1. **uat-approximation-progression.png** - Shows 2, 10, 50 neurons approximating sin(πx)
   - Extract from cell 7 visualization
   - Shows error decreasing with width

2. **width-vs-error.png** - Error vs number of neurons plot (linear and log scale)
   - Extract from cell 7 second plot
   - Shows convergence behavior

3. **contradiction-visualization.png** - 9-panel proof by contradiction visualization
   - Extract from cell 13
   - Shows the impossible detector concept

4. **halfspace-isolation.png** - Shows how half-spaces isolate points
   - Extract from cell 13, panels 4-5
   - Demonstrates the geometric argument

5. **derivative-approximation.png** - Neural network approximating f and f'
   - Extract from cell 16
   - Shows Sobolev space approximation

### From mlp.ipynb
1. **shallow-vs-deep-sin100x.png** - Comparison for high-frequency function
   - Extract from shallow vs deep comparison section
   - Shows depth advantage

### From uat-demo.md (screenshots)
1. **relu-interactive-demo.png** - Screenshot of interactive ReLU builder
   - Capture with components selected/deselected
   - Show neural network architecture below

2. **activation-comparison.png** - Screenshot comparing ReLU, Sigmoid, Parabolic
   - Show different target functions
   - Demonstrate non-universality of parabolic

## Figures to Generate

### Conceptual Diagrams
1. **relu-decomposition-sinpi.png** - Clean diagram showing ReLU decomposition
   - Individual ReLU components
   - How they sum to approximate sin(πx)
   - Mathematical notation

2. **bias-breakpoints-detailed.png** - Illustration of bias as breakpoint control
   - Show ReLU activation at different biases
   - Mark breakpoints x = -b/w
   - Connect to piecewise linear approximation

3. **parabolic-failure.png** - Side-by-side comparison
   - Parabolic approximating sin(πx) successfully (2 units)
   - Parabolic failing on step function
   - Emphasize the coincidence vs general failure

## Generation Instructions

### For Notebook Extractions
```python
# Add to notebook cells for high-quality figure export
fig.savefig('figs/figure-name.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
```

### For Interactive Demo Screenshots
1. Open uat-demo.html in browser
2. Set window to 1920x1080 resolution
3. Configure demo to show key features:
   - ReLU: Select 3-4 components
   - Activation: Show parabolic failing on step
4. Use browser screenshot tool or Selenium for consistency

### For Conceptual Diagrams
Use matplotlib or TikZ to create clean, publication-quality diagrams:

```python
# Example for bias-breakpoints diagram
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Show ReLU with different biases
x = np.linspace(-2, 2, 100)
biases = [0.5, 0, -0.5]
weights = [1, 1, 1]

for ax, b, w in zip(axes, biases, weights):
    y = np.maximum(0, w * x + b)
    breakpoint = -b/w
    
    ax.plot(x, y, 'b-', linewidth=2)
    ax.axvline(breakpoint, color='r', linestyle='--', alpha=0.7)
    ax.scatter([breakpoint], [0], color='r', s=100, zorder=5)
    ax.text(breakpoint, -0.5, f'x = {breakpoint:.1f}', 
            ha='center', fontsize=10)
    ax.set_title(f'ReLU(x + {b:.1f})', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 2)

plt.suptitle('Bias Controls Breakpoint Location', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figs/bias-breakpoints-detailed.png', dpi=150, bbox_inches='tight')
```

## Color Scheme
- Primary: #3498DB (blue)
- Secondary: #E74C3C (red)  
- Success: #2ECC71 (green)
- Warning: #F39C12 (orange)
- Neutral: #34495E (dark gray)

## Quality Requirements
- Resolution: 150-300 DPI for print
- Format: PNG with transparent background where appropriate
- Consistent font: Arial/Helvetica
- Clear labels and legends
- High contrast for projector viewing