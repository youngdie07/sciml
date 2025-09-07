# L1 and L2 Regularization

## The Problem: Overfitting

Neural networks with too many parameters memorize training data instead of learning patterns. **Regularization** adds a penalty to prevent this:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{R}(\theta)$$

## L1 vs L2: The Geometry Tells the Story

<div id="regularization-viz"></div>

The key difference between L1 and L2 regularization lies in their geometry:

- **L1 norm**: $|\theta_1| + |\theta_2| = c$ creates a **diamond**
- **L2 norm**: $\theta_1^2 + \theta_2^2 = c$ creates a **circle**

### Why These Shapes?

**L1 Diamond:** The constraint $|\theta_1| + |\theta_2| = c$ creates four linear boundaries:
- Quadrant I: $\theta_1 + \theta_2 = c$ (slope = -1)
- Quadrant II: $-\theta_1 + \theta_2 = c$ (slope = +1)  
- Quadrant III: $-\theta_1 - \theta_2 = c$ (slope = -1)
- Quadrant IV: $\theta_1 - \theta_2 = c$ (slope = +1)

These connect at corners *exactly on the axes* at $(±c, 0)$ and $(0, ±c)$.

**L2 Circle:** The constraint $\theta_1^2 + \theta_2^2 = c$ is simply a circle with radius $\sqrt{c}$.

### The Crucial Insight: Corners = Sparsity

When we optimize:
1. Cost function contours expand from the unconstrained optimum
2. They first touch the constraint boundary
3. **L1**: Often hits at a corner → one parameter becomes exactly zero → **sparse solution**
4. **L2**: Hits smooth boundary → parameters shrink but stay non-zero → **dense solution**

## Mathematical Details

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Penalty | $\sum\|\theta_i\|$ | $\sum\theta_i^2$ |
| Gradient | $\text{sign}(\theta_i)$ | $2\theta_i$ |
| Effect | Forces weights to 0 | Shrinks weights uniformly |
| Use when | Many irrelevant features | All features matter |

The gradient difference is key:
- **L1**: Constant force regardless of weight size → can push to exactly zero
- **L2**: Force proportional to weight → diminishes near zero

## Implementation

```python
def regularization_loss(weights, reg_type='l2', lambda_reg=0.01):
    if reg_type == 'l1':
        return lambda_reg * sum(torch.abs(w).sum() for w in weights)
    elif reg_type == 'l2':
        return lambda_reg * sum((w**2).sum() for w in weights)
```

## When to Use Which?

**L1 Regularization:**
- Feature selection needed
- Interpretability important
- Sparse data

**L2 Regularization:**
- Multicollinearity present
- Need stable predictions
- All features relevant

**Elastic Net** combines both: $\alpha\sum|\theta_i| + (1-\alpha)\sum\theta_i^2$

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
    const width = 800;
    const height = 400;
    const margin = {top: 40, right: 40, bottom: 60, left: 60};
    const plotWidth = (width - margin.left - margin.right) / 2;
    const plotHeight = height - margin.top - margin.bottom;
    
    const svg = d3.select("#regularization-viz")
        .append("svg")
        .attr("width", width)
        .attr("height", height);
    
    // Create two plots
    const l1Group = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
    
    const l2Group = svg.append("g")
        .attr("transform", `translate(${margin.left + plotWidth + 60}, ${margin.top})`);
    
    // Scales
    const xScale = d3.scaleLinear().domain([-2, 2]).range([0, plotWidth]);
    const yScale = d3.scaleLinear().domain([-2, 2]).range([plotHeight, 0]);
    
    // Add axes and grid for both plots
    [l1Group, l2Group].forEach(group => {
        // Grid
        group.append("g")
            .attr("class", "grid")
            .call(d3.axisBottom(xScale).tickSize(plotHeight).tickFormat(""))
            .style("stroke-dasharray", "3,3").style("opacity", 0.3);
        
        group.append("g")
            .attr("class", "grid")
            .call(d3.axisLeft(yScale).tickSize(-plotWidth).tickFormat(""))
            .style("stroke-dasharray", "3,3").style("opacity", 0.3);
        
        // Axes
        group.append("line")
            .attr("x1", 0).attr("y1", yScale(0))
            .attr("x2", plotWidth).attr("y2", yScale(0))
            .style("stroke", "black").style("stroke-width", 2);
        
        group.append("line")
            .attr("x1", xScale(0)).attr("y1", 0)
            .attr("x2", xScale(0)).attr("y2", plotHeight)
            .style("stroke", "black").style("stroke-width", 2);
    });
    
    // Titles
    l1Group.append("text")
        .attr("x", plotWidth / 2).attr("y", -10)
        .attr("text-anchor", "middle")
        .style("font-weight", "bold")
        .text("L1: |θ₁| + |θ₂| ≤ c");
    
    l2Group.append("text")
        .attr("x", plotWidth / 2).attr("y", -10)
        .attr("text-anchor", "middle")
        .style("font-weight", "bold")
        .text("L2: θ₁² + θ₂² ≤ c");
    
    function updateConstraints(lambda) {
        const c = 1.0 / lambda;  // constraint size
        
        // Clear previous (but not labels)
        l1Group.selectAll(".constraint, .corner, .optimum, .contour, .intersection").remove();
        l2Group.selectAll(".constraint, .corner, .optimum, .contour, .intersection").remove();
        
        // Draw L1 diamond
        const diamond = [[0, c], [c, 0], [0, -c], [-c, 0], [0, c]]
            .map(d => [xScale(d[0]), yScale(d[1])]);
        
        l1Group.append("path")
            .attr("class", "constraint")
            .datum(diamond)
            .attr("d", d3.line())
            .attr("fill", "#2196F3")
            .attr("fill-opacity", 0.15)
            .attr("stroke", "#2196F3")
            .attr("stroke-width", 2);
        
        // Mark corners on L1
        [[0, c], [c, 0], [0, -c], [-c, 0]].forEach(corner => {
            l1Group.append("circle")
                .attr("class", "corner")
                .attr("cx", xScale(corner[0]))
                .attr("cy", yScale(corner[1]))
                .attr("r", 5)
                .attr("fill", "#FF5722");
        });
        
        // Draw L2 circle
        const circle = [];
        for (let angle = 0; angle <= 2 * Math.PI; angle += 0.05) {
            circle.push([xScale(Math.sqrt(c) * Math.cos(angle)), 
                        yScale(Math.sqrt(c) * Math.sin(angle))]);
        }
        
        l2Group.append("path")
            .attr("class", "constraint")
            .datum(circle)
            .attr("d", d3.line())
            .attr("fill", "#4CAF50")
            .attr("fill-opacity", 0.15)
            .attr("stroke", "#4CAF50")
            .attr("stroke-width", 2);
        
        // Draw cost function contours
        const optimalPoint = [1.2, 1.2];
        
        // Calculate the level that touches the constraint
        // For L1: touches at (0, c) so distance from (1.2, 1.2) to (0, c)
        const l1TouchLevel = Math.abs(1.2 - c) / 0.5;  // accounting for ellipse aspect ratio
        // For L2: find where ellipse touches circle
        const l2TouchLevel = (Math.sqrt(c) - 0.3) / 0.5;
        
        // Draw contours that lead up to the constraint
        const levels = [0.3, 0.6, 0.9].concat([l1TouchLevel, l2TouchLevel].filter(l => l > 0 && l < 2.5));
        levels.sort((a, b) => a - b);
        
        levels.forEach(level => {
            const contour = [];
            for (let angle = 0; angle <= 2 * Math.PI; angle += 0.1) {
                const x = optimalPoint[0] + level * 0.7 * Math.cos(angle);
                const y = optimalPoint[1] + level * 0.5 * Math.sin(angle);
                contour.push([xScale(x), yScale(y)]);
            }
            
            [l1Group, l2Group].forEach(group => {
                group.append("path")
                    .attr("class", "contour")
                    .datum(contour)
                    .attr("d", d3.line())
                    .attr("fill", "none")
                    .attr("stroke", "#ff4444")
                    .attr("stroke-width", 1)
                    .attr("opacity", 0.4);
            });
        });
        
        // Mark optimal points
        [l1Group, l2Group].forEach(group => {
            group.append("circle")
                .attr("class", "optimum")
                .attr("cx", xScale(optimalPoint[0]))
                .attr("cy", yScale(optimalPoint[1]))
                .attr("r", 4)
                .attr("fill", "black");
        });
        
        // Mark intersection points
        l1Group.append("circle")
            .attr("class", "intersection")
            .attr("cx", xScale(0))
            .attr("cy", yScale(c))
            .attr("r", 6)
            .attr("fill", "#4CAF50")
            .attr("stroke", "white")
            .attr("stroke-width", 2);
        
        // Update or create sparse label
        let sparseLabel = l1Group.select(".sparse-label");
        if (sparseLabel.empty()) {
            sparseLabel = l1Group.append("text")
                .attr("class", "sparse-label")
                .style("font-size", "12px")
                .style("font-weight", "bold")
                .text("Sparse!");
        }
        sparseLabel
            .attr("x", xScale(0) - 15)
            .attr("y", yScale(c) - 10);
        
        const l2_intersect = 0.4 * Math.sqrt(c);
        l2Group.append("circle")
            .attr("class", "intersection")
            .attr("cx", xScale(l2_intersect))
            .attr("cy", yScale(l2_intersect * 1.8))
            .attr("r", 6)
            .attr("fill", "#4CAF50")
            .attr("stroke", "white")
            .attr("stroke-width", 2);
        
        // Update or create dense label
        let denseLabel = l2Group.select(".dense-label");
        if (denseLabel.empty()) {
            denseLabel = l2Group.append("text")
                .attr("class", "dense-label")
                .style("font-size", "12px")
                .style("font-weight", "bold")
                .text("Dense");
        }
        denseLabel
            .attr("x", xScale(l2_intersect) + 10)
            .attr("y", yScale(l2_intersect * 1.8) - 10);
    }
    
    // Slider for lambda
    const sliderContainer = d3.select("#regularization-viz")
        .append("div")
        .style("text-align", "center")
        .style("margin-top", "20px");
    
    sliderContainer.append("label")
        .text("Regularization strength λ: ");
    
    const slider = sliderContainer.append("input")
        .attr("type", "range")
        .attr("min", 0.5)
        .attr("max", 3)
        .attr("step", 0.1)
        .attr("value", 1.2)
        .style("width", "300px");
    
    const valueDisplay = sliderContainer.append("span")
        .style("margin-left", "10px")
        .text("1.2");
    
    slider.on("input", function() {
        const lambda = +this.value;
        valueDisplay.text(lambda);
        updateConstraints(lambda);
    });
    
    // Initial draw
    updateConstraints(1.2);
    
    // Key insight box
    d3.select("#regularization-viz")
        .append("div")
        .style("margin-top", "20px")
        .style("padding", "15px")
        .style("background-color", "#f5f5f5")
        .style("border-left", "4px solid #2196F3")
        .html(`
            <strong>Key Insight:</strong> L1's diamond has corners on the axes where parameters are zero. 
            When the cost function (red contours) expands from the unconstrained optimum (black dot), 
            it hits L1's corner → sparse solution. L2's smooth circle → dense solution.
        `);
})();
</script>

## Choosing λ

- **Small λ**: Weak regularization → potential overfitting
- **Large λ**: Strong regularization → potential underfitting  
- **Find optimal λ**: Use cross-validation

The constraint size scales as $c = 1/\lambda$:
- L1: Diamond vertices at $(±1/\lambda, 0)$ and $(0, ±1/\lambda)$
- L2: Circle radius = $\sqrt{1/\lambda}$

## Summary

**L1 creates sparsity** because its diamond constraint has corners on the axes.  
**L2 creates density** because its circular constraint is smooth everywhere.

Choose based on your goal: feature selection (L1) or stable predictions (L2).