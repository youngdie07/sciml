import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Configure both subplots
for ax in [ax1, ax2]:
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r'$\theta_1$', fontsize=14)
    ax.set_ylabel(r'$\theta_2$', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')

# ============= L1 Regularization (left plot) =============
ax1.set_title('L1 Regularization', fontsize=16, fontweight='bold')

# Draw L1 constraint (diamond)
diamond = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]])
ax1.plot(diamond[:, 0], diamond[:, 1], 'b-', linewidth=2)
ax1.fill(diamond[:, 0], diamond[:, 1], alpha=0.1, color='blue')

# Unconstrained optimum
theta_hat = [1.4, 1.4]
ax1.plot(theta_hat[0], theta_hat[1], 'ko', markersize=8)
ax1.text(theta_hat[0] + 0.1, theta_hat[1] + 0.1, r'$\hat{\theta}$', fontsize=14)

# L1 parameters - YOUR SPECIFIED VALUES
rotation_L1_deg = 0  # L1 angle = 0 degrees
scale_L1 = 1.75      # L1 scale = 1.75
a_ratio = 0.9
b_ratio = 0.5

# Convert rotation to radians
rotation_L1 = np.radians(rotation_L1_deg)
cos_r = np.cos(rotation_L1)
sin_r = np.sin(rotation_L1)

angles = np.linspace(0, 2*np.pi, 500)

def point_on_ellipse(angle, scale, center, cos_r, sin_r, a, b):
    """Get a point on the rotated ellipse"""
    x_e = scale * a * np.cos(angle)
    y_e = scale * b * np.sin(angle)
    x_rot = center[0] + x_e * cos_r - y_e * sin_r
    y_rot = center[1] + x_e * sin_r + y_e * cos_r
    return x_rot, y_rot

# Draw multiple ellipse levels
levels_L1 = [0.4, 0.7, 1.0, 1.3, scale_L1]

for i, level in enumerate(levels_L1):
    x_ellipse = level * a_ratio * np.cos(angles)
    y_ellipse = level * b_ratio * np.sin(angles)
    
    # Rotate and translate
    x_rot = theta_hat[0] + x_ellipse * cos_r - y_ellipse * sin_r
    y_rot = theta_hat[1] + x_ellipse * sin_r + y_ellipse * cos_r
    
    # Thicker line for the outermost ellipse
    linewidth = 2 if i == len(levels_L1) - 1 else 1
    ax1.plot(x_rot, y_rot, 'r-', linewidth=linewidth, alpha=0.8)

# Find and mark the closest point to (0, 1)
min_dist = float('inf')
closest_point = None
for angle in angles:
    x, y = point_on_ellipse(angle, scale_L1, theta_hat, cos_r, sin_r, a_ratio, b_ratio)
    dist = np.sqrt((x - 0)**2 + (y - 1)**2)
    if dist < min_dist:
        min_dist = dist
        closest_point = [x, y]

# Mark the intersection/tangent point
if abs(min_dist) < 0.01:  # If very close to touching
    ax1.plot(0, 1, 'go', markersize=10)
    ax1.text(-0.3, 1.2, '(0, 1)', fontsize=12, color='blue')

# Add labels
ax1.text(-2.5, 2.5, r'$\lambda||\theta||_1$', fontsize=14)
ax1.text(-2.5, 2, r'$|\theta_1| + |\theta_2|$', fontsize=12)
ax1.text(2.2, 2.5, r'$Cost(y_i, f_\theta(x_i))$', fontsize=12, style='italic')

# ============= L2 Regularization (right plot) =============
ax2.set_title('L2 Regularization', fontsize=16, fontweight='bold')

# Draw L2 constraint (circle)
circle = Circle((0, 0), 1, fill=False, edgecolor='blue', linewidth=2)
ax2.add_patch(circle)
circle_fill = Circle((0, 0), 1, alpha=0.1, color='blue')
ax2.add_patch(circle_fill)

# Unconstrained optimum
ax2.plot(theta_hat[0], theta_hat[1], 'ko', markersize=8)
ax2.text(theta_hat[0] + 0.1, theta_hat[1] + 0.1, r'$\hat{\theta}$', fontsize=14)

# L2 parameters - YOUR SPECIFIED VALUES
rotation_L2_deg = 10.5  # L2 angle = 10.5 degrees
scale_L2 = 1.33         # L2 scale = 1.33

# L2 rotation
rotation_L2 = np.radians(rotation_L2_deg)
cos_r2 = np.cos(rotation_L2)
sin_r2 = np.sin(rotation_L2)

# Find minimum distance point on the ellipse to origin
min_dist = float('inf')
touch_point = None
for angle in angles:
    x, y = point_on_ellipse(angle, scale_L2, theta_hat, cos_r2, sin_r2, a_ratio, b_ratio)
    dist = np.sqrt(x**2 + y**2)
    if dist < min_dist:
        min_dist = dist
        touch_point = [x, y]

# Draw multiple ellipse levels
levels_L2 = [0.4, 0.7, 1.0, scale_L2]

for i, level in enumerate(levels_L2):
    x_ellipse = level * a_ratio * np.cos(angles)
    y_ellipse = level * b_ratio * np.sin(angles)
    
    # Rotate and translate
    x_rot = theta_hat[0] + x_ellipse * cos_r2 - y_ellipse * sin_r2
    y_rot = theta_hat[1] + x_ellipse * sin_r2 + y_ellipse * cos_r2
    
    # Thicker line for the outermost ellipse
    linewidth = 2 if i == len(levels_L2) - 1 else 1
    ax2.plot(x_rot, y_rot, 'r-', linewidth=linewidth, alpha=0.8)

# Mark the tangent point
if touch_point:
    # Normalize to unit circle
    norm = np.sqrt(touch_point[0]**2 + touch_point[1]**2)
    touch_point_normalized = [touch_point[0]/norm, touch_point[1]/norm]
    
    # Plot the intersection point on the circle
    ax2.plot(touch_point_normalized[0], touch_point_normalized[1], 'go', markersize=10)
    
    # Draw red dashed box around the tangent point
    box_size = 0.35
    rect = Rectangle((touch_point_normalized[0] - box_size/2, touch_point_normalized[1] - box_size/2), 
                    box_size, box_size, 
                    fill=False, edgecolor='red', linestyle='--', linewidth=1.5)
    ax2.add_patch(rect)
    
    # Add coordinates text
    ax2.text(touch_point_normalized[0] - 0.5, touch_point_normalized[1] - 0.3, 
            f'({touch_point_normalized[0]:.2f}, {touch_point_normalized[1]:.2f})', 
            fontsize=12, color='black')
    
    print(f"L2 intersection point (green circle): ({touch_point_normalized[0]:.3f}, {touch_point_normalized[1]:.3f})")
    print(f"Distance from ellipse to origin: {min_dist:.4f}")

# Add labels
ax2.text(-2.5, 2.5, r'$\lambda||\theta||_2^2$', fontsize=14)
ax2.text(-2.5, 2, r'$\theta_1^2 + \theta_2^2$', fontsize=12)
ax2.text(2.2, 2.5, r'$Cost(y_i, f_\theta(x_i))$', fontsize=12, style='italic')

plt.tight_layout()

# Save the figure
output_path = '/Users/krishna/courses/CE397-Scientific-MachineLearning/sciml/docs/00-mlp/figs/L1-L2-regularization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {output_path}")

plt.show()