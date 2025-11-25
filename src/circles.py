import numpy as np
import matplotlib.pyplot as plt

def parametric_curve(p1, p2, c=None, K=None, num_points=100):
    """
    Create a curve between two points with curvature parameter.

    Parameters:
    - p1, p2: endpoints as (x, y) tuples
    - c: curvature parameter (chord-normalized)
      * c = 0: straight line
      * c = 1: semicircle (bulging in one direction)
      * c = -1: semicircle (bulging in opposite direction)
    - K: standard curvature (K = 1/R, alternative to c)
      * K = 0: straight line
      * K > 0: curves one direction
      * K < 0: curves opposite direction
    - num_points: number of points to generate

    Note: Provide either c OR K, not both.

    Returns:
    - x, y: arrays of coordinates
    """
    if c is not None and K is not None:
        raise ValueError("Provide either 'c' or 'K', not both")

    if c is None and K is None:
        raise ValueError("Must provide either 'c' or 'K' parameter")

    p1 = np.array(p1)
    p2 = np.array(p2)

    # Convert K to c if K was provided
    if K is not None:
        chord_length = np.linalg.norm(p2 - p1)
        c = K * chord_length / 2
    
    # Vector from p1 to p2
    d = p2 - p1
    distance = np.linalg.norm(d)
    
    if abs(c) < 1e-10:  # Straight line case
        t = np.linspace(0, 1, num_points)
        points = p1[:, None] + t * d[:, None]
        return points[0], points[1]
    
    # For |c| = 1, we want a semicircle (radius = distance/2, h = 0)
    # For a circular arc: h = radius - sqrt(radius^2 - (distance/2)^2)
    # Solving for radius given c: radius = distance / (2 * abs(c))
    # This gives h = distance/(2|c|) - distance/(2|c|) * sqrt(1 - c^2)

    midpoint = (p1 + p2) / 2

    # Perpendicular direction (rotated 90 degrees)
    perp = np.array([-d[1], d[0]]) / distance

    # Calculate radius: for |c|=1 -> semicircle, radius = distance/2
    radius = distance / (2 * abs(c))

    # Perpendicular offset from midpoint to center
    # h = sqrt(radius^2 - (distance/2)^2)
    h = np.sqrt(radius**2 - (distance/2)**2) * np.sign(c)

    # Center of circle
    center = midpoint + h * perp

    # Calculate the angle subtended by the arc
    # Angle from center to p1
    v1 = p1 - center
    v2 = p2 - center

    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    # Ensure we go the right direction
    angle_diff = angle2 - angle1
    if c > 0:
        if angle_diff < 0:
            angle_diff += 2 * np.pi
    else:
        if angle_diff > 0:
            angle_diff -= 2 * np.pi

    # Generate points along the arc
    angles = np.linspace(angle1, angle1 + angle_diff, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    
    return x, y

if __name__ == "__main__":
    # Example usage
    p1 = (0, 0)
    p2 = (4, 0)

    plt.figure(figsize=(12, 8))

    # Test different values of c
    c_values = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(c_values)))

    for c, color in zip(c_values, colors):
        x, y = parametric_curve(p1, p2, c)
        plt.plot(x, y, label=f'c = {c}', linewidth=2, color=color)

    plt.plot(*p1, 'ro', markersize=10, label='Endpoints')
    plt.plot(*p2, 'ro', markersize=10)

    p3 = (0, 0)
    p4 = (2, 2)
    c = np.sin(-np.pi / 4)
    x, y = parametric_curve(p3, p4, c)
    plt.plot(x, y, label=f'c = {c}', linewidth=2)

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Parametric Curves with Different Curvature Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()