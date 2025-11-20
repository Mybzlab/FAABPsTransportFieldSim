# Wall Implementation Documentation

## Overview

Static walls in the FAABP simulation that:
1. Block particle and payload movement via repulsive forces
2. Prevent particle-particle and payload-particle force interactions across walls
3. Support periodic boundaries correctly

---

## 1. Wall Data Structure

### Representation
- **Format**: NumPy array of shape `(n_walls, 4)` containing `[x1, y1, x2, y2]` for each wall segment
- **Type**: `np.float64` for numba compatibility

### Example
```python
box_size = 350
walls = np.array([
    [box_size*0.6, box_size*0.2, box_size*0.6, box_size*0.7],  # Vertical wall
    [box_size*0.3, box_size*0.4, box_size*0.3, box_size*0.9]   # Vertical wall
], dtype=np.float64)

# Or empty for no walls:
walls = np.zeros((0, 4), dtype=np.float64)
```

---

## 2. Implementation

### Geometry Functions ([physics_utils.py](../src/physics_utils.py))

```python
@njit(fastmath=True)
def line_segments_intersect(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y):
    """Check if line segment (p1, p2) intersects with line segment (p3, p4)."""

@njit(fastmath=True)
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calculate minimum distance from point to line segment.
    Returns: (distance, closest_x, closest_y)"""

@njit(fastmath=True)
def line_intersects_any_wall(p1_x, p1_y, p2_x, p2_y, walls):
    """Check if line segment intersects any wall."""

@njit(fastmath=True)
def particles_separated_by_wall_periodic(pos_i, pos_j, walls, box_size):
    """Check if a wall blocks the periodic shortest path between two particles.

    Critical for periodic boundaries: checks the actual shortest periodic path,
    not the straight-line distance."""
```

### Wall Forces ([forces.py](../src/forces.py))

```python
@njit(fastmath=True)
def compute_wall_forces(pos, radius, walls, stiffness):
    """Compute repulsive forces from all walls on a particle/payload.

    For each wall:
    1. Calculate distance from particle center to wall segment
    2. If distance < radius: particle is colliding
    3. Apply force: F = stiffness * (radius - distance) * normal_direction
    """
```

### Force Integration ([simulation.py](../src/simulation.py))

```python
@njit(fastmath=True)
def compute_all_forces(positions, payload_pos, radii, payload_radius,
                       stiffness, n_particles, box_size, walls):
    """Compute all forces with wall interactions.

    - Particle-particle forces: blocked across walls using periodic check
    - Particle-payload forces: blocked across walls using periodic check
    - Particle-wall forces: repulsive collision forces
    - Payload-wall forces: repulsive collision forces
    """
```

---

## 3. Periodic Boundary Wall Blocking

### The Challenge
Particle interactions use the shortest periodic path, so wall blocking must check the same path.

**Example**:
- Particle A at (10, 50), Particle B at (340, 50)
- Box size = 350
- Periodic distance = 20 (wraps around)
- Straight-line distance = 330

If wall check uses straight line (330 units), it may miss walls that block the actual periodic path (20 units).

### The Solution

```python
@njit(fastmath=True)
def particles_separated_by_wall_periodic(pos_i, pos_j, walls, box_size):
    # 1. Compute periodic shortest displacement
    r_ij = compute_minimum_distance(pos_i, pos_j, box_size)

    # 2. Get endpoint following periodic path
    pos_j_periodic = pos_i + r_ij

    # 3. Check if any wall intersects THIS path
    return line_intersects_any_wall(pos_i[0], pos_i[1],
                                     pos_j_periodic[0], pos_j_periodic[1],
                                     walls)
```

Used in:
- Particle-particle forces ([simulation.py:66](../src/simulation.py))
- Particle-payload forces ([simulation.py:28](../src/simulation.py))

---

## 4. Visualization

Walls are rendered as black lines in animations ([visualization.py](../src/visualization.py)).

```python
# Walls drawn with:
for i in range(walls.shape[0]):
    ax.plot([walls[i, 0], walls[i, 2]],
            [walls[i, 1], walls[i, 3]],
            color='black', linewidth=4, zorder=10)
```

---

## 5. Testing

### Unit Tests
Located in [tests/](../tests/)

| Test File | Coverage | Status |
|-----------|----------|--------|
| [test_physics_utils.py](../tests/test_physics_utils.py) | Line intersection, point-to-segment distance, periodic boundaries, wall intersections | ✓ 22/23 passing |
| [test_forces.py](../tests/test_forces.py) | Repulsive forces, wall forces, cell lists | ✓ 11/11 passing |
| [test_simulation.py](../tests/test_simulation.py) | Orientation updates, force computation | ✓ 6/6 passing |
| [test_main.py](../tests/test_main.py) | Integration tests with walls, periodic boundaries | ✓ 3/3 passing |

### Visual Verification
- Walls appear as black lines
- Particles/payload bounce off walls (no clipping)
- Forces blocked across walls

---

## 6. Performance

### Computational Complexity
- **Wall collision forces**: O(N × W) where N = particles, W = walls
- **Wall blocking checks**: O(N × neighbors × W) during force computation
- All functions are `@njit` compiled (zero Python overhead)

### Optimization
- Early rejection for parallel lines in intersection tests
- Wall forces only computed when distance < radius
- No memory allocations in hot loops

---

## 7. Edge Cases

| Case | Handling |
|------|----------|
| Empty walls `(0, 4)` | Loops over 0 walls, no forces/checks applied |
| Zero-length wall | Treated as point, distance calculated correctly |
| Particle exactly on wall | Falls back to wall perpendicular direction |
| Periodic boundaries | Uses `particles_separated_by_wall_periodic()` |
