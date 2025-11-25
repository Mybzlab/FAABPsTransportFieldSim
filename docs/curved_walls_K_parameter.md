# Using Standard Curvature K=1/R for Curved Walls

## Quick Start

You can define curved walls using either:
1. **c parameter** (chord-normalized): value between -1 and 1
2. **K parameter** (standard curvature): K = 1/R where R is radius

## Method 1: Using c (chord-normalized)

```python
import numpy as np

WALLS = np.array([
    [x1, y1, x2, y2, c],
    # c = 0: straight
    # c = ±1: semicircle
])
```

## Method 2: Using K (standard curvature)

### In main.py (when defining walls):

```python
import numpy as np

# Use the K_to_c helper function
R = 50  # radius
K = 1/R  # curvature

WALLS = np.array([
    [0, 0, 200, 0, K_to_c(0, 0, 200, 0, K)],  # curved wall
    [0, 0, 0, 200, 0],  # straight wall (c=0)
])
```

### In visualization code (parametric_curve):

```python
from src.circles import parametric_curve

# Method A: Using c
x, y = parametric_curve((0, 0), (100, 0), c=0.5)

# Method B: Using K
K = 1/50  # radius = 50
x, y = parametric_curve((0, 0), (100, 0), K=K)
```

## Conversion Formula

```
c = K × chord_length / 2

where:
- K = 1/R (standard curvature)
- R = radius of the circular arc
- chord_length = distance between endpoints
```

## Examples

### Example 1: Wall with radius 100

```python
# Wall from (0,0) to (150,0) with radius 100
x1, y1, x2, y2 = 0, 0, 150, 0
R = 100
K = 1/R

WALLS = np.array([
    [x1, y1, x2, y2, K_to_c(x1, y1, x2, y2, K)]
])
```

### Example 2: Multiple walls with different radii

```python
# Three walls with different curvatures
WALLS = np.array([
    # Gentle curve (R=200)
    [0, 0, 100, 0, K_to_c(0, 0, 100, 0, 1/200)],

    # Sharper curve (R=50)
    [0, 100, 100, 100, K_to_c(0, 100, 100, 100, 1/50)],

    # Straight wall
    [100, 0, 100, 100, 0],
])
```

## Key Differences: c vs K

| Aspect | c (chord-normalized) | K (standard curvature) |
|--------|---------------------|----------------------|
| **Range** | -1 ≤ c ≤ 1 | Any value |
| **Interpretation** | Fraction of semicircle | 1/radius |
| **Same value** | Same relative curvature | Same absolute radius |
| **Distance dependent** | No | Yes (K = 2c/chord_length) |

### When to use c:
- Simple cases (straight, semicircle)
- Relative curvature matters more than absolute radius
- Want bounded parameter

### When to use K:
- You know the desired radius R
- Want consistent radius across different chord lengths
- Coming from differential geometry background

## Notes

- Internally, everything uses `c` parameter
- `K` is converted to `c` automatically when provided
- For straight walls: both `c=0` and `K=0` work
- Sign indicates direction: positive/negative bulge opposite sides
