Required Changes

1. Data Structure (MEDIUM difficulty)
Files: main.py, src/runner.py
Change wall representation from (n_walls, 4) to (n_walls, 5) adding c parameter: [x1, y1, x2, y2, c]
Update all wall initialization code
Maintain backward compatibility or migrate existing wall definitions

2. Force Computation (HIGH difficulty)
Files: src/forces.py Currently uses point_to_segment_distance() which calculates distance from a point to a straight line segment. You'd need:
New function: point_to_curve_distance(px, py, x1, y1, x2, y2, c)
For c=0: use existing straight-line logic
For c≠0: calculate distance to circular arc
Find closest point on arc to particle center
Compute normal direction from arc (perpendicular to arc tangent)
This is geometrically complex: requires solving for closest point on a circular arc
Normal direction calculation changes: perpendicular to arc tangent at closest point (not trivial)

3. Wall Intersection Detection (VERY HIGH difficulty)
Files: src/physics_utils.py Currently uses line_segments_intersect() to check if particle-particle paths cross walls. You'd need:
New function: line_intersects_curved_wall(p1, p2, wall_x1, wall_y1, wall_x2, wall_y2, c)
For c=0: use existing line-segment intersection
For c≠0: check if line segment intersects circular arc
Requires solving quadratic equations for line-circle intersection
Then checking if intersection points lie within arc bounds
This affects:
line_intersects_any_wall() physics_utils.py:100
particles_separated_by_wall_periodic() physics_utils.py:134
Critical: used to prevent forces through walls in periodic boundaries

4. Visualization (EASY TO MEDIUM difficulty)
Files: src/visualization.py Currently draws walls as straight lines:
ax.plot([walls[i, 0], walls[i, 2]], [walls[i, 1], walls[i, 3]])
You'd need:
Use your parametric_curve() function to generate arc points
Plot the curved path instead
Adjust line thickness/rendering for curved segments
This is the easiest part since circles.py already provides the solution

5. Testing (MEDIUM difficulty)
Files: tests/test_forces.py, tests/test_physics_utils.py Would need to:
Add tests for curved wall forces
Add tests for curved wall intersection detection
Test edge cases: c=0 (straight), c=±1 (semicircle), large |c|
Verify forces have correct direction (perpendicular to arc)
Key Mathematical Challenges
Point-to-arc distance: Not as simple as point-to-line. Need to:
Project point onto circle
Check if projection lies within arc bounds
If not, distance is to nearest arc endpoint
Arc normal calculation: Normal direction is perpendicular to arc tangent at closest point, requiring:
Find tangent vector to circle at closest point
Rotate 90° for normal
Line-arc intersection: Requires:
Solve for line-circle intersection (quadratic equation)
Filter solutions to only those within arc angular bounds
Handle periodic boundary wraparound
Numba Compatibility Concern
All geometry functions use @njit(fastmath=True) for performance. The curved wall logic must also be numba-compatible, which limits:
No Python objects/classes
No matplotlib functions in force calculations
Must use only numpy operations

Recommendation
Difficulty ranking by component:
Visualization: ⭐⭐ (Easy - already have the curve generator)
Data structure: ⭐⭐⭐ (Medium - straightforward but touches many files)
Force computation: ⭐⭐⭐⭐ (High - complex geometry, normal calculations)
Wall intersection: ⭐⭐⭐⭐⭐ (Very High - hardest part, critical for correctness)
Testing: ⭐⭐⭐ (Medium - need comprehensive coverage)
Overall: This is a substantial refactor affecting core physics calculations. The hardest parts are the geometric algorithms for curved wall interactions, not the data structure changes.