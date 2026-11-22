"""Render an mp4 of a per-maze-cell particle-count histogram evolving over time.

Each bar is one maze cell; bars are sorted left->right by maze (BFS) distance to the
cell containing the goal. Each video frame is one saved state in the npz.

Maze structure is recovered from the npz itself: 'walls' (segments [x1,y1,x2,y2,c]),
'box_size' and 'goal_position'. Interior straight walls of length == one cell, lying on a
grid line, are wall segments between adjacent cells. Boundary walls and corner arcs are ignored.

Usage:
    python miscellaneous/render_cell_histogram.py run.npz out.mp4 [--grid 10] [--fps 15] [--stride 1]
"""
import argparse
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def recover_blocked_edges(walls, box, G):
    """Return set of frozenset({(r,c),(r2,c2)}) pairs of adjacent cells separated by a wall."""
    cell = box / G
    tol = 1e-3 * cell
    blocked = set()
    for x1, y1, x2, y2, c in np.asarray(walls, dtype=float):
        if abs(c) > 1e-9:
            continue  # corner arc
        if abs(y1 - y2) < tol:  # horizontal wall at y = (r+1)*cell spanning column c
            k = y1 / cell
            if abs(abs(x2 - x1) - cell) > tol or abs(k - round(k)) * cell > tol:
                continue
            k = int(round(k))
            if not 0 < k < G:
                continue  # boundary
            col = int(np.floor(min(x1, x2) / cell + 0.5))
            blocked.add(frozenset({(k - 1, col), (k, col)}))
        elif abs(x1 - x2) < tol:  # vertical wall at x = (c+1)*cell spanning row r
            k = x1 / cell
            if abs(abs(y2 - y1) - cell) > tol or abs(k - round(k)) * cell > tol:
                continue
            k = int(round(k))
            if not 0 < k < G:
                continue
            row = int(np.floor(min(y1, y2) / cell + 0.5))
            blocked.add(frozenset({(row, k - 1), (row, k)}))
    return blocked


def bfs_order(blocked, G, goal_cell):
    """Return (order, dist): cells sorted by BFS distance from goal_cell, and the distance map."""
    dist = {goal_cell: 0}
    order = [goal_cell]
    dq = deque([goal_cell])
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if not (0 <= n[0] < G and 0 <= n[1] < G) or n in dist:
                continue
            if frozenset({(r, c), n}) in blocked:
                continue
            dist[n] = dist[(r, c)] + 1
            order.append(n)
            dq.append(n)
    return order, dist


def render(npz_path, out_path, grid=10, fps=15, stride=1, ymax=None):
    data = np.load(npz_path)
    box = float(data['box_size'])
    goal = np.asarray(data['goal_position'], dtype=float)
    positions = data['positions'][::stride]  # (F, N, 2)
    F, N, _ = positions.shape
    cell = box / grid

    blocked = recover_blocked_edges(data['walls'], box, grid)
    goal_cell = (min(int(goal[1] // cell), grid - 1), min(int(goal[0] // cell), grid - 1))
    order, dist = bfs_order(blocked, grid, goal_cell)
    if len(order) != grid * grid:
        print(f"warning: only {len(order)}/{grid * grid} cells reachable from goal (grid={grid} wrong?)")
    rank = {cell_rc: i for i, cell_rc in enumerate(order)}
    n_bars = len(order)

    # Bin every particle of every frame into its bar index (-1 = outside the grid).
    col = np.clip((positions[..., 0] // cell).astype(int), 0, grid - 1)
    row = np.clip((positions[..., 1] // cell).astype(int), 0, grid - 1)
    rank_grid = np.full((grid, grid), -1, dtype=int)
    for (r, c), i in rank.items():
        rank_grid[r, c] = i
    bar_idx = rank_grid[row, col]  # (F, N)
    counts = np.stack([np.bincount(b[b >= 0], minlength=n_bars) for b in bar_idx])  # (F, n_bars)

    if ymax is None:
        ymax = counts.max() * 1.05

    d = np.array([dist[c] for c in order])
    colors = plt.cm.viridis_r(d / max(d.max(), 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(np.arange(n_bars), counts[0], width=0.9, color=colors)
    ax.set_xlim(-0.7, n_bars - 0.3)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("maze cell, sorted by distance to target cell (target at left)")
    ax.set_ylabel("particles in cell")
    ax.set_xticks(np.arange(0, n_bars, 10))
    title = ax.set_title("")
    fig.tight_layout()

    def update(f):
        for b, h in zip(bars, counts[f]):
            b.set_height(h)
        title.set_text(f"saved state {f * stride + 1}/{F * stride}")
        return (*bars, title)

    anim = FuncAnimation(fig, update, frames=F, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=4000, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"wrote {out_path} ({F} frames, {N} particles, {n_bars} cells)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("output")
    ap.add_argument("--grid", type=int, default=10, help="maze is grid x grid cells (default 10)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth saved state")
    ap.add_argument("--ymax", type=float, default=None, help="fixed y-axis max (default: max count)")
    a = ap.parse_args()
    render(a.npz, a.output, a.grid, a.fps, a.stride, a.ymax)
