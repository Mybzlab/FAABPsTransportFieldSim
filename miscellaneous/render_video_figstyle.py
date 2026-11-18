"""Render an mp4 of a full (non-light) NPZ in the style of the paper figures
(figures/fig6/f/render_final_frame.py):

  * particles coloured by score with viridis (scores >= vmax clip to yellow, so a
    swarm with no goal information, all at 9999, shows as uniform yellow)
  * particles drawn 1.5x larger: display radius = max(r, box/300) * 1.5
  * Perlin-noise grey texture on the payload
  * dark-blue payload path
  * grey walls, dark-green goal star

Usage:
    python miscellaneous/render_video_figstyle.py data/run.npz visualizations/run.mp4
    python miscellaneous/render_video_figstyle.py data/run.npz out.mp4 --save-interval 100 --steps-per-frame 1000
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize

# ── Style constants (same values as figures/fig6/f/render_final_frame.py) ─────
MAXCOLOR_SCORE_BY_G = {5: 5, 10: 10, 20: 20, 30: 35}
SCORE_CMAP = plt.get_cmap('viridis')
PARTICLE_ALPHA = 0.7
PARTICLE_RADIUS_SCALE = 1.5
GOAL_COLOR = '#008F25'
PATH_COLOR = '#00008B'
WALL_COLOR = '#505050'
PAYLOAD_GRAY_LIGHT = (0.86, 0.86, 0.86)
PAYLOAD_GRAY_DARK = (0.40, 0.40, 0.40)
PAYLOAD_NOISE_CELLS = 4
PAYLOAD_NOISE_OCTAVES = 6
PAYLOAD_NOISE_PERSIST = 0.65
PAYLOAD_NOISE_SIZE = 768
PAYLOAD_NOISE_SEED = 7


def perlin_noise(size, cells, octaves=1, seed=0, persistence=0.5):
    """Fractal 2D Perlin noise, (size, size) array normalised to [0, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = cells * 2**o
        ang = rng.uniform(0, 2 * np.pi, (n + 1, n + 1))
        gx, gy = np.cos(ang), np.sin(ang)
        t = np.linspace(0, n, size, endpoint=False)
        xi = t.astype(int)
        f = t - xi
        u = f * f * f * (f * (f * 6 - 15) + 10)
        X, Y = np.meshgrid(f, f)
        U, V = np.meshgrid(u, u)
        ix, iy = np.meshgrid(xi, xi)

        def dot(dx, dy):
            return gx[iy + dy, ix + dx] * (X - dx) + gy[iy + dy, ix + dx] * (Y - dy)

        n00, n10, n01, n11 = dot(0, 0), dot(1, 0), dot(0, 1), dot(1, 1)
        val = (n00 * (1 - U) + n10 * U) * (1 - V) + (n01 * (1 - U) + n11 * U) * V
        out += amp * val
        total += amp
        amp *= persistence
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def arc_points(x1, y1, x2, y2, c, n_pts=64):
    """Return (xs, ys) to draw a wall arc. Falls back to a segment when c ~ 0."""
    if abs(c) < 1e-9:
        return [x1, x2], [y1, y2]
    chx, chy = x2 - x1, y2 - y1
    chord_len = np.sqrt(chx**2 + chy**2)
    if chord_len < 1e-10:
        return [x1, x2], [y1, y2]
    R = chord_len / (2.0 * abs(c))
    h = np.sqrt(max(R**2 - (chord_len / 2)**2, 0.0))
    cw_perp = np.array([chy, -chx]) / chord_len
    mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    center = mid + (1.0 if c > 0 else -1.0) * h * cw_perp
    theta1 = np.arctan2(y1 - center[1], x1 - center[0])
    theta2 = np.arctan2(y2 - center[1], x2 - center[0])
    span_ccw = (theta2 - theta1) % (2 * np.pi)
    if span_ccw <= np.pi:
        thetas = np.linspace(theta1, theta1 + span_ccw, n_pts)
    else:
        thetas = np.linspace(theta1, theta1 - (2 * np.pi - span_ccw), n_pts)
    return (center[0] + R * np.cos(thetas)).tolist(), (center[1] + R * np.sin(thetas)).tolist()


def render(npz_path, out_path, save_interval=100, steps_per_frame=1000, fps=15, vmax=None):
    data = np.load(npz_path)
    box = float(data['box_size'])
    payload_radius = float(data['payload_radius'])
    goal = np.asarray(data['goal_position'], dtype=float)
    walls = data['walls']
    payload_traj = data['payload_positions']
    particle_radius = np.atleast_1d(data['particle_radius']).astype(float)

    n_saved = len(payload_traj)
    skip = max(1, int(round(steps_per_frame / save_interval)))
    frames = list(range(0, n_saved, skip))
    if frames[-1] != n_saved - 1:
        frames.append(n_saved - 1)  # always show the final state

    positions = data['positions'][frames]          # (F, N, 2)
    scores = data['particle_scores'][frames]       # (F, N)
    n_particles = positions.shape[1]
    G = int(round(box / 60.0))
    if vmax is None:
        vmax = MAXCOLOR_SCORE_BY_G.get(G, 35)
    norm = Normalize(vmin=0, vmax=vmax, clip=True)
    print(f"{n_saved} saved states -> {len(frames)} video frames ({steps_per_frame} steps/frame), "
          f"G={G}, viridis 0..{vmax}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    # Walls
    wall_lw = 1.3 if box > 1200 else 1.8
    for w in walls:
        xs, ys = arc_points(w[0], w[1], w[2], w[3], w[4])
        ax.plot(xs, ys, color=WALL_COLOR, linewidth=wall_lw, solid_capstyle='round', zorder=10)

    # Goal
    ax.plot(goal[0], goal[1], '*', color=GOAL_COLOR, markersize=18,
            markeredgewidth=0.8, markeredgecolor='#003A0E', zorder=11)

    # Path
    path_line, = ax.plot(payload_traj[0:1, 0], payload_traj[0:1, 1], color=PATH_COLOR,
                         linewidth=3.0, linestyle='-', alpha=0.75, zorder=8)

    # Particles: display radius (data units) -> scatter marker area (pt^2).
    # The axes fill the 10in x 10in figure, so 1 data unit = 720/box points.
    # Minimum display radius per maze size, as in render_final_frame.py
    min_r = box / 600.0 if box > 1200 else (box / 400.0 if box > 600 else box / 300.0)
    scale = PARTICLE_RADIUS_SCALE if box > 120 else 1.0
    display_r = np.maximum(particle_radius, min_r) * scale
    display_r = np.broadcast_to(display_r, (n_particles,))
    marker_area = (2.0 * display_r * 720.0 / box) ** 2

    def colors_for(sc):
        rgba = SCORE_CMAP(norm(sc.astype(float)))
        rgba[:, 3] = PARTICLE_ALPHA
        return rgba

    scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], s=marker_area,
                         c=colors_for(scores[0]), linewidths=0, zorder=9)

    # Payload: Perlin-noise textured disc, clipped to a circle, moved every frame
    noise = perlin_noise(PAYLOAD_NOISE_SIZE, PAYLOAD_NOISE_CELLS, PAYLOAD_NOISE_OCTAVES,
                         PAYLOAD_NOISE_SEED, PAYLOAD_NOISE_PERSIST)
    lo, hi = np.array(PAYLOAD_GRAY_LIGHT), np.array(PAYLOAD_GRAY_DARK)
    tex = lo + (hi - lo) * noise[..., None]
    cx, cy = payload_traj[0]
    im = ax.imshow(tex, extent=(cx - payload_radius, cx + payload_radius,
                                cy - payload_radius, cy + payload_radius),
                   origin='lower', interpolation='bilinear', zorder=7)
    clip = mpatches.Circle((cx, cy), radius=payload_radius, transform=ax.transData)
    im.set_clip_path(clip)
    outline = mpatches.Circle((cx, cy), radius=payload_radius, facecolor='none',
                              edgecolor='#282828', linewidth=1.0, zorder=7.1)
    ax.add_patch(outline)
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_aspect('equal')

    def update(k):
        if k % 50 == 0:
            print(f"Progress: video frame {k}/{len(frames)}")
        s_idx = frames[k]
        px, py = payload_traj[s_idx]
        scatter.set_offsets(positions[k])
        scatter.set_facecolor(colors_for(scores[k]))
        path_line.set_data(payload_traj[:s_idx + 1, 0], payload_traj[:s_idx + 1, 1])
        im.set_extent((px - payload_radius, px + payload_radius,
                       py - payload_radius, py + payload_radius))
        clip.center = (px, py)
        im.set_clip_path(clip)
        outline.center = (px, py)
        return [scatter, path_line, im, outline]

    plt.rcParams['savefig.dpi'] = 150
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=8000, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("output")
    ap.add_argument("--save-interval", type=int, default=100,
                    help="SAVE_INTERVAL the NPZ was recorded with (default 100)")
    ap.add_argument("--steps-per-frame", type=int, default=1000,
                    help="Simulation steps advanced per video frame (default 1000)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--vmax", type=float, default=None,
                    help="Upper end of the viridis score range (default: per-G value used by the figures)")
    a = ap.parse_args()
    render(a.npz, a.output, a.save_interval, a.steps_per_frame, a.fps, a.vmax)
