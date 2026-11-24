"""Animation of the payload transport simulation, in the style of the paper figures.

  * particles coloured by score with viridis (scores >= vmax clip to yellow)
  * particles drawn 1.5x larger: display radius = max(r, box/300) * 1.5
  * Perlin-noise grey texture on the payload
  * dark-blue payload path
  * grey walls, dark-green goal star
  * optional: payload-zoom inset (show_inset), polarity arrows (show_vectors)

Use from code:   create_payload_animation(positions, ..., params, ...)
Use from a file: python src/visualization.py data/run.npz visualizations/run.mp4
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize



# ── Style constants (same values as figures/fig6/f/render_final_frame.py) ─────
MAXCOLOR_SCORE_BY_G = {4:4, 5: 5, 6:6, 10: 10, 20: 20, 30: 35}
SCORE_CMAP = plt.get_cmap('viridis')
NO_SCORE_COLOR = (0.5, 0.5, 0.5, 1.0)
PARTICLE_ALPHA = 0.7
PARTICLE_RADIUS_SCALE = 1.5
GOAL_COLOR = '#008F25'
PATH_COLOR = '#00008B'
WALL_COLOR = '#505050'
NEGATIVE_CURVITY_COLOR = to_rgba("#FF9B00FF") #Dutch Orange
POSITIVE_CURVITY_COLOR = to_rgba("#8ED64CFF") #Snot Green
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


def _payload_texture():
    noise = perlin_noise(PAYLOAD_NOISE_SIZE, PAYLOAD_NOISE_CELLS, PAYLOAD_NOISE_OCTAVES,
                         PAYLOAD_NOISE_SEED, PAYLOAD_NOISE_PERSIST)
    lo, hi = np.array(PAYLOAD_GRAY_LIGHT), np.array(PAYLOAD_GRAY_DARK)
    return lo + (hi - lo) * noise[..., None]


#####################################################
# Animation function                                #
#####################################################

def create_payload_animation(positions, orientations, velocities, payload_positions, params,
                             curvity_values, output_file='visualizations/payload_animation_00.mp4',
                             show_vectors=False, polarity=None, particle_scores=None,
                             particle_curvities=None,
                             show_inset=False, steps_per_frame=100, fps=15, vmax=None,
                             first_update_steps=None):
    """Create an animation of the payload transport simulation.

    orientations, velocities and curvity_values are unused (kept for call compatibility).

    Args:
        particle_scores: (n_frames, n_particles). Particles are coloured by score (viridis);
            if None they are drawn in a single grey.
        particle_curvity: (n_frames, n_particles). Particles edges are coloured by curvity (orange/green);
            if None they are drawn in grey.
        show_vectors: draw the polarity vectors as arrows (needs `polarity`, (n_frames, N, 2)).
        show_inset: draw a zoomed payload-centred view in the top-left corner.
        steps_per_frame: simulation steps advanced per video frame (default 100). Saved states
            are params['save_interval'] steps apart (default 100), so the default draws every
            saved state; use e.g. 1000 for 10x fewer frames.
        vmax: upper end of the viridis score range (default: per maze size G, as in the figures).
        first_update_steps: optional (n_particles,) array; particles are red until their first
            score/polarity update step.
    """
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("visualizations", exist_ok=True)
        output_file = f"visualizations/payload_animation_{timestamp}.mp4"
    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)

    print("Creating animation...")
    start_time = time.time()

    box = float(params['box_size'])
    payload_radius = float(params['payload_radius'])
    goal = np.asarray(params['goal_position'], dtype=float)
    walls = params.get('walls', np.zeros((0, 5), dtype=np.float64))
    save_interval = int(params.get('save_interval', 100))
    n_saved = len(payload_positions)
    n_particles = positions.shape[1]

    skip = max(1, int(round(steps_per_frame / save_interval)))
    frames = list(range(0, n_saved, skip))
    if frames[-1] != n_saved - 1:
        frames.append(n_saved - 1)  # always show the final state

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
    path_line, = ax.plot(payload_positions[0:1, 0], payload_positions[0:1, 1], color=PATH_COLOR,
                         linewidth=3.0, linestyle='-', alpha=0.75, zorder=8)

    # Particles: display radius (data units) -> scatter marker area (pt^2).
    # The axes fill the 10in x 10in figure, so 1 data unit = 720/box points.
    particle_radius = np.atleast_1d(params['particle_radius']).astype(float)
    min_r = box / 600.0 if box > 1200 else (box / 400.0 if box > 600 else box / 300.0)
    scale = PARTICLE_RADIUS_SCALE if box > 120 else 1.0
    display_r = np.broadcast_to(np.maximum(particle_radius, min_r) * scale, (n_particles,))
    marker_area = (2.0 * display_r * 720.0 / box) ** 2

    first_update = None if first_update_steps is None else np.asarray(first_update_steps)
    frame_steps = np.array(frames) * save_interval

    def colors_for(k):
        if particle_scores is None:
            rgba = np.tile(NO_SCORE_COLOR, (n_particles, 1))
        else:
            rgba = SCORE_CMAP(norm(np.asarray(particle_scores[frames[k]], dtype=float)))
        if first_update is not None:
            rgba[first_update > frame_steps[k]] = (1.0, 0.0, 0.0, 1.0)
        rgba[:, 3] = PARTICLE_ALPHA
        return rgba

    #Set up curvity colors for particles edges
    def curvity_color(k):        
        if particle_curvities is None:
            return np.tile(NO_SCORE_COLOR, (n_particles, 1))
        else:
            rgba = np.zeros((n_particles,4))
            c = np.asarray(particle_curvities[frames[k]], dtype=float)
            rgba = np.where(np.asarray(c)[:, None] < 0, NEGATIVE_CURVITY_COLOR, POSITIVE_CURVITY_COLOR)
            return rgba

    edge_color = curvity_color(0)
    face_color = colors_for(0)
    scatter = ax.scatter(positions[frames[0], :, 0], positions[frames[0], :, 1],
                         s=marker_area, c=face_color, edgecolors=edge_color, linewidths=1, zorder=9)
    
    # Payload: Perlin-noise textured disc, clipped to a circle, moved every frame
    tex = _payload_texture()
    cx, cy = payload_positions[0]
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

    # Optional polarity arrows
    quiver = None
    arrow_length = 8.0
    if show_vectors and polarity is not None:
        quiver = ax.quiver(positions[frames[0], :, 0], positions[frames[0], :, 1],
                           polarity[frames[0], :, 0] * arrow_length,
                           polarity[frames[0], :, 1] * arrow_length,
                           angles='xy', scale_units='xy', scale=1, color='darkblue',
                           alpha=0.3, width=0.004, headwidth=3.5, headlength=4.5, zorder=9.5)

    # Optional inset: zoomed view locked to the payload (periodic-corrected, payload at origin)
    inset = None
    if show_inset:
        zoom_half = 40
        inset_ax = fig.add_axes([0.02, 0.73, 0.25, 0.25])
        inset_ax.set_xlim(-zoom_half, zoom_half)
        inset_ax.set_ylim(-zoom_half, zoom_half)
        inset_ax.set_aspect('equal')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title('Payload zoom', fontsize=7, pad=2)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(1.5)
        inset_ax.set_facecolor('white')

        # inset is 0.25*720 pt wide for 2*zoom_half data units
        inset_area = (2.0 * display_r * (0.25 * 720.0 / (2 * zoom_half))) ** 2

        def rel(p, ref):
            d = p - ref
            return d - box * np.round(d / box)

        def inset_walls(px, py):
            out = []
            for w in walls:
                mx, my = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                sx = -box * np.round((mx - px) / box)
                sy = -box * np.round((my - py) / box)
                out.append(arc_points(w[0] - px + sx, w[1] - py + sy,
                                      w[2] - px + sx, w[3] - py + sy, w[4]))
            return out

        px0, py0 = payload_positions[0]
        inset_scatter = inset_ax.scatter(rel(positions[frames[0], :, 0], px0),
                                         rel(positions[frames[0], :, 1], py0),
                                         s=inset_area, c=colors_for(0), linewidths=0, zorder=9)
        inset_im = inset_ax.imshow(tex, extent=(-payload_radius, payload_radius,
                                                -payload_radius, payload_radius),
                                   origin='lower', interpolation='bilinear', zorder=7)
        inset_im.set_clip_path(mpatches.Circle((0, 0), radius=payload_radius,
                                               transform=inset_ax.transData))
        inset_ax.add_patch(mpatches.Circle((0, 0), radius=payload_radius, facecolor='none',
                                           edgecolor='#282828', linewidth=1.0, zorder=7.1))
        inset_goal, = inset_ax.plot(rel(goal[0], px0), rel(goal[1], py0), '*', color=GOAL_COLOR,
                                    markersize=8, markeredgewidth=0.8,
                                    markeredgecolor='#003A0E', zorder=11)
        inset_traj, = inset_ax.plot([0], [0], color=PATH_COLOR, alpha=0.75, linewidth=2.0,
                                    zorder=8)
        inset_wall_lines = []
        for xs, ys in inset_walls(px0, py0):
            line, = inset_ax.plot(xs, ys, color=WALL_COLOR, linewidth=1.5,
                                  solid_capstyle='round', zorder=10)
            inset_wall_lines.append(line)
        inset_ax.set_xlim(-zoom_half, zoom_half)
        inset_ax.set_ylim(-zoom_half, zoom_half)
        inset = True

    def update(k):
        if k % 50 == 0:
            print(f"Progress: video frame {k}/{len(frames)}")
        s_idx = frames[k]
        px, py = payload_positions[s_idx]
        colors = colors_for(k)
        scatter.set_offsets(positions[s_idx])
        scatter.set_facecolor(colors)
        scatter.set_edgecolor(curvity_color(k))
        path_line.set_data(payload_positions[:s_idx + 1, 0], payload_positions[:s_idx + 1, 1])
        im.set_extent((px - payload_radius, px + payload_radius,
                       py - payload_radius, py + payload_radius))
        clip.center = (px, py)
        im.set_clip_path(clip)
        outline.center = (px, py)
        artists = [scatter, path_line, im, outline]

        if quiver is not None:
            quiver.set_offsets(positions[s_idx])
            quiver.set_UVC(polarity[s_idx, :, 0] * arrow_length,
                           polarity[s_idx, :, 1] * arrow_length)
            artists.append(quiver)

        if inset:
            inset_scatter.set_offsets(np.column_stack([rel(positions[s_idx, :, 0], px),
                                                       rel(positions[s_idx, :, 1], py)]))
            inset_scatter.set_facecolor(colors)
            inset_goal.set_data([rel(goal[0], px)], [rel(goal[1], py)])
            inset_traj.set_data(rel(payload_positions[:s_idx + 1, 0], px),
                                rel(payload_positions[:s_idx + 1, 1], py))
            for line, (xs, ys) in zip(inset_wall_lines, inset_walls(px, py)):
                line.set_data(xs, ys)
            artists.extend([inset_scatter, inset_goal, inset_traj])
            artists.extend(inset_wall_lines)
        return artists

    plt.rcParams['savefig.dpi'] = 150
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=8000, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
    anim.save(output_file, writer=writer)
    plt.close(fig)

    print(f"Animation saved as '{output_file}'")
    print(f"Animation creation time: {time.time() - start_time:.2f} seconds")


#####################################################
# Render from a saved full (non-light) NPZ          #
#####################################################

def render_from_npz(npz_path, output_file, red_until_first_update=False, **kwargs):
    """Load a saved NPZ and call create_payload_animation on it."""
    data = np.load(npz_path)
    params = {
        'box_size': float(data['box_size']),
        'payload_radius': float(data['payload_radius']),
        'goal_position': np.asarray(data['goal_position'], dtype=float),
        'walls': data['walls'],
        'particle_radius': data['particle_radius'],
        'save_interval': kwargs.pop('save_interval', 100),
    }
    first_update = None
    if red_until_first_update:
        if 'first_update_steps' not in data.files:
            raise SystemExit("NPZ has no 'first_update_steps'; re-run the simulation with the updated runner.")
        first_update = data['first_update_steps']
    polarity = data['polarity'] if 'polarity' in data.files else None
    particle_curvities = data['curvity_values'] if 'curvity_values' in data.files else None
    create_payload_animation(
        data['positions'], None, None, data['payload_positions'], params, None,
        output_file=output_file, polarity=polarity,
        particle_scores=data['particle_scores'], particle_curvities=particle_curvities,
        first_update_steps=first_update, **kwargs)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render an mp4 from a full (non-light) NPZ.")
    ap.add_argument("npz")
    ap.add_argument("output")
    ap.add_argument("--save-interval", type=int, default=100,
                    help="SAVE_INTERVAL the NPZ was recorded with (default 100)")
    ap.add_argument("--steps-per-frame", type=int, default=100,
                    help="Simulation steps advanced per video frame (default 100)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--vmax", type=float, default=None,
                    help="Upper end of the viridis score range (default: per-G value used by the figures)")
    ap.add_argument("--red-until-first-update", action="store_true",
                    help="Particles are red until their first score/polarity update "
                         "(needs 'first_update_steps' in the NPZ)")
    ap.add_argument("--show-inset", action="store_true", help="Add the payload zoom inset")
    ap.add_argument("--show-vectors", action="store_true", help="Draw polarity arrows")
    a = ap.parse_args()
    render_from_npz(a.npz, a.output, red_until_first_update=a.red_until_first_update,
                    save_interval=a.save_interval, steps_per_frame=a.steps_per_frame,
                    fps=a.fps, vmax=a.vmax, show_inset=a.show_inset,
                    show_vectors=a.show_vectors)
