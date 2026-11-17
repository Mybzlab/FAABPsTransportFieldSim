import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DT = 0.01
V0 = 5.0
R  = 1.0

# OLD hard-coded data (kept for reference, NOT used any more; see data.txt below).
_old_raw_data = {
    4: [13911, 16804, 22685, 18771, 14038],
    25:  [49050, 50387, 40305, 29497, 41081], # 34680
    100: [103190, 118197, 113178, 128281, 117503], # 138170
    400: [720429, 503865, 631420, 567655, 392383], # 635820
    900: [2196953, 2222580, 3315073, 3452369, 5826395], # 2645760
    1225: [865693, 1119207, 1536524, 1552937, 2203070],  # G=35
    # 1600: [],
}

import os
import re
_HERE = os.path.dirname(os.path.abspath(__file__))

# Data now comes from data.txt (lines like "MAZE_RESULT: G=5 run=1 maze_seed=... final_step=... n_steps=... output=...").
# Keys are maze areas (G**2), values are lists of final_step per run.
# Runs that hit the step cap (final_step >= n_steps) or have no final_step are skipped.
def load_raw_data(path):
    out, seen = {}, set()
    with open(path) as f:
        for line in f:
            if not line.startswith("MAZE_RESULT:"):
                continue
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            if "G" not in kv or "final_step" not in kv or not kv["final_step"].isdigit():
                print("skipping (no final_step):", line.strip()); continue
            key = kv.get("output", line.strip())
            if key in seen:
                print("skipping duplicate:", key); continue
            seen.add(key)
            final, cap = int(kv["final_step"]), int(kv.get("n_steps", 0) or 0)
            if cap and final >= cap:
                print("skipping (hit step cap):", key); continue
            out.setdefault(int(kv["G"]) ** 2, []).append(final)
    return out

raw_data = load_raw_data(os.path.join(_HERE, "data.txt"))
print({a: len(v) for a, v in sorted(raw_data.items())}, "trials per maze area")

data ={s: [v * DT * V0 / R for v in vals] for s, vals in raw_data.items()}

sizes = np.array(sorted(data.keys()))
means = np.array([np.mean(data[s]) for s in sizes])
stds  = np.array([np.std(data[s], ddof=1) if len(data[s]) > 1 else 0 for s in sizes])

coeffs = np.polyfit(np.log10(sizes), np.log10(means), 1)
exponent, intercept = coeffs
x_fit = np.logspace(np.log10(sizes[0] * 0.9), np.log10(sizes[-1] * 1.1), 200)
y_fit = 10**intercept * x_fit**exponent

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x_fit, y_fit, '--', color='steelblue', alpha=0.5,
        label=f'$t^*$ ∝ $N^{{{exponent:.2f}}}$')
ax.errorbar(sizes, means, yerr=stds, fmt='o', markersize=8,
            color='steelblue', capsize=4,) # label='mean ± std'

#offsets = [(8, 0)] * (len(sizes) - 1) + [(-55, 0)]
#for x, y, (dx, dy) in zip(sizes, means, offsets):
#    ax.annotate(f'{y:,.0f}', (x, y), textcoords='offset points',
#                xytext=(dx, dy), va='center', fontsize=9)

ax.tick_params(axis='both', labelsize=16)

ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_xticks(sizes)
#ax.set_xticklabels([str(s) for s in sizes])
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_ylim(100, 1000000)
ax.set_xlabel('Maze Area (# Cells)',fontsize=18)
ax.set_ylabel(r'Time to Target  ($t^* = t\, v_0 / r$)',fontsize=18)
#ax.set_title(r'Time to Reach Goal vs Maze Area ($t^* = t\, v_0 / r$)')
ax.legend(fontsize=16)


ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.set_xlim(1,1500)
plt.tight_layout()
plt.savefig(os.path.join(_HERE, 'maze_scaling_plot.png'), dpi=300) 
plt.show()
