#!/usr/bin/python3
import os
import sys
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.plotting import PLOT_STYLES, style_axes, save_fig
from hassan_functions.style    import apply_style
from hassan_functions.db       import table_exists, column_exists
from hassan_functions.jstats   import cubic_mean, cubic_dispersion

apply_style("default")

def print_stats(j, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  N interactions: {j.size}")
    if j.size == 0:
        return
    print(f"  Cubic mean: {cubic_mean(j):.4f} Hz")
    print(f"  Cubic disp: {cubic_dispersion(j):.4f} Hz")
    print(f"  Mean:       {np.mean(j):.4f} Hz")
    print(f"  Median:     {np.median(j):.4f} Hz")
    print(f"  Std dev:    {np.std(j):.4f} Hz")
    print(f"  Min:        {np.min(j):.4f} Hz")
    print(f"  Max:        {np.max(j):.4f} Hz")

DB_PATH     = "nmr_jcoupling.db"
PLOT_DIR    = "plots"
VARIANT     = "TZ2P_FC"
J_COL       = f"J_{VARIANT}"
COMMENT_COL = f"comment_{VARIANT}"
NH_BOND     = 1.3   # Å; N-H closer than this is the bonded pair (1J), else other-N H

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
if not table_exists(cur, "nh_intra_coupling"):
    conn.close()
    print("nh_intra_coupling table absent -> no intra-urea N-H couplings read yet; skipping.")
    sys.exit(0)

comment_sel = COMMENT_COL if column_exists(cur, "nh_intra_coupling", COMMENT_COL) else "NULL"
rows = cur.execute(
    f"SELECT n_step, N_pert, H_resp, distance, {J_COL}, {comment_sel} "
    f"FROM nh_intra_coupling WHERE {J_COL} IS NOT NULL").fetchall()
conn.close()

# SCF-flagged couplings (comment set) are nonphysical -> kept out of the stats.
# No |J| ceiling here: the bonded 1J(N-H) is legitimately tens of Hz, unlike the
# through-space contacts the J_PHYSICAL_MAX_HZ guard was written for.
clean = [r for r in rows if r[5] is None]
if not clean:
    print("no clean intra-urea N-H couplings in nh_intra_coupling yet; skipping.")
    sys.exit(0)

j = np.abs(np.array([r[4] for r in clean]))
d = np.array([r[3] for r in clean])

print_stats(j, f"N-H5 intra-urea {VARIANT.replace('_', ' ')}")

# The urea N couples to its own two bonded H's (1J) and the two on the other N
# (geminal, through 2/3 bonds); the bond length separates the two groups cleanly.
TYPES = [
    (d < NH_BOND,  r"N$-$H5 (bonded)",   "tab:blue"),
    (d >= NH_BOND, r"N$\cdots$H5 (other N)", "tab:orange"),
]
print("  per pair type (|J|):")
for m, lbl, _ in TYPES:
    if m.any():
        print(f"    {lbl:<22} n={int(m.sum()):4d}  cubic={cubic_mean(j[m]):.4f}  "
              f"mean={j[m].mean():.4f} Hz")

groups = defaultdict(list)
for n_step, n, h, dist, jv, _ in clean:
    groups[(n_step, n)].append(jv)
sizes = np.array([len(v) for v in groups.values()])
print(f"\n  groups (step,N): {len(groups)}  responders/group: "
      f"mean={sizes.mean():.1f}  min={sizes.min()}  max={sizes.max()}")

for lc, transparent, suffix in PLOT_STYLES:
    fig, ax = plt.subplots(figsize=(8, 5))

    handles, labels = [], []
    for m, lbl, col in TYPES:
        if m.sum() < 2:
            continue
        cm   = cubic_mean(j[m])
        disp = cubic_dispersion(j[m])
        sns.kdeplot(j[m], ax=ax, color=col, linewidth=2, clip=(0, None))
        handles.append(ax.lines[-1])
        labels.append(rf"{lbl}: $\langle J\rangle = {cm:.2f}\pm{disp:.2f}$ Hz")
        ax.axvline(cm, color=col, linestyle=':', linewidth=1.2)
    ax.set_xlabel(r"$|J_{\mathrm{NH}}|$ (Hz)")
    ax.set_ylabel("density")
    ax.set_title("N(urea) - H5 intra-urea coupling")
    ax.set_xlim(left=0)

    style_axes(ax, lc, transparent)

    leg   = ax.legend(handles, labels, loc="upper right", frameon=True, fontsize=11)
    frame = leg.get_frame()
    frame.set_facecolor("none" if transparent else (1.0, 1.0, 1.0, 1.0))
    frame.set_edgecolor(lc)
    frame.set_alpha(1.0)
    frame.set_linewidth(1.0)
    for t in leg.get_texts():
        t.set_color(lc)

    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/nh_intra_coupling_{VARIANT}{suffix}", transparent)
