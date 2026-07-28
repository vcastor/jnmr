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
from hassan_functions.db       import column_exists
from hassan_functions.jstats   import cubic_mean, cubic_dispersion
from hassan_functions.params   import J_PHYSICAL_MAX_HZ

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

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
# comment_{variant} holds the SCF-convergence warning of the step (as for the HH
# snapshots); older DBs may not have the column yet -> everything counts as clean.
comment_sel = COMMENT_COL if column_exists(cur, "ch_coupling", COMMENT_COL) else "NULL"
htype_sel   = "h_type" if column_exists(cur, "ch_coupling", "h_type") else "NULL"
rows = cur.execute(
    f"SELECT n_step, C_pert, H_resp, distance, {J_COL}, {comment_sel}, {htype_sel} FROM ch_coupling "
    f"WHERE {J_COL} IS NOT NULL").fetchall()
conn.close()

# SCF-flagged couplings (comment set) are nonphysical -> kept out of the stats.
clean = [r for r in rows if r[5] is None]
# Second line of defence on top of the SCF flag: a converged-but-garbage step can
# still emit |J| well over 100 Hz for these through-space contacts; drop the
# nonphysical tail (reported, not silent).
n_before  = len(clean)
clean     = [r for r in clean if abs(r[4]) <= J_PHYSICAL_MAX_HZ]
n_dropped = n_before - len(clean)
if n_dropped:
    print(f"dropped {n_dropped} nonphysical |J| > {J_PHYSICAL_MAX_HZ:.0f} Hz coupling(s)")

# |J| everywhere: the sign has no physical meaning against the experimental data,
# so (as for every other J analysis) the stats and plots use the absolute value.
j     = np.abs(np.array([r[4] for r in clean]))
htype = np.array([r[6] for r in clean])

# ── overall distribution ─────────────────────────────────────────────────────
# Effective coupling = the "cubic" power mean (p=2.25), same as the HH J couplings.
print_stats(j, f"C H {VARIANT.replace('_', ' ')}")

# Each responding H sits on one choline group; split every stat/plot by that group.
# Console tags are the raw DB h_type; the LaTeX labels feed the plot legends.
# H1-H5 shorthand in the legends (H1=CH3, H2=CH2-N, H3=CH2-O, H4=OH). DB h_type unchanged.
TYPES = [
    ("H(CH3)",  r"C$-$H1", "tab:blue"),
    ("H(CH2N)", r"C$-$H2", "tab:green"),
    ("H(CH2O)", r"C$-$H3", "tab:orange"),
    ("H(O)",    r"C$-$H4", "tab:red"),
]
if any(h is not None for h in htype):
    print("  per H type (|J|):")
    for tag, _, _ in TYPES:
        m = htype == tag
        if m.any():
            print(f"    {tag:<8} n={int(m.sum()):4d}  cubic={cubic_mean(j[m]):.4f}  "
                  f"mean={j[m].mean():.4f} Hz")

# ── geometric vs arithmetic mean within each (step, C) group ──────────────────
# A large arith/geom gap means one close H dominates the contact, so a plain
# arithmetic average over the group misrepresents the effective coupling.
groups = defaultdict(list)
for n_step, c, h, d, jv, _, _ht in clean:
    groups[(n_step, c)].append(jv)

sizes = np.array([len(v) for v in groups.values()])
arith, geom = [], []
for v in groups.values():
    a  = np.abs(v)
    nz = a[a > 0]                       # |J|=0 is a numerical floor -> drop it from the log/geom
    arith.append(a.mean())
    geom.append(np.exp(np.mean(np.log(nz))) if nz.size else 0.0)
arith, geom = np.array(arith), np.array(geom)
mask  = geom > 0
ratio = arith[mask]/geom[mask]

print(f"\n  groups (step,C): {len(groups)}  H-partners/group: "
      f"mean={sizes.mean():.1f}  min={sizes.min()}  max={sizes.max()}")
print(f"  |J| per group:   arith={arith.mean():.3f}  geom={geom.mean():.3f}")
print(f"  arith/geom:      mean={ratio.mean():.2f}  median={np.median(ratio):.2f}  "
      f"max={ratio.max():.2f}")

# ── plot: |J| distribution split by choline H type ────────────────────────────
# Modelled on the HH J-coupling histogram: one KDE per H group with its effective
# (cubic) coupling shown in the boxed legend. No distance panel.
series = [(lbl, col, htype == tag) for tag, lbl, col in TYPES]
other  = np.array([h not in {t[0] for t in TYPES} for h in htype])
if other.any():                       # h_type not populated yet -> one grey series
    series.append((r"H (unclassified)", "tab:gray", other))

for lc, transparent, suffix in PLOT_STYLES:
    fig, ax = plt.subplots(figsize=(8, 5))

    handles, labels = [], []
    for lbl, col, m in series:
        if m.sum() < 2:
            continue
        cm   = cubic_mean(j[m])
        disp = cubic_dispersion(j[m])
        # |J| >= 0: clip the KDE to positive support so no unphysical negative tail
        # is drawn (the negative part only adds noise).
        sns.kdeplot(j[m], ax=ax, color=col, linewidth=2, clip=(0, None))
        handles.append(ax.lines[-1])
        labels.append(rf"{lbl}: $\langle J\rangle = {cm:.2f}\pm{disp:.2f}$ Hz")
        ax.axvline(cm, color=col, linestyle=':', linewidth=1.2)
    ax.set_xlabel(r"$|J_{\mathrm{CH}}|$ (Hz)")
    ax.set_ylabel("density")
    ax.set_title("CH coupling")
    ax.set_xlim(left=0)

    style_axes(ax, lc, transparent)

    # cubic means boxed in the legend, like the HH J-coupling histogram
    leg   = ax.legend(handles, labels, loc="upper right", frameon=True, fontsize=11)
    frame = leg.get_frame()
    frame.set_facecolor("none" if transparent else (1.0, 1.0, 1.0, 1.0))
    frame.set_edgecolor(lc)
    frame.set_alpha(1.0)
    frame.set_linewidth(1.0)
    for t in leg.get_texts():
        t.set_color(lc)

    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/ch_coupling_{VARIANT}{suffix}", transparent)
