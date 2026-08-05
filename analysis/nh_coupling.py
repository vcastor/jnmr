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
# The nh outputs may not be downloaded / read yet; leave the analysis pipeline green
# until there is data (nothing to do rather than crash on an empty/missing table).
if not table_exists(cur, "nh_coupling"):
    conn.close()
    print("nh_coupling table absent -> no N(urea) couplings read yet; skipping.")
    sys.exit(0)

# comment_{variant} holds the SCF-convergence warning of the step (as for the HH
# snapshots); older DBs may not have the column yet -> everything counts as clean.
comment_sel = COMMENT_COL if column_exists(cur, "nh_coupling", COMMENT_COL) else "NULL"
rtype_sel   = "resp_type" if column_exists(cur, "nh_coupling", "resp_type") else "NULL"
rows = cur.execute(
    f"SELECT n_step, N_pert, resp, distance, {J_COL}, {comment_sel}, {rtype_sel} FROM nh_coupling "
    f"WHERE {J_COL} IS NOT NULL").fetchall()
conn.close()

# SCF-flagged couplings (comment set) are nonphysical -> kept out of the stats.
clean = [r for r in rows if r[5] is None]
# Second line of defence: a converged-but-garbage step can still emit |J| well over
# 100 Hz for these through-space N-H contacts; drop the nonphysical tail (reported,
# not silent) so the |J| stats/plots are not dominated by a handful of artefacts.
n_before  = len(clean)
clean     = [r for r in clean if abs(r[4]) <= J_PHYSICAL_MAX_HZ]
n_dropped = n_before - len(clean)
if n_dropped:
    print(f"dropped {n_dropped} nonphysical |J| > {J_PHYSICAL_MAX_HZ:.0f} Hz coupling(s)")
if not clean:
    print("no clean N(urea) couplings in nh_coupling yet; skipping.")
    sys.exit(0)

# |J| everywhere: the sign has no physical meaning against the experimental data,
# so (as for every other J analysis) the stats and plots use the absolute value.
j     = np.abs(np.array([r[4] for r in clean]))
rtype = np.array([r[6] for r in clean])

# ── overall distribution ─────────────────────────────────────────────────────
# Effective coupling = the "cubic" power mean (p=CUBIC_P, see hassan_functions/jstats.py), same as the HH J couplings.
print_stats(j, f"N {VARIANT.replace('_', ' ')}")

# The urea N couples to the choline methyl H's [H(CH3)], the two CH2 hydrogens
# [H(CH2-N)/H(CH2-O)] and the two CH2 carbons [C(CH2-N)/C(CH2-O)]; split every
# stat/plot by that responder group. Console tags are the raw DB resp_type; the LaTeX
# labels feed the plot legends.
# H1-H5 shorthand in the legends (H1=CH3, H2=CH2-N, H3=CH2-O); the CH2 carbons are not
# hydrogens, so they keep a descriptive C(CH2-*) label. DB resp_type values are unchanged.
TYPES = [
    ("H(CH3)",  r"N$-$H1",                    "tab:blue"),
    ("H(CH2N)", r"N$-$H2",                    "tab:green"),
    ("H(CH2O)", r"N$-$H3",                    "tab:orange"),
    ("C(CH2N)", r"N$-$C($\mathrm{CH_2}$-N)",  "tab:red"),
    ("C(CH2O)", r"N$-$C($\mathrm{CH_2}$-O)",  "tab:purple"),
]
if any(r is not None for r in rtype):
    print("  per responder type (|J|):")
    for tag, _, _ in TYPES:
        m = rtype == tag
        if m.any():
            print(f"    {tag:<8} n={int(m.sum()):4d}  cubic={cubic_mean(j[m]):.4f}  "
                  f"mean={j[m].mean():.4f} Hz")

# ── geometric vs arithmetic mean within each (step, N) group ──────────────────
# A large arith/geom gap means one close responder dominates the contact, so a plain
# arithmetic average over the group misrepresents the effective coupling.
groups = defaultdict(list)
for n_step, n, r, d, jv, _, _rt in clean:
    groups[(n_step, n)].append(jv)

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

print(f"\n  groups (step,N): {len(groups)}  responders/group: "
      f"mean={sizes.mean():.1f}  min={sizes.min()}  max={sizes.max()}")
print(f"  |J| per group:   arith={arith.mean():.3f}  geom={geom.mean():.3f}")
print(f"  arith/geom:      mean={ratio.mean():.2f}  median={np.median(ratio):.2f}  "
      f"max={ratio.max():.2f}")

# ── plot: |J| distribution split by responder group ───────────────────────────
# Modelled on the HH J-coupling histogram: one KDE per responder group with its
# effective (cubic) coupling shown in the boxed legend. No distance panel.
series = [(lbl, col, rtype == tag) for tag, lbl, col in TYPES]
other  = np.array([r not in {t[0] for t in TYPES} for r in rtype])
if other.any():                       # resp_type not populated yet -> one grey series
    series.append((r"other", "tab:gray", other))

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
    ax.set_xlabel(r"$|J_{\mathrm{N}}|$ (Hz)")
    ax.set_ylabel("density")
    ax.set_title("N(urea) coupling")
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
    save_fig(fig, f"{PLOT_DIR}/nh_coupling_{VARIANT}{suffix}", transparent)
