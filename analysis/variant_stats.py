#!/usr/bin/python3
"""Estimator diagnostics for the INTER couplings only — the BCP-uncertain
through-space case this analysis exists for. Per variant: p-sweep of the power mean,
the exponent p* and scale factor X reproducing experiment, and zero-inflation stats.
TZ2P_FC and TZ2PJ_all printed as reference."""
import os
import sys
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.db       import table_exists, column_exists
from hassan_functions.plotting import PLOT_STYLES, style_axes, save_fig
from hassan_functions.style    import apply_style
from hassan_functions.jstats   import (cubic_mean, effective_n, cubic_mean_ci, CUBIC_P)
from hassan_functions.params   import J_PHYSICAL_MAX_HZ

apply_style("notex")

PLOT_DIR = "plots"
DB_PATH  = "nmr_jcoupling.db"

VARIANTS = [
    ("TZ2P_FC",   "TZ2P FC",   "steelblue"),
    ("TZ2P_all",  "TZ2P all",  "deepskyblue"),
    ("TZ2PJ_FC",  "TZ2PJ FC",  "darkorange"),
    ("TZ2PJ_all", "TZ2PJ all", "crimson"),
]
EXP = {"inter": (1.104, 0.031)}

ZERO_EPS = 0.05        # Hz; |J| below this counts as a numerical zero
PS       = np.arange(0.5, 6.01, 0.05)

def get_processed_steps(cursor, variant):
    comment_col = f"comment_{variant}"
    if not column_exists(cursor, "snapshots", comment_col):
        return []
    cursor.execute(f"SELECT n_step FROM snapshots WHERE {comment_col} IS NULL")
    return [row[0] for row in cursor.fetchall()]

def collect(cursor, steps, table_type, variant):
    """(|J|, snapshot) arrays."""
    j_col = f"J_{variant}"
    vals, stp = [], []
    for n_step in steps:
        table = f"step_{n_step}_{table_type}"
        if not table_exists(cursor, table) or not column_exists(cursor, table, j_col):
            continue
        cursor.execute(f"SELECT {j_col} FROM {table} WHERE {j_col} IS NOT NULL")
        for row in cursor.fetchall():
            vals.append(abs(row[0]))
            stp.append(n_step)
    j, s = np.array(vals, float), np.array(stp, int)
    keep = j <= J_PHYSICAL_MAX_HZ
    return j[keep], s[keep]

def p_star(j, target):
    """Exponent where the power mean crosses the experimental value (monotonic in p),
    or None when no p in the sweep reaches it — i.e. no exponent can fix the data."""
    if j.size == 0:
        return None
    means = np.array([cubic_mean(j, p=p) for p in PS])
    if target < means.min() or target > means.max():
        return None
    return float(np.interp(target, means, PS))

def kde_mode(j):
    if j.size < 2:
        return None
    x = np.linspace(0, j.max(), 500)
    return float(x[np.argmax(gaussian_kde(j, bw_method=0.3)(x))])

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

for table_type in ("inter",):
    exp, exp_err = EXP[table_type]
    print(f"\n{'='*72}")
    print(f"  {table_type} — exp {exp} +- {exp_err} Hz   (p={CUBIC_P} is the headline)")
    print(f"{'='*72}")
    print(f"  {'variant':<10} {'n':>5} {'snap':>5} {'ess':>5}  {'cubic':>7} "
          f"{'95% CI':>17}  {'p*':>5} {'X':>6}  {'f0':>6} {'tail':>7} {'mode':>6}")

    sweep = []
    for variant, label, color in VARIANTS:
        steps = get_processed_steps(cursor, variant)
        j, s  = collect(cursor, steps, table_type, variant)
        if j.size == 0:
            continue
        cm     = cubic_mean(j)
        lo, hi = cubic_mean_ci(j, s)
        ps     = p_star(j, exp)
        x      = exp/cm if cm > 0 else np.nan       # multiplicative adjustment
        f0     = np.mean(j < ZERO_EPS)              # zero-inflated fraction
        tail   = j[j >= ZERO_EPS]
        cmtail = cubic_mean(tail) if tail.size else np.nan
        mode   = kde_mode(j)
        sweep.append((label, color, j))
        print(f"  {label:<10} {j.size:>5} {np.unique(s).size:>5} "
              f"{effective_n(j, s):>5.1f}  {cm:>7.3f} [{lo:>7.3f},{hi:>7.3f}]  "
              f"{ps if ps is None else round(ps, 2)!s:>5} {x:>6.3f}  "
              f"{f0:>6.1%} {cmtail:>7.3f} {mode:>6.3f}")
    print(f"  p*: exponent reproducing exp (None = out of reach for any p)")
    print(f"  X: scale on the p={CUBIC_P} cubic mean; f0: fraction |J| < {ZERO_EPS} Hz; "
          f"tail: cubic mean of |J| >= {ZERO_EPS} Hz")

    for lc, transparent, suffix in PLOT_STYLES:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, color, j in sweep:
            ax.plot(PS, [cubic_mean(j, p=p) for p in PS], color=color,
                    linewidth=2, label=label)
        ax.axhline(exp, color=lc, ls="--", linewidth=1.5)
        ax.axhspan(exp-exp_err, exp+exp_err, color=lc, alpha=0.12, zorder=0)
        ax.axvline(CUBIC_P, color=lc, ls=":", linewidth=1.2)
        ax.set_xlabel("power-mean exponent p")
        ax.set_ylabel("effective J (Hz)")
        ax.set_title(f"{table_type}molecular J vs exponent (exp dashed, p={CUBIC_P:g} dotted)")
        style_axes(ax, lc, transparent)
        leg   = ax.legend(loc="upper left", frameon=True, fontsize=11)
        frame = leg.get_frame()
        frame.set_facecolor("none" if transparent else (1.0, 1.0, 1.0, 1.0))
        frame.set_edgecolor(lc)
        frame.set_alpha(1.0)
        frame.set_linewidth(1.0)
        for t in leg.get_texts():
            t.set_color(lc)
        fig.tight_layout()
        save_fig(fig, f"{PLOT_DIR}/j_vs_power_{table_type}{suffix}", transparent)

conn.close()
