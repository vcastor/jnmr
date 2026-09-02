#!/usr/bin/python3
import os
import sys
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.db import table_exists, column_exists
from hassan_functions.plotting import PLOT_STYLES, style_axes
from hassan_functions.style    import apply_style
from hassan_functions.jstats   import (cubic_mean, cubic_dispersion,
                                       effective_n, cubic_mean_ci, CUBIC_P)
from hassan_functions.params   import J_PHYSICAL_MAX_HZ

apply_style("notex")

PLOT_DIR = "plots"
DB_PATH  = "nmr_jcoupling.db"

# (variant, label, color)
VARIANTS = [
    ("TZ2P_FC",   "TZ2P FC",   "steelblue"),
    ("TZ2P_all",  "TZ2P all",  "deepskyblue"),
    ("TZ2PJ_FC",  "TZ2PJ FC",  "darkorange"),
    ("TZ2PJ_all", "TZ2PJ all", "crimson"),
]

# experimental values
EXP_INTRA     = 5.90
EXP_INTRA_ERR = 0.24
EXP_INTER     = 1.104
EXP_INTER_ERR = 0.031

def get_processed_steps(cursor, basis_cont):
    if not table_exists(cursor, "snapshots"):
        return []
    comment_col = f"comment_{basis_cont}"
    if not column_exists(cursor, "snapshots", comment_col):
        return []
    cursor.execute(f"SELECT n_step FROM snapshots WHERE {comment_col} IS NULL")
    return [row[0] for row in cursor.fetchall()]

def collect_j_values(cursor, steps, table_type, basis_cont, main_only=False):
    """Return (|J| array, owning-snapshot array) across interactions over all steps.
    The snapshot ids travel with the values because the snapshot, not the individual
    H-H pair, is the independent sampling unit: pairs within one snapshot share a
    geometry and must be resampled together in any error estimate.
"""
    j_col = f"J_{basis_cont}"
    all_values, all_steps = [], []
    for n_step in steps:
        table_name = f"step_{n_step}_{table_type}"
        if not table_exists(cursor, table_name):
            continue
        if not column_exists(cursor, table_name, j_col):
            continue
        where = f"WHERE {j_col} IS NOT NULL"
        if main_only and column_exists(cursor, table_name, "is_main"):
            where += " AND is_main = 1"
        cursor.execute(f"SELECT {j_col} FROM {table_name} {where}")
        for row in cursor.fetchall():
            all_values.append(abs(row[0]))
            all_steps.append(n_step)
    return np.array(all_values, dtype=float), np.array(all_steps, dtype=int)

def collect_j_with_distance(cursor, steps, basis_cont):
    j_col = f"J_{basis_cont}"
    j_values, distances = [], []
    for n_step in steps:
        table_name = f"step_{n_step}_inter"
        if not table_exists(cursor, table_name):
            continue
        if not column_exists(cursor, table_name, j_col):
            continue
        if not column_exists(cursor, table_name, "distance"):
            continue
        cursor.execute(
            f"SELECT {j_col}, distance FROM {table_name} WHERE {j_col} IS NOT NULL"
        )
        rows = cursor.fetchall()
        if rows:
            best = max(rows, key=lambda r: abs(r[0]))
            j_values.append(abs(best[0]))
            distances.append(best[1])
    return np.array(j_values, dtype=float), np.array(distances, dtype=float)

def print_stats(j_values, steps, label, p=CUBIC_P):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  N interactions: {j_values.size}")
    if j_values.size == 0:
        return
    n_snap = np.unique(steps).size
    ess    = effective_n(j_values, steps, p=p)
    lo, hi = cubic_mean_ci(j_values, steps, p=p)
    print(f"  N snapshots: {n_snap}  (effective for p={p:g}: {ess:.1f})")
    print(f"  Power mean (p={p:g}): {cubic_mean(j_values, p=p):.4f} Hz  "
          f"95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  Disp (p={p:g}): {cubic_dispersion(j_values, p=p):.4f} Hz")
    print(f"  Mean:       {np.mean(j_values):.4f} Hz")
    print(f"  Median:     {np.median(j_values):.4f} Hz")
    print(f"  Std dev:    {np.std(j_values):.4f} Hz")
    print(f"  Min:        {np.min(j_values):.4f} Hz")
    print(f"  Max:        {np.max(j_values):.4f} Hz")

def fit_bimodal(j_values, label):
    """Fit a 2-component GMM. Returns (means, per-peak Gaussian width sigma) sorted by
    peak position — the sigma is the width of each fitted Gaussian, used to annotate the
    split of the (bimodal) intra J distribution."""
    if j_values.size < 10:
        return None
    gmm = GaussianMixture(n_components=2, random_state=0, max_iter=300)
    gmm.fit(j_values.reshape(-1, 1))
    idx    = np.argsort(gmm.means_.ravel())
    means  = gmm.means_.ravel()[idx]
    sigmas = np.sqrt(gmm.covariances_.ravel())[idx]   # width of each Gaussian component
    print(f"\n  -- Bimodal fit: {label} --")
    print(f"  Peak 1:      {means[0]:.4f} ± {sigmas[0]:.4f} Hz  (Gaussian width)")
    print(f"  Peak 2:      {means[1]:.4f} ± {sigmas[1]:.4f} Hz  (Gaussian width)")
    print(f"  Global mean: {np.mean(j_values):.4f} Hz")
    print(f"  MAE:         {np.mean(np.abs(j_values - np.mean(j_values))):.4f} Hz")
    return means, sigmas

def _leg_row(label, mean, mae, precision):
    """Build a monospace-aligned legend row."""
    label_width = 9
    value_width = precision + 4
    return (
        f"{label:<{label_width}} \u27e8J\u27e9 = "
        f"{mean:{value_width}.{precision}f} \u00b1 "
        f"{mae:{value_width}.{precision}f} Hz"
    )

def plot_overlay(variant_data, title, output, exp_mean=None, exp_std=None,
                 gmm_peaks=None, value_precision=2, peak_text_offset=0.03):
    """variant_data: list of (label, j_values, color, snapshot_ids, mean); the KDE uses
    j_values, the legend the precomputed mean (cutoffs apply to the mean only).
    gmm_peaks: dict label -> (means, per-peak Gaussian width sigma) for peak annotations."""
    finite = [v for _, v, _, _, _ in variant_data if v.size > 0]
    if not finite:
        return
    xmax = max(np.max(v) for v in finite)*1.05
    if exp_mean is not None:
        xmax = max(xmax, exp_mean*1.05)
    x = np.linspace(0, xmax, 500)

    fig, ax = plt.subplots(figsize=(8, 5))

    kde_store   = {}  # label -> (x, kde_y, color)
    leg_handles = []
    leg_labels  = []

    for label, vals, color, stp, mean in variant_data:
        if vals.size < 2:
            continue
        mae  = np.mean(np.abs(vals - np.mean(vals)))
        kde  = gaussian_kde(vals, bw_method=0.3)
        y    = kde(x)
        kde_store[label] = (x, y, color)
        line, = ax.plot(x, y, color=color, linewidth=2.5)
        ax.axvline(mean, color=color, linestyle=":", linewidth=1.5, alpha=0.8)
        leg_handles.append(line)
        leg_labels.append(_leg_row(label, mean, mae, value_precision))

    if exp_mean is not None:
        exp_line = ax.axvline(exp_mean, color=LETTER_COLOUR, ls="--", linewidth=1.8)
        if exp_std is not None:
            ax.axvspan(exp_mean-exp_std, exp_mean+exp_std,
                       color=LETTER_COLOUR, alpha=0.12, zorder=0)
        exp_txt = _leg_row("Exp", exp_mean, exp_std, value_precision)
        leg_handles.append(exp_line)
        leg_labels.append(exp_txt)

    all_y = [y for _, y, _ in kde_store.values()]
    ymax  = max(np.max(y) for y in all_y)
    ax.set_ylim(0, ymax*1.15)

    if gmm_peaks is not None:
        for label, (peaks_m, peaks_sig) in gmm_peaks.items():
            if label not in kde_store:
                continue
            xc, yc, color = kde_store[label]
            for pm, psig in zip(peaks_m, peaks_sig):
                # KDE height at the GMM mean position
                y_at_pm = float(np.interp(pm, xc, yc))
                # vertical line from 0 to the curve height only
                ax.vlines(pm, 0, y_at_pm, color=color, linestyle="--",
                          linewidth=1.0, alpha=0.6)
                # text uses GMM mean ± the Gaussian width (sigma of the component)
                txt = f"{pm:.2f}\u00b1{psig:.2f}"
                ax.text(pm+0.3, y_at_pm + ymax*0.03, txt,
                        ha="center", va="bottom", fontsize=12, color=LETTER_COLOUR,
                        bbox={
                            "boxstyle": "round,pad=0.20",
                            "facecolor": "white" if not TRANSPARENT else "none",
                            "edgecolor": color,
                            "alpha": 0.4,
                            "linewidth": 0.8,
                            },
                        )

    ax.set_xlabel("J coupling (Hz)")
    ax.set_ylabel("Relative frequency")
    ax.set_title(title)
    ax.set_xlim(0, xmax)

    style_axes(ax, LETTER_COLOUR)

    if TRANSPARENT:
        # transparent panel, but keep the box outline so it stays delimited
        leg = ax.legend(leg_handles, leg_labels, loc="upper right", frameon=True,
                        prop={"family": "monospace", "size": 12})
        frame = leg.get_frame()
        frame.set_facecolor("none")
        frame.set_edgecolor(LETTER_COLOUR)
        frame.set_alpha(1.0)
        frame.set_linewidth(1.0)
    else:
        leg = ax.legend(leg_handles, leg_labels, loc="upper right", frameon=True,
                        prop={"family": "monospace", "size": 12})
        frame = leg.get_frame()
        frame.set_facecolor((1.0, 1.0, 1.0, 1.0))
        frame.set_edgecolor(LETTER_COLOUR)
        frame.set_alpha(1.0)
        frame.set_linewidth(1.0)
    for t in leg.get_texts():
        t.set_color(LETTER_COLOUR)
    leg.set_zorder(20)

    fig.tight_layout()
    ext = "svg" if TRANSPARENT else "pdf"
    fig.savefig(f"{PLOT_DIR}/{output}.{ext}", dpi=150, transparent=TRANSPARENT)
    plt.close(fig)

def plot_j_vs_distance(j_values, distances, output):
    if j_values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distances, j_values, alpha=0.4, s=10, color="steelblue")
    ax.set_xlabel("H-H distance (A)")
    ax.set_ylabel("J coupling (Hz)")
    ax.set_title("Inter-molecular J coupling vs distance")
    style_axes(ax, LETTER_COLOUR)
    fig.tight_layout()
    ext = "svg" if TRANSPARENT else "pdf"
    fig.savefig(f"{PLOT_DIR}/{output}.{ext}", dpi=150, transparent=TRANSPARENT)
    plt.close(fig)

# ============================== #
#              Main
# ============================== #

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# intra · all variants overlay
intra_data  = []
intra_peaks = {}
for variant, label, color in VARIANTS:
    # all four variants now, incl. the all-contribution (_all) J's — not just FC
    steps    = get_processed_steps(cursor, variant)
    j, j_stp = collect_j_values(cursor, steps, "intra", variant)
    print_stats(j, j_stp, f"Intra · {label}")
    result = fit_bimodal(j, f"Intra · {label}")
    if result is not None:
        means, sigmas = result
        intra_peaks[label] = (means, sigmas)
    intra_data.append((label, j, color, j_stp, cubic_mean(j) if j.size else 0.0))

# inter · all variants overlay
inter_data = []
for variant, label, color in VARIANTS:
    steps = get_processed_steps(cursor, variant)
    j, j_stp = collect_j_values(cursor, steps, "inter", variant, main_only=True)
    keep = j <= J_PHYSICAL_MAX_HZ
    j, j_stp = j[keep], j_stp[keep]
    print_stats(j, j_stp, f"Inter · {label}")
    inter_data.append((label, j, color, j_stp, cubic_mean(j) if j.size else 0.0))

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    plot_overlay(intra_data, "Intramolecular J coupling (CH2-CH2)",
                 f"hist_intra{SUFFIX}", exp_mean=EXP_INTRA, exp_std=EXP_INTRA_ERR,
                 gmm_peaks=intra_peaks, value_precision=2, peak_text_offset=0.06)
    plot_overlay(inter_data, "Intermolecular J coupling (NH2-CH3)",
                 f"hist_inter{SUFFIX}", exp_mean=EXP_INTER, exp_std=EXP_INTER_ERR,
                 value_precision=3)

# # Sensitivity of effective J to power-mean exponent (TZ2P_FC reference)
# basis_cont = "TZ2P_FC"
# steps  = get_processed_steps(cursor, basis_cont)
# jvals  = collect_j_values(cursor, steps, "inter", basis_cont)
# ps     = np.arange(1, 6.1, 0.25)
# means  = [cubic_mean(jvals, p=p) if jvals.size else 0 for p in ps]
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(ps, means, "o-", color="darkorange")
# ax.set_xlabel("Power mean exponent p")
# ax.set_ylabel("Effective J (Hz)")
# ax.set_title("Sensitivity of effective J to power mean exponent")
# ax.axhline(EXP_INTER, color=LETTER_COLOUR, ls="--",
#            label=f"Exp = {EXP_INTER}±{EXP_INTER_ERR} Hz")
# ax.axhspan(EXP_INTER-EXP_INTER_ERR, EXP_INTER+EXP_INTER_ERR,
#            color=LETTER_COLOUR, alpha=0.12)
# ax.legend()
# style_axes(ax)
# fig.tight_layout()
# fig.savefig(f"{PLOT_DIR}/j_vs_power.pdf", dpi=150, transparent=TRANSPARENT)
# plt.close(fig)

conn.close()

