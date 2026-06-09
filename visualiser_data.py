#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from hassan_functions.db import table_exists, column_exists
from hassan_functions.plotting import PLOT_STYLES, style_axes

plt.rcParams['font.size']       = 16
plt.rcParams['axes.titlesize']  = 19
plt.rcParams['axes.labelsize']  = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

PLOT_DIR = "plots"
DB_PATH  = "nmr_jcoupling.db"

# (variant, label, color)
VARIANTS = [
    ("TZ2P_FC",   "TZ2P FC",  "steelblue"),
    ("TZ2P_all",  "TZ2P   ",  "deepskyblue"),
    ("TZ2PJ_FC",  "TZ2PJ FC",  "darkorange"),
    ("TZ2PJ_all", "TZ2PJ   ",  "crimson"),
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

def collect_j_values(cursor, steps, table_type, basis_cont):
    """Return flat array of |J| across every interaction over all steps."""
    j_col = f"J_{basis_cont}"
    all_values = []
    for n_step in steps:
        table_name = f"step_{n_step}_{table_type}"
        if not table_exists(cursor, table_name):
            continue
        if not column_exists(cursor, table_name, j_col):
            continue
        cursor.execute(f"SELECT {j_col} FROM {table_name} WHERE {j_col} IS NOT NULL")
        all_values.extend(abs(row[0]) for row in cursor.fetchall())
    return np.array(all_values, dtype=float)

def cubic_mean(vals, p=2.25):
    return float((np.mean(vals**p))**(1./p))

def cubic_dispersion(vals, p=2.25):
    cm = cubic_mean(vals, p=p)
    return float((np.mean(np.abs(vals-cm)**p))**(1./(p)))

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

def print_stats(j_values, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  N interactions: {j_values.size}")
    if j_values.size == 0:
        return
    print(f"  Cubic mean: {cubic_mean(j_values):.4f} Hz")
    print(f"  Cubic disp: {cubic_dispersion(j_values):.4f} Hz")
    print(f"  Mean:       {np.mean(j_values):.4f} Hz")
    print(f"  Median:     {np.median(j_values):.4f} Hz")
    print(f"  Std dev:    {np.std(j_values):.4f} Hz")
    print(f"  Min:        {np.min(j_values):.4f} Hz")
    print(f"  Max:        {np.max(j_values):.4f} Hz")

def fit_bimodal(j_values, label):
    """Fit a 2-component GMM. Returns (means, ci95, mae) sorted by peak position."""
    if j_values.size < 10:
        return None
    gmm = GaussianMixture(n_components=2, random_state=0, max_iter=300)
    gmm.fit(j_values.reshape(-1, 1))
    idx   = np.argsort(gmm.means_.ravel())
    means = gmm.means_.ravel()[idx]
    stds  = np.sqrt(gmm.covariances_.ravel()[idx])
    ci95  = 1.96*stds
    mae   = np.mean(np.abs(j_values - np.mean(j_values)))
    print(f"\n  -- Bimodal fit: {label} --")
    print(f"  Peak 1:      {means[0]:.4f} ± {ci95[0]:.4f} Hz  (95% CI, σ={stds[0]:.4f})")
    print(f"  Peak 2:      {means[1]:.4f} ± {ci95[1]:.4f} Hz  (95% CI, σ={stds[1]:.4f})")
    print(f"  Global mean: {np.mean(j_values):.4f} Hz")
    print(f"  MAE:         {mae:.4f} Hz")
    return means, ci95, mae

def _leg_row(label, mean, mae):
    """Build a monospace-aligned legend string.
    Columns: basis(5) | ⟨J⟩ = | value(5) | ± | err(5) | Hz
    Both '=' and '±' align vertically across all rows."""
    return f"{label} \u27e8J\u27e9 = {mean:5.2f} \u00b1 {mae:5.2f} Hz"

def plot_overlay(variant_data, title, output, exp_mean=None, exp_std=None, gmm_peaks=None):
    """variant_data: list of (label, j_values, color).
    gmm_peaks: dict label -> (means, ci95) for peak annotations."""
    finite = [v for _, v, _ in variant_data if v.size > 0]
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

    for label, vals, color in variant_data:
        if vals.size < 2:
            continue
        mean = cubic_mean(vals)
        mae  = np.mean(np.abs(vals - np.mean(vals)))
        kde  = gaussian_kde(vals, bw_method=0.3)
        y    = kde(x)
        kde_store[label] = (x, y, color)
        line, = ax.plot(x, y, color=color, linewidth=2.5)
        ax.axvline(mean, color=color, linestyle=":", linewidth=1.5, alpha=0.8)
        leg_handles.append(line)
        leg_labels.append(_leg_row(label, mean, mae))

    if exp_mean is not None:
        exp_line = ax.axvline(exp_mean, color=LETTER_COLOUR, ls="--", linewidth=1.8)
        if exp_std is not None:
            ax.axvspan(exp_mean-exp_std, exp_mean+exp_std,
                       color=LETTER_COLOUR, alpha=0.12, zorder=0)
        # align exp row: same column widths, 'J exp' in place of 'TZ2PJ ⟨J⟩'
        exp_txt = f"J exp     = {exp_mean:5.3f} \u00b1 {exp_std:5.3f} Hz"
        leg_handles.append(exp_line)
        leg_labels.append(exp_txt)

    all_y = [y for _, y, _ in kde_store.values()]
    ymax  = max(np.max(y) for y in all_y)
    ax.set_ylim(0, ymax*1.1)

    if gmm_peaks is not None:
        for label, (peaks_m, peaks_ci) in gmm_peaks.items():
            if label not in kde_store:
                continue
            xc, yc, color = kde_store[label]
            for pm, pci in zip(peaks_m, peaks_ci):
                # KDE height at the GMM mean position
                y_at_pm = float(np.interp(pm, xc, yc))
                # vertical line from 0 to the curve height only
                ax.vlines(pm, 0, y_at_pm, color=color, linestyle="--",
                          linewidth=1.0, alpha=0.6)
                # text uses GMM mean ± ci95, placed above the curve
                txt = f"{pm:.2f}\u00b1{pci:.2f}"
                ax.text(pm, y_at_pm + ymax*0.03, txt,
                        ha="center", va="bottom", fontsize=12, color=color)

    ax.set_xlabel("J coupling (Hz)")
    ax.set_ylabel("Relative frequency")
    ax.set_title(title)
    ax.set_xlim(0, xmax)

    style_axes(ax, LETTER_COLOUR)

    if TRANSPARENT:
        leg = ax.legend(leg_handles, leg_labels, loc="upper right", frameon=False,
                        prop={"family": "monospace", "size": 12})
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor("none")
        leg.get_frame().set_alpha(0)
    else:
        leg = ax.legend(leg_handles, leg_labels, loc="upper right", frameon=True,
                        prop={"family": "monospace", "size": 12})
        frame = leg.get_frame()
        frame.set_facecolor((1.0, 1.0, 1.0, 1.0))
        frame.set_edgecolor(LETTER_COLOUR)
        frame.set_alpha(1.0)
        frame.set_linewidth(1.0)
    leg.set_zorder(20)

    fig.tight_layout()
    fig.savefig(PLOT_DIR + "/" + output, dpi=150, transparent=TRANSPARENT)
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
    fig.savefig(PLOT_DIR + "/" + output, dpi=150, transparent=TRANSPARENT)
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
    if variant in ["TZ2PJ_all", "TZ2P_all"]:
        continue
    steps  = get_processed_steps(cursor, variant)
    j      = collect_j_values(cursor, steps, "intra", variant)
    print_stats(j, f"Intra · {label}")
    result = fit_bimodal(j, f"Intra · {label}")
    if result is not None:
        means, ci95, _ = result
        intra_peaks[label] = (means, ci95)
    intra_data.append((label, j, color))

# inter · all variants overlay
inter_data = []
for variant, label, color in VARIANTS:
    steps = get_processed_steps(cursor, variant)
    j     = collect_j_values(cursor, steps, "inter", variant)
    if variant == "TZ2PJ_all":
        j = j[j <= 8.0]
    print_stats(j, f"Inter · {label}")
    inter_data.append((label, j, color))

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    plot_overlay(intra_data, "Intramolecular J coupling (CH2-CH2)",
                 f"hist_intra{SUFFIX}.pdf", exp_mean=EXP_INTRA, exp_std=EXP_INTRA_ERR,
                 gmm_peaks=intra_peaks)
    plot_overlay(inter_data, "Intermolecular J coupling (NH2-CH3)",
                 f"hist_inter{SUFFIX}.pdf", exp_mean=EXP_INTER, exp_std=EXP_INTER_ERR)

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

