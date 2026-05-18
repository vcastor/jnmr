#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PLOT_DIR    = "plots"
DB_PATH     = "nmr_jcoupling.db"
PLOT_STYLES = [('black', False, ''), ('white', True, '_transparent')]

# (variant, label, color)
VARIANTS = [
    ("TZ2P_FC",   "TZ2P FC",   "steelblue"),
    ("TZ2P_all",  "TZ2P all",  "deepskyblue"),
    ("TZ2PJ_FC",  "TZ2PJ FC",  "darkorange"),
    ("TZ2PJ_all", "TZ2PJ all", "crimson"),
]

# experimental values
EXP_INTRA     = 5.9
EXP_INTER     = 1.104
EXP_INTER_ERR = 0.031
EXP_INTRA_ERR = 0.24

def table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    )
    return cursor.fetchone() is not None

def column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())

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

def style_axes(ax):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(LETTER_COLOUR)
    ax.tick_params(colors=LETTER_COLOUR, which="both")
    ax.xaxis.label.set_color(LETTER_COLOUR)
    ax.yaxis.label.set_color(LETTER_COLOUR)
    ax.title.set_color(LETTER_COLOUR)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor(LETTER_COLOUR)
        for t in leg.get_texts():
            t.set_color(LETTER_COLOUR)

def plot_overlay(variant_data, title, output, exp_mean=None, exp_std=None):
    """variant_data: list of (label, j_values, color)."""
    finite = [v for _, v, _ in variant_data if v.size > 0]
    if not finite:
        return
    xmax = max(np.max(v) for v in finite)*1.05
    if exp_mean is not None:
        xmax = max(xmax, exp_mean*1.05)
    x = np.linspace(0, xmax, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals, color in variant_data:
        if vals.size < 2:
            continue
        mean = cubic_mean(vals)
        disp = cubic_dispersion(vals)
        leg  = f"{label}  ⟨J⟩={mean:.2f}±{disp:.2f} Hz"
        kde  = gaussian_kde(vals, bw_method=0.3)
        y    = kde(x)
        kde = gaussian_kde(vals, bw_method=0.3)
        ax.plot(x, kde(x), color=color, linewidth=2.5, label=leg)
        ax.axvline(mean, color=color, linestyle=":", linewidth=1.5, alpha=0.8)
    if exp_mean is not None:
        exp_label = (f"Exp = {exp_mean}±{exp_std} Hz" if exp_std is not None
                     else f"Exp = {exp_mean} Hz")
        ax.axvline(exp_mean, color=LETTER_COLOUR, ls="--", linewidth=1.8, label=exp_label)
        if exp_std is not None:
            ax.axvspan(exp_mean - exp_std, exp_mean + exp_std,
                       color=LETTER_COLOUR, alpha=0.12)
    ax.set_xlabel("J coupling (Hz)")
    ax.set_ylabel("Relative frequency")
    ax.set_title(title)
    ax.set_xlim(0, xmax)
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR+"/"+output, dpi=150, transparent=TRANSPARENT)
    plt.close(fig)

def plot_j_vs_distance(j_values, distances, output):
    if j_values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distances, j_values, alpha=0.4, s=10, color="steelblue")
    ax.set_xlabel("H-H distance (A)")
    ax.set_ylabel("J coupling (Hz)")
    ax.set_title("Inter-molecular J coupling vs distance")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR+output, dpi=150, transparent=TRANSPARENT)
    plt.close(fig)

# ============================== #
#              Main
# ============================== #

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# intra · all variants overlay
intra_data = []
for variant, label, color in VARIANTS:
    if variant in ["TZ2PJ_all", "TZ2P_all"]:
        continue
    steps = get_processed_steps(cursor, variant)
    j     = collect_j_values(cursor, steps, "intra", variant)
    print_stats(j, f"Intra · {label}")
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
                 f"hist_intra{SUFFIX}.pdf", exp_mean=EXP_INTRA)
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
# ax.axhspan(EXP_INTER - EXP_INTER_ERR, EXP_INTER + EXP_INTER_ERR,
#            color=LETTER_COLOUR, alpha=0.12)
# ax.legend()
# style_axes(ax)
# fig.tight_layout()
# fig.savefig(f"{PLOT_DIR}/j_vs_power.pdf", dpi=150, transparent=TRANSPARENT)
# plt.close(fig)

conn.close()

