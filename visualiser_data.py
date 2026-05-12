#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DB_PATH = "nmr_jcoupling.db"

# (variant, label, color, draw_bars)
VARIANTS = [
    ("TZ2P_FC",   "TZ2P FC",   "steelblue",  True),
    ("TZ2P_all",  "TZ2P all",  "deepskyblue", False),
    ("TZ2PJ_FC",  "TZ2PJ FC",  "darkorange", False),
    ("TZ2PJ_all", "TZ2PJ all", "crimson",    False),
]

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

def collect_j_values(cursor, steps, table_type, basis_cont, p=3):
    j_col = f"J_{basis_cont}"
    all_values = []
    for n_step in steps:
        table_name = f"step_{n_step}_{table_type}"
        if not table_exists(cursor, table_name):
            continue
        if not column_exists(cursor, table_name, j_col):
            continue
        cursor.execute(f"SELECT {j_col} FROM {table_name} WHERE {j_col} IS NOT NULL")
        vals = np.array([abs(row[0]) for row in cursor.fetchall()])
        if vals.size > 0:
            j_eff = (np.mean(vals**p))**(1.0/p)
            all_values.append(j_eff)
    return np.array(all_values, dtype=float)

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
    print(f"  N samples:  {j_values.size}")
    if j_values.size == 0:
        return
    print(f"  Mean:       {np.mean(j_values):.4f} Hz")
    print(f"  Median:     {np.median(j_values):.4f} Hz")
    print(f"  Std dev:    {np.std(j_values):.4f} Hz")
    print(f"  Min:        {np.min(j_values):.4f} Hz")
    print(f"  Max:        {np.max(j_values):.4f} Hz")

def style_axes(ax):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor("white")
        for t in leg.get_texts():
            t.set_color("white")

def plot_overlay(variant_data, title, output, bin_width=0.4):
    """variant_data: list of (label, j_values, color, draw_bars)."""
    finite = [v for _, v, _, _ in variant_data if v.size > 0]
    if not finite:
        return
    xmax = max(np.max(v) for v in finite)*1.05
    bins = np.arange(0, xmax + bin_width, bin_width)
    x    = np.linspace(0, xmax, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals, color, draw_bars in variant_data:
        if vals.size == 0:
            continue
        if draw_bars:
            ax.hist(vals, bins=bins, edgecolor="white", alpha=0.4, color=color)
        kde = gaussian_kde(vals, bw_method=0.3)
        ax.plot(x, kde(x)*vals.size*bin_width,
                color=color, linewidth=2.5, label=label)
    ax.set_xlabel("J coupling (Hz)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(0, xmax)
    ax.legend()
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150, transparent=True)
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
    fig.savefig(output, dpi=150, transparent=True)
    plt.close(fig)

# ============================== #
#              Main
# ============================== #

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# intra · all variants overlay
intra_data = []
for variant, label, color, bars in VARIANTS:
    steps = get_processed_steps(cursor, variant)
    j     = collect_j_values(cursor, steps, "intra", variant)
    print_stats(j, f"Intra · {label}")
    intra_data.append((label, j, color, bars))
plot_overlay(intra_data, "Intra-molecular J coupling (CH2-CH2)", "hist_intra.pdf")

# inter · all variants overlay
inter_data = []
for variant, label, color, bars in VARIANTS:
    steps = get_processed_steps(cursor, variant)
    j     = collect_j_values(cursor, steps, "inter", variant)
    print_stats(j, f"Inter · {label}")
    inter_data.append((label, j, color, bars))
plot_overlay(inter_data, "Inter-molecular J coupling (NH2-CH3)", "hist_inter.pdf")

# Sensitivity of effective J to power-mean exponent (TZ2P_FC reference)
basis_cont = "TZ2P_FC"
steps = get_processed_steps(cursor, basis_cont)
fig, ax = plt.subplots(figsize=(8, 5))
ps    = np.arange(1, 6.1, 0.25)
means = []
for p in ps:
    jvals = collect_j_values(cursor, steps, "inter", basis_cont, p=p)
    means.append(np.mean(jvals) if jvals.size else 0)
ax.plot(ps, means, "o-", color="darkorange")
ax.set_xlabel("Power mean exponent p")
ax.set_ylabel("Mean effective J (Hz)")
ax.set_title("Sensitivity of effective J to power mean exponent")
ax.axhline(1.1, color="white", ls="--", label="Experimental")
ax.legend()
style_axes(ax)
fig.tight_layout()
fig.savefig("j_vs_power.pdf", dpi=150, transparent=True)
plt.close(fig)

conn.close()
