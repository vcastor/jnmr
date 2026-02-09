#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt


def get_processed_steps(cursor):
    cursor.execute("SELECT n_step FROM snapshots WHERE comment IS NULL")
    return [row[0] for row in cursor.fetchall()]


def collect_j_values(cursor, steps, table_type):
    max_per_system = []
    for n_step in steps:
        table_name = f"step_{n_step}_{table_type}"
        cursor.execute(f"""
            SELECT MAX(ABS(J_fermi))
            FROM {table_name}
            WHERE J_fermi IS NOT NULL
        """)
        value = cursor.fetchone()[0]
        if value is not None:
            max_per_system.append(value)
    return np.array(max_per_system)


def collect_j_with_distance(cursor, steps):
    j_values  = []
    distances = []
    for n_step in steps:
        table_name = f"step_{n_step}_inter"
        cursor.execute(f"""
            SELECT J_fermi, distance
            FROM {table_name}
            WHERE J_fermi IS NOT NULL
        """)
        for j,d in cursor.fetchall():
            j_values.append(j)
            distances.append(d)
    return np.array(j_values), np.array(distances)


def print_stats(j_values, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  N samples:  {j_values.size}")
    print(f"  Mean:       {np.mean(j_values):.4f} Hz")
    print(f"  Median:     {np.median(j_values):.4f} Hz")
    print(f"  Std dev:    {np.std(j_values):.4f} Hz")
    print(f"  Min:        {np.min(j_values):.4f} Hz")
    print(f"  Max:        {np.max(j_values):.4f} Hz")


def plot_histogram(j_values, title, color, output):
    bin_width = 0.5
    bins = np.arange(0, np.max(j_values)+bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(j_values, bins=bins, edgecolor='black', alpha=0.7, color=color)
    ax.axvline(np.mean(j_values), linestyle='--', label=f"mean = {np.mean(j_values):.2f} Hz")
    ax.set_xlabel("J coupling (Hz)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(0, np.max(j_values)*1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_j_vs_distance(j_values, distances, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distances, j_values, alpha=0.4, s=10)
    ax.set_xlabel("H-H distance (A)")
    ax.set_ylabel("J coupling (Hz)")
    ax.set_title("Inter-molecular J coupling vs distance")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# ============================== #
#              Main
# ============================== #

conn = sqlite3.connect("nmr_jcoupling.db")
cursor = conn.cursor()

steps = get_processed_steps(cursor)

j_intra = collect_j_values(cursor, steps, "intra")
print_stats(j_intra, "Intra-molecular (CH2-CH2)")
plot_histogram(
    j_intra,
    "Intra-molecular J coupling (CH2-CH2)",
    "steelblue",
    "hist_intra.pdf"
)

j_inter = collect_j_values(cursor, steps, "inter")
print_stats(j_inter, "Inter-molecular (NH2-CH3)")
plot_histogram(
    j_inter,
    "Inter-molecular J coupling (NH2-CH3)",
    "darkorange",
    "hist_inter.pdf"
)

# j_inter_d, distances = collect_j_with_distance(cursor, steps)
# plot_j_vs_distance(j_inter_d, distances, "j_vs_distance.pdf")

conn.close()

