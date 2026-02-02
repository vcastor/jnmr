#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def get_processed_steps(cursor):
    cursor.execute("SELECT n_step FROM snapshots WHERE comment IS NOT 'Error processing snapshot'")
    return [row[0] for row in cursor.fetchall()]


def collect_j_values(cursor, steps, table_type):
    j_values = []
    for n_step in steps:
        table_name = f"step_{n_step}_{table_type}"
        cursor.execute(f"SELECT J_fermi FROM {table_name} WHERE J_fermi IS NOT NULL")
        j_values.extend([row[0] for row in cursor.fetchall()])
    return np.array(j_values)


def collect_j_with_distance(cursor, steps):
    data = []
    for n_step in steps:
        table_name = f"step_{n_step}_inter"
        cursor.execute(f"SELECT J_fermi, distance FROM {table_name} WHERE J_fermi IS NOT NULL")
        data.extend(cursor.fetchall())
    j_values  = np.array([d[0] for d in data])
    distances = np.array([d[1] for d in data])
    return j_values, distances


def print_stats(j_values, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  N samples:  {len(j_values)}")
    print(f"  Mean:       {np.mean(j_values):.4f} Hz")
    print(f"  Median:     {np.median(j_values):.4f} Hz")
    print(f"  Std dev:    {np.std(j_values):.4f} Hz")
    print(f"  Min:        {np.min(j_values):.4f} Hz")
    print(f"  Max:        {np.max(j_values):.4f} Hz")


def plot_intra_histogram(j_values, output="hist_intra.pdf"):
    bin_width=0.5
    bins=np.arange(np.min(j_values),np.max(j_values)+bin_width,bin_width)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(j_values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel("J coupling (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Intra-molecular J coupling (CH2-CH2)")
    ax.axvline(np.mean(j_values), color='red', linestyle='--', label=f"mean = {np.mean(j_values):.2f} Hz")
    ax.set_xlim(-10, 10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_inter_histogram(j_values, output="hist_inter.pdf"):
    bin_width=0.5
    bins=np.arange(np.min(j_values),np.max(j_values)+bin_width,bin_width)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(j_values, bins=bins, edgecolor='black', alpha=0.7, color='darkorange')
    ax.set_xlabel("J coupling (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-molecular J coupling (NH2-CH3)")
    ax.axvline(np.mean(j_values), color='red', linestyle='--', label=f"mean = {np.mean(j_values):.2f} Hz")
    ax.set_xlim(-10, 10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_j_vs_distance(j_values, distances, output="j_vs_distance.pdf"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distances, j_values, alpha=0.4, s=10, color='darkorange')
    ax.set_xlabel("H-H distance (A)")
    ax.set_ylabel("J coupling (Hz)")
    ax.set_title("Inter-molecular J coupling vs distance")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# ============================== #
#             Main
# ============================== #

conn = sqlite3.connect("nmr_jcoupling.db")
cursor = conn.cursor()

steps = get_processed_steps(cursor)

# Intra-molecular
j_intra = collect_j_values(cursor, steps, "intra")
print_stats(j_intra, "Intra-molecular (CH2-CH2)")
plot_intra_histogram(j_intra)

# Inter-molecular (written but not called)
j_inter = collect_j_values(cursor, steps, "inter")
print_stats(j_inter, "Inter-molecular (NH2-CH3)")
plot_inter_histogram(j_inter)

j_inter_d, distances = collect_j_with_distance(cursor, steps)
plot_j_vs_distance(j_inter_d, distances)

conn.close()

