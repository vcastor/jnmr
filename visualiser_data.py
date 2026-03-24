#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt


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

def collect_j_values(cursor, steps, table_type, basis_cont, p=2.7):
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

# def collect_j_values(cursor, steps, table_type, basis_cont):
#     j_col = f"J_{basis_cont}"
#     all_values = []

#     for n_step in steps:
#         table_name = f"step_{n_step}_{table_type}"
#         if not table_exists(cursor, table_name):
#             continue
#         if not column_exists(cursor, table_name, j_col):
#             continue

#         cursor.execute(f"SELECT {j_col} FROM {table_name} WHERE {j_col} IS NOT NULL")
#         vals = [row[0] for row in cursor.fetchall()]
#         if vals:
#             strongest = max(vals, key=abs)
#             all_values.append(abs(strongest))

#     return np.array(all_values, dtype=float)

def collect_j_with_distance(cursor, steps, basis_cont):
    j_col = f"J_{basis_cont}"
    j_values = []
    distances = []

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


def plot_histogram(j_values, title, color, output):
    if j_values.size == 0:
        return
    mean = np.mean(j_values)
    std = np.std(j_values)
    bin_width = 0.5
    bins = np.arange(0, np.max(j_values) + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(j_values, bins=bins, edgecolor="black", alpha=0.7, color=color)
    ax.axvline(mean, color="black", linestyle="--", label=f"mean = {mean:.2f} Hz")
    ax.axvspan(
        mean - std, mean + std, alpha=0.15, color="black", label=f"±1σ = {std:.2f} Hz"
    )
    ax.set_xlabel("J coupling (Hz)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(0, np.max(j_values) * 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_j_vs_distance(j_values, distances, output):
    if j_values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distances, j_values, alpha=0.4, s=10)
    ax.set_xlabel("H-H distance (Å)")
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

basis_cont = "TZ2P_FC"
steps = get_processed_steps(cursor, basis_cont)

j_intra = collect_j_values(cursor, steps, "intra", basis_cont)
print_stats(j_intra, "Intra-molecular (CH2-CH2) – main only")
plot_histogram(
    j_intra,
    "Intra-molecular J coupling (CH2-CH2) – main only",
    "steelblue",
    "hist_intra.pdf",
)

# Experimental code for the average
fig, ax = plt.subplots(figsize=(8, 5))
ps = np.arange(1, 6.1, 0.25)
means = []
for p in ps:
    jvals = collect_j_values(cursor, steps, "inter", basis_cont, p=p)
    means.append(np.mean(jvals) if jvals.size else 0)
ax.plot(ps, means, "o-", color="darkorange")
ax.set_xlabel("Power mean exponent p")
ax.set_ylabel("Mean effective J (Hz)")
ax.set_title("Sensitivity of effective J to power mean exponent")
ax.axhline(1.1, color="gray", ls="--", label="Experimental")
ax.legend()
fig.tight_layout()
fig.savefig("j_vs_power.pdf", dpi=150)
plt.close(fig)

j_inter = collect_j_values(cursor, steps, "inter", basis_cont)
print_stats(j_inter, "Inter-molecular (NH2-CH3) – main only")
plot_histogram(
    j_inter,
    "Inter-molecular J coupling (NH2-CH3) – main only",
    "darkorange",
    "hist_inter.pdf",
)

# j_inter_d, distances = collect_j_with_distance(cursor, steps, basis_cont)
# plot_j_vs_distance(j_inter_d, distances, "j_vs_distance.pdf")

conn.close()

