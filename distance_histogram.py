#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = "nmr_jcoupling.db"

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'step_%_inter'"
)
tables = [r[0] for r in cursor.fetchall()]

dists = []
for t in tables:
    cursor.execute(f"SELECT distance FROM {t} WHERE distance IS NOT NULL")
    dists.extend(r[0] for r in cursor.fetchall())

conn.close()

d = np.asarray(dists, dtype=float)
n = d.size

stats = {
    "count":   n,
    "mean":    d.mean(),
    "median":  np.median(d),
    "std":     d.std(ddof=1),
    "min":     d.min(),
    "max":     d.max(),
    "p05":     np.percentile(d,  5),
    "p25":     np.percentile(d, 25),
    "p75":     np.percentile(d, 75),
    "p95":     np.percentile(d, 95),
}

print(f"tables scanned : {len(tables)}")
print(f"non-null count : {n}")
for k in ("mean", "median", "std", "min", "max", "p05", "p25", "p75", "p95"):
    print(f"{k:>7} : {stats[k]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(d, bins=80, color="steelblue", edgecolor="black", alpha=0.85)
ax.axvline(stats["median"], color="crimson",  linestyle="--", linewidth=1.5,
           label=f"median = {stats['median']:.3f}")
ax.axvline(stats["mean"],   color="darkorange", linestyle=":", linewidth=1.5,
           label=f"mean = {stats['mean']:.3f}")
ax.set_xlabel("inter distance")
ax.set_ylabel("count")
ax.set_title(f"Inter distance distribution ({n} values from {len(tables)} tables)")
ax.legend()
fig.tight_layout()
fig.savefig("distance_histogram.png", dpi=150)

