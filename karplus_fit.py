#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR    = "plots"
DB_PATH     = "nmr_jcoupling.db"
VARIANTS    = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]
PLOT_STYLES = [("black", False, ""), ("white", True, "_transparent")]

def column_exists(cursor, table, col):
    cursor.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cursor.fetchall())

def karplus(theta_rad, alpha, beta, gamma):
    c = np.cos(theta_rad)
    return alpha + beta*c + gamma*c*c

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

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

for variant in VARIANTS:
    j_col = f"J_{variant}"
    cursor.execute(f"SELECT n_step FROM snapshots WHERE comment_{variant} IS NULL")
    steps = sorted(r[0] for r in cursor.fetchall())

    thetas, js = [], []
    for n_step in steps:
        t = f"step_{n_step}_intra"
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (t,),
        )
        if cursor.fetchone() is None:
            continue
        if not column_exists(cursor, t, "dihedral"):
            continue
        if not column_exists(cursor, t, j_col):
            continue
        cursor.execute(
            f"SELECT dihedral, {j_col} FROM {t} "
            f"WHERE dihedral IS NOT NULL AND {j_col} IS NOT NULL"
        )
        for theta, j in cursor.fetchall():
            thetas.append(theta)
            js.append(j)

    if not thetas:
        print(f"{variant}: no data")
        continue

    thetas = np.asarray(thetas, dtype=float)
    js     = np.asarray(js,     dtype=float)
    c      = np.cos(thetas)

    # J = a + b·cos(θ) + g·cos²(θ)  → polyfit in c, deg 2 → [g, b, a]
    coeffs, cov = np.polyfit(c, js, 2, cov=True)
    g, b, a     = coeffs
    da, db, dg  = np.sqrt(np.diag(cov))[::-1]
    j_fit       = a + b*c + g*c*c
    rmse        = np.sqrt(np.mean((js - j_fit)**2))

    # Drop the worst residuals and refit (db untouched).
    if variant == "TZ2P_FC":
        worst = np.argsort(np.abs(js - j_fit))[-10:]
    else:
        worst = np.argsort(np.abs(js - j_fit))[-2:]
    mask = np.ones(thetas.size, dtype=bool)
    mask[worst] = False

    c2, js2 = c[mask], js[mask]
    coeffs2, cov2 = np.polyfit(c2, js2, 2, cov=True)
    g2, b2, a2    = coeffs2
    da2, db2, dg2 = np.sqrt(np.diag(cov2))[::-1]
    rmse2 = np.sqrt(np.mean((js2 - (a2 + b2*c2 + g2*c2*c2))**2))

    print(f"\n{variant}: n = {thetas.size}")
    print("  -- raw fit --")
    print(f"  alpha = {a:+.4f} ± {da:.4f} Hz")
    print(f"  beta  = {b:+.4f} ± {db:.4f} Hz")
    print(f"  gamma = {g:+.4f} ± {dg:.4f} Hz")
    print(f"  RMSE  = {rmse:.4f} Hz")
    print("  -- outliers removed --")
    print(f"  alpha = {a2:+.4f} ± {da2:.4f} Hz")
    print(f"  beta  = {b2:+.4f} ± {db2:.4f} Hz")
    print(f"  gamma = {g2:+.4f} ± {dg2:.4f} Hz")
    print(f"  RMSE  = {rmse2:.4f} Hz")

    for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(thetas[mask],  js[mask],
                   alpha=0.25, s=8,  color="steelblue", label="data")
        ax.scatter(thetas[~mask], js[~mask],
                   alpha=0.9,  s=45, color="red", marker="x", linewidths=1.8,
                   label="outliers (excluded)")
        tc = np.linspace(0, np.pi, 400)
        ax.plot(tc, karplus(tc, a2, b2, g2), color="crimson", linewidth=2,
                label=fr"${a2:+.2f}{b2:+.2f}\cos\theta{g2:+.2f}\cos^2\theta$")
        ax.set_xlabel(r"|H-C-C-H| dihedral $\theta$ (rad)")
        ax.set_ylabel(f"J ({variant}) (Hz)")
        ax.set_title(f"Karplus fit · {variant}  (RMSE {rmse2:.2f} Hz, 5 outliers excluded)")
        ax.set_xlim(0, np.pi)
        ax.legend()
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/karplus_{variant}{SUFFIX}.pdf", transparent=TRANSPARENT)
        plt.close(fig)

conn.close()

