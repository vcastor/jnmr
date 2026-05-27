#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

from hassan_functions.db import column_exists
from hassan_functions.plotting import PLOT_STYLES, style_axes

plt.rcParams["text.usetex"]         = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

PLOT_DIR = "plots"
DB_PATH  = "nmr_jcoupling.db"
VARIANT  = "TZ2P_FC"
J_COL    = f"J_{VARIANT}"

DIST_FORMS = {
    "linear":    lambda d: d,
    "quadratic": lambda d: d*d,
    "cubic":     lambda d: d*d*d,
    "inverse":   lambda d: 1/d,
    "sqrt":      lambda d: np.sqrt(d),
}

def fit_lstsq(X, y):
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid  = y - X@coeffs
    n, p   = X.shape
    sigma2 = (resid@resid)/(n - p)
    errs   = np.sqrt(np.diag(sigma2*np.linalg.inv(X.T@X)))
    rmse   = np.sqrt(np.mean(resid**2))
    return coeffs, errs, rmse

def print_fit(label, names, coeffs, errs, rmse):
    parts = "  ".join(f"{n}={v:+.4f}±{e:.4f}"
                      for n, v, e in zip(names, coeffs, errs))
    print(f"  [{label:<26}] {parts}  RMSE={rmse:.4f} Hz")

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute(f"SELECT n_step FROM snapshots WHERE comment_{VARIANT} IS NULL")
steps = sorted(r[0] for r in cursor.fetchall())

thetas, dists, dis, js = [], [], [], []
for n_step in steps:
    t = f"step_{n_step}_intra"
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (t,),
    )
    if cursor.fetchone() is None:                 continue
    if not column_exists(cursor, t, "dihedral"):  continue
    if not column_exists(cursor, t, "distance"):  continue
    if not column_exists(cursor, t, "DI"):        continue
    if not column_exists(cursor, t, J_COL):       continue
    cursor.execute(
        f"SELECT dihedral, distance, DI, {J_COL} FROM {t} "
        f"WHERE dihedral IS NOT NULL AND distance IS NOT NULL "
        f"AND DI IS NOT NULL AND {J_COL} IS NOT NULL"
    )
    for theta, d, di, j in cursor.fetchall():
        thetas.append(theta)
        dists.append(d)
        dis.append(di)
        js.append(j)

conn.close()

thetas = np.asarray(thetas, dtype=float)
dists  = np.asarray(dists,  dtype=float)
dis    = np.asarray(dis,    dtype=float)
js     = np.abs(np.asarray(js, dtype=float))
c      = np.cos(thetas)
ones   = np.ones_like(c)

print(f"=== {VARIANT}: n = {thetas.size}, "
      f"d in [{dists.min():.3f}, {dists.max():.3f}], "
      f"DI in [{dis.min():.3f}, {dis.max():.3f}] ===")

results = {}

# --- step 1: pure Karplus ---
print("\n--- Karplus ---")
X = np.column_stack([ones, c, c*c])
results["karplus"] = fit_lstsq(X, js)
print_fit("karplus", ["a", "b", "g"], *results["karplus"])

# --- step 2: Karplus + f(d) ---
print("\n--- Karplus + f(d) ---")
for name, func in DIST_FORMS.items():
    fd = func(dists)
    X  = np.column_stack([ones, c, c*c, fd])
    results[f"karplus+{name}"] = fit_lstsq(X, js)
    print_fit(f"karplus+{name}(d)", ["a", "b", "g", "d"],
              *results[f"karplus+{name}"])

# --- step 3: Karplus + DI ---
print("\n--- Karplus + DI ---")
X = np.column_stack([ones, c, c*c, dis])
results["karplus+DI"] = fit_lstsq(X, js)
print_fit("karplus+DI", ["a", "b", "g", "e"], *results["karplus+DI"])

# --- step 4: Karplus + DI + f(d) ---
print("\n--- Karplus + DI + f(d) ---")
for name, func in DIST_FORMS.items():
    fd = func(dists)
    X  = np.column_stack([ones, c, c*c, dis, fd])
    results[f"karplus+DI+{name}"] = fit_lstsq(X, js)
    print_fit(f"karplus+DI+{name}(d)", ["a", "b", "g", "e", "d"],
              *results[f"karplus+DI+{name}"])

print("\n=== RMSE summary (Hz) ===")
for key, (_, _, rmse) in results.items():
    print(f"  {key:<26} {rmse:.4f}")

# --- plotting ---
xticks  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
xlabels = ["0", r"$\sfrac\pi4$", r"$\sfrac\pi2$",
           r"$\sfrac{3\pi}4$", r"$\pi$"]

best_d  = min(DIST_FORMS, key=lambda n: results[f"karplus+{n}"][2])
best_ed = min(DIST_FORMS, key=lambda n: results[f"karplus+DI+{n}"][2])
d_mean  = dists.mean()
di_mean = dis.mean()

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, ax = plt.subplots(figsize=(9, 6))
    tc   = np.linspace(0, np.pi, 400)
    ctc  = np.cos(tc)
    ctc2 = ctc*ctc

    ax.scatter(thetas, js, alpha=0.25, s=10, color="tab:gray", label="data")

    a, b, g = results["karplus"][0]
    rmse    = results["karplus"][2]
    ax.plot(tc, a + b*ctc + g*ctc2, color="tab:green", linewidth=2,
            label=fr"Karplus (RMSE {rmse:.2f})")

    a, b, g, dlt = results[f"karplus+{best_d}"][0]
    rmse         = results[f"karplus+{best_d}"][2]
    fd_m         = DIST_FORMS[best_d](d_mean)
    ax.plot(tc, a + b*ctc + g*ctc2 + dlt*fd_m,
            color="tab:blue", linewidth=2,
            label=fr"Karplus + {best_d}(d) (RMSE {rmse:.2f})")

    a, b, g, e = results["karplus+DI"][0]
    rmse       = results["karplus+DI"][2]
    ax.plot(tc, a + b*ctc + g*ctc2 + e*di_mean,
            color="tab:orange", linewidth=2,
            label=fr"Karplus + DI (RMSE {rmse:.2f})")

    a, b, g, e, dlt = results[f"karplus+DI+{best_ed}"][0]
    rmse            = results[f"karplus+DI+{best_ed}"][2]
    fd_m            = DIST_FORMS[best_ed](d_mean)
    ax.plot(tc, a + b*ctc + g*ctc2 + e*di_mean + dlt*fd_m,
            color="tab:purple", linewidth=2,
            label=fr"Karplus + DI + {best_ed}(d) (RMSE {rmse:.2f})")

    ax.set_xlabel(r"$\vert$H-C-C-H$\vert$ dihedral angle (rad)")
    ax.set_ylabel("J (Hz)")
    ax.set_title(r"Stepwise Karplus fits (curves at $\bar d$, $\overline{DI}$)")
    ax.set_xlim(0, np.pi)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.legend(loc="best", fontsize=9)
    style_axes(ax, LETTER_COLOUR)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/karplus_stages_{VARIANT}{SUFFIX}.pdf",
                transparent=TRANSPARENT)
    plt.close(fig)

