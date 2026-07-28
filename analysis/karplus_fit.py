#!/usr/bin/python3
import os
import sys
import sqlite3
import itertools
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.plotting import PLOT_STYLES, style_axes, save_fig
from hassan_functions.style    import apply_style

apply_style("large")

PLOT_DIR = "plots"
DB_PATH  = "nmr_jcoupling.db"
VARIANT  = "TZ2P_FC"
J_COL    = f"J_{VARIANT}"

# descriptors that augment the Karplus base, each tried in every form below
FEATURES = ["DI", "distance", "angle_hcc", "angle_cch", "chi"]
FORMS = {
    "linear":    lambda x: x,
    "inverse":   lambda x: 1/x,
    "sqrt":      lambda x: np.sqrt(x),
    "quadratic": lambda x: x*x,
}


def fit_lstsq(X, y):
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid  = y - X@coeffs
    n, p   = X.shape
    sigma2 = (resid@resid)/(n - p)
    errs   = np.sqrt(np.diag(sigma2*np.linalg.inv(X.T@X)))
    rmse   = np.sqrt(np.mean(resid**2))
    return coeffs, errs, rmse


def table_cols(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cursor.fetchall()}


conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute(f"SELECT n_step FROM snapshots WHERE comment_{VARIANT} IS NULL")
steps = sorted(r[0] for r in cursor.fetchall())

read_cols = ["dihedral"] + FEATURES + [J_COL]
data = {c: [] for c in read_cols}
for n_step in steps:
    t = f"step_{n_step}_intra"
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (t,),
    )
    if cursor.fetchone() is None:        continue
    present = table_cols(cursor, t)
    if J_COL not in present:             continue
    sel = [c for c in read_cols if c in present]
    cursor.execute(f"SELECT {', '.join(sel)} FROM {t} WHERE {J_COL} IS NOT NULL")
    for row in cursor.fetchall():
        vals = dict(zip(sel, row))
        for c in read_cols:
            data[c].append(vals.get(c))

conn.close()

data = {c: np.array(v, dtype=float) for c, v in data.items()}   # None -> nan
js   = np.abs(data[J_COL])
c    = np.cos(data["dihedral"])
ones = np.ones_like(c)
n    = js.size

print(f"=== {VARIANT}: n = {n} (|{J_COL}|), "
      f"dihedral in [{data['dihedral'].min():.3f}, {data['dihedral'].max():.3f}] rad ===")
print("    per-feature non-null rows: "
      + ", ".join(f"{f}={np.isfinite(data[f]).sum()}" for f in FEATURES))

# --- pure Karplus (this is what gets plotted) ---
X_k = np.column_stack([ones, c, c*c])
karplus = fit_lstsq(X_k, js)
a, b, g = karplus[0]
print("\n--- pure Karplus: a + b*cos(phi) + g*cos^2(phi) ---")
ea, eb, eg = karplus[1]
print(f"  a={a:+.4f}±{ea:.4f}  b={b:+.4f}±{eb:.4f}  g={g:+.4f}±{eg:.4f}"
      f"  RMSE={karplus[2]:.4f} Hz")

# --- every Karplus + form(feature) combination ---
# per feature: absent, or one of the four forms. fit on rows where every used
# feature is non-null; skip forms that turn non-finite on those rows.
results = []
for choice in itertools.product([None] + list(FORMS), repeat=len(FEATURES)):
    used = [(f, form) for f, form in zip(FEATURES, choice) if form is not None]
    if not used:
        continue
    mask = np.isfinite(c)
    for f, _ in used:
        mask = mask & np.isfinite(data[f])
    nrows  = int(mask.sum())
    nparam = 3 + len(used)
    if nrows < nparam + 1:
        continue
    cols  = [ones[mask], c[mask], (c*c)[mask]]
    bad   = False
    label = "karplus"
    for f, form in used:
        with np.errstate(invalid="ignore", divide="ignore"):
            v = FORMS[form](data[f][mask])
        if not np.all(np.isfinite(v)):
            bad = True
            break
        cols.append(v)
        label += f"+{form}({f})"
    if bad:
        continue
    coeffs, errs, rmse = fit_lstsq(np.column_stack(cols), js[mask])
    results.append((label, nrows, nparam, rmse))

results.sort(key=lambda r: r[3])
print(f"\n--- Karplus + form(feature): top 5 of {len(results)} combinations "
      f"(training RMSE; n varies with descriptor availability) ---")
print(f"  {'#':<3}{'model':<48}{'n':>5}{'p':>4}{'RMSE (Hz)':>12}")
for i, (label, nrows, nparam, rmse) in enumerate(results[:5], 1):
    print(f"  {i:<3}{label:<48}{nrows:>5}{nparam:>4}{rmse:>12.4f}")

# --- plot: pure Karplus only ---
xticks  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
xlabels = ["0", r"$\sfrac\pi4$", r"$\sfrac\pi2$", r"$\sfrac{3\pi}4$", r"$\pi$"]
tc      = np.linspace(0, np.pi, 400)
ctc     = np.cos(tc)

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(data["dihedral"], js, alpha=0.25, s=10, color=LETTER_COLOUR,
               label="data")
    ax.plot(tc, a + b*ctc + g*ctc*ctc, color="orange", linewidth=2,
            label=fr"Karplus (RMSE {karplus[2]:.2f} Hz)")
    ax.set_xlabel(r"$\vert$H-C-C-H$\vert$ dihedral angle (rad)")
    ax.set_ylabel(r"$J$ (Hz)")
    ax.set_title(r"Karplus Fit")
    ax.set_xlim(0, np.pi)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.legend(loc="best", fontsize=16)
    style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/karplus_fit_{VARIANT}{SUFFIX}", TRANSPARENT)

