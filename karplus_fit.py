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

DI_FORMS = {
    "linear":    lambda x: x,
    "quadratic": lambda x: x*x,
    "cubic":     lambda x: x*x*x,
    "inverse":   lambda x: 1/x,
    "sqrt":      lambda x: np.sqrt(x),
    "exp":       lambda x: np.exp(x),
    "log":       lambda x: np.log(x),
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
    print(f"  [{label:<34}] {parts}  RMSE={rmse:.4f} Hz")


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

# precompute transformed features, skip forms producing non-finite values
def transform_feats(forms, x):
    out, skipped = {}, []
    for n, f in forms.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            v = f(x)
        if np.all(np.isfinite(v)):
            out[n] = v
        else:
            skipped.append(n)
    return out, skipped

dist_feats, dist_skipped = transform_feats(DIST_FORMS, dists)
di_feats,   di_skipped   = transform_feats(DI_FORMS,   dis)
if dist_skipped:
    print(f"  skipped dist forms (non-finite): {dist_skipped}")
if di_skipped:
    print(f"  skipped DI forms (non-finite): {di_skipped}")

results = {}

# --- step 1: pure Karplus ---
print("\n--- Karplus ---")
X = np.column_stack([ones, c, c*c])
results["karplus"] = fit_lstsq(X, js)
print_fit("karplus", ["a", "b", "g"], *results["karplus"])

# --- step 2: Karplus + f(d) ---
for dn, fd in dist_feats.items():
    X = np.column_stack([ones, c, c*c, fd])
    results[f"karplus+{dn}(d)"] = fit_lstsq(X, js)
best_d_key = min((k for k in results if k.endswith("(d)") and "DI" not in k),
                 key=lambda k: results[k][2])
print(f"\n--- Karplus + f(d): best of {len(dist_feats)} ---")
print_fit(best_d_key, ["a", "b", "g", "d"], *results[best_d_key])

# --- step 3: Karplus + g(DI) ---
for en, gdi in di_feats.items():
    X = np.column_stack([ones, c, c*c, gdi])
    results[f"karplus+{en}(DI)"] = fit_lstsq(X, js)
best_di_key = min((k for k in results if k.endswith("(DI)")),
                  key=lambda k: results[k][2])
print(f"\n--- Karplus + g(DI): best of {len(di_feats)} ---")
print_fit(best_di_key, ["a", "b", "g", "e"], *results[best_di_key])

# --- step 4: Karplus + g(DI) + f(d) ---
for en, gdi in di_feats.items():
    for dn, fd in dist_feats.items():
        X = np.column_stack([ones, c, c*c, gdi, fd])
        results[f"karplus+{en}(DI)+{dn}(d)"] = fit_lstsq(X, js)
best_combo_key = min((k for k in results if "(DI)+" in k),
                     key=lambda k: results[k][2])
n_combos = len(di_feats)*len(dist_feats)
print(f"\n--- Karplus + g(DI) + f(d): best of {n_combos} ---")
print_fit(best_combo_key, ["a", "b", "g", "e", "d"], *results[best_combo_key])

# --- summary ---
print("\n=== All RMSE sorted (Hz) ===")
for key in sorted(results, key=lambda k: results[k][2]):
    print(f"  {key:<34} {results[key][2]:.4f}")

# --- plotting ---
xticks  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
xlabels = ["0", r"$\sfrac\pi4$", r"$\sfrac\pi2$",
           r"$\sfrac{3\pi}4$", r"$\pi$"]

d_mean  = dists.mean()
di_mean = dis.mean()


def parse_form(key, marker):
    # extract "<form>" from "...+<form>(<marker>)..."
    seg = key.split(f"({marker})")[0]
    return seg.split("+")[-1]


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

    best_d       = parse_form(best_d_key, "d")
    a, b, g, dlt = results[best_d_key][0]
    rmse         = results[best_d_key][2]
    fd_m         = DIST_FORMS[best_d](d_mean)
    ax.plot(tc, a + b*ctc + g*ctc2 + dlt*fd_m,
            color="tab:blue", linewidth=2,
            label=fr"Karplus + {best_d}(d) (RMSE {rmse:.2f})")

    best_di    = parse_form(best_di_key, "DI")
    a, b, g, e = results[best_di_key][0]
    rmse       = results[best_di_key][2]
    gdi_m      = DI_FORMS[best_di](di_mean)
    ax.plot(tc, a + b*ctc + g*ctc2 + e*gdi_m,
            color="tab:orange", linewidth=2,
            label=fr"Karplus + {best_di}(DI) (RMSE {rmse:.2f})")

    bc_di           = parse_form(best_combo_key, "DI")
    bc_d            = parse_form(best_combo_key.split("+")[-1], "d")
    a, b, g, e, dlt = results[best_combo_key][0]
    rmse            = results[best_combo_key][2]
    gdi_m           = DI_FORMS[bc_di](di_mean)
    fd_m            = DIST_FORMS[bc_d](d_mean)
    ax.plot(tc, a + b*ctc + g*ctc2 + e*gdi_m + dlt*fd_m,
            color="tab:purple", linewidth=2,
            label=fr"Karplus + {bc_di}(DI) + {bc_d}(d) (RMSE {rmse:.2f})")

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

