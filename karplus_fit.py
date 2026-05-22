#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"]         = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

PLOT_DIR    = "plots"
DB_PATH     = "nmr_jcoupling.db"
VARIANTS    = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]
COLORS      = {"TZ2P_FC":  "tab:green",
               "TZ2P_all": "tab:blue",
               "TZ2PJ_FC": "tab:orange",
               "TZ2PJ_all":"tab:purple"}

DIST_FORMS = {
    "none":      None,
    "linear":    lambda d: d,
    "quadratic": lambda d: d*d,
    "cubic":     lambda d: d*d*d,
    "inverse":   lambda d: 1/d,
    "sqrt":      lambda d: np.sqrt(d),
}

PLOT_STYLES = [("black", False, ""), ("white", True, "_transparent")]


def column_exists(cursor, table, col):
    cursor.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cursor.fetchall())


def fit_model(c, fd, js):
    if fd is None:
        X = np.column_stack([np.ones_like(c), c, c*c])
    else:
        X = np.column_stack([np.ones_like(c), c, c*c, fd])
    coeffs, *_ = np.linalg.lstsq(X, js, rcond=None)
    resid  = js - X@coeffs
    n, p   = X.shape
    sigma2 = (resid@resid)/(n - p)
    errs   = np.sqrt(np.diag(sigma2*np.linalg.inv(X.T@X)))
    rmse   = np.sqrt(np.mean(resid**2))
    return coeffs, errs, rmse


def style_axes(ax, letter_colour):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(letter_colour)
    ax.tick_params(colors=letter_colour, which="both")
    ax.xaxis.label.set_color(letter_colour)
    ax.yaxis.label.set_color(letter_colour)
    ax.title.set_color(letter_colour)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor(letter_colour)
        for t in leg.get_texts():
            t.set_color(letter_colour)


conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

results = {}
for variant in VARIANTS:
    j_col = f"J_{variant}"
    cursor.execute(f"SELECT n_step FROM snapshots WHERE comment_{variant} IS NULL")
    steps = sorted(r[0] for r in cursor.fetchall())

    thetas, dists, js = [], [], []
    for n_step in steps:
        t = f"step_{n_step}_intra"
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (t,),
        )
        if cursor.fetchone() is None:                 continue
        if not column_exists(cursor, t, "dihedral"):  continue
        if not column_exists(cursor, t, "distance"):  continue
        if not column_exists(cursor, t, j_col):       continue
        cursor.execute(
            f"SELECT dihedral, distance, {j_col} FROM {t} "
            f"WHERE dihedral IS NOT NULL AND distance IS NOT NULL "
            f"AND {j_col} IS NOT NULL"
        )
        for theta, d, j in cursor.fetchall():
            thetas.append(theta)
            dists.append(d)
            js.append(j)

    if not thetas:
        print(f"{variant}: no data")
        continue

    thetas = np.asarray(thetas, dtype=float)
    dists  = np.asarray(dists,  dtype=float)
    js     = np.abs(np.asarray(js, dtype=float))
    c      = np.cos(thetas)

    print(f"\n=== {variant}: n = {thetas.size}, "
          f"d in [{dists.min():.3f}, {dists.max():.3f}] ===")

    variant_fits = {}
    for name, func in DIST_FORMS.items():
        fd = None if func is None else func(dists)
        coeffs, errs, rmse = fit_model(c, fd, js)
        variant_fits[name] = (coeffs, errs, rmse)
        if func is None:
            a, b, g       = coeffs
            da, db, dg    = errs
            print(f"  [{name:<9}] a={a:+.4f}±{da:.4f}  "
                  f"b={b:+.4f}±{db:.4f}  g={g:+.4f}±{dg:.4f}  "
                  f"RMSE={rmse:.4f} Hz")
        else:
            a, b, g, dlt   = coeffs
            da, db, dg, dd = errs
            print(f"  [{name:<9}] a={a:+.4f}±{da:.4f}  "
                  f"b={b:+.4f}±{db:.4f}  g={g:+.4f}±{dg:.4f}  "
                  f"d={dlt:+.4f}±{dd:.4f}  RMSE={rmse:.4f} Hz")

    results[variant] = dict(thetas=thetas, dists=dists, js=js, fits=variant_fits)

conn.close()

print("\n=== RMSE summary (Hz) ===")
print(f"{'variant':<12}" + "".join(f"{n:>11}" for n in DIST_FORMS))
for variant in VARIANTS:
    if variant not in results:
        continue
    row = f"{variant:<12}"
    for name in DIST_FORMS:
        row += f"{results[variant]['fits'][name][2]:>11.4f}"
    print(row)

xticks  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
xlabels = ["0", r"$\sfrac\pi4$", r"$\sfrac\pi2$",
           r"$\sfrac{3\pi}4$", r"$\pi$"]

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, ax = plt.subplots(figsize=(9, 6))
    tc = np.linspace(0, np.pi, 400)
    for variant in VARIANTS:
        if variant not in results or not variant.endswith("_FC"):
            continue
        r   = results[variant]
        col = COLORS[variant]
        best = min((n for n in DIST_FORMS if n != "none"),
                   key=lambda n: r["fits"][n][2])
        coeffs, _, rmse = r["fits"][best]
        a, b, g, dlt = coeffs
        fd_mean = DIST_FORMS[best](r["dists"].mean())
        ax.scatter(r["thetas"], r["js"], alpha=0.3, s=10, color=col)
        label = (fr"{variant.replace('_', ' ')} + {best}(d)  "
                 fr"(RMSE {rmse:.2f} Hz)")
        ax.plot(tc, a + b*np.cos(tc) + g*np.cos(tc)**2 + dlt*fd_mean,
                color=col, linewidth=2, label=label)
    ax.set_xlabel(r"$\vert$H-C-C-H$\vert$ dihedral angle (rad)")
    ax.set_ylabel("J (Hz)")
    ax.set_title(r"Karplus + distance fits (curve at $\bar d$)")
    ax.set_xlim(0, np.pi)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.legend(loc="best", fontsize=9)
    style_axes(ax, LETTER_COLOUR)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/karplus_dist_combined{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

