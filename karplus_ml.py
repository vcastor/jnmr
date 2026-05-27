#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm             import SVR
from sklearn.kernel_ridge    import KernelRidge
from sklearn.gaussian_process         import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import KFold, cross_val_score

from hassan_functions.db       import column_exists
from hassan_functions.plotting import PLOT_STYLES, style_axes

plt.rcParams["text.usetex"]         = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

PLOT_DIR = "plots"
DB_PATH  = "nmr_jcoupling.db"
VARIANT  = "TZ2P_FC"
J_COL    = f"J_{VARIANT}"
RNG      = 42
N_SPLITS = 5

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

X = np.column_stack([thetas, dists, dis])
y = js

print(f"=== {VARIANT}: n = {y.size}, features: [phi, distance, DI] ===")

kernel = (ConstantKernel(1.0, (1e-3, 1e3))
          *RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
          + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2)))

models = {
    "RF":  RandomForestRegressor(n_estimators=500, random_state=RNG, n_jobs=-1),
    "GBR": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                     max_depth=4, random_state=RNG),
    "SVR": Pipeline([
        ("sc",  StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)),
    ]),
    "KRR-RBF": Pipeline([
        ("sc",  StandardScaler()),
        ("krr", KernelRidge(kernel="rbf", alpha=1e-2, gamma=1.0)),
    ]),
    "GPR": Pipeline([
        ("sc",  StandardScaler()),
        ("gpr", GaussianProcessRegressor(kernel=kernel,
                                         normalize_y=True,
                                         n_restarts_optimizer=2,
                                         random_state=RNG)),
    ]),
}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)

print(f"\n=== {N_SPLITS}-fold CV RMSE (Hz) ===")
cv_rmse = {}
for name, model in models.items():
    neg_mse = cross_val_score(model, X, y, cv=kf,
                              scoring="neg_mean_squared_error", n_jobs=-1)
    rmse    = np.sqrt(-neg_mse)
    cv_rmse[name] = rmse
    print(f"  {name:<10}  {rmse.mean():.4f} ± {rmse.std():.4f}")

predictions = {}
for name, model in models.items():
    model.fit(X, y)
    predictions[name] = model.predict(X)

print("\n=== train RMSE (Hz) ===")
for name, ypred in predictions.items():
    print(f"  {name:<10}  {np.sqrt(np.mean((y - ypred)**2)):.4f}")

# --- parity plot ---
n_models = len(models)
cols     = 3
rows     = (n_models + cols - 1)//cols

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (name, ypred) in zip(axes, predictions.items()):
        ax.scatter(y, ypred, alpha=0.3, s=10, color="tab:blue")
        lo = min(y.min(), ypred.min())
        hi = max(y.max(), ypred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel(r"$J_\mathrm{ref}$ (Hz)")
        ax.set_ylabel(r"$J_\mathrm{pred}$ (Hz)")
        ax.set_title(fr"{name}  CV RMSE {cv_rmse[name].mean():.2f} Hz")
        style_axes(ax, LETTER_COLOUR)
    for ax in axes[n_models:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/ml_parity_{VARIANT}{SUFFIX}.pdf",
                transparent=TRANSPARENT)
    plt.close(fig)

