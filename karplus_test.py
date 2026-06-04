#!/usr/bin/python3
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm              import SVR
from sklearn.kernel_ridge     import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import make_pipeline
from sklearn.model_selection  import RepeatedKFold, KFold, cross_validate, cross_val_predict

from hassan_functions.db       import column_exists
from hassan_functions.plotting import PLOT_STYLES, style_axes

plt.rcParams["text.usetex"]         = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

PLOT_DIR  = "plots"
DB_PATH   = "nmr_jcoupling.db"
VARIANT   = "TZ2P_FC"
J_COL     = f"J_{VARIANT}"
RNG       = 42
N_SPLITS  = 5
N_REPEATS = 5
N_JOBS    = os.cpu_count() - 10

CONFIGS = {
    "phi":     ["dihedral"],
    "geom":    ["dihedral", "angle_hcc", "angle_cch", "distance"],
    "geom_DI": ["dihedral", "angle_hcc", "angle_cch", "distance", "DI"],
    "phi_chi": ["dihedral", "chi"],
    "phi_DI":  ["dihedral", "DI"],
    "DI":      ["DI"],
}
DESCRIPTORS = sorted({c for cols in CONFIGS.values() for c in cols})

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute(f"SELECT n_step FROM snapshots WHERE comment_{VARIANT} IS NULL")
steps = sorted(r[0] for r in cursor.fetchall())

data = {c: [] for c in DESCRIPTORS}
js   = []
for n_step in steps:
    t = f"step_{n_step}_intra"
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (t,),
    )
    if cursor.fetchone() is None:            continue
    if not column_exists(cursor, t, J_COL):  continue
    present = [c for c in DESCRIPTORS if column_exists(cursor, t, c)]
    cursor.execute(f"SELECT {', '.join(present + [J_COL])} FROM {t} WHERE {J_COL} IS NOT NULL")
    for row in cursor.fetchall():
        vals = dict(zip(present, row))
        for c in DESCRIPTORS:
            data[c].append(vals.get(c))
        js.append(row[-1])

conn.close()

data = {c: np.array(v, dtype=float) for c, v in data.items()}
y    = np.abs(np.asarray(js, dtype=float))

datasets = {}
for cfg_name, cols in CONFIGS.items():
    M    = np.column_stack([data[c] for c in cols])
    mask = ~np.isnan(M).any(axis=1)
    datasets[cfg_name] = (M[mask], y[mask])

models = {
    "RF":      RandomForestRegressor(),
    "GBR":     GradientBoostingRegressor(),
    "SVR":     make_pipeline(StandardScaler(), SVR()),
    "KRR-RBF": make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
    "GPR":     make_pipeline(StandardScaler(), GaussianProcessRegressor()),
}

cv      = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RNG)
cv_pred = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)

mae_mean = {}
for cfg_name, (X, yc) in datasets.items():
    print(f"\n=== {VARIANT} / {cfg_name}: n = {yc.size}, features: {CONFIGS[cfg_name]} ===")
    print(f"{'model':<10}{'MAE':>10}{'±std':>9}{'min':>9}{'max':>9}{'RMSE':>10}")
    mae_mean[cfg_name] = {}
    for name, model in models.items():
        sc = cross_validate(model, X, yc, cv=cv,
                            scoring=("neg_mean_absolute_error",
                                     "neg_root_mean_squared_error"),
                            n_jobs=N_JOBS)
        mae  = -sc["test_neg_mean_absolute_error"]
        rmse = -sc["test_neg_root_mean_squared_error"]
        mae_mean[cfg_name][name] = mae.mean()
        print(f"{name:<10}{mae.mean():>10.4f}{mae.std():>9.4f}"
              f"{mae.min():>9.4f}{mae.max():>9.4f}{rmse.mean():>10.4f}")

best = min(((c, m, v) for c, mm in mae_mean.items() for m, v in mm.items()),
           key=lambda x: x[2])
print(f"\nbest (defaults): {best[1]} on '{best[0]}'  MAE = {best[2]:.4f} Hz "
      f"(mean over {N_SPLITS*N_REPEATS} splits)")

n_models = len(models)
cols     = 3
rows     = (n_models + cols - 1)//cols

for cfg_name, (X, yc) in datasets.items():
    predictions = {name: cross_val_predict(model, X, yc, cv=cv_pred, n_jobs=N_JOBS)
                   for name, model in models.items()}
    for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, (name, ypred) in zip(axes, predictions.items()):
            ax.scatter(yc, ypred, alpha=0.4, s=12, color="tab:blue")
            lo = min(yc.min(), ypred.min())
            hi = max(yc.max(), ypred.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlabel(r"$J_\mathrm{ref}$ (Hz)")
            ax.set_ylabel(r"$J_\mathrm{pred}$ (Hz)")
            ax.set_title(fr"{name}  MAE {mae_mean[cfg_name][name]:.2f} Hz")
            style_axes(ax, LETTER_COLOUR)
        for ax in axes[n_models:]:
            ax.axis("off")
        fig.suptitle(cfg_name)
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/ml_quick_{VARIANT}_{cfg_name}{SUFFIX}.pdf",
                    transparent=TRANSPARENT)
        plt.close(fig)

