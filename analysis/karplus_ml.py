#!/usr/bin/python3
import os
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
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from hassan_functions.plotting import PLOT_STYLES, style_axes

plt.rcParams["text.usetex"]         = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xfrac}"

PLOT_DIR = "plots"
DB_PATH  = "nmr_jcoupling.db"
VARIANT  = "TZ2P_FC"
J_COL    = f"J_{VARIANT}"
RNG      = 69
N_SPLITS = 5
N_JOBS   = -1                  # all available cores; 64 procs on the main compute node

GEOM      = ["dihedral", "angle_hcc", "angle_cch", "distance"]
DI_TERMS  = [f"DI_{t}"  for t in ["HpHr", "HpCp", "HpCr", "CpCr", "HrCp", "HrCr"]]
CHI_TERMS = [f"chi_{t}" for t in ["HpHr", "HpCp", "HpCr", "CpCr", "HrCp", "HrCr"]]

CONFIGS = {
    "phi":              ["dihedral"],
    "geom":             GEOM,
    "geom_DI_HpHr":     GEOM + ["DI_HpHr"],
    "geom_chi_HpHr":    GEOM + ["chi_HpHr"],
    "geom_DI_chi_HpHr": GEOM + ["DI_HpHr", "chi_HpHr"],
    "geom_DI_all":      GEOM + DI_TERMS,
    "geom_chi_all":     GEOM + CHI_TERMS,
    "geom_DI_chi_all":  GEOM + DI_TERMS + CHI_TERMS,
}
DESCRIPTORS = sorted({c for cols in CONFIGS.values() for c in cols})

def table_cols(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cursor.fetchall()}

def flag_outliers(vals):
    # modified z-score (median/MAD); robust for the handful of folds we have
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        std = vals.std()
        if std == 0:
            return [], med
        z = (vals - med)/std
        return [i for i in range(vals.size) if abs(z[i]) > 2.5], med
    z = 0.6745*(vals - med)/mad
    return [i for i in range(vals.size) if abs(z[i]) > 3.5], med

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
    if cursor.fetchone() is None:    continue
    present = table_cols(cursor, t)
    if J_COL not in present:         continue
    sel = [c for c in DESCRIPTORS if c in present]
    cursor.execute(f"SELECT {', '.join(sel + [J_COL])} FROM {t} WHERE {J_COL} IS NOT NULL")
    for row in cursor.fetchall():
        vals = dict(zip(sel + [J_COL], row))
        for c in DESCRIPTORS:
            data[c].append(vals.get(c))
        js.append(vals[J_COL])

conn.close()

data = {c: np.array(v, dtype=float) for c, v in data.items()}   # None -> nan
y    = np.abs(np.asarray(js, dtype=float))

datasets = {}
for cfg_name, cols in CONFIGS.items():
    X    = np.column_stack([data[c] for c in cols])
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    datasets[cfg_name] = (X[mask], y[mask])

kernel = (ConstantKernel(1.0, (1e-3, 1e3))
          *RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
          + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2)))

models = {
    "RF":  RandomForestRegressor(n_estimators=500, random_state=RNG, n_jobs=1),
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

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)

print(f"=== {VARIANT}: {N_SPLITS} random splits (KFold shuffle, seed {RNG}), "
      f"n_jobs={N_JOBS} ===")
print(f"target: |{J_COL}|  range {y.min():.4f} -> {y.max():.4f}")

mae_mean = {}
flagged  = []                  # (cfg, model, fold, mae, median)
for cfg_name, (X, yc) in datasets.items():
    print(f"\n--- {cfg_name}: n = {yc.size}, features: {', '.join(CONFIGS[cfg_name])} ---")
    print(f"{'model':<10}{'MAE':>9}{'±std':>9}{'min':>9}{'max':>9}{'RMSE':>9}  outliers")
    mae_mean[cfg_name] = {}
    for name, model in models.items():
        sc = cross_validate(model, X, yc, cv=cv,
                            scoring=("neg_mean_absolute_error",
                                     "neg_root_mean_squared_error"),
                            n_jobs=N_JOBS)
        mae  = -sc["test_neg_mean_absolute_error"]
        rmse = -sc["test_neg_root_mean_squared_error"]
        mae_mean[cfg_name][name] = mae.mean()

        out, med = flag_outliers(mae)
        for i in out:
            flagged.append((cfg_name, name, i, mae[i], med))
        tag = "" if not out else "  *folds " + ",".join(
            f"{i}({mae[i]:.2f})" for i in out)

        print(f"{name:<10}{mae.mean():>9.4f}{mae.std():>9.4f}"
              f"{mae.min():>9.4f}{mae.max():>9.4f}{rmse.mean():>9.4f}{tag}")

best = min(((c, m, v) for c, mm in mae_mean.items() for m, v in mm.items()),
           key=lambda x: x[2])
print(f"\nbest: {best[1]} on '{best[0]}'  MAE = {best[2]:.4f} Hz "
      f"(mean over {N_SPLITS} splits)")

if flagged:
    print(f"\n=== fold-MAE outliers (modified z-score > 3.5 vs the other folds) ===")
    print(f"{'config':<13}{'model':<10}{'fold':>5}{'MAE':>9}{'median':>9}{'ratio':>8}")
    for cfg_name, name, fold, mae_v, med in flagged:
        print(f"{cfg_name:<13}{name:<10}{fold:>5}{mae_v:>9.4f}{med:>9.4f}"
              f"{mae_v/med:>8.2f}")
else:
    print("\nno fold-MAE outliers: every split is within trend for all models.")

# --- parity plots (out-of-fold predictions, one grid per config) ---
n_models = len(models)
ncols    = 3
nrows    = (n_models + ncols - 1)//ncols

for cfg_name, (X, yc) in datasets.items():
    predictions = {name: cross_val_predict(model, X, yc, cv=cv, n_jobs=N_JOBS)
                   for name, model in models.items()}
    for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, (name, ypred) in zip(axes, predictions.items()):
            ax.scatter(yc, ypred, alpha=0.3, s=10, color="tab:blue")
            lo = min(yc.min(), ypred.min())
            hi = max(yc.max(), ypred.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlabel(r"$\vert J_\mathrm{ref}\vert$ (Hz)")
            ax.set_ylabel(r"$\vert J_\mathrm{pred}\vert$ (Hz)")
            ax.set_title(fr"{name}  MAE {mae_mean[cfg_name][name]:.2f} Hz")
            style_axes(ax, LETTER_COLOUR)
        for ax in axes[n_models:]:
            ax.axis("off")
        fig.suptitle(cfg_name.replace("_", r"\_"))
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/ml_parity_{VARIANT}_{cfg_name}{SUFFIX}.pdf",
                    transparent=TRANSPARENT)
        plt.close(fig)

