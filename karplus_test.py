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
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import r2_score, mean_absolute_error, root_mean_squared_error

PLOT_DIR = "plots"
DB_PATH  = "nmr_jcoupling.db"
VARIANT  = "TZ2P_FC"
J_COL    = f"J_{VARIANT}"
RNG      = 42

CONFIGS = {
    "phi":         ["dihedral"],
    "geom":        ["dihedral", "angle_hcc", "angle_cch", "distance"],
    "geom_DI":     ["dihedral", "angle_hcc", "angle_cch", "distance", "DI"],
    "geom_chi":    ["dihedral", "angle_hcc", "angle_cch", "distance", "chi"],
    "geom_DI_chi": ["dihedral", "angle_hcc", "angle_cch", "distance", "DI", "chi"],
}
DESCRIPTORS = sorted({c for cols in CONFIGS.values() for c in cols})


def table_cols(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cursor.fetchall()}


os.makedirs(PLOT_DIR, exist_ok=True)

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

print(f"=== {VARIANT}: single train/test split (test=0.2, seed {RNG}) ===")
print(f"target: |{J_COL}|  range {y.min():.4f} -> {y.max():.4f}\n")

models = {
    "RF":      RandomForestRegressor(random_state=RNG),
    "GBR":     GradientBoostingRegressor(random_state=RNG),
    "SVR":     make_pipeline(StandardScaler(), SVR()),
    "KRR-RBF": make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
    "GPR":     make_pipeline(StandardScaler(),
                             GaussianProcessRegressor(random_state=RNG,
                                                      normalize_y=True)),
}

rows = []
for cfg_name, cols in CONFIGS.items():
    X    = np.column_stack([data[c] for c in cols])
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    Xc, yc = X[mask], y[mask]

    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=RNG)
    print(f"{cfg_name}: n={yc.size} (train {ytr.size}, test {yte.size}), "
          f"features = {', '.join(cols)}")

    for name, model in models.items():
        model.fit(Xtr, ytr)
        yp   = model.predict(Xte)
        r2   = r2_score(yte, yp)
        mae  = mean_absolute_error(yte, yp)
        rmse = root_mean_squared_error(yte, yp)
        rows.append((cfg_name, name, yc.size, r2, mae, rmse))

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(yte, yp, s=20, alpha=0.7)
        lo = min(yte.min(), yp.min())
        hi = max(yte.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        ax.set_xlabel(f"reference |{J_COL}|")
        ax.set_ylabel(f"predicted |{J_COL}|")
        ax.set_title(f"{cfg_name} / {name}  (MAE {mae:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"test_parity_{cfg_name}_{name}.png"), dpi=300)
        plt.close(fig)

print(f"\n{'config':<13}{'model':<10}{'n':>6}{'R2':>9}{'MAE':>9}{'RMSE':>9}")
print("-"*56)
for cfg_name, name, ncfg, r2, mae, rmse in rows:
    print(f"{cfg_name:<13}{name:<10}{ncfg:>6}{r2:>9.4f}{mae:>9.4f}{rmse:>9.4f}")
