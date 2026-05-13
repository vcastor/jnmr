#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, spearmanr
from scipy.optimize import curve_fit
from pathlib import Path

DB_PATH = Path("nmr_jcoupling.db")
VARIANT = "TZ2P_FC"
J_COL   = f"J_{VARIANT}"

def good_steps(conn):
    cur = conn.execute("SELECT n_step FROM snapshots WHERE comment IS NULL")
    return [row[0] for row in cur.fetchall()]

def table_exists(conn, name):
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,)
    )
    return cur.fetchone() is not None

def columns_of(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def print_sample(conn, kind="intra", limit=12):
    steps = good_steps(conn)
    for n_step in steps:
        t = f"step_{n_step}_{kind}"
        if not table_exists(conn, t):
            continue
        cols = columns_of(conn, t)
        cur  = conn.execute(
            f"SELECT * FROM {t} "
            f"WHERE DI IS NOT NULL AND {J_COL} IS NOT NULL "
            f"LIMIT {limit}"
        )
        rows = cur.fetchall()
        if not rows:
            continue
        print(f"\n--- sample from {t}  (kind={kind}) ---")
        print("  ".join(cols))
        for r in rows:
            print("  ".join(f"{v}" if v is not None else "None" for v in r))
        return

def fetch_di_j(conn, kind):
    di, j = [], []
    for n_step in good_steps(conn):
        t = f"step_{n_step}_{kind}"
        if not table_exists(conn, t):
            continue
        cur = conn.execute(
            f"SELECT DI, {J_COL} FROM {t} "
            f"WHERE DI IS NOT NULL AND {J_COL} IS NOT NULL"
        )
        for d, jv in cur.fetchall():
            di.append(d)
            j.append(jv)
    return np.array(di), np.array(j)

conn = sqlite3.connect(DB_PATH)

print_sample(conn, "intra")
print_sample(conn, "inter")

di_intra, j_intra = fetch_di_j(conn, "intra")
di_inter, j_inter = fetch_di_j(conn, "inter")
conn.close()

di    = np.concatenate([di_intra, di_inter])
j_abs = np.abs(np.concatenate([j_intra, j_inter]))

mask  = (di>0) & (j_abs>0)
di_p  = di[mask]
j_p   = j_abs[mask]

print(f"\nn total      = {len(di)}")
print(f"n positive   = {len(di_p)}  (used for power/log fits)")

pear = pearsonr(di_p, j_p)
spea = spearmanr(di_p, j_p)
print(f"Pearson  r   = {pear.statistic:+.4f}   p = {pear.pvalue:.2e}")
print(f"Spearman rho = {spea.statistic:+.4f}   p = {spea.pvalue:.2e}")

def r2(y, yhat):
    ss_res = np.sum((y-yhat)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - ss_res/ss_tot

# 1) linear  |J| = a*DI + b
lin = linregress(di_p, j_p)
yhat_lin = lin.slope*di_p + lin.intercept
print(f"\n[linear]      |J| = {lin.slope:+.3f}*DI {lin.intercept:+.3f}")
print(f"              r2 = {r2(j_p, yhat_lin):.4f}")

# 2) power   |J| = a*DI^b   (log-log linear)
log_di = np.log(di_p)
log_j  = np.log(j_p)
pw = linregress(log_di, log_j)
a_pw = np.exp(pw.intercept)
b_pw = pw.slope
yhat_pw = a_pw*di_p**b_pw
print(f"\n[power]       |J| = {a_pw:.3f} * DI^{b_pw:+.3f}")
print(f"              r2(log-log) = {pw.rvalue**2:.4f}")
print(f"              r2(linear scale) = {r2(j_p, yhat_pw):.4f}")

# 3) exponential  |J| = a*exp(b*DI)   (semi-log linear)
ex = linregress(di_p, log_j)
a_ex = np.exp(ex.intercept)
b_ex = ex.slope
yhat_ex = a_ex*np.exp(b_ex*di_p)
print(f"\n[exponential] |J| = {a_ex:.3f} * exp({b_ex:+.3f}*DI)")
print(f"              r2(semi-log) = {ex.rvalue**2:.4f}")
print(f"              r2(linear scale) = {r2(j_p, yhat_ex):.4f}")

# 4) quadratic  |J| = a*DI^2 + b*DI + c
quad_coef = np.polyfit(di_p, j_p, 2)
yhat_q = np.polyval(quad_coef, di_p)
print(f"\n[quadratic]   |J| = {quad_coef[0]:+.3f}*DI^2 {quad_coef[1]:+.3f}*DI {quad_coef[2]:+.3f}")
print(f"              r2 = {r2(j_p, yhat_q):.4f}")

# 5) shifted power  |J| = a*(DI+c)^b   (non-linear; helpful when DI small/noisy)
def shifted_power(x, a, b, c):
    return a*(x+c)**b
p0   = [a_pw, b_pw, 1e-3]
popt, _ = curve_fit(shifted_power, di_p, j_p, p0=p0, maxfev=20000)
yhat_sp = shifted_power(di_p, *popt)
print(f"\n[shifted pwr] |J| = {popt[0]:.3f}*(DI {popt[2]:+.4f})^{popt[1]:+.3f}")
print(f"              r2 = {r2(j_p, yhat_sp):.4f}")

# ---------------- plots ----------------
xgrid = np.linspace(di_p.min(), di_p.max(), 400)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
ax.scatter(di_intra, np.abs(j_intra), s=14, alpha=0.4, color="tab:orange",
           label=f"intra (n={len(di_intra)})")
ax.scatter(di_inter, np.abs(j_inter), s=14, alpha=0.4, color="tab:blue",
           label=f"inter (n={len(di_inter)})")
ax.plot(xgrid, lin.slope*xgrid+lin.intercept, "k-",  lw=1.3,
        label=f"linear  r²={r2(j_p, yhat_lin):.3f}")
ax.plot(xgrid, a_pw*xgrid**b_pw, "r--", lw=1.3,
        label=f"power  b={b_pw:+.2f}  r²={r2(j_p, yhat_pw):.3f}")
ax.plot(xgrid, a_ex*np.exp(b_ex*xgrid), "g-.", lw=1.3,
        label=f"exp  b={b_ex:+.2f}  r²={r2(j_p, yhat_ex):.3f}")
ax.plot(xgrid, np.polyval(quad_coef, xgrid), "m:", lw=1.6,
        label=f"quad  r²={r2(j_p, yhat_q):.3f}")
ax.set_xlabel("DI (delocalization index)")
ax.set_ylabel(fr"$|J_{{\mathrm{{{VARIANT}}}}}|$  [Hz]")
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.legend(loc="best", fontsize=8)
ax.set_title("linear scale")

ax = axes[1]
ax.scatter(di_p, j_p, s=14, alpha=0.4, color="tab:gray")
ax.plot(xgrid, a_pw*xgrid**b_pw, "r--", lw=1.4,
        label=f"power: slope(log-log)={b_pw:+.2f}")
ax.plot(xgrid, a_ex*np.exp(b_ex*xgrid), "g-.", lw=1.4,
        label=f"exp: b={b_ex:+.2f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("DI (log)")
ax.set_ylabel(fr"$|J|$ (log)")
ax.legend(loc="best", fontsize=8)
ax.set_title("log-log: straight line ⇒ power law")

plt.tight_layout()
plt.savefig("regression_j_di.pdf")

