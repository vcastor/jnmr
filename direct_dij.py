#!/usr/bin/env python3
import csv
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DB_PATH = Path("nmr_jcoupling.db")
XYZ_DIR = Path("mdStepsxyz")

# External field direction in the lab frame
B0 = np.array([0.0, 0.0, 1.0], dtype=float)

# Set to False if H_pert / H_resp are already 0-based in the database
DB_INDICES_ARE_1_BASED = True

ISOTOPE_FOR_SYMBOL = {
   "H":  "1H",
   "C":  "13C",
   "N":  "15N",
   "O":  "17O",
   "Cl": "35Cl",
}

# Gyromagnetic ratios rad s^-1 T^-1
GAMMA = {
   "1H":  2.6752218744e8,
   "13C": 6.728284e7,
   "15N": -2.71261804e7,
   "17O": -3.62808e7,
   "35Cl": 2.624198e7,
}

MU0_OVER_4PI  = 1.0e-7
HBAR          = 1.054571817e-34
TWO_PI        = 2.0*np.pi
ANGSTROM_TO_M = 1.0e-10

def table_exists(con, name):
   cur = con.execute(
      "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
      (name,)
   )
   return cur.fetchone()[0] > 0

def normalise_symbol(sym):
   return sym[0].upper() + sym[1:].lower()

def read_xyz(path):
   lines  = path.read_text().splitlines()
   natoms = int(lines[0].strip())

   symbols, coords = [], []

   for line in lines[2:2+natoms]:
      parts = line.split()
      sym = normalise_symbol(parts[0])
      xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
      symbols.append(sym)
      coords.append(xyz)

   return symbols, np.array(coords, dtype=float)

def db_index_to_xyz_index(idx):
   return idx - 1 if DB_INDICES_ARE_1_BASED else idx


def dipolar_direct_hz(ri_ang, rj_ang, isotope_i, isotope_j, b0):
   b0 = np.array(b0, dtype=float)
   b0 = b0/np.linalg.norm(b0)

   rij_ang = rj_ang - ri_ang
   r_ang   = np.linalg.norm(rij_ang)
   u = rij_ang/r_ang
   cos_theta = np.dot(u, b0)
   r_m = r_ang * ANGSTROM_TO_M

   pref = -(MU0_OVER_4PI)*(GAMMA[isotope_i]*GAMMA[isotope_j]*HBAR)/TWO_PI
   geom =  (3.0*cos_theta*cos_theta - 1.0)/(2.0 * r_m**3)
   D_hz = pref*geom

   return D_hz, r_ang, cos_theta, geom

def fetch_steps(con):
   cur = con.execute("SELECT n_step FROM snapshots ORDER BY n_step")
   return [row[0] for row in cur.fetchall()]

def fetch_pairs_for_step(con, nstep, mode="inter"):
   table = f"step_{nstep}_{mode}"
   cur = con.execute(
      f'SELECT DISTINCT H_pert, H_resp FROM "{table}" ORDER BY H_pert, H_resp'
   )
   return cur.fetchall()

def sem(x):
   if len(x) < 2:
      return 0.0
   return np.std(x, ddof=1)/np.sqrt(len(x))

def ci95(x):
   if len(x) < 2:
      return 0.0
   return 1.96*sem(x)

con   = sqlite3.connect(DB_PATH)
steps = fetch_steps(con)

per_snapshot_rows, grouped = [], {}

mode = "inter"

for nstep in steps:
   table_name = f"step_{nstep}_{mode}"
   if not table_exists(con, table_name):
      continue

   xyz_path = XYZ_DIR/f"MDStep{nstep}.xyz"
   assert xyz_path.exists(), f"Missing xyz file: {xyz_path}"

   symbols, coords = read_xyz(xyz_path)
   pairs = fetch_pairs_for_step(con, nstep, mode=mode)

   for h_pert, h_resp in pairs:
      i = db_index_to_xyz_index(h_pert)
      j = db_index_to_xyz_index(h_resp)

      assert 0 <= i < len(symbols), f"Index out of range in step {nstep}: H_pert={h_pert}"
      assert 0 <= j < len(symbols), f"Index out of range in step {nstep}: H_resp={h_resp}"

      sym_i = symbols[i]
      sym_j = symbols[j]

      assert sym_i in ISOTOPE_FOR_SYMBOL, f"No isotope configured for symbol {sym_i}"
      assert sym_j in ISOTOPE_FOR_SYMBOL, f"No isotope configured for symbol {sym_j}"

      iso_i = ISOTOPE_FOR_SYMBOL[sym_i]
      iso_j = ISOTOPE_FOR_SYMBOL[sym_j]

      D_hz, r_ang, cos_theta, geom = dipolar_direct_hz(
         coords[i], coords[j], iso_i, iso_j, B0
      )

      theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

      row = {
         "n_step": nstep,
         "H_pert_db": h_pert,
         "H_resp_db": h_resp,
         "i_xyz": i,
         "j_xyz": j,
         "sym_i": sym_i,
         "sym_j": sym_j,
         "iso_i": iso_i,
         "iso_j": iso_j,
         "r_ang": r_ang,
         "cos_theta": cos_theta,
         "theta_deg": theta_deg,
         "geom_factor_m-3": geom,
         "D_direct_Hz": D_hz,
      }

      per_snapshot_rows.append(row)

      key = (min(h_pert, h_resp), max(h_pert, h_resp))
      grouped.setdefault(key, []).append(row)

con.close()

assert len(per_snapshot_rows) > 0, "No data collected"
assert len(grouped) > 0, "No grouped pair data collected"

summary_rows = []
for (a, b), rows in sorted(grouped.items()):
   D_vals = np.array([r["D_direct_Hz"] for r in rows], dtype=float)
   r_vals = np.array([r["r_ang"] for r in rows], dtype=float)
   cos_vals = np.array([r["cos_theta"] for r in rows], dtype=float)
   theta_vals = np.array([r["theta_deg"] for r in rows], dtype=float)
   geom_vals = np.array([r["geom_factor_m-3"] for r in rows], dtype=float)

   summary_rows.append({
      "H_pert_db": a,
      "H_resp_db": b,
      "sym_i": rows[0]["sym_i"],
      "sym_j": rows[0]["sym_j"],
      "iso_i": rows[0]["iso_i"],
      "iso_j": rows[0]["iso_j"],
      "n_snapshots": len(D_vals),
      "D_mean_Hz": np.mean(D_vals),
      "D_std_Hz": np.std(D_vals, ddof=1) if len(D_vals) > 1 else 0.0,
      "D_sem_Hz": sem(D_vals),
      "D_ci95_Hz": ci95(D_vals),
      "D_median_Hz": np.median(D_vals),
      "D_min_Hz": np.min(D_vals),
      "D_max_Hz": np.max(D_vals),
      "D_abs_mean_Hz": np.mean(np.abs(D_vals)),
      "r_mean_ang": np.mean(r_vals),
      "r_std_ang": np.std(r_vals, ddof=1) if len(r_vals) > 1 else 0.0,
      "cos_theta_mean": np.mean(cos_vals),
      "theta_mean_deg": np.mean(theta_vals),
      "geom_mean_m-3": np.mean(geom_vals),
      "geom_std_m-3": np.std(geom_vals, ddof=1) if len(geom_vals) > 1 else 0.0,
   })

all_D = np.array([r["D_direct_Hz"] for r in per_snapshot_rows], dtype=float)
pair_means = np.array([r["D_mean_Hz"] for r in summary_rows], dtype=float)
pair_abs_means = np.array([r["D_abs_mean_Hz"] for r in summary_rows], dtype=float)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.hist(all_D, bins="auto", edgecolor="black")
ax.set_xlabel(r"$D_{ij}^{\mathrm{direct}}$ (Hz)")
ax.set_ylabel("Count")
ax.set_title("All snapshot values")
ax.axvline(np.mean(all_D), color="red", linestyle="--", label=f"mean = {np.mean(all_D):.4f} Hz")
ax.axvline(np.median(all_D), color="blue", linestyle=":", label=f"median = {np.median(all_D):.4f} Hz")
ax.legend()

ax = axes[1]
ax.hist(pair_means, bins="auto", edgecolor="black")
ax.set_xlabel(r"$\langle D_{ij}^{\mathrm{direct}} \rangle$ (Hz)")
ax.set_ylabel("Count")
ax.set_title("Pair means")
ax.axvline(np.mean(pair_means), color="red", linestyle="--", label=f"mean = {np.mean(pair_means):.4f} Hz")
ax.axvline(np.median(pair_means), color="blue", linestyle=":", label=f"median = {np.median(pair_means):.4f} Hz")
ax.legend()

ax = axes[2]
ax.hist(pair_abs_means, bins="auto", edgecolor="black")
ax.set_xlabel(r"$\langle |D_{ij}^{\mathrm{direct}}| \rangle$ (Hz)")
ax.set_ylabel("Count")
ax.set_title("Pair mean magnitudes")
ax.axvline(np.mean(pair_abs_means), color="red", linestyle="--", label=f"mean = {np.mean(pair_abs_means):.4f} Hz")
ax.axvline(np.median(pair_abs_means), color="blue", linestyle=":", label=f"median = {np.median(pair_abs_means):.4f} Hz")
ax.legend()

plt.tight_layout()
plt.savefig("direct_dij_histogram.pdf")

global_mean = np.mean(all_D)
global_std = np.std(all_D, ddof=1) if len(all_D) > 1 else 0.0
global_sem = sem(all_D)
global_ci95 = ci95(all_D)

print(f"Pairs: {len(summary_rows)}")
print(f"Snapshots with pair entries: {len(per_snapshot_rows)}")
print(f"Global signed mean over all entries: {global_mean:.8f} Hz")
print(f"Global signed std over all entries:  {global_std:.8f} Hz")
print(f"Global signed sem over all entries:  {global_sem:.8f} Hz")
print(f"Global signed 95% CI half-width:    {global_ci95:.8f} Hz")
print(f"Mean of pair means:                 {np.mean(pair_means):.8f} Hz")
print(f"Median of pair means:               {np.median(pair_means):.8f} Hz")
print(f"Mean of pair |mean| values:         {np.mean(pair_abs_means):.8f} Hz")
print("Wrote dipolar_direct_summary.csv")
print("Wrote direct_dij_histogram.pdf")

