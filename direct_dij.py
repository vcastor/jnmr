#!/usr/bin/python3
import csv
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DB_PATH = Path("nmr_jcoupling.db")
XYZ_DIR = Path("mdStepsxyz")

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

MODE = "inter"

# Direction sampling for the external field
RANDOM_FIELD_SEED = 12345
N_RANDOM_FIELD_DIRECTIONS = 24

ROT_X_ANGLES = [
   np.pi/6.0,
   np.pi/4.0,
   np.pi/3.0,
   np.pi/2.0,
   2.0*np.pi/3.0,
]

ROT_Y_ANGLES = [
   np.pi/5.0,
   np.pi/3.0,
   np.pi/2.0,
   3.0*np.pi/5.0,
]

ROT_Z_ANGLES = [
   np.pi/5.0,
   2.0*np.pi/5.0,
   np.pi/3.0,
   np.pi/2.0,
   3.0*np.pi/4.0,
]

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

def sem(x):
   if len(x) < 2:
      return 0.0
   return np.std(x, ddof=1)/np.sqrt(len(x))

def ci95(x):
   if len(x) < 2:
      return 0.0
   return 1.96*sem(x)

def rx(angle):
   c = np.cos(angle)
   s = np.sin(angle)
   return np.array([
      [1.0, 0.0, 0.0],
      [0.0, c,  -s ],
      [0.0, s,   c ],
   ], dtype=float)

def ry(angle):
   c = np.cos(angle)
   s = np.sin(angle)
   return np.array([
      [ c, 0.0, s],
      [0.0, 1.0, 0.0],
      [-s, 0.0, c],
   ], dtype=float)

def rz(angle):
   c = np.cos(angle)
   s = np.sin(angle)
   return np.array([
      [c, -s, 0.0],
      [s,  c, 0.0],
      [0.0, 0.0, 1.0],
   ], dtype=float)

def unit_vector(vec):
   vec = np.array(vec, dtype=float)
   norm = np.linalg.norm(vec)
   assert norm > 0.0, "Zero-length direction"
   return vec/norm

def unique_direction_entries(entries, tol=1.0e-12):
   unique = []
   for label, vec in entries:
      u = unit_vector(vec)
      is_duplicate = False
      for _, v in unique:
         if np.linalg.norm(u - v) < tol:
            is_duplicate = True
            break
      if not is_duplicate:
         unique.append((label, u))
   return unique


def build_field_directions():
   ez = np.array([0.0, 0.0, 1.0], dtype=float)
   entries = [
      ("+x", np.array([ 1.0,  0.0,  0.0], dtype=float)),
      ("-x", np.array([-1.0,  0.0,  0.0], dtype=float)),
      ("+y", np.array([ 0.0,  1.0,  0.0], dtype=float)),
      ("-y", np.array([ 0.0, -1.0,  0.0], dtype=float)),
      ("+z", np.array([ 0.0,  0.0,  1.0], dtype=float)),
      ("-z", np.array([ 0.0,  0.0, -1.0], dtype=float)),
   ]

   for ax in ROT_X_ANGLES:
      for az in ROT_Z_ANGLES:
         vec = rz(az) @ rx(ax) @ ez
         entries.append((f"Rz({az/np.pi:.6f}pi)Rx({ax/np.pi:.6f}pi)+", vec))
         entries.append((f"Rz({az/np.pi:.6f}pi)Rx({ax/np.pi:.6f}pi)-", -vec))

   for ay in ROT_Y_ANGLES:
      for az in ROT_Z_ANGLES:
         vec = rz(az) @ ry(ay) @ ez
         entries.append((f"Rz({az/np.pi:.6f}pi)Ry({ay/np.pi:.6f}pi)+", vec))
         entries.append((f"Rz({az/np.pi:.6f}pi)Ry({ay/np.pi:.6f}pi)-", -vec))

   rng = np.random.default_rng(RANDOM_FIELD_SEED)
   for n in range(N_RANDOM_FIELD_DIRECTIONS):
      vec = rng.normal(size=3)
      vec = unit_vector(vec)
      entries.append((f"rand_{n+1:03d}+", vec))
      entries.append((f"rand_{n+1:03d}-", -vec))

   return unique_direction_entries(entries)


def dipolar_direct_hz(ri_ang, rj_ang, isotope_i, isotope_j, b0):
   b0 = unit_vector(b0)

   rij_ang = rj_ang - ri_ang
   r_ang   = np.linalg.norm(rij_ang)
   u = rij_ang/r_ang
   cos_theta = np.dot(u, b0)
   r_m = r_ang*ANGSTROM_TO_M

   pref = -(MU0_OVER_4PI)*(GAMMA[isotope_i]*GAMMA[isotope_j]*HBAR)/TWO_PI
   geom =  (3.0*cos_theta*cos_theta - 1.0)/(2.0*r_m**3)
   D_hz =   pref*geom

   return D_hz, r_ang, cos_theta, geom


def evaluate_over_fields(ri_ang, rj_ang, isotope_i, isotope_j, field_directions):
   D_vals, geom_vals = [], []
   cos_vals, theta_vals = [], []
   r_ang = None

   for _, b0 in field_directions:
      D_hz, r_ang, cos_theta, geom = dipolar_direct_hz(
         ri_ang, rj_ang, isotope_i, isotope_j, b0
      )
      theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

      D_vals.append(D_hz)
      cos_vals.append(cos_theta)
      theta_vals.append(theta_deg)
      geom_vals.append(geom)

   return (
      np.array(D_vals, dtype=float),
      np.array(cos_vals, dtype=float),
      np.array(theta_vals, dtype=float),
      np.array(geom_vals, dtype=float),
      r_ang,
   )

def fetch_steps(con):
   cur = con.execute("SELECT n_step FROM snapshots ORDER BY n_step")
   return [row[0] for row in cur.fetchall()]

def fetch_pairs_for_step(con, nstep, mode="inter"):
   table = f"step_{nstep}_{mode}"
   cur = con.execute(
      f'SELECT DISTINCT H_pert, H_resp FROM "{table}" ORDER BY H_pert, H_resp'
   )
   return cur.fetchall()

def write_csv(path, rows):
   rows = list(rows)
   assert len(rows) > 0, f"No rows to write for {path}"

   with path.open("w", newline="") as fh:
      writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)

field_directions = build_field_directions()

field_rows = []
for idx, (label, vec) in enumerate(field_directions, start=1):
   field_rows.append({
      "field_id": idx,
      "label": label,
      "Bx": vec[0],
      "By": vec[1],
      "Bz": vec[2],
   })

con = sqlite3.connect(DB_PATH)
steps = fetch_steps(con)

per_snapshot_rows = []
grouped = {}

for nstep in steps:
   table_name = f"step_{nstep}_{MODE}"
   if not table_exists(con, table_name):
      continue

   xyz_path = XYZ_DIR/f"MDStep{nstep}.xyz"
   assert xyz_path.exists(), f"Missing xyz file: {xyz_path}"

   symbols, coords = read_xyz(xyz_path)
   pairs = fetch_pairs_for_step(con, nstep, mode=MODE)

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

      D_vals, cos_vals, theta_vals, geom_vals, r_ang = evaluate_over_fields(
         coords[i], coords[j], iso_i, iso_j, field_directions
      )

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
         "n_B_directions": len(field_directions),
         "r_ang": r_ang,
         "D_direct_mean_over_B_Hz": np.mean(D_vals),
         "D_direct_std_over_B_Hz": np.std(D_vals, ddof=1) if len(D_vals) > 1 else 0.0,
         "D_direct_sem_over_B_Hz": sem(D_vals),
         "D_direct_ci95_over_B_Hz": ci95(D_vals),
         "D_direct_median_over_B_Hz": np.median(D_vals),
         "D_direct_min_over_B_Hz": np.min(D_vals),
         "D_direct_max_over_B_Hz": np.max(D_vals),
         "D_direct_abs_mean_over_B_Hz": np.mean(np.abs(D_vals)),
         "cos_theta_mean_over_B": np.mean(cos_vals),
         "theta_mean_deg_over_B": np.mean(theta_vals),
         "geom_mean_over_B_m-3": np.mean(geom_vals),
         "geom_std_over_B_m-3": np.std(geom_vals, ddof=1) if len(geom_vals) > 1 else 0.0,
         "D_direct_for_+z_Hz": D_vals[next(k for k, (lab, _) in enumerate(field_directions) if lab == "+z")],
      }

      per_snapshot_rows.append(row)

      key = (min(h_pert, h_resp), max(h_pert, h_resp))
      grouped.setdefault(key, []).append(row)

con.close()

assert len(per_snapshot_rows) > 0, "No data collected"
assert len(grouped) > 0, "No grouped pair data collected"

summary_rows = []
for (a, b), rows in sorted(grouped.items()):
   D_vals = np.array([r["D_direct_mean_over_B_Hz"] for r in rows], dtype=float)
   D_abs_vals = np.array([r["D_direct_abs_mean_over_B_Hz"] for r in rows], dtype=float)
   D_dir_std_vals = np.array([r["D_direct_std_over_B_Hz"] for r in rows], dtype=float)
   r_vals = np.array([r["r_ang"] for r in rows], dtype=float)
   geom_vals = np.array([r["geom_mean_over_B_m-3"] for r in rows], dtype=float)

   summary_rows.append({
      "H_pert_db": a,
      "H_resp_db": b,
      "sym_i": rows[0]["sym_i"],
      "sym_j": rows[0]["sym_j"],
      "iso_i": rows[0]["iso_i"],
      "iso_j": rows[0]["iso_j"],
      "n_snapshots": len(D_vals),
      "n_B_directions": rows[0]["n_B_directions"],
      "D_mean_of_snapshot_means_Hz": np.mean(D_vals),
      "D_std_of_snapshot_means_Hz": np.std(D_vals, ddof=1) if len(D_vals) > 1 else 0.0,
      "D_sem_of_snapshot_means_Hz": sem(D_vals),
      "D_ci95_of_snapshot_means_Hz": ci95(D_vals),
      "D_median_of_snapshot_means_Hz": np.median(D_vals),
      "D_min_of_snapshot_means_Hz": np.min(D_vals),
      "D_max_of_snapshot_means_Hz": np.max(D_vals),
      "D_abs_mean_of_snapshot_means_Hz": np.mean(np.abs(D_vals)),
      "mean_of_snapshot_abs_means_Hz": np.mean(D_abs_vals),
      "mean_directional_std_Hz": np.mean(D_dir_std_vals),
      "r_mean_ang": np.mean(r_vals),
      "r_std_ang": np.std(r_vals, ddof=1) if len(r_vals) > 1 else 0.0,
      "geom_mean_over_snapshots_m-3": np.mean(geom_vals),
      "geom_std_over_snapshots_m-3": np.std(geom_vals, ddof=1) if len(geom_vals) > 1 else 0.0,
   })

all_D_avgB = np.array([r["D_direct_mean_over_B_Hz"] for r in per_snapshot_rows], dtype=float)
all_D_abs_avgB = np.array([r["D_direct_abs_mean_over_B_Hz"] for r in per_snapshot_rows], dtype=float)
pair_means = np.array([r["D_mean_of_snapshot_means_Hz"] for r in summary_rows], dtype=float)
pair_abs_means = np.array([r["mean_of_snapshot_abs_means_Hz"] for r in summary_rows], dtype=float)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.hist(all_D_avgB, bins="auto", edgecolor="black")
ax.set_xlabel(r"$\langle D_{ij}^{\mathrm{direct}} \rangle_{B}$ (Hz)")
ax.set_ylabel("Count")
ax.set_title("Snapshot values averaged over field directions")
ax.axvline(np.mean(all_D_avgB), color="red", linestyle="--", label=f"mean = {np.mean(all_D_avgB):.4f} Hz")
ax.axvline(np.median(all_D_avgB), color="blue", linestyle=":", label=f"median = {np.median(all_D_avgB):.4f} Hz")
ax.legend()

ax = axes[1]
ax.hist(pair_means, bins="auto", edgecolor="black")
ax.set_xlabel(r"$\langle \langle D_{ij}^{\mathrm{direct}} \rangle_{B} \rangle_{\mathrm{snap}}$ (Hz)")
ax.set_ylabel("Count")
ax.set_title("Pair means")
ax.axvline(np.mean(pair_means), color="red", linestyle="--", label=f"mean = {np.mean(pair_means):.4f} Hz")
ax.axvline(np.median(pair_means), color="blue", linestyle=":", label=f"median = {np.median(pair_means):.4f} Hz")
ax.legend()

ax = axes[2]
ax.hist(pair_abs_means, bins="auto", edgecolor="black")
ax.set_xlabel(r"$\langle |\langle D_{ij}^{\mathrm{direct}} \rangle_{B}| \rangle_{\mathrm{snap}}$ (Hz)")
ax.set_ylabel("Count")
ax.set_title("Pair mean magnitudes")
ax.axvline(np.mean(pair_abs_means), color="red", linestyle="--", label=f"mean = {np.mean(pair_abs_means):.4f} Hz")
ax.axvline(np.median(pair_abs_means), color="blue", linestyle=":", label=f"median = {np.median(pair_abs_means):.4f} Hz")
ax.legend()

plt.tight_layout()
plt.savefig("direct_dij_histogram.pdf")

write_csv(Path("direct_dij_field_directions.csv"), field_rows)
write_csv(Path("direct_dij_per_snapshot_avgB.csv"), per_snapshot_rows)
write_csv(Path("dipolar_direct_summary.csv"), summary_rows)

global_mean = np.mean(all_D_avgB)
global_std = np.std(all_D_avgB, ddof=1) if len(all_D_avgB) > 1 else 0.0
global_sem = sem(all_D_avgB)
global_ci95 = ci95(all_D_avgB)

print(f"Field directions used:               {len(field_directions)}")
print(f"Pairs:                               {len(summary_rows)}")
print(f"Snapshots with pair entries:         {len(per_snapshot_rows)}")
print(f"Global mean over snapshot B-averages:{global_mean:.8f} Hz")
print(f"Global std over snapshot B-averages: {global_std:.8f} Hz")
print(f"Global sem over snapshot B-averages: {global_sem:.8f} Hz")
print(f"Global 95% CI half-width:            {global_ci95:.8f} Hz")
print(f"Mean of pair means:                  {np.mean(pair_means):.8f} Hz")
print(f"Median of pair means:                {np.median(pair_means):.8f} Hz")
print(f"Mean of pair |mean| values:          {np.mean(pair_abs_means):.8f} Hz")
print("Wrote direct_dij_field_directions.csv")
print("Wrote direct_dij_per_snapshot_avgB.csv")
print("Wrote dipolar_direct_summary.csv")
print("Wrote direct_dij_histogram.pdf")

