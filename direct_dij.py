#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DB_PATH  = Path("nmr_jcoupling.db")
XYZ_DIR  = Path("clusters")
PLOT_DIR = "plots"

DB_INDICES_ARE_1_BASED = True

ISOTOPE_FOR_SYMBOL = {
   "H":  "1H",
   "C":  "13C",
   "N":  "15N",
   "O":  "17O",
   "Cl": "35Cl",
}

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

def rx(angle):
   c, s = np.cos(angle), np.sin(angle)
   return np.array([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]], dtype=float)

def ry(angle):
   c, s = np.cos(angle), np.sin(angle)
   return np.array([[c,0.0,s],[0.0,1.0,0.0],[-s,0.0,c]], dtype=float)

def rz(angle):
   c, s = np.cos(angle), np.sin(angle)
   return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], dtype=float)

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
   return pref*geom, r_ang

def evaluate_over_fields(ri_ang, rj_ang, isotope_i, isotope_j, field_directions):
   D_vals = []
   r_ang = None
   for _, b0 in field_directions:
      D_hz, r_ang = dipolar_direct_hz(ri_ang, rj_ang, isotope_i, isotope_j, b0)
      D_vals.append(D_hz)
   return np.array(D_vals, dtype=float), r_ang

def fetch_steps(con):
   cur = con.execute("SELECT n_step FROM snapshots ORDER BY n_step")
   return [row[0] for row in cur.fetchall()]

def fetch_pairs_for_step(con, nstep, mode="inter"):
   table = f"step_{nstep}_{mode}"
   cur = con.execute(
      f'SELECT DISTINCT H_pert, H_resp FROM "{table}" ORDER BY H_pert, H_resp'
   )
   return cur.fetchall()

field_directions = build_field_directions()
con = sqlite3.connect(DB_PATH)
steps = fetch_steps(con)

all_absD = []
all_info = []

for nstep in steps:
   table_name = f"step_{nstep}_{MODE}"
   if not table_exists(con, table_name):
      continue

   xyz_path = XYZ_DIR/f"MDStep{nstep}_cluster.xyz"
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

      D_vals, r_ang = evaluate_over_fields(
         coords[i], coords[j], iso_i, iso_j, field_directions
      )

      # absD_mean = np.mean(D_vals)
      absD_mean = np.mean(np.abs(D_vals))
      all_absD.append(absD_mean)
      all_info.append({
         "n_step": nstep,
         "xyz_file": str(xyz_path),
         "H_pert_db": h_pert,
         "H_resp_db": h_resp,
         "i_xyz": i,
         "j_xyz": j,
         "sym_i": sym_i,
         "sym_j": sym_j,
         "r_ang": r_ang,
         "absD_mean_Hz": absD_mean,
      })

con.close()
assert len(all_absD) > 0, "No data collected"

all_absD = np.array(all_absD, dtype=float)

for LETTER_COLOUR, TRANSPARENT, SUFFIX in [("black", False, ""), ("white", True, "_transparent")]:
   fig, ax = plt.subplots(figsize=(8, 5))
   ax.hist(all_absD, bins="auto", edgecolor=LETTER_COLOUR)
   ax.set_xlabel(r"$\langle |D_{ij}^{\mathrm{direct}}| \rangle_{B}$ (Hz)")
   ax.set_ylabel("Count")
   ax.set_title("Direct dipolar coupling (absolute) averaged over field directions")
   ax.axvline(np.mean(all_absD), color="red", linestyle="--",
              label=f"mean = {np.mean(all_absD):.4f} Hz")
   ax.axvline(np.median(all_absD), color="blue", linestyle=":",
              label=f"median = {np.median(all_absD):.4f} Hz")
   ax.legend()
   ax.set_facecolor("none")
   for spine in ax.spines.values():
      spine.set_color(LETTER_COLOUR)
   ax.tick_params(colors=LETTER_COLOUR, which="both")
   ax.xaxis.label.set_color(LETTER_COLOUR)
   ax.yaxis.label.set_color(LETTER_COLOUR)
   ax.title.set_color(LETTER_COLOUR)
   leg = ax.get_legend()
   leg.get_frame().set_facecolor("none")
   leg.get_frame().set_edgecolor(LETTER_COLOUR)
   for t in leg.get_texts():
      t.set_color(LETTER_COLOUR)
   plt.tight_layout()
   plt.savefig(f"{PLOT_DIR}/direct_dij_histogram{SUFFIX}.pdf", transparent=TRANSPARENT)
   plt.close(fig)

print(f"Field directions used:  {len(field_directions)}")
print(f"Total entries:          {len(all_absD)}")
print(f"Global mean:            {np.mean(all_absD):.8f} Hz")
print(f"Global median:          {np.median(all_absD):.8f} Hz")
print(f"Global std:             {np.std(all_absD, ddof=1):.8f} Hz")
print()

top3_idx = np.argsort(all_absD)[-3:][::-1]
print("Top 3 largest |D| values:")
for rank, k in enumerate(top3_idx, 1):
   info = all_info[k]
   print(f"  {rank}. |D| = {info['absD_mean_Hz']:.4f} Hz  "
         f"step={info['n_step']}  file={info['xyz_file']}  "
         f"pair=({info['sym_i']}[{info['i_xyz']}], {info['sym_j']}[{info['j_xyz']}])  "
         f"r={info['r_ang']:.4f} A")

