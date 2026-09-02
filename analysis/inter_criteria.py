#!/usr/bin/python3
"""Two trial selections for the inter J_HH, per variant, vs exp:
A) contacts with a real N...H-C interaction: an NH2-CH3 contact qualifies when its
   closest CH3 H sits within NHC_CUT of the urea N; that anchor H is excluded and of
   the remaining NH2 x CH3 pairs (4) only the strongest |J| per contact is kept.
B) per snapshot the strongest |J|, averaged across snapshots."""
import os
import sys
import glob
import sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reader import read_out_atoms, molecule_ids
from hassan_functions.db     import table_exists, column_exists
from hassan_functions.jstats import cubic_mean, cubic_mean_ci, effective_n, CUBIC_P
from hassan_functions.params import J_PHYSICAL_MAX_HZ

DB_PATH  = "nmr_jcoupling.db"
VARIANTS = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]
EXP, EXP_ERR = 1.104, 0.031
NHC_CUT  = 3.0   # Å; N...H(CH3) below this counts as the non-covalent contact
XH_BOND  = 1.3

def out_atoms(n_step):
    for v in VARIANTS:
        p = f"amsoutput/{v}/MDStep{n_step}_cluster.out"
        if os.path.exists(p):
            return read_out_atoms(p)
    return None

def stats_line(tag, j, s):
    j = np.asarray(j, float)
    if j.size == 0:
        print(f"  {tag:<28} n=0")
        return
    s = np.asarray(s)
    lo, hi = cubic_mean_ci(j, s)
    print(f"  {tag:<28} n={j.size:<5d} snap={np.unique(s).size:<4d} "
          f"cubic={cubic_mean(j):.3f} [{lo:.3f},{hi:.3f}]  "
          f"mean={j.mean():.3f}  median={np.median(j):.3f}")

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

atoms_cache = {}
print(f"inter J_HH selections — exp {EXP} +- {EXP_ERR} Hz, "
      f"N...H cut {NHC_CUT} A, p={CUBIC_P:g}")

for variant in VARIANTS:
    j_col, c_col = f"J_{variant}", f"comment_{variant}"
    if not column_exists(cursor, "snapshots", c_col):
        continue
    cursor.execute(f"SELECT n_step FROM snapshots WHERE {c_col} IS NULL")
    steps = [r[0] for r in cursor.fetchall()]

    sel_j, sel_s = [], []          # A: N...H-C contacts
    max_j, max_s = [], []          # B: strongest |J| per snapshot
    for n_step in steps:
        table = f"step_{n_step}_inter"
        if not table_exists(cursor, table) or not column_exists(cursor, table, j_col):
            continue
        cursor.execute(
            f"SELECT H_pert, H_resp, {j_col} FROM {table} WHERE {j_col} IS NOT NULL")
        rows = [(hp, hr, abs(j)) for hp, hr, j in cursor.fetchall()
                if abs(j) <= J_PHYSICAL_MAX_HZ]
        if not rows:
            continue

        max_j.append(max(j for _, _, j in rows))
        max_s.append(n_step)

        if n_step not in atoms_cache:
            atoms_cache[n_step] = (out_atoms(n_step),)
        atoms = atoms_cache[n_step][0]
        if atoms is None:
            continue
        mol_of = molecule_ids(atoms)

        # contact = (urea N bonded to H_pert, responding choline); keep it when one
        # of its CH3 H's is the N...H-C partner
        contacts = {}
        for hp, hr, j in rows:
            hx = atoms[hp][1]
            n_id = min((n for n, (sym, _) in atoms.items() if sym == 'N'),
                       key=lambda n: np.linalg.norm(atoms[n][1] - hx))
            if np.linalg.norm(atoms[n_id][1] - hx) > XH_BOND:
                continue
            contacts.setdefault((n_id, mol_of[hr]), []).append((hp, hr, j))
        for (n_id, _mol), crows in contacts.items():
            danchor, anchor = min((np.linalg.norm(atoms[n_id][1] - atoms[hr][1]), hr)
                                  for _, hr, _ in crows)
            if danchor > NHC_CUT:
                continue
            rest = [j for _, hr, j in crows if hr != anchor]
            if rest:
                sel_j.append(max(rest))
                sel_s.append(n_step)

    print(f"\n{variant}")
    stats_line("A: max |J| per N...H-C contact", sel_j, sel_s)
    stats_line("B: max |J| per snapshot", max_j, max_s)

conn.close()
