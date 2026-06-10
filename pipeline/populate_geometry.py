#!$AMSBIN/plams
import os
import glob
import sqlite3
import numpy as np
from hassan_functions import *

DB_PATH      = "nmr_jcoupling.db"
CLUSTERS_DIR = "clusters"
SPECIES      = ['urea', 'choline', 'chloride']
GEOM_COLS    = ("dihedral", "distance", "angle_hcc", "angle_cch")

init()

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    n_step = get_step_from_filename(xf)
    table  = f"step_{n_step}_intra"

    if not table_exists(cursor, table):
        continue
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue

    for col in GEOM_COLS:
        if not column_exists(cursor, table, col):
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")

    # check the DB first: skip the PLAMS work if no geometry value is NULL
    null_cond = " OR ".join(f"{c} IS NULL" for c in GEOM_COLS)
    cursor.execute(f"SELECT 1 FROM {table} WHERE {null_cond} LIMIT 1")
    if cursor.fetchone() is None:
        continue

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs = compute_offsets(mol_data, SPECIES)
    cholines = mol_data['choline']
    cho_off  = offs['choline']

    for ci, choline in enumerate(cholines):
        off = cho_off[ci]
        for c1, h1s, c2, h2s in find_adjacent_xh_pairs(choline, 'C', 2):
            for h1 in h1s:
                gi1 = off + choline.atoms.index(h1) + 1
                for h2 in h2s:
                    gi2   = off + choline.atoms.index(h2) + 1
                    theta = abs(dihedral(h1, c1, c2, h2))
                    dis   = distance(h1, h2)
                    a_hcc = angle(h1, c1, c2)
                    a_cch = angle(c1, c2, h2)
                    cursor.execute(
                        f"UPDATE {table} SET dihedral = ?, distance = ?, "
                        f"angle_hcc = ?, angle_cch = ? "
                        f"WHERE H_pert = ? AND H_resp = ?",
                        (theta, dis, a_hcc, a_cch, gi1, gi2),
                    )

conn.commit()
conn.close()
finish()

