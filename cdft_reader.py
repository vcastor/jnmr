#!/usr/bin/python3
import os
import re
import glob
import sqlite3
import numpy as np
from hassan_functions.io import get_step_from_filename

DB_PATH  = "nmr_jcoupling.db"
CDFT_DIR = "amsresults/cdft"
HEADER   = "CONDENSED LINEAR RESPONSE FUNCTION (MATRIX ELEMENTS)"

def read_chi_matrix(path):
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if HEADER in l) + 3

    atom_labels = []
    rows = []
    i = start
    while i < len(lines):
        m = re.match(r"\s*(\d+)\s+\w+\s*:(.*)$", lines[i])
        if m is None:
            break
        atom_labels.append(int(m.group(1)))
        nvals = len(atom_labels)
        vals = [float(x) for x in m.group(2).split()]
        i += 1
        while len(vals) < nvals:
            vals.extend(float(x) for x in lines[i].split())
            i += 1
        rows.append(vals[:nvals])

    n = len(atom_labels)
    matrix = np.zeros((n, n))
    for k, vals in enumerate(rows):
        matrix[k, :k + 1] = vals
        matrix[:k + 1, k] = vals
    return atom_labels, matrix

def step_tables(conn):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name GLOB 'step_*_intra' OR name GLOB 'step_*_inter')"
    )
    return [row[0] for row in cur.fetchall()]

def add_chi_column(conn, table):
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if "chi" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN chi REAL")

def update_chi_for_step(conn, nstep, atom_labels, matrix):
    idx_of = {atom: i for i, atom in enumerate(atom_labels)}
    for kind in ("intra", "inter"):
        table = f"step_{nstep}_{kind}"
        rows = conn.execute(f"SELECT id, H_pert, H_resp FROM {table}").fetchall()
        for rid, h_pert, h_resp in rows:
            chi = float(matrix[idx_of[h_pert], idx_of[h_resp]])
            conn.execute(f"UPDATE {table} SET chi = ? WHERE id = ?", (chi, rid))

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    for table in step_tables(conn):
        add_chi_column(conn, table)
    conn.commit()

    for path in sorted(glob.glob(os.path.join(CDFT_DIR, "*.out"))):
        nstep = get_step_from_filename(path)
        atom_labels, matrix = read_chi_matrix(path)
        update_chi_for_step(conn, nstep, atom_labels, matrix)

    conn.commit()
    conn.close()

