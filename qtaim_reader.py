#!/usr/bin/python3
import os
import re
import glob
import sqlite3
import numpy as np
from hassan_functions.io import get_step_from_filename

DB_PATH   = "nmr_jcoupling.db"
QTAIM_DIR = "amsoutput/qtaim"
HEADER    = "LOCALIZATION AND DELOCALIZATION INDEXES (MATRIX ELEMENTS)"

def read_di_matrix(path):
    """Read the DI matrix from a QTAIM .out file.
    Returns (atom_labels, matrix) where atom_labels[i] is the absolute atom
    number of row/col i in the symmetric (n, n) matrix."""
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

def add_di_column(conn, table):
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if "DI" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN DI REAL")

def update_di_for_step(conn, nstep, atom_labels, matrix):
    idx_of = {atom: i for i, atom in enumerate(atom_labels)}
    for kind in ("intra", "inter"):
        table = f"step_{nstep}_{kind}"
        rows = conn.execute(f"SELECT id, H_pert, H_resp FROM {table}").fetchall()
        for rid, h_pert, h_resp in rows:
            di = float(matrix[idx_of[h_pert], idx_of[h_resp]])
            conn.execute(f"UPDATE {table} SET DI = ? WHERE id = ?", (di, rid))

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    for table in step_tables(conn):
        add_di_column(conn, table)
    conn.commit()

    for path in sorted(glob.glob(os.path.join(QTAIM_DIR, "*.out"))):
        nstep = get_step_from_filename(path)
        atom_labels, matrix = read_di_matrix(path)
        update_di_for_step(conn, nstep, atom_labels, matrix)

    conn.commit()
    conn.close()

