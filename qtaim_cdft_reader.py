#!/usr/bin/python3
import os
import glob
import sqlite3

from hassan_functions.io import get_step_from_filename, read_labeled_matrix
from hassan_functions.db import column_exists

DB_PATH    = "nmr_jcoupling.db"
QTAIM_DIR  = "amsoutput/qtaim"
CDFT_DIR   = "amsoutput/cdft"
DI_HEADER  = "LOCALIZATION AND DELOCALIZATION INDEXES (MATRIX ELEMENTS)"
CHI_HEADER = "CONDENSED LINEAR RESPONSE FUNCTION (MATRIX ELEMENTS)"

def step_tables(conn):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name GLOB 'step_*_intra' OR name GLOB 'step_*_inter')"
    )
    return [row[0] for row in cur.fetchall()]

def add_real_column(conn, table, col):
    if not column_exists(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")

def update_matrix_for_step(conn, nstep, atom_labels, matrix, col):
    idx_of = {atom: i for i, atom in enumerate(atom_labels)}
    for kind in ("intra", "inter"):
        table = f"step_{nstep}_{kind}"
        rows = conn.execute(f"SELECT id, H_pert, H_resp FROM {table}").fetchall()
        for rid, h_pert, h_resp in rows:
            val = float(matrix[idx_of[h_pert], idx_of[h_resp]])
            conn.execute(f"UPDATE {table} SET {col} = ? WHERE id = ?", (val, rid))

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    for table in step_tables(conn):
        add_real_column(conn, table, "DI")
        add_real_column(conn, table, "chi")
    conn.commit()

    for path in sorted(glob.glob(os.path.join(QTAIM_DIR, "*.out"))):
        nstep = get_step_from_filename(path)
        atom_labels, di = read_labeled_matrix(path, DI_HEADER)
        update_matrix_for_step(conn, nstep, atom_labels, di, "DI")

    for path in sorted(glob.glob(os.path.join(CDFT_DIR, "*.out"))):
        nstep = get_step_from_filename(path)
        atom_labels, chi = read_labeled_matrix(path, CHI_HEADER)
        update_matrix_for_step(conn, nstep, atom_labels, chi, "chi")

    conn.commit()
    conn.close()

