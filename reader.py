#!/usr/bin/python3
import os
import glob
import sqlite3
import subprocess
from typing import Optional
from hassan_functions.db import table_exists, column_exists
from hassan_functions.io import get_step_from_filename, read_labeled_matrix
from hassan_functions.criann import SCF_WARNINGS

DB_PATH    = "nmr_jcoupling.db"
VARIANTS   = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]
QTAIM_DIR  = "amsoutput/qtaim"
CDFT_DIR   = "amsoutput/cdft"
DI_HEADER  = "LOCALIZATION AND DELOCALIZATION INDEXES (MATRIX ELEMENTS)"
CHI_HEADER = "CONDENSED LINEAR RESPONSE FUNCTION (MATRIX ELEMENTS)"

# ── SCF convergence warnings ────────────────────────────────────────────────

def mark_warnings(cursor):
    """Set comment_{variant} for steps whose SCF did not (fully) converge."""
    for variant in VARIANTS:
        output_dir  = f"amsoutput/{variant}"
        comment_col = f"comment_{variant}"

        # Clear stale warnings first: a previously-failing calc may have been
        # rerun and now converges, in which case the old comment must not stick
        # around (otherwise the step is silently excluded downstream).
        cursor.execute(f"UPDATE snapshots SET {comment_col} = NULL")

        for pattern in SCF_WARNINGS:
            result = subprocess.run(
                f"grep -rl '{pattern}' {output_dir}",
                shell=True, capture_output=True, text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                continue
            for filepath in result.stdout.strip().split('\n'):
                filename = filepath.split('/')[-1]
                n_step = int(filename.replace('MDStep', '').replace('_cluster.out', ''))
                cursor.execute(
                    f"UPDATE snapshots SET {comment_col} = ? WHERE n_step = ?",
                    (pattern, n_step),
                )

# ── J-coupling values ───────────────────────────────────────────────────────

def get_pending_steps(cursor):
    cursor.execute("SELECT n_step FROM snapshots WHERE comment IS NULL")
    return [row[0] for row in cursor.fetchall()]

def check_jcolumn_filled(cursor, table_name, j_col) -> bool:
    """Check if a specific J column is fully filled in a table."""
    if not table_exists(cursor, table_name):
        return False
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {j_col} IS NOT NULL")
    filled = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total = cursor.fetchone()[0]
    return filled == total and total > 0

def find_j_value(filepath, atom1, atom2) -> Optional[float]:
    """J coupling for a pair of atoms in an ADF output file.

    Searches for 'Atom input numbers in the ADF calculation' containing both
    atoms, then reads the J value 9 lines later.
    """
    result = subprocess.run(
        f"grep -n 'Atom input numbers in the ADF calculation' {filepath} "
        f"| grep {atom1} | grep {atom2}",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    line_num = int(result.stdout.strip().split(':')[0])
    result = subprocess.run(
        f"sed -n '{line_num + 9}p' {filepath}",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip().split()[-1])
    return None

def fill_j(conn, cursor):
    """Fill J_{variant} in step_{n}_intra / step_{n}_inter from ADF output."""
    for n_step in get_pending_steps(cursor):
        filename = f"MDStep{n_step}_cluster.out"

        for variant in VARIANTS:
            filepath = os.path.join(f"amsoutput/{variant}", filename)
            j_col    = f"J_{variant}"

            if not os.path.exists(filepath):
                continue
            if (check_jcolumn_filled(cursor, f"step_{n_step}_intra", j_col)
                    and check_jcolumn_filled(cursor, f"step_{n_step}_inter", j_col)):
                continue

            for suffix in ("intra", "inter"):
                table = f"step_{n_step}_{suffix}"
                if not table_exists(cursor, table):
                    continue
                cursor.execute(
                    f"SELECT id, H_pert, H_resp FROM {table} WHERE {j_col} IS NULL")
                for row_id, h_pert, h_resp in cursor.fetchall():
                    j_value = find_j_value(filepath, h_pert, h_resp)
                    if j_value is not None:
                        cursor.execute(
                            f"UPDATE {table} SET {j_col} = ? WHERE id = ?",
                            (j_value, row_id))
            conn.commit()

# ── QTAIM (DI) and CDFT (chi) matrices ──────────────────────────────────────

def step_tables(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name GLOB 'step_*_intra' OR name GLOB 'step_*_inter')")
    return [row[0] for row in cursor.fetchall()]

def update_matrix_for_step(conn, nstep, atom_labels, matrix, col):
    idx_of = {atom: i for i, atom in enumerate(atom_labels)}
    for kind in ("intra", "inter"):
        table = f"step_{nstep}_{kind}"
        rows = conn.execute(f"SELECT id, H_pert, H_resp FROM {table}").fetchall()
        for rid, h_pert, h_resp in rows:
            val = float(matrix[idx_of[h_pert], idx_of[h_resp]])
            conn.execute(f"UPDATE {table} SET {col} = ? WHERE id = ?", (val, rid))

def fill_di_chi(conn, cursor):
    for table in step_tables(cursor):
        for col in ("DI", "chi"):
            if not column_exists(cursor, table, col):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
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

if __name__ == "__main__":
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    mark_warnings(cursor)
    conn.commit()
    fill_j(conn, cursor)
    fill_di_chi(conn, cursor)

    conn.close()

