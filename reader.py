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
GRID_WARNING = "WARNING: Nr of shared arrays in use at the end"

INTRA_TERMS = ["HpHr", "HpCp", "HpCr", "CpCr", "HrCp", "HrCr"]
INTER_TERMS = ["HpHr"]

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

            if (check_jcolumn_filled(cursor, f"step_{n_step}_intra", j_col)
                    and check_jcolumn_filled(cursor, f"step_{n_step}_inter", j_col)):
                continue
            if not os.path.exists(filepath):
                continue

            for suffix in ("intra", "inter"):
                table = f"step_{n_step}_{suffix}"
                if not table_exists(cursor, table):
                    continue
                cursor.execute(
                    f"SELECT H_pert, H_resp FROM {table} WHERE {j_col} IS NULL")
                for h_pert, h_resp in cursor.fetchall():
                    j_value = find_j_value(filepath, h_pert, h_resp)
                    if j_value is not None:
                        cursor.execute(
                            f"UPDATE {table} SET {j_col} = ? WHERE H_pert = ? AND H_resp = ?",
                            (j_value, h_pert, h_resp))
            conn.commit()

# ── QTAIM (DI) and CDFT (\chi) ───────────────────────────────────────────────────

def step_tables(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name GLOB 'step_*_intra' OR name GLOB 'step_*_inter')")
    return [row[0] for row in cursor.fetchall()]

def table_exists_conn(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,)).fetchone() is not None

def update_matrix_for_step(conn, nstep, atom_labels, matrix, prop):
    idx = {atom: i for i, atom in enumerate(atom_labels)}
    def v(a, b):
        return float(matrix[idx[a], idx[b]])

    inter = f"step_{nstep}_inter"
    if table_exists_conn(conn, inter):
        for rid, hp, hr in conn.execute(
                f"SELECT id, H_pert, H_resp FROM {inter}").fetchall():
            conn.execute(
                f"UPDATE {inter} SET {prop}_HpHr = ? WHERE id = ?", (v(hp, hr), rid))

    intra = f"step_{nstep}_intra"
    if table_exists_conn(conn, intra):
        for rid, hp, hr, cp, cr in conn.execute(
                f"SELECT id, H_pert, H_resp, C_pert, C_resp FROM {intra}").fetchall():
            conn.execute(
                f"UPDATE {intra} SET {prop}_HpHr = ?, {prop}_HpCp = ?, {prop}_HpCr = ?, "
                f"{prop}_CpCr = ?, {prop}_HrCp = ?, {prop}_HrCr = ? WHERE id = ?",
                (v(hp, hr), v(hp, cp), v(hp, cr), v(cp, cr), v(hr, cp), v(hr, cr), rid))

def check_output_ok(path):
    if subprocess.run(["grep", "-qF", GRID_WARNING, path]).returncode == 0:
        print(f"  grid warning: {os.path.basename(path)}")
        return False
    return subprocess.run(["grep", "-qF", "NORMAL TERMINATION", path]).returncode == 0

def _step_needs_prop(cursor, nstep, prop):
    for suffix in ("intra", "inter"):
        table = f"step_{nstep}_{suffix}"
        if not table_exists(cursor, table):
            continue
        terms = INTRA_TERMS if suffix == "intra" else INTER_TERMS
        for col in [f"{prop}_{t}" for t in terms]:
            if not column_exists(cursor, table, col):
                return True
            cursor.execute(f"SELECT 1 FROM {table} WHERE {col} IS NULL LIMIT 1")
            if cursor.fetchone() is not None:
                return True
    return False

def fill_di_chi(conn, cursor):
    for table in step_tables(cursor):
        terms = INTRA_TERMS if table.endswith("_intra") else INTER_TERMS
        for col in [f"{p}_{t}" for p in ("DI", "chi") for t in terms]:
            if not column_exists(cursor, table, col):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
    conn.commit()

    for path in sorted(glob.glob(os.path.join(QTAIM_DIR, "*.out"))):
        nstep = get_step_from_filename(path)
        if not _step_needs_prop(cursor, nstep, "DI"):
            continue
        if not check_output_ok(path):
            continue
        atom_labels, di = read_labeled_matrix(path, DI_HEADER)
        update_matrix_for_step(conn, nstep, atom_labels, di, "DI")

    for path in sorted(glob.glob(os.path.join(CDFT_DIR, "*.out"))):
        nstep = get_step_from_filename(path)
        if not _step_needs_prop(cursor, nstep, "chi"):
            continue
        if not check_output_ok(path):
            continue
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

