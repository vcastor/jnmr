#!/usr/bin/python3
import os
import sqlite3
import subprocess
from typing import Optional, List, Tuple

VARIANTS = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]


def get_pending_steps(cursor) -> List[int]:
    cursor.execute("SELECT n_step FROM snapshots WHERE comment IS NULL")
    return [row[0] for row in cursor.fetchall()]


def table_exists(cursor, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,))
    return cursor.fetchone()[0] > 0


def check_jcolumn_filled(cursor, table_name: str, j_col: str) -> bool:
    """Check if a specific J column is fully filled in a table."""
    if not table_exists(cursor, table_name):
        return False
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {j_col} IS NOT NULL")
    filled = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total = cursor.fetchone()[0]
    return filled == total and total > 0


def find_j_value(filepath: str, atom1: int, atom2: int) -> Optional[float]:
    """
    Find J coupling value for a pair of atoms in output file.
    Searches for 'Atom input numbers in the ADF calculation:' line containing both atoms,
    then reads J value 9 lines later.
    """
    search_pattern = "Atom input numbers in the ADF calculation"

    result = subprocess.run(
        f"grep -n '{search_pattern}' {filepath} | grep {atom1} | grep {atom2}",
        shell=True, capture_output=True, text=True
    )

    if result.returncode != 0 or not result.stdout.strip():
        return None

    line_num = int(result.stdout.strip().split(':')[0])
    j_line_num = line_num + 9

    result = subprocess.run(
        f"sed -n '{j_line_num}p' {filepath}",
        shell=True, capture_output=True, text=True
    )

    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip().split()[-1])

    return None


def get_null_pairs(cursor, table_name: str, j_col: str) -> List[Tuple[int, int, int]]:
    """Get (id, H_pert, H_resp) pairs where the given J column is NULL."""
    cursor.execute(
        f"SELECT id, H_pert, H_resp FROM {table_name} WHERE {j_col} IS NULL")
    return cursor.fetchall()


def update_j_value(cursor, table_name: str, j_col: str,
                   row_id: int, j_value: float) -> None:
    """Update a specific J column for a row."""
    cursor.execute(
        f"UPDATE {table_name} SET {j_col} = ? WHERE id = ?",
        (j_value, row_id))


def process_output_file(cursor, n_step: int, variant: str, filepath: str):
    """
    Process output file for a given step and variant.
    Fills the J_{variant} column in step_{n_step}_intra and step_{n_step}_inter.
    """
    j_col = f"J_{variant}"

    for suffix in ("intra", "inter"):
        table_name = f"step_{n_step}_{suffix}"
        if not table_exists(cursor, table_name):
            continue
        pairs = get_null_pairs(cursor, table_name, j_col)
        for row_id, h_pert, h_resp in pairs:
            j_value = find_j_value(filepath, h_pert, h_resp)
            if j_value is not None:
                update_j_value(cursor, table_name, j_col, row_id, j_value)

# ============================== #
#             Main
# ============================== #

config = {
    "db_path": "nmr_jcoupling.db",
    "variants": [
        {"name": "TZ2P_FC",   "output_dir": "amsoutput/TZ2P_FC"},
        {"name": "TZ2P_all",  "output_dir": "amsoutput/TZ2P_all"},
        {"name": "TZ2PJ_FC",  "output_dir": "amsoutput/TZ2PJ_FC"},
        {"name": "TZ2PJ_all", "output_dir": "amsoutput/TZ2PJ_all"},
    ],
}

conn   = sqlite3.connect(config["db_path"])
cursor = conn.cursor()

pending_steps = get_pending_steps(cursor)

for n_step in pending_steps:
    filename = f"MDStep{n_step}_cluster.out"

    for var in config["variants"]:
        variant    = var["name"]
        filepath   = os.path.join(var["output_dir"], filename)
        j_col      = f"J_{variant}"

        if not os.path.exists(filepath):
            continue

        # Check if already filled for this variant (both tables)
        intra_ok = check_jcolumn_filled(cursor, f"step_{n_step}_intra", j_col)
        inter_ok = check_jcolumn_filled(cursor, f"step_{n_step}_inter", j_col)
        if intra_ok and inter_ok:
            continue

        process_output_file(cursor, n_step, variant, filepath)
        conn.commit()

conn.close()
