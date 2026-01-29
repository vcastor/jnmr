#!/usr/bin/python3
import os
import sqlite3
import subprocess
from typing import Optional, List, Tuple

def get_pending_steps(cursor) -> List[int]:
    cursor.execute("SELECT n_step FROM snapshots WHERE comment IS NULL")
    return [row[0] for row in cursor.fetchall()]


def check_jvalues_filled(cursor, n_step: int, table_type: str) -> bool:
    """Check if J_fermi values are already filled for this step."""
    table_name = f"step_{n_step}_{table_type}"
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE J_fermi IS NOT NULL")
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
        f"grep -n '{search_pattern}' {filepath} | grep '{atom1}' | grep '{atom2}'",
        capture_output=True, text=True
    )

    if result.returncode != 0 or not result.stdout.strip():
        return None

    line_num = int(result.stdout.strip().split(':')[0])
    j_line_num = line_num + 9

    result = subprocess.run(
        f"sed -n '{j_line_num}p' {filepath}",
        capture_output=True, text=True
    )
    
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip().split()[-1])

    return None


def get_atom_pairs_intra(cursor, n_step: int) -> List[Tuple[int, int, int]]:
    """Get (id, H_pert, H_resp) pairs from intra table."""
    table_name = f"step_{n_step}_intra"
    cursor.execute(f"SELECT id, H_pert, H_resp FROM {table_name} WHERE J_fermi IS NULL")
    return cursor.fetchall()


def get_atom_pairs_inter(cursor, n_step: int) -> List[Tuple[int, int, int]]:
    """Get (id, H_pert, H_resp) pairs from inter table."""
    table_name = f"step_{n_step}_inter"
    cursor.execute(f"SELECT id, H_pert, H_resp FROM {table_name} WHERE J_fermi IS NULL")
    return cursor.fetchall()


def update_j_value(cursor, n_step: int, table_type: str, row_id: int, j_value: float) -> None:
    """Update J_fermi value for a specific row."""
    table_name = f"step_{n_step}_{table_type}"
    cursor.execute(f"UPDATE {table_name} SET J_fermi = ? WHERE id = ?", (j_value, row_id))


def process_output_file(
        cursor,
        n_step: int,
        filepath: str) -> Tuple[int, int, int]:
    """
    Process output file for a given step.
    Returns (n_intra_updated, n_inter_updated, n_errors).
    """
    
    # Process intra-molecular interactions
    intra_pairs = get_atom_pairs_intra(cursor, n_step)
    for row_id, h_pert, h_resp in intra_pairs:
        j_value = find_j_value(filepath, h_pert, h_resp)
        if j_value is not None:
            update_j_value(cursor, n_step, "intra", row_id, j_value)
    
    # Process inter-molecular interactions
    inter_pairs = get_atom_pairs_inter(cursor, n_step)
    for row_id, h_pert, h_resp in inter_pairs:
        j_value = find_j_value(filepath, h_pert, h_resp)
        if j_value is not None:
            update_j_value(cursor, n_step, "inter", row_id, j_value)

# ============================== #
#             Main 
# ============================== #

config = {
    "db_path": "nmr_jcoupling.db",
    "output_dir": "ouputAMS",
}

conn   = sqlite3.connect(config["db_path"])
cursor = conn.cursor()

# Get pending steps
pending_steps = get_pending_steps(cursor)

for n_step in pending_steps:
    filename = f"MDStep{n_step}_cluster.out"
    filepath = os.path.join(config["output_dir"], filename)
    
    if not os.path.exists(filepath):
        continue
    
    # Check if already filled
    intra_filled = check_jvalues_filled(cursor, n_step, "intra")
    inter_filled = check_jvalues_filled(cursor, n_step, "inter")
    
    if intra_filled and inter_filled:
        conn.commit()
        continue
    
    process_output_file(cursor, n_step, filepath)
    conn.commit()

conn.close()


