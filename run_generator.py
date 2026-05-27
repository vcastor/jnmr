import os
import glob
import sqlite3
import numpy as np
from itertools import groupby
from typing import List, Tuple, Dict
from hassan_functions import *

SPECIES = ['urea', 'choline', 'chloride']

def intra_choline_interactions(
        mol_data: dict,
        choline_offsets: List[int]) -> Dict[int, List[Tuple[int, List[int]]]]:
    """
    Find CH2-CH2 intra-molecular interactions for each choline.
    Returns {choline_index: [(H1_global, [H2_globals]), ...]}
    """
    cholines = mol_data['choline']
    intra    = {}

    for ci, choline in enumerate(cholines):
        offset = choline_offsets[ci]
        interactions = []
        for _, h1_list, _, h2_list in find_adjacent_xh_pairs(choline, 'C', 2, return_indices=True):
            for h1 in h1_list:
                global_h1  = offset + h1 + 1
                global_h2s = [offset + h2 + 1 for h2 in h2_list]
                interactions.append((global_h1, global_h2s))
        intra[ci] = interactions

    return intra


def inter_nh2_ch3_interactions(
        mol_data: dict,
        urea_offsets: List[int],
        choline_offsets: List[int],
        distance_threshold: float = 5.0) -> List[Dict]:
    """
    Find NH2-CH3 inter-molecular interactions.
    The distance threshold is between the H atoms.

    Each interaction dict represents one H_urea → H_choline pair with its
    distance.  An extra 'contact_id' key groups rows that belong to the same
    NH2-(CH3)3 contact.  Within each contact the closest H-H pair is flagged
    with is_main = 1 (all others 0).  A cluster can have several contacts,
    so the inter table can have several is_main = 1 rows.
    """
    ureas    = mol_data['urea']
    cholines = mol_data['choline']

    contacts: Dict[Tuple[int, int], List[Dict]] = {}

    for ui, urea in enumerate(ureas):
        nh2_bonds = find_xh_bonds(urea, 'N', return_indices=True)

        for ci, choline in enumerate(cholines):
            ch3_groups = find_xh_groups(choline, 'C', 3, neighbour_symbol='N', return_indices=True)
            if not ch3_groups:
                continue
            choline_h_indices = [h for _, hs in ch3_groups for h in hs]

            for _, urea_h_idx in nh2_bonds:
                urea_h_coord = np.array(urea.atoms[urea_h_idx].coords)

                for ch3_h_idx in choline_h_indices:
                    ch3_h_coord = np.array(choline.atoms[ch3_h_idx].coords)
                    dist = np.linalg.norm(urea_h_coord - ch3_h_coord)

                    if dist > distance_threshold:
                        continue

                    global_h_urea = urea_offsets[ui] + urea_h_idx + 1
                    global_ch3_h  = choline_offsets[ci] + ch3_h_idx + 1

                    key = (ui, ci)
                    contacts.setdefault(key, []).append({
                        'urea_idx':    ui,
                        'choline_idx': ci,
                        'H_urea':      global_h_urea,
                        'H_choline':   global_ch3_h,
                        'distance':    dist,
                    })

    interactions = []
    for contact_id, ((ui, ci), rows) in enumerate(contacts.items()):
        min_dist = min(r['distance'] for r in rows)
        for r in rows:
            r['contact_id'] = contact_id
            r['is_main'] = 1 if r['distance'] == min_dist else 0
        interactions.extend(rows)

    return interactions

def write_adf_input(
        sorted_mols: List['Molecule'], filename: str,
        intra_interactions: Dict[int, List[Tuple[int, List[int]]]],
        inter_interactions: List[Dict],
        contributions = False,
        j_basis = False) -> None:
    """
    Write ADF input file with NMR coupling calculations
    """

    basename = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, 'w') as f:
        f.write("#!/bin/sh\n\n")
        f.write(f"export AMS_JOBNAME={basename}\n")
        f.write(f"$AMSBIN/ams <<eor\n")
        f.write("System\n")
        f.write("  Atoms\n")
        for mol in sorted_mols:
            for atom in mol.atoms:
                f.write(f"    {atom.symbol:>4s} {atom.x:>14.8f} {atom.y:>14.8f} {atom.z:>14.8f}\n")
        f.write("  End\n")
        f.write("End\n\n")
        f.write("Task SinglePoint\n\n")
        f.write("Engine ADF\n")
        f.write(f"  title {basename}\n")
        f.write("  NumericalQuality Excellent\n")
        f.write("  Basis\n")
        if j_basis:
            f.write("    Type TZ2P-J\n")
        else:
            f.write("    Type TZ2P\n")
        f.write("    core None\n")
        f.write("  End\n")
        f.write("  save TAPE10\n")
        f.write("  symmetry NOSYM\n")
        f.write("  XC\n")
        f.write("    GGA PBE\n")
        f.write("  End\n")
        f.write("  Relativity\n")
        f.write("    Level None\n")
        f.write("  End\n")
        f.write("EndEngine\n")
        f.write("eor\n\n")
        #
        # NMR J-coupling section
        #
        f.write("#\n# NMR J-coupling calculation\n#\n\n")
        # Intra-molecular CH2-CH2 interactions
        f.write("# Intra-molecular J coupling (CH2-CH2 in choline)\n")
        for ci in sorted(intra_interactions.keys()):
            f.write(f"# Choline {ci + 1}\n")
            for h1, h2_list in intra_interactions[ci]:
                h2_str = " ".join(str(h) for h in h2_list)
                f.write(f"$AMSBIN/cpl << eor\n")
                f.write(f"  adffile {basename}.results/adf.rkf\n")
                f.write(f"  tape10file {basename}.results/TAPE10\n")
                f.write(f"  nmrcoupling\n")
                if contributions:
                    f.write(f"    dso\n")
                    f.write(f"    pso\n")
                    f.write(f"    sd\n")
                f.write(f"    atompert {h1}\n")
                f.write(f"    atomresp {h2_str}\n")
                f.write(f"  end\n")
                f.write(f"eor\n\n")
        # Inter-molecular NH2-CH3 interactions
        f.write("# Inter-molecular J coupling (NH2-CH3)\n")
        sorted_inter = sorted(inter_interactions,
                              key=lambda x: (x['urea_idx'], x['choline_idx'],
                                             x['H_urea']))
        current_pair = None
        for h_urea, group in groupby(sorted_inter,
                                      key=lambda x: x['H_urea']):
            rows = list(group)
            ui, ci = rows[0]['urea_idx'], rows[0]['choline_idx']
            if (ui, ci) != current_pair:
                current_pair = (ui, ci)
                f.write(f"# Urea {ui + 1} - Choline {ci + 1}\n")
            h_choline_str = " ".join(str(r['H_choline']) for r in rows)
            f.write(f"$AMSBIN/cpl << eor\n")
            f.write(f"  adffile {basename}.results/adf.rkf\n")
            f.write(f"  tape10file {basename}.results/TAPE10\n")
            f.write(f"  nmrcoupling\n")
            if contributions:
                f.write(f"    dso\n")
                f.write(f"    pso\n")
                f.write(f"    sd\n")
            f.write(f"    atompert {h_urea}\n")
            f.write(f"    atomresp {h_choline_str}\n")
            f.write(f"  end\n")
            f.write(f"eor\n\n")

def add_snapshot_to_db(
        db_path: str,
        n_step: int,
        n_choline: int,
        intra_interactions: Dict[int, List[Tuple[int, List[int]]]],
        inter_interactions: List[Dict]) -> None:
    """
    Populate unified tables (one pair per step, not per variant):
      step_{n_step}_intra   – with 4 J columns (one per variant)
      step_{n_step}_inter   – with 4 J columns + geometry columns
    """

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    n_inter = len(inter_interactions)

    cursor.execute('''
        INSERT OR IGNORE INTO snapshots (n_step, n_choline, n_inter)
        VALUES (?, ?, ?)
    ''', (n_step, n_choline, n_inter))

    # ── intra table ─────────────────────────────────────────────────────
    # Only populate on first creation; re-running this script must NOT wipe
    # J / dihedral / distance / DI values filled in by the reader scripts.
    intra_table = f"step_{n_step}_intra"
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (intra_table,))
    if cursor.fetchone() is None:
        cursor.execute(f'''
            CREATE TABLE {intra_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                H_pert      INTEGER,
                H_resp      INTEGER,
                J_TZ2P_FC   REAL,
                J_TZ2P_all  REAL,
                J_TZ2PJ_FC  REAL,
                J_TZ2PJ_all REAL
            )
        ''')
        for ci in sorted(intra_interactions.keys()):
            for h1, h2_list in intra_interactions[ci]:
                for h2 in h2_list:
                    cursor.execute(f'''
                        INSERT INTO {intra_table} (H_pert, H_resp)
                        VALUES (?, ?)
                    ''', (h1, h2))

    # ── inter table ─────────────────────────────────────────────────────
    inter_table = f"step_{n_step}_inter"
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (inter_table,))
    if cursor.fetchone() is None:
        cursor.execute(f'''
            CREATE TABLE {inter_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                H_pert      INTEGER,
                H_resp      INTEGER,
                distance    REAL,
                is_main     INTEGER DEFAULT 0,
                J_TZ2P_FC   REAL,
                J_TZ2P_all  REAL,
                J_TZ2PJ_FC  REAL,
                J_TZ2PJ_all REAL
            )
        ''')
        for inter in inter_interactions:
            cursor.execute(f'''
                INSERT INTO {inter_table}
                    (H_pert, H_resp, distance, is_main)
                VALUES (?, ?, ?, ?)
            ''', (inter['H_urea'], inter['H_choline'],
                  inter['distance'], inter['is_main']))

    conn.commit()
    conn.close()

def add_snapshot_to_db_error(db_path: str, n_step: int) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO snapshots (n_step, n_choline, n_inter, comment)
        VALUES (?, 0, 0, 'Error processing snapshot')
        ON CONFLICT(n_step) DO UPDATE SET comment = 'Error processing snapshot'
    ''', (n_step,))

    conn.commit()
    conn.close()

def analyse_snapshot(input_xyz: str, distance_threshold: float = 5.0):
    """
    Parse the XYZ file, classify molecules, compute interactions.
    Returns (sorted_mols, counts, intra, inter) or None on error.
    """
    n_atoms = sum(1 for line in open(input_xyz)) - 2
    if n_atoms < 1:
        return None

    cluster = Molecule(input_xyz)
    centre  = np.mean(cluster.as_array(), axis=0)

    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs     = compute_offsets(mol_data, SPECIES)
    sorted_mols = [m for name in SPECIES for m in mol_data[name]]
    counts      = {name: len(mol_data[name]) for name in SPECIES}

    intra = intra_choline_interactions(mol_data, offs['choline'])
    inter = inter_nh2_ch3_interactions(mol_data, offs['urea'],
                                       offs['choline'], distance_threshold)

    return sorted_mols, counts, intra, inter


# =========================================================================== #
#                                  Main                                       #
# =========================================================================== #

init()

config = {
    "clusters_xyz": "clusters",
    "run_dirs":    ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"],
    "output_dirs": ["amsoutput/TZ2P_FC",  "amsoutput/TZ2P_all",
                    "amsoutput/TZ2PJ_FC", "amsoutput/TZ2PJ_all"],
    "opts": [
        {"contributions": False, "basisJ": False},
        {"contributions": True,  "basisJ": False},
        {"contributions": False, "basisJ": True},
        {"contributions": True,  "basisJ": True},
    ],
    "db_path": "nmr_jcoupling.db",
    "distance_threshold": 5.0,
}

xyz_files = sorted(glob.glob(os.path.join(config["clusters_xyz"], "*.xyz")))

for xyz_file in xyz_files:
    basename = os.path.splitext(os.path.basename(xyz_file))[0]
    n_step   = get_step_from_filename(xyz_file)

    # ── Geometry analysis (once per snapshot) ────────────────────────────
    analysis = analyse_snapshot(xyz_file, config["distance_threshold"])

    if analysis is None:
        add_snapshot_to_db_error(config["db_path"], n_step)
        continue

    sorted_mols, counts, intra, inter = analysis

    # ── Populate DB tables (once – geometry only, J columns start NULL) ──
    add_snapshot_to_db(config["db_path"], n_step, counts['choline'],
                       intra, inter)

    # ── Generate .run files (one per variant) ────────────────────────────
    for run_dir, out_dir, opts in zip(
            config["run_dirs"], config["output_dirs"], config["opts"]):

        out_file   = os.path.join(out_dir, f"{basename}.out")
        run_script = os.path.join("run_scripts", run_dir, f"{basename}.run")

        if os.path.exists(out_file) or os.path.exists(run_script):
            continue

        write_adf_input(sorted_mols, run_script, intra, inter,
                        opts["contributions"],
                        opts["basisJ"])

finish()

