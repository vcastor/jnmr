#!$AMSBIN/plams
import os
import glob
import sqlite3
import numpy as np
from itertools import groupby
from typing import List, Tuple, Dict
from hassan_functions import *

SPECIES = ['urea', 'choline', 'chloride']
DB_PATH = "nmr_jcoupling.db"
CLUSTERS_DIR = "clusters"

# main NMR workflow: (variant, dso/pso/sd contributions, TZ2P-J basis)
VARIANTS = [
    ("TZ2P_FC",   False, False),
    ("TZ2P_all",  True,  False),
    ("TZ2PJ_FC",  False, True),
    ("TZ2PJ_all", True,  True),
]
DISTANCE_THRESHOLD = 5.0

# C(urea)-H(choline) coupling for small clusters → court partition
CH_OUT_DIR   = "run_scripts/ch"
CH_PARTITION = "court"

def classify_cluster(xyz):
    """Classify a cluster into SPECIES, tag each atom with its global 1-based
    cluster_id, and return (mol_data, offsets, sorted_mols, counts)."""
    cluster = Molecule(xyz)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs = compute_offsets(mol_data, SPECIES)
    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1
    sorted_mols = [m for name in SPECIES for m in mol_data[name]]
    counts      = {name: len(mol_data[name]) for name in SPECIES}
    return mol_data, offs, sorted_mols, counts

def write_system_engine(f, sorted_mols, basename, j_basis=False):
    """System/Atoms + ADF engine block (saves TAPE10 for the cpl runs)."""
    f.write("#!/bin/sh\n\n")
    f.write(f"export AMS_JOBNAME={basename}\n")
    f.write("$AMSBIN/ams <<eor\n")
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
    f.write(f"    Type {'TZ2P-J' if j_basis else 'TZ2P'}\n")
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
    f.write("eor\n")

def write_cpl(f, basename, pert, resp, contributions=False):
    """One $AMSBIN/cpl nmrcoupling block: pert against the resp list."""
    resp_str = " ".join(str(r) for r in resp)
    f.write("$AMSBIN/cpl << eor\n")
    f.write(f"  adffile {basename}.results/adf.rkf\n")
    f.write(f"  tape10file {basename}.results/TAPE10\n")
    f.write("  nmrcoupling\n")
    if contributions:
        f.write("    dso\n")
        f.write("    pso\n")
        f.write("    sd\n")
    f.write(f"    atompert {pert}\n")
    f.write(f"    atomresp {resp_str}\n")
    f.write("  end\n")
    f.write("eor\n")

def intra_choline_interactions(mol_data, choline_offsets):
    """CH2-CH2 intra interactions per choline:
    {choline_index: [(H1_global, [H2_globals]), ...]}."""
    intra = {}
    for ci, choline in enumerate(mol_data['choline']):
        offset = choline_offsets[ci]
        interactions = []
        for _, h1_list, _, h2_list in find_adjacent_xh_pairs(choline, 'C', 2, return_indices=True):
            for h1 in h1_list:
                global_h1  = offset + h1 + 1
                global_h2s = [offset + h2 + 1 for h2 in h2_list]
                interactions.append((global_h1, global_h2s))
        intra[ci] = interactions
    return intra

def inter_nh2_ch3_interactions(mol_data, urea_offsets, choline_offsets,
                               distance_threshold=5.0):
    """NH2-CH3 inter interactions (threshold between H atoms). Rows of the same
    NH2-(CH3)3 contact share a contact_id; the closest H-H pair gets is_main=1."""
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
                    contacts.setdefault((ui, ci), []).append({
                        'urea_idx':    ui,
                        'choline_idx': ci,
                        'H_urea':      urea_offsets[ui] + urea_h_idx + 1,
                        'H_choline':   choline_offsets[ci] + ch3_h_idx + 1,
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

def write_main_run(sorted_mols, filename, intra_interactions, inter_interactions,
                   contributions=False, j_basis=False):
    """ADF input with intra CH2-CH2 and inter NH2-CH3 NMR coupling blocks."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, 'w') as f:
        write_system_engine(f, sorted_mols, basename, j_basis)
        f.write("\n#\n# NMR J-coupling calculation\n#\n")

        f.write("\n# Intra-molecular J coupling (CH2-CH2 in choline)\n")
        for ci in sorted(intra_interactions.keys()):
            f.write(f"# Choline {ci + 1}\n")
            for h1, h2_list in intra_interactions[ci]:
                write_cpl(f, basename, h1, h2_list, contributions)
                f.write("\n")

        f.write("# Inter-molecular J coupling (NH2-CH3)\n")
        sorted_inter = sorted(inter_interactions,
                              key=lambda x: (x['urea_idx'], x['choline_idx'], x['H_urea']))
        current_pair = None
        for h_urea, group in groupby(sorted_inter, key=lambda x: x['H_urea']):
            rows = list(group)
            ui, ci = rows[0]['urea_idx'], rows[0]['choline_idx']
            if (ui, ci) != current_pair:
                current_pair = (ui, ci)
                f.write(f"# Urea {ui + 1} - Choline {ci + 1}\n")
            write_cpl(f, basename, h_urea, [r['H_choline'] for r in rows], contributions)
            f.write("\n")

def add_snapshot_to_db(db_path, n_step, n_choline, intra_interactions, inter_interactions):
    """Seed step_{n}_intra / step_{n}_inter (geometry only; J columns stay NULL).
    Only populated on first creation — re-running must not wipe reader values."""
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO snapshots (n_step, n_choline, n_inter) VALUES (?, ?, ?)",
        (n_step, n_choline, len(inter_interactions)))

    intra_table = f"step_{n_step}_intra"
    if not table_exists(cursor, intra_table):
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
                    cursor.execute(
                        f"INSERT INTO {intra_table} (H_pert, H_resp) VALUES (?, ?)",
                        (h1, h2))

    inter_table = f"step_{n_step}_inter"
    if not table_exists(cursor, inter_table):
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
            cursor.execute(
                f"INSERT INTO {inter_table} (H_pert, H_resp, distance, is_main) "
                f"VALUES (?, ?, ?, ?)",
                (inter['H_urea'], inter['H_choline'], inter['distance'], inter['is_main']))

    conn.commit()
    conn.close()

def add_snapshot_to_db_error(db_path, n_step):
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO snapshots (n_step, n_choline, n_inter, comment)
        VALUES (?, 0, 0, 'Error processing snapshot')
        ON CONFLICT(n_step) DO UPDATE SET comment = 'Error processing snapshot'
    ''', (n_step,))
    conn.commit()
    conn.close()

def ch_interactions(mol_data):
    """C(urea)-H(CH2,CH3 of choline) coupling blocks keyed by cluster_id."""
    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    blocks = []
    for ui, u in enumerate(ureas):
        c_urea = next(at for at in u.atoms if at.symbol == 'C')
        for ci, ch in enumerate(cholines):
            pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            if pair is None:
                continue
            _, hN, _, hO = pair
            ch3_hs = [h for _, hs in find_xh_groups(ch, 'C', 3, neighbour_symbol='N')
                      for h in hs]
            resp = hN + hO + ch3_hs
            if not resp:
                continue
            blocks.append({
                'label': f"Urea {ui + 1} - Choline {ci + 1}  [C(urea)-H(CH2,CH3)]",
                'pert':  c_urea.cluster_id,
                'resp':  [h.cluster_id for h in resp],
            })
    return blocks

def write_ch_run(filename, basename, sorted_mols, cpl_blocks):
    with open(filename, "w") as f:
        write_system_engine(f, sorted_mols, basename)
        for block in cpl_blocks:
            f.write(f"\n# {block['label']}\n")
            write_cpl(f, basename, block['pert'], block['resp'])

def warning_steps(cursor, variant):
    """Steps whose SCF warned for this variant (excluded from submission)."""
    placeholders = ",".join("?" for _ in SCF_WARNINGS)
    cursor.execute(
        f"SELECT n_step FROM snapshots WHERE comment_{variant} IN ({placeholders})",
        tuple(SCF_WARNINGS))
    return {r[0] for r in cursor.fetchall()}

# ── main ──────────────────────────────────────────────────────────────────

init()
os.makedirs(CH_OUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
warned = {variant: warning_steps(conn.cursor(), variant) for variant, _, _ in VARIANTS}
conn.close()

for xyz_file in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    basename = os.path.splitext(os.path.basename(xyz_file))[0]
    n_step   = get_step_from_filename(xyz_file)

    if sum(1 for _ in open(xyz_file)) - 2 < 1:
        add_snapshot_to_db_error(DB_PATH, n_step)
        continue

    mol_data, offs, sorted_mols, counts = classify_cluster(xyz_file)

    # ── 4-variant NMR workflow: seed DB, write .run files ──
    intra = intra_choline_interactions(mol_data, offs['choline'])
    inter = inter_nh2_ch3_interactions(mol_data, offs['urea'], offs['choline'],
                                       DISTANCE_THRESHOLD)
    add_snapshot_to_db(DB_PATH, n_step, counts['choline'], intra, inter)

    for variant, contributions, j_basis in VARIANTS:
        out_file   = os.path.join("amsoutput", variant, f"{basename}.out")
        run_script = os.path.join("run_scripts", variant, f"{basename}.run")
        sl_script  = os.path.join("run_scripts", variant, f"{basename}.sl")

        if not (os.path.exists(out_file) or os.path.exists(run_script)):
            write_main_run(sorted_mols, run_script, intra, inter, contributions, j_basis)

        # .sl: skip submission once the output is in or the SCF warned
        if os.path.exists(out_file) or n_step in warned[variant]:
            if os.path.exists(sl_script):
                os.remove(sl_script)
        else:
            cfg = VARIANT_SLURM[variant]
            with open(sl_script, "w") as f:
                f.write(slurm_script(str(n_step), basename, cfg["partition"], cfg["walltime"]))

    # ── C(urea)-H coupling: write .run + .sl ──
    ch_run = os.path.join(CH_OUT_DIR, f"{basename}.run")
    ch_out = os.path.join("amsoutput", "ch", f"{basename}.out")
    if not (os.path.exists(ch_out) or os.path.exists(ch_run)):
        cpl_blocks = ch_interactions(mol_data)
        if cpl_blocks:
            write_ch_run(ch_run, basename, sorted_mols, cpl_blocks)
            with open(os.path.join(CH_OUT_DIR, f"{basename}.sl"), "w") as f:
                f.write(slurm_script(f"ch{n_step}", basename, CH_PARTITION))

finish()

