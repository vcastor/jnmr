import os
import sys
import glob
import sqlite3
import numpy as np
from typing import List, Tuple, Dict

def get_step_from_filename(filename: str) -> int:
    basename = os.path.splitext(os.path.basename(filename))[0]
    import re
    match = re.search(r'(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0

def get_canonical_atom_order(mol: 'Molecule', formula: str) -> List[int]:
    """
    Return atom indices in canonical order based on connectivity.
    """
    natoms = len(mol.atoms)
    
    if formula == 'Cl':
        return [0]
    
    adj: Dict[int, List[int]] = {i: [] for i in range(natoms)}
    for bond in mol.bonds:
        i = mol.atoms.index(bond.atom1)
        j = mol.atoms.index(bond.atom2)
        adj[i].append(j)
        adj[j].append(i)
    
    symbol_counts = {}
    for at in mol.atoms:
        symbol_counts[at.symbol] = symbol_counts.get(at.symbol, 0) + 1
    
    start_idx = 0
    start_priority = (symbol_counts[mol.atoms[0].symbol], mol.atoms[0].symbol)
    for i, at in enumerate(mol.atoms):
        priority = (symbol_counts[at.symbol], at.symbol)
        if priority < start_priority:
            start_priority = priority
            start_idx = i
    
    visited = [False] * natoms
    order   = []
    queue   = [start_idx]
    visited[start_idx] = True
    
    while queue:
        current = queue.pop(0)
        order.append(current)
        neighbors = adj[current]
        unvisited = [n for n in neighbors if not visited[n]]
        unvisited.sort(key=lambda n: (mol.atoms[n].symbol, len(adj[n])))
        for n in unvisited:
            if not visited[n]:
                visited[n] = True
                queue.append(n)
    
    return order


def reorder_molecule_atoms(mol: 'Molecule') -> 'Molecule':
    """
    Create new molecule with atoms in canonical order
    """
    formula = mol.get_formula()
    order   = get_canonical_atom_order(mol, formula)
    
    new_mol = Molecule()
    new_mol.properties = mol.properties.copy()
    
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    
    for old_idx in order:
        at = mol.atoms[old_idx]
        new_at = Atom(atnum=at.atnum, coords=at.coords)
        new_at.properties = at.properties.copy()
        new_mol.add_atom(new_at)
    
    for bond in mol.bonds:
        i = mol.atoms.index(bond.atom1)
        j = mol.atoms.index(bond.atom2)
        new_mol.add_bond(new_mol.atoms[old_to_new[i]], new_mol.atoms[old_to_new[j]], bond.order)
    
    return new_mol


def get_nh2_hydrogens(urea: 'Molecule') -> List[Tuple[int, int]]:
    """
    Return H atoms bonded to N in urea
    (0-based)
    Returns [(H_index, N_index), ...]
    """
    nh2_hydrogens = []
    for i, at in enumerate(urea.atoms):
        if at.symbol != 'N':
            continue
        for bond in at.bonds:
            neighbor = bond.other_end(at)
            if neighbor.symbol == 'H':
                h_idx = urea.atoms.index(neighbor)
                nh2_hydrogens.append((h_idx, i))
    return nh2_hydrogens


def get_ch3_hydrogens(choline: 'Molecule') -> Tuple[int, List[int]]:
    """
    Return N atom index and all 9 H atoms from CH3 groups
    (0-based)
    Returns {N_index: [H_indices]}
    """
    for i, at in enumerate(choline.atoms):
        if at.symbol != 'N':
            continue

        h_indices = []
        for bond in at.bonds:
            c_atom = bond.other_end(at)
            if c_atom.symbol != 'C':
                continue
            c_neighbors = [b.other_end(c_atom) for b in c_atom.bonds]
            c_h_atoms   = [n for n in c_neighbors if n.symbol == 'H']
            if len(c_h_atoms) == 3:
                h_indices.extend([choline.atoms.index(h) for h in c_h_atoms])

        if len(h_indices) == 9:
            return i, h_indices

    return -1, []


def get_ch2_pairs(choline: 'Molecule') -> List[Tuple[List[int], List[int]]]:
    """
    Return pairs of H indices for adjacent CH2 groups in choline
    """
    ch2_carbons = []
    for i, at in enumerate(choline.atoms):
        if at.symbol != 'C':
            continue
        neighbors = [b.other_end(at) for b in at.bonds]
        h_neighbors = [n for n in neighbors if n.symbol == 'H']
        if len(h_neighbors) == 2:
            h_indices = [choline.atoms.index(h) for h in h_neighbors]
            ch2_carbons.append((at, h_indices))
    
    pairs = []
    for idx1, (c1, h1) in enumerate(ch2_carbons):
        c1_neighbors = [b.other_end(c1) for b in c1.bonds]
        for idx2, (c2, h2) in enumerate(ch2_carbons):
            if idx2 <= idx1:
                continue
            if c2 in c1_neighbors:
                pairs.append((h1, h2))
    return pairs


def classify_sort_canonical(molecules: List['Molecule'],
                            centre: np.ndarray) -> Tuple[List['Molecule'], dict, dict]:
    """
    Sort molecules by type and distance to centre, canonicalize atoms
    """
    urea, choline, chloride = [], [], []

    for mol in molecules:
        formula = mol.get_formula()
        com     = np.array(mol.get_center_of_mass())
        dist    = np.linalg.norm(com - centre)
        canonical_mol = reorder_molecule_atoms(mol)
        
        if formula == 'CH4N2O':
            urea.append((canonical_mol, dist))
        elif formula == 'C5H14NO':
            choline.append((canonical_mol, dist))
        elif formula == 'Cl':
            chloride.append((canonical_mol, dist))

    urea.sort(key=lambda x: x[1])
    choline.sort(key=lambda x: x[1])
    chloride.sort(key=lambda x: x[1])

    sorted_mols = [m for m, _ in urea + choline + chloride]
    
    counts = {
        'urea':     len(urea),
        'choline':  len(choline),
        'chloride': len(chloride),
    }
    
    mol_data = {
        'urea':     [m for m, _ in urea],
        'choline':  [m for m, _ in choline],
        'chloride': [m for m, _ in chloride]
    }
    
    return sorted_mols, counts, mol_data


def compute_offsets(mol_data: dict) -> Tuple[List[int], List[int], List[int]]:
    """Compute global atom offsets for each molecule."""
    ureas     = mol_data['urea']
    cholines  = mol_data['choline']
    chlorides = mol_data['chloride']
    
    urea_offsets = []
    offset = 0
    for u in ureas:
        urea_offsets.append(offset)
        offset += len(u)
    
    choline_offsets = []
    for c in cholines:
        choline_offsets.append(offset)
        offset += len(c)
    
    chloride_offsets = []
    for cl in chlorides:
        chloride_offsets.append(offset)
        offset += len(cl)
    
    return urea_offsets, choline_offsets, chloride_offsets


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
        offset    = choline_offsets[ci]
        ch2_pairs = get_ch2_pairs(choline)
        
        interactions = []
        for h1_list, h2_list in ch2_pairs:
            for h1 in h1_list:
                global_h1 = offset + h1 + 1
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
    """
    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    
    interactions = []
    
    for ui, urea in enumerate(ureas):
        nh2_h_list = get_nh2_hydrogens(urea)
        
        for ci, choline in enumerate(cholines):
            choline_n_idx, choline_h_indices = get_ch3_hydrogens(choline)
            if choline_n_idx < 0:
                continue

            for urea_h_idx, urea_n_list in nh2_h_list:
                urea_h_coord = np.array(urea.atoms[urea_h_idx].coords)

                # Find hydrogens bellow the threshold distance
                close_h = []
                for ch3_h_idx in choline_h_indices:
                    ch3_h_coord = np.array(choline.atoms[ch3_h_idx].coords)
                    dist = np.linalg.norm(urea_h_coord - ch3_h_coord)
                    if dist <= distance_threshold:
                        global_ch3_h = choline_offsets[ci] + ch3_h_idx + 1
                        close_h.append((global_ch3_h, dist))

                if not close_h:
                    continue

                close_h.sort(key=lambda x: x[1])

                # Global indices (1-based)
                global_h_urea = urea_offsets[ui] + urea_h_idx + 1
                global_n_urea = urea_offsets[ci] + choline_n_idx + 1
                global_n_choline = choline_offsets[ci] + choline_n_idx + 1

                interactions.append({
                    'urea_idx': ui,
                    'choline_idx': ci,
                    'H_urea': global_h_urea,
                    'N_urea': global_n_urea,
                    'N_choline': global_n_choline,
                    'H_choline': [h for h, _ in close_h],
                    'distances': [d for _, d in close_h]
                })
    
    return interactions


def write_adf_input(
        sorted_mols: List['Molecule'], filename: str,
        intra_interactions: Dict[int, List[Tuple[int, List[int]]]],
        inter_interactions: List[Dict]) -> None:
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
        f.write("  beckegrid\n")
        f.write("    quality good\n")
        f.write("  End\n")
        f.write("  Basis\n")
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
                f.write(f"    atompert {h1}\n")
                f.write(f"    atomresp {h2_str}\n")
                f.write(f"  end\n")
                f.write(f"eor\n\n")
        # Inter-molecular NH2-CH3 interactions
        f.write("# Inter-molecular J coupling (NH2-CH3)\n")
        current_pair = None
        for inter in inter_interactions:
            ui, ci = inter['urea_idx'], inter['choline_idx']
            if (ui, ci) != current_pair:
                current_pair = (ui, ci)
                f.write(f"# Urea {ui + 1} - Choline {ci + 1}\n")
            h_urea = inter['H_urea']
            h_choline_str = " ".join(str(h) for h in inter['H_choline'])
            f.write(f"$AMSBIN/cpl << eor\n")
            f.write(f"  adffile {basename}.results/adf.rkf\n")
            f.write(f"  tape10file {basename}.results/TAPE10\n")
            f.write(f"  nmrcoupling\n")
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

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count interactions
    n_inter = sum(len(inter['H_choline']) for inter in inter_interactions)
    
    # Insert into main table
    cursor.execute('''
        INSERT OR REPLACE INTO snapshots (n_step, n_choline, n_inter)
        VALUES (?, ?, ?)
    ''', (n_step, n_choline, n_inter))
    
    # Create and populate intra table
    intra_table = f"step_{n_step}_intra"
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {intra_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            H_pert INTEGER,
            H_resp INTEGER,
            J_fermi REAL
        )
    ''')
    
    # Clear existing data for this step
    cursor.execute(f"DELETE FROM {intra_table}")
    
    for ci in sorted(intra_interactions.keys()):
        for h1, h2_list in intra_interactions[ci]:
            for h2 in h2_list:
                cursor.execute(f'''
                    INSERT INTO {intra_table} (H_pert, H_resp, J_fermi)
                    VALUES (?, ?, NULL)
                ''', (h1, h2))
    
    # Create and populate inter table
    inter_table = f"step_{n_step}_inter"
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {inter_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            H_pert INTEGER,
            H_resp INTEGER,
            distance REAL,
            J_fermi REAL
        )
    ''')
    
    # Clear existing data for this step
    cursor.execute(f"DELETE FROM {inter_table}")
    
    for inter in inter_interactions:
        ui = inter['urea_idx'] + 1
        ci = inter['choline_idx'] + 1
        h_urea = inter['H_urea']
        
        for h_ch, dist in zip(inter['H_choline'], inter['distances']):
            cursor.execute(f'''
                INSERT INTO {inter_table} (H_pert, H_resp, distance, J_fermi)
                VALUES (?, ?, ?, NULL)
            ''', (h_urea, h_ch, dist))
    
    conn.commit()
    conn.close()

def add_snapshot_to_db_error(db_path: str, n_step: int) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO snapshots (n_step, n_choline, n_inter, comment)
        VALUES (?, 0, 0, 'Error processing snapshot')
    ''', (n_step,))
    
    conn.commit()
    conn.close()

def process_snapshot(
        input_xyz: str,
        output_run: str,
        db_path: str,
        distance_threshold: float = 5.0) -> bool:

    n_step  = get_step_from_filename(input_xyz)
    cluster = Molecule(input_xyz)
    centre  = np.mean(cluster.as_array(), axis=0)

    # Safe check [zero atoms?]
    if len(cluster.atoms) == 0:
        add_snapshot_to_db_error(db_path, n_step)
        return True

    cluster.guess_bonds()
    molecules = cluster.separate()

    sorted_mols, counts, mol_data = classify_sort_canonical(molecules, centre)
    urea_offsets, choline_offsets, chloride_offsets = compute_offsets(mol_data)

    # zero-based indices
    intra = intra_choline_interactions(mol_data, choline_offsets)
    inter = inter_nh2_ch3_interactions(mol_data, urea_offsets,
                                       choline_offsets, distance_threshold)

    write_adf_input(sorted_mols, output_run, intra, inter)
    add_snapshot_to_db(db_path, n_step, counts['choline'], intra, inter)
    
    return True

# =========================================================================== #
#                                  Main                                       #
# =========================================================================== #

init()

config = {
    "clusters_xyz": "clusters",
    "output_dir": "run_scripts",
    "db_path": "nmr_jcoupling.db",
    "distance_threshold": 5.0,
}

xyz_files = sorted(glob.glob(os.path.join(config["clusters_xyz"], "*.xyz")))

for xyz_file in xyz_files:
    basename   = os.path.splitext(os.path.basename(xyz_file))[0]
    output_run = os.path.join(config["output_dir"], f"{basename}.run")
    if not os.path.exists(output_run):
        process_snapshot(xyz_file, output_run, config["db_path"], config["distance_threshold"])

finish()

