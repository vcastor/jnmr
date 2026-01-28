import sys
import os
import glob
import numpy as np
from typing import List, Tuple, Dict
from scm.plams import Molecule, Atom, init, finish


def get_canonical_atom_order(mol: 'Molecule', formula: str) -> List[int]:
    """
    Return atom indices in canonical order based on connectivity.
    Order is deterministic: start from unique atom, BFS by symbol priority.
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
    
    visited = [False]*natoms
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
    """Create new molecule with atoms in canonical order."""
    formula = mol.get_formula()
    order = get_canonical_atom_order(mol, formula)
    
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

def get_ch2_hydrogens(choline: 'Molecule') -> List[List[int]]:
    """Return indices of H atoms bonded in CH2 [choline] (0-based within molecule)."""
    h_indices = []
    for i, at in enumerate(choline.atoms):
        if at.symbol != 'H':
            continue
        c_atom = at.bonds[0].other_end(at)
        h_at_c = []
        for next_bond in c_atom.bonds:
            neighbor = next_bond.other_end(c_atom)
            if neighbor.symbol == 'C':
                h_at_c.append(i)
                break
        h_indices.append(h_at_c)
    return h_indices

def get_nh2_hydrogens(urea: 'Molecule') -> Dict[int, List[int]]:
    """Return indices of H atoms bonded to N in urea (0-based within molecule).
    {at index of N: [list of H indices]}"""
    h_indices = {}
    for i, at in enumerate(urea.atoms):
        if at.symbol != 'N':
            continue
        h_idx = []
        for bond in at.bonds:
            neighbor = bond.other_end(at)
            if neighbor.symbol == 'H':
                h_idx.append(i)
        h_indices[i] = h_idx
    return h_indices


def get_ch3_hydrogens(choline: 'Molecule') -> Dict[int, List[int]]:
    """Return indices of H atoms in CH3 groups in choline (0-based within molecule).
    {at index of C: [list of H indices]}"""
    h_indices = {}
    for i, at in enumerate(choline.atoms):
        if at.symbol != 'C':
            continue
        for bond in at.bonds:
            neighbor = bond.other_end(at)
            if neighbor.symbol != 'N':
                continue
            h_idx = [atom for atom in at.bonds if atom.other_end(at).symbol == 'H']
        h_indices[i] = h_idx
    return h_indices


def find_nh2_ch3_pairs(urea: 'Molecule', choline: 'Molecule', 
                       urea_offset: int, choline_offset: int,
                       distance_threshold: float = 6.0) -> List[Tuple[int, int, float]]:
    """
    Find H(NH2)-H(CH3) pairs within distance threshold.
    Returns list of interacting atom
    H from NH2 perturbing the 9 H from CH3.
    """
    nh2_h = get_nh2_hydrogens(urea)
    ch3_h = get_ch3_hydrogens(choline)
    
    pairs = []
    for n, nhs in nh2_h:
        urea_h_coord = np.array(urea.atoms[n].coords)
        for c, chs in ch3_h:
            choline_h_coord = np.array(choline.atoms[c].coords)
            dist = np.linalg.norm(urea_h_coord - choline_h_coord)
            if dist < distance_threshold:
                h_h2  = nhs
                c_ch3 = chs
    
    return pairs


def classify_sort_canonical(molecules: List['Molecule'], centre: np.ndarray) -> Tuple[List['Molecule'], dict, List[str]]:
    """
    Sort molecules by type, then by distance to centre.
    Canonicalize atom order within each molecule.
    """
    urea, choline, chloride = [], [], []

    for mol in molecules:
        formula = mol.get_formula()
        com = np.array(mol.get_center_of_mass())
        dist = np.linalg.norm(com - centre)
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
        'urea': len(urea),
        'choline': len(choline),
        'chloride': len(chloride),
    }
    
    atom_labels = []
    for mol_type, mol_list in [('U', urea), ('Ch', choline), ('Cl', chloride)]:
        for i, (mol, _) in enumerate(mol_list):
            for j, at in enumerate(mol.atoms):
                label = f"{mol_type}{i+1}_{at.symbol}{j+1}"
                atom_labels.append(label)
    
    # Store molecule lists for interaction finding
    mol_data = {
        'urea': [m for m, _ in urea],
        'choline': [m for m, _ in choline],
        'chloride': [m for m, _ in chloride]
    }
    
    return sorted_mols, counts, atom_labels, mol_data


def find_all_interactions(mol_data: dict, counts: dict, distance_threshold: float = 4.0) -> List[Tuple[int, int, float, str]]:
    """
    Find all NH2-CH3 interactions between urea and choline molecules.
    Returns list of (urea_H_global_idx, choline_H_global_idx, distance, description).
    Indices are 1-based for output.
    """
    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    
    # Compute global offsets
    urea_atoms = sum(len(u) for u in ureas)
    
    # Inner interactions
    inner = []
    for choline in cholines:
        h2 = get_h2_hydrogens(choline)
        inner = combine(h2)

    interactions = []
    
    for ui, urea in enumerate(ureas):
        urea_offset = sum(len(ureas[k]) for k in range(ui))
        
        for ci, choline in enumerate(cholines):
            choline_offset = urea_atoms + sum(len(cholines[k]) for k in range(ci))
            
            # pairs = find_nh2_ch3_pairs(urea, choline, urea_offset, choline_offset, distance_threshold)
            
            for urea_h, choline_h, dist in pairs:
                desc = f"U{ui+1}_H(NH2) -- Ch{ci+1}_H(CH3)"
                # Convert to 1-based
                interactions.append((urea_h + 1, choline_h + 1, dist, desc))
    
    return interactions

def write_molecules_xyz(cluster, filename: str) -> None:
    basename = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, 'w') as f:
        f.write("#!/bin/sh\n\n")
        f.write(f"AMSJOBNAME={basename} $AMSBIN/ams <<eor\n")
        f.write(f"System\n")
        f.write(f"  Atoms\n")
        for atom in cluster.atoms:
            f.write(f"    {atom.symbol:>4s} {atom.x:>14.8f} {atom.y:>14.8f} {atom.z:>14.8f}\n")
        f.write(f"  End\n")
        f.write(f"End\n\n")
        f.write(f"Task SinglePoint\n\n")
        f.write(f"Engine ADF\n")
        f.write(f"  title {basename}\n")
        f.write(f"  beckegrid\n")
        f.write(f"    quality good\n")
        f.write(f"  End\n")
        f.write(f"  basis\n")
        f.write(f"    type TZ2P\n")
        f.write(f"    core None\n")
        f.write(f"  End\n")
        f.write(f"  save TAPE10\n")
        f.write(f"  symmetry NOSYM\n")
        f.write(f"  XC\n")
        f.write(f"    GGA PBE\n")
        f.write(f"  Relativity\n")
        f.write(f"    Level None\n")
        f.write(f"  End\n")
        f.write(f"EndEngine\n")
        f.write(f"eor\n\n")
        f.write(f"# The next block computes the nmr coupling\n\n")
        f.write(f"$AMSBIN/cpl << eor\n")
        f.write(f"  adffile {basename}.results/adf.rkf\n")
        f.write(f"  tape10file {basename}.results/TAPE10\n")
        f.write(f"  nmrcoupling\n")
        f.write(f"    dso\n")
        f.write(f"    pso\n")
        # CH2-CH2 interactions
        for ai, aj, dist, desc in interactions:
            f.write(f"    nuclei interaction {ai} {aj}\n")
        # NH2-CH3 interactions
        for ai, aj, dist, desc in interactions:
            f.write(f"    nuclei interaction {ai} {aj}\n")
        f.write(f"  End\n")
        f.write(f"eor\n")

def write_interactions_file(interactions, filename) -> None:
    """Write auxiliar file with (NH2-CH3 & CH2-CH2) interaction pairs for J-coupling."""
    with open(filename, 'w') as f:
        f.write("# NH2(urea) - CH3(choline) interactions for J-coupling\n")
        f.write("# atom_i  atom_j  distance(A)  description\n")
        for ai, aj, dist, desc in interactions:
            f.write(f"{ai:6d}  {aj:6d}  {dist:8.3f}    {desc}\n")
        f.write(f"\n# CH2 - CH2 interactions for J-coupling\n")
        for ai, aj, dist, desc in interactions:
            f.write(f"{ai:6d}  {aj:6d}  {dist:8.3f}    {desc}\n")


def process_snapshot(input_xyz: str, output_xyz: str, output_interactions: str,
                     distance_threshold: float = 6.0) -> Tuple[dict, bool]:
    """
    Process single snapshot.
    Returns (counts, success).
    """
    cluster = Molecule(input_xyz)
    centre  = np.mean(cluster.as_array(), axis=0)

    cluster.guess_bonds()
    molecules = cluster.separate()

    sorted_mols, counts, atom_labels, mol_data = classify_sort_canonical(molecules, centre)

    # Find interactions
    interactions = find_all_interactions(mol_data, counts, distance_threshold)
    
    # Write outputs
    write_adf_input(sorted_mols, output_xyz, comment=f"Reordered: {os.path.basename(input_xyz)}")
    write_interactions_file(interactions, output_interactions)
    
    return counts, True


# =============================================================================
#                                      Main
# =============================================================================
init()

config = {
    "input_dir": "input_xyz",
    "output_dir": "output_reordered",
    "distance_threshold": 6.0,
    "test": False
}

for arg in sys.argv:
    if '=' in arg:
        key, value = arg.split('=', 1)
        if key in config:
            if key == "distance_threshold":
                config[key] = float(value)
            else:
                config[key] = value

# Test; test; test #
if config["test"]:
    xyzfile = config["test"]
    output_xyz = xyzfile.replace(".xyz", "_reordered.xyz")
    counts, success = process_snapshot(xyzfile, output_xyz, "test_interactions.txt",
                                       config["distance_threshold"])
    quit()
# Test; test; test #

os.makedirs(output_dir, exist_ok=True)

xyz_files = sorted(glob.glob(os.path.join(input_dir, "*.xyz")))
processed, skipped = 0, 0

for xyz_file in xyz_files:
    basename = os.path.basename(xyz_file)
    name_no_ext = os.path.splitext(basename)[0]
    
    output_xyz = os.path.join(output_dir, basename)
    output_interactions = os.path.join(output_dir, f"{name_no_ext}_interactions.txt")
    
    counts, success = process_snapshot(xyz_file, output_xyz, output_interactions,
                                       distance_threshold)

    if success:
        processed += 1
    else:
        print(f"WARNING: {basename} skipped - count mismatch")
        skipped += 1

finish()

