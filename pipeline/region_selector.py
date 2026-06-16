#!$AMSBIN/plams
import os
import sys
import numpy as np
from typing import List

def centre_of_cage(coords: np.ndarray) -> np.ndarray:
    return np.mean(coords, axis=0)

def closer_to_origin(coords: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(coords, axis=1)
    min_index = np.argmin(distances)
    return coords[min_index]

def atoms_inner_radius(radius: float,
                       centre: np.ndarray,
                       coords: np.ndarray) -> List[int]:
    distances = np.linalg.norm(coords - centre, axis=1)
    return np.where(distances < radius)[0].tolist()

def molecules_inner_radius(
        molecule_indices: List[List[int]],
        atoms_in: List[int]) -> List[int]:
    """Return indices of molecules containing any atom from atoms_in."""
    atoms_in_set = set(atoms_in)
    selected = []
    for mol_idx, mol_atoms in enumerate(molecule_indices):
        if atoms_in_set.intersection(mol_atoms):
            selected.append(mol_idx)
    return selected

def check_solution_ratio(
        molecules: List['Molecule'],
        target_ratio: tuple,
        centre: np.ndarray) -> List['Molecule']:
    """
    Balance molecules to match target_ratio (urea : choline_chloride).
    Removes molecules farthest from centre first.
    """
    # Classify molecules and compute distances from centre
    urea, choline, chloride = [], [], []

    for mol in molecules:
        formula = mol.get_formula()
        com = np.array(mol.get_center_of_mass())
        dist = np.linalg.norm(com - centre)

        if formula == 'CH4N2O':
            urea.append((mol, dist))
        elif formula == 'C5H14NO':
            choline.append((mol, dist))
        elif formula == 'Cl':
            chloride.append((mol, dist))

    # Sort by distance (farthest first for removal)
    urea.sort(key=lambda x: -x[1])
    choline.sort(key=lambda x: -x[1])
    chloride.sort(key=lambda x: -x[1])

    # Balance choline and chloride counts (remove farthest excess)
    while len(choline) > len(chloride):
        choline.pop(0)
    while len(chloride) > len(choline):
        chloride.pop(0)

    # Now balance urea:choline_chloride to target_ratio
    urea_target, cc_target = target_ratio
    n_cc = len(choline)
    n_urea = len(urea)

    # Determine which to trim
    desired_urea = (n_cc*urea_target)//cc_target
    desired_cc = (n_urea*cc_target)//urea_target

    if n_urea > desired_urea:
        # Remove excess urea
        urea = urea[n_urea - desired_urea:]
    elif n_cc > desired_cc:
        # Remove excess choline chloride pairs (farthest combined)
        excess = n_cc - desired_cc
        for _ in range(excess):
            choline.pop(0)
            chloride.pop(0)

    # Collect remaining molecules
    filtered = [m for m, _ in urea + choline + chloride]
    return filtered

def write_molecules_xyz(
        molecules: List['Molecule'],
        filename: str,
        comment: str = "") -> None:
    total_atoms = sum(len(mol) for mol in molecules)
    with open(filename, 'w') as f:
        f.write(f"{total_atoms}\n")
        f.write(f"{comment}\n")
        for mol in molecules:
            for atom in mol.atoms:
                f.write(f"{atom.symbol:>4s} {atom.x:>14.8f} {atom.y:>14.8f} {atom.z:>14.8f}\n")

# ── main ──────────────────────────────────────────────────────────────────
init()

config_input = {
    'xyz_dir': 'mdSteps/xyz',
    'cluster_dir': 'clusters',
}

for xyz in os.listdir(config_input['xyz_dir']):
    xyzpath = os.path.join(config_input['xyz_dir'], xyz)
    clusterpath = os.path.join(config_input['cluster_dir'], xyz.replace('.xyz', '_cluster.xyz'))

    if os.path.isfile(clusterpath):
        continue

    cluster = Molecule(xyzpath)
    cluster.guess_bonds()

    # Get molecule indices BEFORE separate() - maps atom indices to molecules
    molecule_indices = cluster.get_molecule_indices()  # [[0,1,2], [3,4,5], ...]
    all_molecules = cluster.separate()

    coords = cluster.as_array()
    centre = centre_of_cage(coords)

    atoms_in    = atoms_inner_radius(10.0, centre, coords)
    mol_indices = molecules_inner_radius(molecule_indices, atoms_in)

    # Get the actual Molecule objects
    molecules = [all_molecules[i] for i in mol_indices]

    # Check ratio of the solution [ONLY THE CURRENTY SYSTEM OF UREA + CHOLINE CHLORIDE]
    molecules = check_solution_ratio(molecules, target_ratio=(2, 1), centre=centre)

    write_molecules_xyz(molecules, clusterpath, comment="DES inner sphere")

finish()

