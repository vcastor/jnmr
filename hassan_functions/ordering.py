import numpy as np
from scm.plams import Molecule, Atom

def canonical_order(mol):
    """BFS atom indices in a deterministic order driven by connectivity."""
    n = len(mol.atoms)
    if mol.get_formula() == 'Cl':
        return [0]
    adj = {i: [] for i in range(n)}
    for bond in mol.bonds:
        i = mol.atoms.index(bond.atom1)
        j = mol.atoms.index(bond.atom2)
        adj[i].append(j)
        adj[j].append(i)
    counts = {}
    for at in mol.atoms:
        counts[at.symbol] = counts.get(at.symbol, 0) + 1
    start = 0
    pri = (counts[mol.atoms[0].symbol], mol.atoms[0].symbol)
    for i, at in enumerate(mol.atoms):
        p = (counts[at.symbol], at.symbol)
        if p < pri:
            pri, start = p, i
    visited = [False]*n
    order = []
    queue = [start]
    visited[start] = True
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        nbrs = sorted([k for k in adj[cur] if not visited[k]],
                      key=lambda k: (mol.atoms[k].symbol, len(adj[k])))
        for k in nbrs:
            visited[k] = True
            queue.append(k)
    return order

def reorder(mol):
    """Return a copy of mol with atoms in canonical_order(mol)."""
    order = canonical_order(mol)
    new = Molecule()
    new.properties = mol.properties.copy()
    old_to_new = {old: i for i, old in enumerate(order)}
    for old in order:
        at = mol.atoms[old]
        new_at = Atom(atnum=at.atnum, coords=at.coords)
        new_at.properties = at.properties.copy()
        new.add_atom(new_at)
    for bond in mol.bonds:
        i = mol.atoms.index(bond.atom1)
        j = mol.atoms.index(bond.atom2)
        new.add_bond(new.atoms[old_to_new[i]], new.atoms[old_to_new[j]], bond.order)
    return new

def classify_sort(molecules, centre, formulas):
    """Split molecules by formula, sort each list by COM distance to centre,
    canonicalise atom order.

    formulas: {species_name: chemical_formula}, e.g. {'urea': 'CH4N2O', ...}.
    Returns {species_name: [Molecule, ...]} in nearest-first order.
    """
    buckets = {name: [] for name in formulas}
    for mol in molecules:
        f = mol.get_formula()
        com = np.array(mol.get_center_of_mass())
        d = np.linalg.norm(com - centre)
        for name, target in formulas.items():
            if f == target:
                buckets[name].append((reorder(mol), d))
                break
    out = {}
    for name, items in buckets.items():
        items.sort(key=lambda x: x[1])
        out[name] = [m for m, _ in items]
    return out

def compute_offsets(mol_data, species_order):
    """Global atom-index offsets for each molecule, walked in species_order.

    Returns {species_name: [offset, ...]} aligned with mol_data[species_name].
    """
    offsets = {name: [] for name in species_order}
    off = 0
    for name in species_order:
        for mol in mol_data.get(name, []):
            offsets[name].append(off)
            off += len(mol)
    return offsets

