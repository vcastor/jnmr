"""Locate the named NMR sites (H1-H5, Nurea) inside a classified molecule.

The labels are defined in `constants.SITE_LABELS`. These finders work on PLAMS
Molecule objects that already have bonds (`guess_bonds()`), which is what
`pipeline/coupling_generator.py` has by the time it writes the cpl blocks.

`reader.py` cannot use these — it only has the .out geometry echo, no PLAMS — so it
classifies sites from interatomic distances instead (`reader.classify_site`). The two
must agree; both are driven by the same structural rules:

    H1  H on a C carrying 3 H            (choline methyl; urea has no C-H)
    H2  H on a C carrying 2 H bonded N   (choline, next to the quaternary N)
    H3  H on a C carrying 2 H bonded O   (choline, next to the hydroxyl O)
    H4  H on an O                        (choline hydroxyl; the urea O carries no H)
    H5  H on an N                        (urea amine; the choline N+ carries no H)
    Nurea  an N carrying H               (urea; the choline N+ carries none)
"""
from .finders import find_xh_groups, find_xh_bonds


def choline_sites(mol):
    """{'H1': [Atom, ...], 'H2': [...], 'H3': [...], 'H4': [...]} for one choline.

    A distorted or mis-carved choline may be missing a group; the key is then an
    empty list rather than absent, so callers can iterate without guarding."""
    methyl = [h for _c, hs in find_xh_groups(mol, 'C', 3, neighbour='N') for h in hs]
    ch2_n  = [h for _c, hs in find_xh_groups(mol, 'C', 2, neighbour='N') for h in hs]
    ch2_o  = [h for _c, hs in find_xh_groups(mol, 'C', 2, neighbour='O') for h in hs]
    oh     = [h for _o, hs in find_xh_groups(mol, 'O', 1) for h in hs]
    return {'H1': methyl, 'H2': ch2_n, 'H3': ch2_o, 'H4': oh}


def urea_sites(mol):
    """{'H5': [Atom, ...], 'Nurea': [Atom, ...]} for one urea."""
    nh = find_xh_bonds(mol, 'N')
    return {
        'H5':    [h for _n, h in nh],
        'Nurea': sorted({n for n, _h in nh}, key=lambda a: mol.atoms.index(a)),
    }


def cluster_sites(mol_data):
    """Sites of every molecule in a classified cluster, as
    {'H1': [...], ..., 'Nurea': [...]} of flat Atom lists, plus a per-molecule view.

    Returns (flat, per_mol) where per_mol is
    [(species, molecule_index, {site: [Atom, ...]}), ...] — the per-molecule view is
    what distinguishes an intra pair (same entry) from an inter one."""
    flat = {k: [] for k in ('H1', 'H2', 'H3', 'H4', 'H5', 'Nurea')}
    per_mol = []
    for mi, mol in enumerate(mol_data.get('choline', [])):
        s = choline_sites(mol)
        per_mol.append(('choline', mi, s))
        for k, v in s.items():
            flat[k].extend(v)
    for mi, mol in enumerate(mol_data.get('urea', [])):
        s = urea_sites(mol)
        per_mol.append(('urea', mi, s))
        for k, v in s.items():
            flat[k].extend(v)
    return flat, per_mol
