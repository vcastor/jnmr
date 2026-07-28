#!$AMSBIN/plams
import os
import sys
import glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance, dihedral
from hassan_functions.finders   import find_xh_bonds, find_adjacent_xh_pair_anchored
from hassan_functions.constants import FORMULAS
from hassan_functions.cache     import save_cache

CLUSTERS_DIR = "clusters"

intra_NH        = []
intra_Nu_Hother = []   # urea N to the two H's on the OTHER N (non-bonded urea H's)
intra_CH_urea   = []
intra_HH      = []
intra_HH_dih  = []
intra_CH_N    = []
intra_CH_O    = []
dihedrals     = []

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue

    cluster = Molecule(xf)
    cluster.guess_bonds()
    mols    = cluster.separate()

    ureas    = [m for m in mols if m.get_formula() == FORMULAS['urea']]
    cholines = [m for m in mols if m.get_formula() == FORMULAS['choline']]

    for u in ureas:
        c_u     = next(at for at in u.atoms if at.symbol == 'C')
        hs_u    = [at for at in u.atoms if at.symbol == 'H']
        n_atoms = [at for at in u.atoms if at.symbol == 'N']
        for n, h in find_xh_bonds(u, 'N'):
            intra_NH.append(distance(n, h))
        for h in hs_u:
            intra_CH_urea.append(distance(c_u, h))
        # each urea N to the two H's on the OTHER N (the urea H's it is not bonded to)
        for n in n_atoms:
            for other in n_atoms:
                if other is n:
                    continue
                for b in other.bonds:
                    h = b.other_end(other)
                    if h.symbol == 'H':
                        intra_Nu_Hother.append(distance(n, h))

    for ch in cholines:
        pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
        if pair is None:
            continue
        c_N, hN, c_O, hO = pair
        for a in hN:
            for b in hO:
                intra_HH.append(distance(a, b))
                intra_HH_dih.append(abs(dihedral(a, c_N, c_O, b)))
        for h in hN:
            intra_CH_N.append(distance(c_N, h))
        for h in hO:
            intra_CH_O.append(distance(c_O, h))
        n_at = next((b.other_end(c_N) for b in c_N.bonds if b.other_end(c_N).symbol == 'N'), None)
        o_at = next((b.other_end(c_O) for b in c_O.bonds if b.other_end(c_O).symbol == 'O'), None)
        if n_at is not None and o_at is not None:
            dihedrals.append(abs(dihedral(n_at, c_N, c_O, o_at)))

finish()

save_cache("distance_intra", {
    "intra_NH":        intra_NH,
    "intra_Nu_Hother": intra_Nu_Hother,
    "intra_CH_urea":   intra_CH_urea,
    "intra_HH":      intra_HH,
    "intra_HH_dih":  intra_HH_dih,
    "intra_CH_N":    intra_CH_N,
    "intra_CH_O":    intra_CH_O,
    "dihedrals":     dihedrals,
})
