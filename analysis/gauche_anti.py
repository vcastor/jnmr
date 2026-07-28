#!$AMSBIN/plams
import os
import sys
import glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import dihedral
from hassan_functions.finders   import find_adjacent_xh_pair_anchored
from hassan_functions.ordering  import classify_sort
from hassan_functions.constants import FORMULAS
from hassan_functions.cache     import save_cache

CLUSTERS_DIR = "clusters"
SPECIES      = ['urea', 'choline', 'chloride']

def ncco_dihedral(ch):
    pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
    if pair is None:
        return None
    c1, _, c2, _ = pair
    n_at = next((b.other_end(c1) for b in c1.bonds if b.other_end(c1).symbol == 'N'), None)
    o_at = next((b.other_end(c2) for b in c2.bonds if b.other_end(c2).symbol == 'O'), None)
    if n_at is None or o_at is None:
        return None
    return dihedral(n_at, c1, c2, o_at)

dihedrals = []; n_systems_used = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    n_systems_used += 1

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})

    for ch in mol_data['choline']:
        d = ncco_dihedral(ch)
        if d is not None:
            dihedrals.append(abs(d))

finish()

save_cache("gauche_anti", {"dihedrals": dihedrals, "n_systems_used": n_systems_used})
