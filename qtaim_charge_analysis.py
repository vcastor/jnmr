#!$AMSBIN/plams
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from hassan_functions.finders import find_adjacent_xh_pair_anchored
from hassan_functions.ordering import classify_sort, compute_offsets
from hassan_functions.io import read_qtaim_charges
from hassan_functions.constants import FORMULAS

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "amsoutput/qtaim"
SPECIES      = ['urea', 'choline', 'chloride']

q_Nside = []   # QTAIM net charge of H from CH2 next to N+
q_Oside = []   # QTAIM net charge of H from CH2 next to O

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    base  = os.path.splitext(os.path.basename(xf))[0]
    qpath = os.path.join(QTAIM_DIR, f"{base}.out")
    if not os.path.exists(qpath):
        continue

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs = compute_offsets(mol_data, SPECIES)

    # Tag atoms with the global 1-based index they have in the reordered
    # cluster (which is what run_generator.py wrote to the ADF input, and
    # therefore what the QTAIM output's atom numbers refer to).
    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1

    cholines = mol_data['choline']
    charges  = read_qtaim_charges(qpath)

    for ch in cholines:
        _, hN, _, hO = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
        for h in hN:
            q = charges.get(h.cluster_id)
            if q is not None:
                q_Nside.append(q)
        for h in hO:
            q = charges.get(h.cluster_id)
            if q is not None:
                q_Oside.append(q)

finish()

q_N = np.array(q_Nside)
q_O = np.array(q_Oside)
print(f"q (CH2 near N+): n={len(q_N)}, mean={q_N.mean():.6f}, std={q_N.std():.6f}")
print(f"q (CH2 near O ): n={len(q_O)}, mean={q_O.mean():.6f}, std={q_O.std():.6f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(q_Nside, bins=20, color="steelblue", edgecolor="black", alpha=0.6,
        label=f"CH2 near N+ (n={len(q_Nside)})")
ax.hist(q_Oside, bins=20, color="crimson", edgecolor="black", alpha=0.6,
        label=f"CH2 near O  (n={len(q_Oside)})")
ax.set_title("QTAIM net charge of H in CH2")
ax.set_xlabel("q (au)")
ax.set_ylabel("count")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

