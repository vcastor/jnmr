#!$AMSBIN/plams
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance, angle
from hassan_functions.finders   import find_xh_groups
from hassan_functions.ordering  import classify_sort, compute_offsets
from hassan_functions.constants import FORMULAS

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "/Users/vcastor/Desktop/backup_qtaimcdft/qtaim" #tmp
PLOT_DIR     = "plots"
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"
SPECIES      = ['urea', 'choline', 'chloride']
D_CUTOFF     = 3.0   # only look at N...H contacts closer than this (A)

def read_bcp_pairs(path):
    """Set of frozenset({atomA, atomB}) for every bond path (BCP) in a QTAIM out."""
    with open(path) as f:
        lines = f.readlines()
    bp_start = next(i for i, l in enumerate(lines) if BP_HEADER in l)
    pairs = set()
    for line in lines[bp_start + 4:]:
        if line.strip().startswith("---"):
            break
        parts = line.split()
        pairs.add(frozenset({int(parts[2]), int(parts[4])}))
    return pairs

# each entry: (distance, C-H..N angle, cluster base, N cluster_id, H cluster_id)
bcp_contacts   = []
nobcp_contacts = []

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
    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1

    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    bcps     = read_bcp_pairs(qpath)

    for u in ureas:
        n_urea = [at for at in u.atoms if at.symbol == 'N']
        for ch in cholines:
            for n in n_urea:
                for c, hs in find_xh_groups(ch, 'C', 3, neighbour='N'):
                    for h in hs:
                        d = distance(n, h)
                        if d > D_CUTOFF:
                            continue
                        ang = angle(c, h, n)
                        rec = (d, ang, base, n.cluster_id, h.cluster_id)
                        if frozenset({n.cluster_id, h.cluster_id}) in bcps:
                            bcp_contacts.append(rec)
                        else:
                            nobcp_contacts.append(rec)

finish()

bcp_contacts.sort()
nobcp_contacts.sort()
d_min_bcp = bcp_contacts[0][0]

ang_bcp   = np.array([r[1] for r in bcp_contacts])
ang_nobcp = np.array([r[1] for r in nobcp_contacts])

print(f"N(urea) - H(CH3 choline) contacts within {D_CUTOFF:.1f} A")
print(f"  with a BCP   : {len(bcp_contacts):4d}   "
      f"angle C-H..N: mean={ang_bcp.mean():.1f}  min={ang_bcp.min():.1f}  max={ang_bcp.max():.1f}")
print(f"  without a BCP: {len(nobcp_contacts):4d}   "
      f"angle C-H..N: mean={ang_nobcp.mean():.1f}  min={ang_nobcp.min():.1f}  max={ang_nobcp.max():.1f}")
print(f"  shortest contact WITH a BCP (the ~2.2 A in the plot): {d_min_bcp:.3f} A")
print()

short = [r for r in nobcp_contacts if r[0] <= d_min_bcp]
print(f"contacts WITHOUT a BCP yet shorter than {d_min_bcp:.3f} A: {len(short)}")
print(f"{'d (A)':>8}  {'C-H..N':>7}  {'cluster':<22} {'N#':>5} {'H#':>5}")
for d, ang, base, n_id, h_id in short:
    print(f"{d:8.3f}  {ang:7.1f}  {base:<22} {n_id:5d} {h_id:5d}")

# distance vs C-H..N angle: BCP vs no-BCP, shows why short contacts miss a BCP
fig, ax = plt.subplots(figsize=(8, 6))
nb = np.array([(r[0], r[1]) for r in nobcp_contacts])
b  = np.array([(r[0], r[1]) for r in bcp_contacts])
ax.scatter(nb[:, 0], nb[:, 1], s=26, c="crimson",   alpha=0.5,
           edgecolor='black', linewidth=0.3, label="no BCP")
ax.scatter(b[:, 0],  b[:, 1],  s=26, c="seagreen",  alpha=0.6,
           edgecolor='black', linewidth=0.3, label="BCP")
ax.axvline(d_min_bcp, color="grey", linestyle="--", linewidth=1)
ax.set_title(r"N(urea) $\cdots$ H(CH$_3$): contact vs linearity")
ax.set_xlabel(r"N$\cdots$H distance (Å)")
ax.set_ylabel(r"C–H$\cdots$N angle")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/qtaim_nh_smalldist.pdf")
plt.close(fig)

