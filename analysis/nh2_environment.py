#!$AMSBIN/plams
"""Where the urea NH2 hydrogens point: per H5 the nearest atom of each acceptor
class present in its cluster (geometry, every cluster xyz), plus every H5 BCP from
the QTAIM outputs. Writes analysis/cache/nh2_environment.pkl;
nh2_environment_plot.py draws it."""
import os
import sys
import glob
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance
from hassan_functions.ordering  import classify_sort, compute_offsets
from hassan_functions.sites     import cluster_sites
from hassan_functions.constants import FORMULAS
from hassan_functions.cache     import save_cache

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "amsoutput/qtaim"
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"
SPECIES      = ['urea', 'choline', 'chloride']
CONTACT      = 3.0   # Å; H5...acceptor within this counts as a contact

# acceptor classes the NH2 can donate to
ACCEPTORS = ('Cl', 'Och', 'Ou', 'H1')   # chloride, choline OH oxygen, other-urea O,
                                        # choline methyl H (the NH2-CH3 J contact)

def read_bcps(path):
    """{frozenset({atomA, atomB}): {'rho','gb','vb','ratio'}} — same reader as
    analysis/qtaim_analysis.py, kept local so the two stay independent."""
    with open(path) as f:
        lines = f.readlines()
    bp_start = next(i for i, l in enumerate(lines) if BP_HEADER in l)

    pair_by_cp = {}
    for line in lines[bp_start + 4:]:
        if line.strip().startswith("---"):
            break
        parts = line.split()
        pair_by_cp[int(parts[1])] = frozenset({int(parts[2]), int(parts[4])})

    bcps = {}
    for i in range(bp_start):
        s = lines[i].strip()
        if not s.startswith("CP #"):
            continue
        cp_num = int(s.split()[2])
        if cp_num not in pair_by_cp:
            continue
        rho   = float(lines[i + 24].split()[2])
        gb    = float(lines[i + 45].split()[2])
        vb    = float(lines[i + 46].split()[2])
        ratio = np.abs(vb)/gb if gb != 0 else 0.0
        bcps[pair_by_cp[cp_num]] = {'rho': rho, 'gb': gb, 'vb': vb, 'ratio': ratio}
    return bcps

d_acc   = {k: [] for k in ACCEPTORS}   # per H5: distance to the nearest atom of the class
n_close = {k: [] for k in ACCEPTORS}   # per H5: atoms of the class within CONTACT
nearest = defaultdict(int)             # per H5: class owning the closest atom
bcp     = defaultdict(list)            # partner tag -> (d, rho, gb, vb, ratio)
n_sys = n_h5 = n_q = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    base  = os.path.splitext(os.path.basename(xf))[0]
    qpath = os.path.join(QTAIM_DIR, f"{base}.out")

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

    ureas = mol_data['urea']
    if not ureas:
        continue
    n_sys += 1
    flat, per_mol = cluster_sites(mol_data)
    cls   = [at for m in mol_data['chloride'] for at in m.atoms]
    och   = [at for ch in mol_data['choline'] for at in ch.atoms if at.symbol == 'O']
    u_o   = {id(u): next(a for a in u.atoms if a.symbol == 'O') for u in ureas}

    for u in ureas:
        h5s     = [a for a in u.atoms if a.symbol == 'H']
        other_o = [u_o[id(v)] for v in ureas if v is not u]
        for h5 in h5s:
            n_h5 += 1
            pools = {'Cl': cls, 'Och': och, 'Ou': other_o, 'H1': flat['H1']}
            best  = (None, np.inf)
            for k in ACCEPTORS:
                n_close[k].append(sum(1 for a in pools[k]
                                      if distance(h5, a) <= CONTACT))
                if not pools[k]:
                    continue
                dmin = min(distance(h5, a) for a in pools[k])
                d_acc[k].append(dmin)
                if dmin < best[1]:
                    best = (k, dmin)
            if best[0] is not None:
                nearest[best[0]] += 1

    if not os.path.exists(qpath):
        continue
    n_q += 1
    bcps   = read_bcps(qpath)
    h5_ids = {a.cluster_id for u in ureas for a in u.atoms if a.symbol == 'H'}
    tag_of = {}
    for at in cls:
        tag_of[at.cluster_id] = 'Cl'
    for at in och:
        tag_of[at.cluster_id] = 'Och'
    for u in ureas:
        tag_of[u_o[id(u)].cluster_id] = 'Ou'
        for a in u.atoms:
            if a.symbol == 'N':
                tag_of[a.cluster_id] = 'Nu'
    for k in ('H1', 'H2', 'H3', 'H4'):
        for a in flat[k]:
            tag_of[a.cluster_id] = k
    at_of, mol_of = {}, {}
    for mi, mol in enumerate(m for name in SPECIES for m in mol_data[name]):
        for at in mol.atoms:
            at_of[at.cluster_id]  = at
            mol_of[at.cluster_id] = mi
    for pair, p in bcps.items():
        if len(pair) != 2 or not (pair & h5_ids):
            continue
        a, b = sorted(pair)
        if mol_of[a] == mol_of[b]:           # covalent N-H etc.: only inter counts
            continue
        pid = a if b in h5_ids else b
        if pid in h5_ids and a in h5_ids:
            tag = 'H5'                       # NH2...NH2 dihydrogen contact
        else:
            tag = tag_of.get(pid, at_of[pid].symbol)
        bcp[tag].append((distance(at_of[a], at_of[b]),
                         p['rho'], p['gb'], p['vb'], p['ratio']))

finish()

def _dstats(v):
    v = np.array(v)
    return f"n={v.size:5d}  {v.mean():.3f} +- {v.std():.3f} A  min {v.min():.3f}"

print(f"\n{'='*62}")
print("  Urea NH2 environment")
print(f"{'='*62}")
print(f"  clusters with urea:  {n_sys}   NH2 hydrogens: {n_h5}   QTAIM outputs: {n_q}")
print("\n  nearest acceptor per class (per H5, when the cluster has the class):")
for k in ACCEPTORS:
    if d_acc[k]:
        print(f"    H5...{k:<4} {_dstats(d_acc[k])}")
print(f"\n  closest partner of each H5:")
for k, v in sorted(nearest.items(), key=lambda x: -x[1]):
    print(f"    {k:<4} {v:5d}  {100*v/n_h5:5.1f}%")
print(f"\n  contacts within {CONTACT} A (mean per H5):")
for k in ACCEPTORS:
    if n_close[k]:
        print(f"    {k:<4} {np.mean(n_close[k]):.2f}")
if bcp:
    print("\n  H5 BCPs by partner:")
    for tag, rows in sorted(bcp.items(), key=lambda x: -len(x[1])):
        d   = np.array([r[0] for r in rows])
        rho = np.array([r[1] for r in rows])
        print(f"    H5...{tag:<4} n={len(rows):4d}  d={d.mean():.3f} +- {d.std():.3f} A  "
              f"rho={rho.mean():.4f} a.u.")
print()

save_cache("nh2_environment", {
    "n_sys":   n_sys,
    "n_h5":    n_h5,
    "n_q":     n_q,
    "d_acc":   {k: v for k, v in d_acc.items()},
    "n_close": {k: v for k, v in n_close.items()},
    "nearest": dict(nearest),
    "bcp":     dict(bcp),
    "contact": CONTACT,
})
