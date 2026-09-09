#!$AMSBIN/plams
"""Where the chlorides sit: per Cl the nearest atom of every named site present in
its cluster and the contacts within CONTACT (geometry, every cluster xyz), plus
every Cl BCP from the QTAIM outputs. Writes analysis/cache/cl_environment.pkl;
cl_environment_plot.py draws it."""
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
SITES        = ('H1', 'H2', 'H3', 'H4', 'H5', 'Nurea')
H_SITES      = ('H1', 'H2', 'H3', 'H4', 'H5')
CONTACT      = 3.0   # Å; Cl...H within this counts as a contact (vdW Cl+H ~2.95)

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

d_site   = {k: [] for k in SITES}    # per Cl: distance to the nearest atom of the site
n_close  = {k: [] for k in H_SITES}  # per Cl: H's of the site within CONTACT
nearest  = defaultdict(int)          # per Cl: which site owns the closest atom
d_ncho   = []                        # per Cl: nearest choline N+ (electrostatics)
d_clcl   = []                        # per Cl: nearest other Cl in the cluster
bcp      = defaultdict(list)         # partner tag -> (d, rho, gb, vb, ratio)
n_sys = n_cl = n_q = 0

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

    cls = [at for m in mol_data['chloride'] for at in m.atoms]
    if not cls:
        continue
    n_sys += 1
    flat, _  = cluster_sites(mol_data)
    cho_n    = [at for ch in mol_data['choline'] for at in ch.atoms
                if at.symbol == 'N']

    for cl in cls:
        n_cl += 1
        best = (None, np.inf)
        for k in SITES:
            if not flat[k]:
                continue
            dmin = min(distance(cl, a) for a in flat[k])
            d_site[k].append(dmin)
            if dmin < best[1]:
                best = (k, dmin)
        if best[0] is not None:
            nearest[best[0]] += 1
        for k in H_SITES:
            n_close[k].append(sum(1 for a in flat[k] if distance(cl, a) <= CONTACT))
        if cho_n:
            d_ncho.append(min(distance(cl, n) for n in cho_n))
        others = [o for o in cls if o is not cl]
        if others:
            d_clcl.append(min(distance(cl, o) for o in others))

    if not os.path.exists(qpath):
        continue
    n_q += 1
    bcps    = read_bcps(qpath)
    cl_ids  = {at.cluster_id for at in cls}
    site_of = {a.cluster_id: k for k in SITES for a in flat[k]}
    at_of   = {at.cluster_id: at for name in SPECIES
               for m in mol_data[name] for at in m.atoms}
    for pair, p in bcps.items():
        if len(pair) != 2 or not (pair & cl_ids):
            continue
        a, b = sorted(pair)
        pid  = a if b in cl_ids else b
        if pid in cl_ids and a in cl_ids:      # Cl...Cl BCP
            tag = 'Cl'
        else:
            tag = site_of.get(pid, at_of[pid].symbol)
        bcp[tag].append((distance(at_of[a], at_of[b]),
                         p['rho'], p['gb'], p['vb'], p['ratio']))

finish()

def _dstats(v):
    v = np.array(v)
    return f"n={v.size:5d}  {v.mean():.3f} +- {v.std():.3f} A  min {v.min():.3f}"

print(f"\n{'='*62}")
print("  Chloride environment")
print(f"{'='*62}")
print(f"  clusters with Cl:  {n_sys}   chlorides: {n_cl}   QTAIM outputs: {n_q}")
print("\n  nearest atom per site (per Cl, when the cluster has the site):")
for k in SITES:
    if d_site[k]:
        print(f"    Cl...{k:<6} {_dstats(d_site[k])}")
if d_ncho:
    print(f"    Cl...N+     {_dstats(d_ncho)}")
if d_clcl:
    print(f"    Cl...Cl     {_dstats(d_clcl)}")
print(f"\n  closest partner of each Cl:")
for k, v in sorted(nearest.items(), key=lambda x: -x[1]):
    print(f"    {k:<6} {v:5d}  {100*v/n_cl:5.1f}%")
print(f"\n  H contacts within {CONTACT} A (mean per Cl):")
for k in H_SITES:
    if n_close[k]:
        print(f"    {k:<6} {np.mean(n_close[k]):.2f}")
if bcp:
    print("\n  Cl BCPs by partner:")
    for tag, rows in sorted(bcp.items(), key=lambda x: -len(x[1])):
        d   = np.array([r[0] for r in rows])
        rho = np.array([r[1] for r in rows])
        print(f"    Cl...{tag:<6} n={len(rows):4d}  d={d.mean():.3f} +- {d.std():.3f} A  "
              f"rho={rho.mean():.4f} a.u.")
print()

save_cache("cl_environment", {
    "n_sys":   n_sys,
    "n_cl":    n_cl,
    "n_q":     n_q,
    "d_site":  {k: v for k, v in d_site.items()},
    "n_close": {k: v for k, v in n_close.items()},
    "nearest": dict(nearest),
    "d_ncho":  d_ncho,
    "d_clcl":  d_clcl,
    "bcp":     dict(bcp),
    "contact": CONTACT,
})
