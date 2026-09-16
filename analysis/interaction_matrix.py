#!$AMSBIN/plams
import os
import sys
import glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance
from hassan_functions.ordering  import classify_sort, compute_offsets
from hassan_functions.sites     import choline_sites, urea_sites, cluster_sites
from hassan_functions.constants import FORMULAS
from hassan_functions.cache     import save_cache

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "amsoutput/qtaim"
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"
SPECIES      = ['urea', 'choline', 'chloride']
D_ON = 2.5   # Å; H...O / H...N contact ceiling
D_CL = 3.3   # Å; H...Cl contact ceiling (BCP Cl...H populations reach ~3.3)
D_CLCL = 3.7 # Å; Cl...Cl ion-ion contact (BCP d = 3.42 +- 0.14)

SP  = ('choline', 'urea', 'chloride')
HT  = ('H1', 'H2', 'H3', 'H4', 'H5')
S5  = ('Ou', 'Nu', 'Och', 'Cl')   # H5 acceptor sites
S5B = S5 + ('other',)
SITE = {('urea', 'O'): 'Ou', ('urea', 'N'): 'Nu', ('choline', 'O'): 'Och',
        ('chloride', 'Cl'): 'Cl'}

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
        l = lines[i].strip()
        if not l.startswith("CP #"):
            continue
        cp_num = int(l.split()[2])
        if cp_num not in pair_by_cp:
            continue
        rho   = float(lines[i + 24].split()[2])
        gb    = float(lines[i + 45].split()[2])
        vb    = float(lines[i + 46].split()[2])
        ratio = np.abs(vb)/gb if gb != 0 else 0.0
        bcps[pair_by_cp[cp_num]] = {'rho': rho, 'gb': gb, 'vb': vb, 'ratio': ratio}
    return bcps

matrix = {a: {b: 0 for b in SP} for a in SP}   # row: species, col: partner species
zoom   = {h: {b: 0 for b in SP} for h in HT}   # row: donor H type, col: acceptor species
matrix_b = {a: {b: 0 for b in SP} for a in SP}   # same tables from the BCPs
zoom_b   = {h: {b: 0 for b in SP} for h in HT}
dist_b   = {a: {b: [] for b in SP} for a in SP}  # BCP distances per cell (row view)
zoom5    = {s: 0 for s in S5}                    # H5 contacts per acceptor site
dist5    = {s: [] for s in S5}
zoom5b   = {s: 0 for s in S5B}                   # H5 BCPs per partner site
dist5b   = {s: [] for s in S5B}                  # (d, rho)
n_sys = n_contact = n_q = n_bcp = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    n_sys += 1

    # (H atom, H type, molecule id) donors; (atom, species, molecule id, cutoff) acceptors
    donors, acceptors = [], []
    mid = 0
    for ch in mol_data['choline']:
        s = choline_sites(ch)
        for h in ('H1', 'H2', 'H3', 'H4'):
            donors += [(a, h, mid) for a in s[h]]
        o = next((a for a in ch.atoms if a.symbol == 'O'), None)
        if o is not None:
            acceptors.append((o, 'choline', mid, D_ON, 'Och'))
        mid += 1
    for u in mol_data['urea']:
        s = urea_sites(u)
        donors += [(a, 'H5', mid) for a in s['H5']]
        acceptors.append((next(a for a in u.atoms if a.symbol == 'O'),
                          'urea', mid, D_ON, 'Ou'))
        acceptors += [(a, 'urea', mid, D_ON, 'Nu') for a in s['Nurea']]
        mid += 1
    cl_atoms = []
    for m in mol_data['chloride']:
        for a in m.atoms:
            acceptors.append((a, 'chloride', mid, D_CL, 'Cl'))
            cl_atoms.append((a, mid))
        mid += 1

    donor_sp = {'H1': 'choline', 'H2': 'choline', 'H3': 'choline',
                'H4': 'choline', 'H5': 'urea'}
    for h_at, ht, dm in donors:
        for ac, sp, am, cut, st in acceptors:
            d = distance(h_at, ac)
            if am == dm or d > cut:
                continue
            n_contact += 1
            zoom[ht][sp] += 1
            if ht == 'H5':
                zoom5[st] += 1
                dist5[st].append(d)
            sd = donor_sp[ht]
            a, b = sorted((sd, sp), key=SP.index)
            matrix[a][b] += 1

    # ion-ion contacts have no donor H; count Cl...Cl pairs directly
    for i, (a, am) in enumerate(cl_atoms):
        for b, bm in cl_atoms[i + 1:]:
            if am != bm and distance(a, b) <= D_CLCL:
                n_contact += 1
                matrix['chloride']['chloride'] += 1

    base  = os.path.splitext(os.path.basename(xf))[0]
    qpath = os.path.join(QTAIM_DIR, f"{base}.out")
    if not os.path.exists(qpath):
        continue
    n_q += 1
    offs = compute_offsets(mol_data, SPECIES)
    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1
    flat, _ = cluster_sites(mol_data)
    ht_of   = {a.cluster_id: k for k in HT for a in flat[k]}
    at_of, mol_id, sp_of, site_of = {}, {}, {}, {}
    for bi, (name, mol) in enumerate((n, m) for n in SPECIES for m in mol_data[n]):
        for at in mol.atoms:
            at_of[at.cluster_id]  = at
            mol_id[at.cluster_id] = bi
            sp_of[at.cluster_id]  = name
            site_of[at.cluster_id] = SITE.get((name, at.symbol), 'other')
    for pair, p in read_bcps(qpath).items():
        if len(pair) != 2:
            continue
        a, b = sorted(pair)
        if mol_id[a] == mol_id[b]:
            continue
        n_bcp += 1
        sa, sb = sorted((sp_of[a], sp_of[b]), key=SP.index)
        d = distance(at_of[a], at_of[b])
        matrix_b[sa][sb] += 1
        dist_b[sa][sb].append(d)
        if a in ht_of:
            zoom_b[ht_of[a]][sp_of[b]] += 1
        if b in ht_of:
            zoom_b[ht_of[b]][sp_of[a]] += 1
        for h, o in ((a, b), (b, a)):
            if ht_of.get(h) == 'H5':
                zoom5b[site_of[o]] += 1
                dist5b[site_of[o]].append((d, p['rho']))

finish()

def row(vals):
    tot = sum(vals)
    pct = "  ".join(f"{100*v/tot:7.1f}%" if tot else f"{'n/a':>8}" for v in vals)
    return f"{pct}   (n={tot})"

def triangle(mat, total, label):
    """Upper-triangle species table: each unique pair once, all cells sum to 100%."""
    print(f"  {label} (all cells sum to 100%, n={total}):")
    print(f"  {'':>9} " + "  ".join(f"{b:>8}" for b in SP))
    for i, a in enumerate(SP):
        cells = []
        for j, b in enumerate(SP):
            if j < i:
                cells.append(f"{'':>8}")
            else:
                v = mat[a][b]
                cells.append(f"{100*v/total:7.1f}%" if total else f"{'n/a':>8}")
        print(f"  {a:>9} " + "  ".join(cells))

print(f"\n{'='*66}")
print("  Species interactions — unique pairs, one global 100%")
print(f"{'='*66}")
print(f"  clusters: {n_sys}   contacts: {n_contact}   "
      f"cutoffs O/N {D_ON} A, Cl {D_CL} A, Cl-Cl {D_CLCL} A")
triangle(matrix, n_contact, "geometric contacts")

print(f"\n  zoom — donor H type vs acceptor species (rows sum to 100%):")
print(f"  {'':>9} " + "  ".join(f"{b:>8}" for b in SP))
for h in HT:
    print(f"  {h:>9} " + row([zoom[h][b] for b in SP]))

print(f"\n  H5 by acceptor site (row sums to 100%):")
print(f"  {'':>9} " + "  ".join(f"{s:>8}" for s in S5))
print(f"  {'H5':>9} " + row([zoom5[s] for s in S5]))
for s in S5:
    v = np.array(dist5[s])
    if v.size:
        print(f"    H5...{s:<5} n={v.size:5d}  d={v.mean():.3f} +- {v.std():.3f} A")

print(f"\n{'='*66}")
print("  Same tables from the QTAIM BCPs (inter-molecular only)")
print(f"{'='*66}")
print(f"  QTAIM outputs: {n_q}   inter BCPs: {n_bcp}")
triangle(matrix_b, n_bcp, "BCP interactions")
print(f"\n  BCP distances per pair (A, mean +- std):")
print(f"  {'':>9} " + "  ".join(f"{b:>13}" for b in SP))
for i, a in enumerate(SP):
    cells = []
    for j, b in enumerate(SP):
        if j < i:
            cells.append(f"{'':>13}")
        else:
            v = np.array(dist_b[a][b])
            cells.append(f"{v.mean():5.2f} +- {v.std():4.2f}" if v.size else f"{'-':>13}")
    print(f"  {a:>9} " + "  ".join(cells))
print(f"\n  zoom — H type vs BCP partner species (rows sum to 100%):")
print(f"  {'':>9} " + "  ".join(f"{b:>8}" for b in SP))
for h in HT:
    print(f"  {h:>9} " + row([zoom_b[h][b] for b in SP]))
print(f"\n  H5 BCPs by partner site (row sums to 100%):")
print(f"  {'':>9} " + "  ".join(f"{s:>8}" for s in S5B))
print(f"  {'H5':>9} " + row([zoom5b[s] for s in S5B]))
for s in S5B:
    v = np.array(dist5b[s])
    if v.size:
        print(f"    H5...{s:<5} n={len(v):5d}  d={v[:,0].mean():.3f} +- {v[:,0].std():.3f} A  "
              f"rho={v[:,1].mean():.4f} a.u.")
print()

save_cache("interaction_matrix", {
    "matrix":  matrix,
    "zoom":    zoom,
    "matrixb": matrix_b,
    "zoomb":   zoom_b,
    "distb":   dist_b,
    "zoom5":   zoom5,
    "dist5":   dist5,
    "zoom5b":  zoom5b,
    "dist5b":  dist5b,
    "n_sys":   n_sys,
    "n":       n_contact,
    "nq":      n_q,
    "nbcp":    n_bcp,
    "don":     D_ON,
    "dcl":     D_CL,
})
