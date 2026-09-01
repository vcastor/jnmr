#!$AMSBIN/plams
"""Choline fold: O(-H4)...H1 ring closure vs the same contact to ANOTHER choline.
BCP evidence (QTAIM outputs) split intra/inter, then the intra-BCP distance range
calibrates a geometry-only criterion applied to every cluster xyz."""
import os
import sys
import glob
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance
from hassan_functions.ordering  import classify_sort, compute_offsets
from hassan_functions.sites     import choline_sites
from hassan_functions.constants import FORMULAS
from hassan_functions.cache     import save_cache

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "amsoutput/qtaim"
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"
SPECIES      = ['urea', 'choline', 'chloride']

# A ring critical point is accepted as "belonging to" a candidate ring when it sits
# within this distance of the ring atoms' centroid.
RCP_CENTROID_TOL = 1.6   # Å

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

def read_ring_cps(path):
    """[(x, y, z), ...] for every (3,+1) ring critical point. A ring CP carries no
    atom labels, so it is matched to a candidate ring by centroid proximity."""
    with open(path) as f:
        lines = f.readlines()
    coords = []
    for i, l in enumerate(lines):
        if not l.strip().startswith("CP #"):
            continue
        sig_line = lines[i + 1] if i + 1 < len(lines) else ""
        if "(3,+1)" not in sig_line:
            continue
        for j in range(i + 2, min(i + 8, len(lines))):
            if "CP COORDINATES" in lines[j]:
                parts = lines[j].split(":")[1].split()
                coords.append(np.array([float(v) for v in parts[:3]]))
                break
    return coords

def ring_path_atoms(mol, o_at, h1_at):
    """Atoms of the ring closed by an O...H1 contact (covalent path from the methyl C
    back to the O, plus the two contacting atoms), or None if no path exists."""
    start = h1_at.bonds[0].other_end(h1_at)
    prev, queue = {start: None}, [start]
    while queue:
        cur = queue.pop(0)
        if cur is o_at:
            break
        for b in cur.bonds:
            nb = b.other_end(cur)
            if nb.symbol == 'H' or nb in prev:
                continue
            prev[nb] = cur
            queue.append(nb)
    if o_at not in prev:
        return None
    path, node = [], o_at
    while node is not None:
        path.append(node)
        node = prev[node]
    return path + [h1_at]

# ── accumulators ────────────────────────────────────────────────────────────
n_systems  = 0            # clusters with a QTAIM output
n_cholines = 0            # cholines in those clusters
n_intra_o  = 0            # cholines with an intra O...H1 BCP (ring closed)
n_intra_h  = 0            # cholines with an intra H4...H1 BCP
n_intra    = 0            # either intra contact
n_with_rcp = 0            # intra O...H1 folds confirmed by a ring CP
n_inter_o  = 0            # cholines whose O has a BCP to another choline's H1
n_inter_h  = 0            # cholines whose H4 has a BCP to another choline's H1
n_inter    = 0            # either inter contact

intra_o_bcp = []          # (distance, rho, gb, vb, ratio)
intra_h_bcp = []
inter_o_bcp = []
inter_h_bcp = []
ring_sizes  = defaultdict(int)
fold_steps  = []

# geometry over EVERY xyz (QTAIM output or not): per choline the closest O...H1
# distance, own methyls vs other cholines' methyls
geo = []                  # (step, dmin_intra, dmin_inter)  dmin_inter None if single choline

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    base  = os.path.splitext(os.path.basename(xf))[0]
    qpath = os.path.join(QTAIM_DIR, f"{base}.out")
    step  = int(base.split("MDStep")[1].split("_")[0]) if "MDStep" in base else None

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

    ch_sites = [choline_sites(ch) for ch in mol_data['choline']]
    have_q   = os.path.exists(qpath)
    if have_q:
        bcps = read_bcps(qpath)
        rcps = read_ring_cps(qpath)
        n_systems += 1

    for ci, (ch, sites) in enumerate(zip(mol_data['choline'], ch_sites)):
        if not sites['H1'] or not sites['H4']:
            continue
        h4   = sites['H4'][0]
        o_at = h4.bonds[0].other_end(h4)
        own_h1   = sites['H1']
        other_h1 = [h for cj, s in enumerate(ch_sites) if cj != ci for h in s['H1']]

        geo.append((step,
                    min(distance(o_at, h) for h in own_h1),
                    min((distance(o_at, h) for h in other_h1), default=None)))

        if not have_q:
            continue
        n_cholines += 1
        hit_io = hit_ih = hit_eo = hit_eh = False

        for h1 in own_h1:
            p = bcps.get(frozenset({o_at.cluster_id, h1.cluster_id}))
            if p is not None:
                hit_io = True
                intra_o_bcp.append((distance(o_at, h1), p['rho'], p['gb'],
                                    p['vb'], p['ratio']))
                ring = ring_path_atoms(ch, o_at, h1)
                if ring:
                    ring_sizes[len(ring)] += 1
                    cen = np.mean([np.array(a.coords) for a in ring], axis=0)
                    if any(np.linalg.norm(r - cen) <= RCP_CENTROID_TOL for r in rcps):
                        n_with_rcp += 1
            q = bcps.get(frozenset({h4.cluster_id, h1.cluster_id}))
            if q is not None:
                hit_ih = True
                intra_h_bcp.append((distance(h4, h1), q['rho'], q['gb'],
                                    q['vb'], q['ratio']))

        for h1 in other_h1:
            p = bcps.get(frozenset({o_at.cluster_id, h1.cluster_id}))
            if p is not None:
                hit_eo = True
                inter_o_bcp.append((distance(o_at, h1), p['rho'], p['gb'],
                                    p['vb'], p['ratio']))
            q = bcps.get(frozenset({h4.cluster_id, h1.cluster_id}))
            if q is not None:
                hit_eh = True
                inter_h_bcp.append((distance(h4, h1), q['rho'], q['gb'],
                                    q['vb'], q['ratio']))

        n_intra_o += hit_io
        n_intra_h += hit_ih
        n_inter_o += hit_eo
        n_inter_h += hit_eh
        if hit_io or hit_ih:
            n_intra += 1
            if step is not None:
                fold_steps.append(step)
        if hit_eo or hit_eh:
            n_inter += 1

finish()

def _pct(a, b):
    return f"{100*a/b:5.1f}%" if b else "  n/a"

def _dstats(rows, label):
    if not rows:
        return
    d, rho = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    print(f"  {label}  d = {d.mean():.3f} +- {d.std():.3f} A, rho = {rho.mean():.4f} a.u.")

print(f"\n{'='*62}")
print("  Choline O(-H4)...H1 contacts — QTAIM BCPs, intra vs inter")
print(f"{'='*62}")
print(f"  QTAIM outputs used:            {n_systems}")
print(f"  cholines examined:             {n_cholines}")
print(f"  intra O...H1 BCP (ring):       {n_intra_o:5d}  {_pct(n_intra_o, n_cholines)}")
print(f"  intra H4...H1 BCP:             {n_intra_h:5d}  {_pct(n_intra_h, n_cholines)}")
print(f"  intra either (folded):         {n_intra:5d}  {_pct(n_intra, n_cholines)}")
print(f"  ...also showing a ring CP:     {n_with_rcp:5d}  {_pct(n_with_rcp, n_intra_o)}"
      f"  (of the intra O...H1 folds)")
print(f"  inter O...H1 BCP:              {n_inter_o:5d}  {_pct(n_inter_o, n_cholines)}")
print(f"  inter H4...H1 BCP:             {n_inter_h:5d}  {_pct(n_inter_h, n_cholines)}")
print(f"  inter either:                  {n_inter:5d}  {_pct(n_inter, n_cholines)}")
_dstats(intra_o_bcp, "intra O...H1 ")
_dstats(intra_h_bcp, "intra H4...H1")
_dstats(inter_o_bcp, "inter O...H1 ")
_dstats(inter_h_bcp, "inter H4...H1")
if ring_sizes:
    print("  ring size (atoms) -> count:    "
          + ", ".join(f"{k}:{v}" for k, v in sorted(ring_sizes.items())))

# ── geometry-only extension to every xyz ─────────────────────────────────────
# The intra-BCP O...H1 distances calibrate the criterion: a choline whose closest
# own O...H1 distance is within the range QTAIM certifies as bonded counts as
# folded, with no QTAIM output needed. Same threshold for the inter contact.
if intra_o_bcp:
    dthr = float(np.array([r[0] for r in intra_o_bcp]).max())
    n_all       = len(geo)
    n_geo_intra = sum(1 for _, di, _de in geo if di <= dthr)
    with_pair   = [(s, di, de) for s, di, de in geo if de is not None]
    n_geo_inter = sum(1 for _, _di, de in with_pair if de <= dthr)
    steps_intra = sorted({s for s, di, _de in geo if s is not None and di <= dthr})
    print(f"\n  geometry criterion d(O...H1) <= {dthr:.3f} A "
          f"(max intra-BCP distance), all xyz:")
    print(f"  cholines (all clusters):       {n_all}")
    print(f"  folded (intra, geometric):     {n_geo_intra:5d}  {_pct(n_geo_intra, n_all)}")
    print(f"  inter contact (geometric):     {n_geo_inter:5d}  "
          f"{_pct(n_geo_inter, len(with_pair))}  (of {len(with_pair)} with a partner choline)")
print()

save_cache("choline_fold", {
    "n_systems":   n_systems,
    "n_cholines":  n_cholines,
    "n_intra_o":   n_intra_o,
    "n_intra_h":   n_intra_h,
    "n_intra":     n_intra,
    "n_with_rcp":  n_with_rcp,
    "n_inter_o":   n_inter_o,
    "n_inter_h":   n_inter_h,
    "n_inter":     n_inter,
    "intra_o_bcp": intra_o_bcp,
    "intra_h_bcp": intra_h_bcp,
    "inter_o_bcp": inter_o_bcp,
    "inter_h_bcp": inter_h_bcp,
    "ring_sizes":  dict(ring_sizes),
    "fold_steps":  sorted(set(fold_steps)),
    "geo":         geo,
})
