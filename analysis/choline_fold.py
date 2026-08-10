#!$AMSBIN/plams
"""Is the choline folded? — QTAIM evidence for an intramolecular hydrogen bond
between the hydroxyl end and the trimethylammonium head.

Context: the intramolecular J(H1,H3) and J(H1,H4) couplings only make sense
alongside the conformation they come from. A folded choline closes a ring
    O(-H4) ... H1-C-N-C-C(-O)
so the QTAIM partition should show (a) a bond critical point between the choline's
own O (or its H4) and one of its methyl H1's, and (b) a ring critical point, since
closing a ring is what creates one. Both are read from the same QTAIM outputs the
rest of the analysis uses; nothing about how the J's are computed depends on this.

Two contacts are counted separately because they are different interactions:
    O...H1   the hydroxyl OXYGEN accepting from a methyl H  (the real H-bond)
    H4...H1  the hydroxyl HYDROGEN close to a methyl H      (a contact, not an H-bond)

Writes analysis/cache/choline_fold.pkl and prints a summary table.
"""
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
# within this distance of the ring atoms' centroid. Generous, because the CP sits
# wherever the density says, not at the geometric centre.
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
    """[(x, y, z), ...] for every (3,+1) ring critical point.

    Unlike a BCP, a ring CP is not labelled with the atoms it belongs to, so it is
    located by coordinate and matched to a candidate ring by centroid proximity."""
    with open(path) as f:
        lines = f.readlines()
    coords = []
    for i, l in enumerate(lines):
        if not l.strip().startswith("CP #"):
            continue
        # the (RANK,SIGNATURE) line follows the CP # line, coordinates two below it
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
    """The atoms of the ring closed by an O...H1 contact: the covalent path from the
    methyl H's carbon back to the O, plus the two contacting atoms. Returned as a
    list of Atom objects, or None if no path exists (distorted choline)."""
    start = h1_at.bonds[0].other_end(h1_at)          # the methyl carbon
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
n_systems       = 0
n_cholines      = 0
n_folded_o      = 0      # cholines with an O...H1 BCP
n_folded_h      = 0      # cholines with an H4...H1 BCP
n_folded_either = 0
n_with_rcp      = 0      # folded AND a ring CP near the closed ring's centroid

o_h1_bcp   = []   # (distance, rho, gb, vb, ratio)
h4_h1_bcp  = []
ring_sizes = defaultdict(int)     # n atoms in the closed ring -> count
fold_steps = []                   # steps showing a fold, for cross-referencing J

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

    bcps = read_bcps(qpath)
    rcps = read_ring_cps(qpath)
    n_systems += 1
    step = int(base.split("MDStep")[1].split("_")[0]) if "MDStep" in base else None

    for ch in mol_data['choline']:
        sites = choline_sites(ch)
        if not sites['H1'] or not sites['H4']:
            continue
        n_cholines += 1
        h4    = sites['H4'][0]
        o_at  = h4.bonds[0].other_end(h4)
        hit_o = hit_h = False

        for h1 in sites['H1']:
            p = bcps.get(frozenset({o_at.cluster_id, h1.cluster_id}))
            if p is not None:
                hit_o = True
                o_h1_bcp.append((distance(o_at, h1), p['rho'], p['gb'],
                                 p['vb'], p['ratio']))
                ring = ring_path_atoms(ch, o_at, h1)
                if ring:
                    ring_sizes[len(ring)] += 1
                    cen = np.mean([np.array(a.coords) for a in ring], axis=0)
                    if any(np.linalg.norm(r - cen) <= RCP_CENTROID_TOL for r in rcps):
                        n_with_rcp += 1
            q = bcps.get(frozenset({h4.cluster_id, h1.cluster_id}))
            if q is not None:
                hit_h = True
                h4_h1_bcp.append((distance(h4, h1), q['rho'], q['gb'],
                                  q['vb'], q['ratio']))

        n_folded_o += hit_o
        n_folded_h += hit_h
        if hit_o or hit_h:
            n_folded_either += 1
            if step is not None:
                fold_steps.append(step)

finish()

def _pct(a, b):
    return f"{100*a/b:5.1f}%" if b else "  n/a"

print(f"\n{'='*62}")
print("  Folded choline — QTAIM intramolecular contacts")
print(f"{'='*62}")
print(f"  QTAIM outputs used:            {n_systems}")
print(f"  cholines examined:             {n_cholines}")
print(f"  with an O...H(CH3) BCP:        {n_folded_o:5d}  {_pct(n_folded_o, n_cholines)}")
print(f"  with an H(O)...H(CH3) BCP:     {n_folded_h:5d}  {_pct(n_folded_h, n_cholines)}")
print(f"  folded by either contact:      {n_folded_either:5d}  "
      f"{_pct(n_folded_either, n_cholines)}")
print(f"  ...also showing a ring CP:     {n_with_rcp:5d}  {_pct(n_with_rcp, n_folded_o)}"
      f"  (of the O...H1 folds)")
if o_h1_bcp:
    d, rho = np.array([r[0] for r in o_h1_bcp]), np.array([r[1] for r in o_h1_bcp])
    print(f"  O...H1  d = {d.mean():.3f} +- {d.std():.3f} A, "
          f"rho = {rho.mean():.4f} a.u.")
if h4_h1_bcp:
    d, rho = np.array([r[0] for r in h4_h1_bcp]), np.array([r[1] for r in h4_h1_bcp])
    print(f"  H4...H1 d = {d.mean():.3f} +- {d.std():.3f} A, "
          f"rho = {rho.mean():.4f} a.u.")
if ring_sizes:
    print("  ring size (atoms) -> count:    "
          + ", ".join(f"{k}:{v}" for k, v in sorted(ring_sizes.items())))
print()

save_cache("choline_fold", {
    "n_systems":       n_systems,
    "n_cholines":      n_cholines,
    "n_folded_o":      n_folded_o,
    "n_folded_h":      n_folded_h,
    "n_folded_either": n_folded_either,
    "n_with_rcp":      n_with_rcp,
    "o_h1_bcp":        o_h1_bcp,
    "h4_h1_bcp":       h4_h1_bcp,
    "ring_sizes":      dict(ring_sizes),
    "fold_steps":      sorted(set(fold_steps)),
})
