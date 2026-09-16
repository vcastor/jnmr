#!$AMSBIN/plams
"""Find the smallest cluster (few molecules) showing every main interaction at
once — Cl-H1, H2/H3 to the urea O/N, O(urea)-H4, NH2-CH3, Ch-Ch, urea-urea,
Cl-H4, Cl-H5 — for the reduced visualisation the experimental team asked for.
Prints the ranking and writes the winner as repr_MDStep<n>.xyz plus
_noCh / _noUrea variants (cwd)."""
import os
import sys
import glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance
from hassan_functions.ordering  import classify_sort, compute_offsets
from hassan_functions.sites     import choline_sites, urea_sites
from hassan_functions.constants import FORMULAS
from hassan_functions.cache     import save_cache

CLUSTERS_DIR = "clusters"
SPECIES      = ['urea', 'choline', 'chloride']
D_HB = 2.5   # Å; heavy...H hydrogen-bond ceiling (OH/NH donors)
D_CH = 3.0   # Å; heavy...H ceiling for the weaker C-H donors (H1/H2/H3)
D_CL = 3.3   # Å; Cl...H contact ceiling (BCP Cl...H populations reach ~3.3)
D_HH = 3.0   # Å; H5...H1 (the NH2-CH3 J contact)

MOTIFS = ('ClH1', 'H2Urea', 'H3Urea', 'NH1', 'NH2', 'NH3', 'ChUrea',
          'NH2CH3', 'ChCh', 'UreaUrea', 'ClCh', 'ClUrea')

def motif_distances(mol_data):
    """{motif: min distance or None} for one cluster."""
    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    cls      = [at for m in mol_data['chloride'] for at in m.atoms]
    u_sites  = [urea_sites(u) for u in ureas]
    ch_sites = [choline_sites(ch) for ch in cholines]
    u_o      = [next(a for a in u.atoms if a.symbol == 'O') for u in ureas]
    ch_o     = [next((a for a in ch.atoms if a.symbol == 'O'), None) for ch in cholines]
    h5       = [h for s in u_sites for h in s['H5']]
    h1       = [h for s in ch_sites for h in s['H1']]

    def dmin(pairs):
        best = None
        for a, b in pairs:
            d = distance(a, b)
            if best is None or d < best[0]:
                best = (d, a, b)
        return best

    u_n = [a for s in u_sites for a in s['Nurea']]

    m = {}
    # chloride against the choline methyl H's (top Cl BCP partner)
    m['ClH1'] = dmin([(cl, h) for cl in cls for h in h1])
    # choline CH2 H's against the urea O/N
    m['H2Urea'] = dmin([(t, h) for t in u_o + u_n
                        for s in ch_sites for h in s['H2']])
    m['H3Urea'] = dmin([(t, h) for t in u_o + u_n
                        for s in ch_sites for h in s['H3']])
    # urea N against each choline H type
    m['NH1'] = dmin([(n, h) for n in u_n for h in h1])
    m['NH2'] = dmin([(n, h) for n in u_n
                     for s in ch_sites for h in s['H2']])
    m['NH3'] = dmin([(n, h) for n in u_n
                     for s in ch_sites for h in s['H3']])
    # urea C=O accepting the choline OH
    m['ChUrea'] = dmin([(o, h) for o in u_o
                        for s in ch_sites for h in s['H4']])
    # the NH2-CH3 J contact
    m['NH2CH3'] = dmin([(a, b) for a in h5 for b in h1])
    # choline hydroxyl O accepting from another choline (H1 or H4)
    m['ChCh'] = dmin([(o, h) for ci, o in enumerate(ch_o) if o is not None
                      for cj, s in enumerate(ch_sites) if cj != ci
                      for h in s['H1'] + s['H4']])
    # urea O accepting another urea's NH
    m['UreaUrea'] = dmin([(o, h) for ui, o in enumerate(u_o)
                          for uj, s in enumerate(u_sites) if uj != ui
                          for h in s['H5']])
    # chloride accepting the choline OH
    m['ClCh'] = dmin([(cl, h) for cl in cls
                      for s in ch_sites for h in s['H4']])
    # chloride accepting the urea NH
    m['ClUrea'] = dmin([(cl, h) for cl in cls for h in h5])
    return m

def motif_ok(m):
    cut = {'ClH1': D_CL, 'H2Urea': D_CH, 'H3Urea': D_CH,
           'NH1': D_CH, 'NH2': D_CH, 'NH3': D_CH,
           'ChUrea': D_HB, 'NH2CH3': D_HH, 'ChCh': D_HB,
           'UreaUrea': D_HB, 'ClCh': D_CL, 'ClUrea': D_CL}
    return {k: (m[k] is not None and m[k] <= cut[k]) for k in MOTIFS}

def motif_dists(mp):
    return {k: (mp[k][0] if mp[k] is not None else None) for k in MOTIFS}

def write_xyz(path, mols, comment):
    atoms = [a for mol in mols for a in mol.atoms]
    with open(path, 'w') as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for a in atoms:
            x, y, z = a.coords
            f.write(f"{a.symbol:>4s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")

ranking = []   # (n_motifs, natoms, step, ok dict, distances dict)

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    natoms = sum(1 for _ in open(xf)) - 2
    if natoms < 1:
        continue
    base = os.path.splitext(os.path.basename(xf))[0]
    if "MDStep" not in base:
        continue
    step = int(base.split("MDStep")[1].split("_")[0])

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    n_mol = sum(len(mol_data[name]) for name in SPECIES)
    m  = motif_dists(motif_distances(mol_data))
    ok = motif_ok(m)
    ranking.append((sum(ok.values()), n_mol, natoms, step, ok, m, xf))

# MOTIF="A,B": rank by those interactions, any present (shortest first);
# MOTIF="A+B": only clusters showing ALL of them, tightest sum first
targets = os.environ.get("MOTIF")
if targets:
    need_all = "+" in targets
    targets  = targets.replace("+", ",").split(",")
    if need_all:
        ranking = [r for r in ranking if all(r[4].get(t) for t in targets)]
        ranking.sort(key=lambda r: (sum(r[5][t] for t in targets), r[1], r[2]))
    else:
        ranking = [r for r in ranking if any(r[4].get(t) for t in targets)]
        ranking.sort(key=lambda r: (min(r[5][t] for t in targets
                                        if r[5][t] is not None), r[1], r[2]))
else:
    ranking.sort(key=lambda r: (-r[0], r[1], r[2]))

print(f"\n{'='*78}")
print("  Representative cluster — all motifs at once, smallest first")
print(f"{'='*78}")
print(f"  cutoffs: O/N-H {D_HB} A, C-H {D_CH} A, Cl...H {D_CL} A, H5...H1 {D_HH} A")
print(f"  {'step':>10} {'mols':>4} {'atoms':>5} {'motifs':>6}  " +
      "  ".join(f"{k:>9}" for k in MOTIFS))
for n_ok, n_mol, natoms, step, ok, m, _xf in ranking[:10]:
    row = "  ".join(f"{m[k]:>8.2f}{'*' if ok[k] else ' '}" if m[k] is not None
                    else f"{'-':>9}" for k in MOTIFS)
    print(f"  {step:>10} {n_mol:>4} {natoms:>5} {n_ok:>6}  {row}")
print("  (* = within cutoff)")

best = ranking[0]
n_ok, n_mol, natoms, step, ok, m, xf = best
cluster = Molecule(xf)
centre  = np.mean(cluster.as_array(), axis=0)
cluster.guess_bonds()
mol_data = classify_sort(cluster.separate(), centre,
                         {k: FORMULAS[k] for k in SPECIES})
everything = [mol for name in SPECIES for mol in mol_data[name]]
no_ch      = [mol for name in ('urea', 'chloride') for mol in mol_data[name]]
no_urea    = [mol for name in ('choline', 'chloride') for mol in mol_data[name]]
write_xyz(f"repr_MDStep{step}.xyz", everything, f"MDStep{step} {n_ok}/{len(MOTIFS)} motifs")
write_xyz(f"repr_MDStep{step}_noCh.xyz", no_ch, f"MDStep{step} without cholines")
write_xyz(f"repr_MDStep{step}_noUrea.xyz", no_urea, f"MDStep{step} without ureas")
print(f"\n  wrote repr_MDStep{step}.xyz (+_noCh, +_noUrea)")

# closest pair of each motif, numbered as in repr_MDStep<n>.xyz (1-based)
mp  = motif_distances(mol_data)
idx = {id(a): i + 1
       for i, a in enumerate(a for mol in everything for a in mol.atoms)}
print(f"  motif pairs in repr_MDStep{step}.xyz (atom numbers, * = within cutoff):")
for k in MOTIFS:
    if mp[k] is None:
        continue
    d, a, b = mp[k]
    print(f"    {k:>9}{'*' if ok[k] else ' '} "
          f"{a.symbol}{idx[id(a)]} - {b.symbol}{idx[id(b)]}  {d:.2f} A")

finish()

save_cache("representative_cluster", {
    "ranking": [(n, nm, na, st, ok, m) for n, nm, na, st, ok, m, _ in ranking[:50]],
    "best":    step,
})
