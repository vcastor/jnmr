#!$AMSBIN/plams
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from hassan_functions.geometry import distance
from hassan_functions.finders import find_xh_groups, find_adjacent_xh_pair_anchored
from hassan_functions.ordering import classify_sort, compute_offsets
from hassan_functions.constants import FORMULAS

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "amsoutput/qtaim"
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"
SPECIES      = ['urea', 'choline', 'chloride']

def read_bcps(path):
    """Return {frozenset({atomA, atomB}): {'rho', 'gb', 'vb', 'ratio'}} from QTAIM output."""
    with open(path) as f:
        lines = f.readlines()

    bp_start = next(i for i, l in enumerate(lines) if BP_HEADER in l)

    # first pass: BP summary -> {cp_num: frozenset({atomA, atomB})}
    pair_by_cp = {}
    for line in lines[bp_start + 4:]:
        if line.strip().startswith("---"):
            break
        parts = line.split()
        pair_by_cp[int(parts[1])] = frozenset({int(parts[2]), int(parts[4])})

    # second pass: rho, Gb, Vb for each BCP
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

def bcp_props(bcps, a, b):
    return bcps.get(frozenset({a.cluster_id, b.cluster_id}))

# accumulators: each entry is (distance, rho, gb, vb, ratio) ---------------
nh2_ch3       = []
o_hch2_all    = []
o_hch2_N      = []   # H from CH2 next to N+
o_hch2_O      = []   # H from CH2 next to O
o_hch2_bridge = []
o_hch2_single = []
o_hoh              = []   # H from OH of choline
o_ch2_oh_bridge    = []   # O sees H(CH2, any) and H(OH)
o_ch2N_oh_bridge   = []   # O sees H(CH2-N) and H(OH)
o_ch2O_oh_bridge   = []   # O sees H(CH2-O) and H(OH)

n_systems_used          = 0
n_pairs_total           = 0
n_pairs_nh2_ch3         = 0
n_pairs_o_any           = 0
n_pairs_o_hch2_N        = 0
n_pairs_o_hch2_O        = 0
n_pairs_o_bridge        = 0
n_pairs_o_single        = 0
n_pairs_o_same_ch2      = 0
n_pairs_o_hoh           = 0
n_pairs_o_ch2_oh_bridge = 0
n_pairs_o_ch2N_oh       = 0
n_pairs_o_ch2O_oh       = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    base  = os.path.splitext(os.path.basename(xf))[0]
    qpath = os.path.join(QTAIM_DIR, f"{base}.out")
    if not os.path.exists(qpath):
        continue
    n_systems_used += 1

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs = compute_offsets(mol_data, SPECIES)

    # Tag atoms with the global 1-based index they have in the reordered
    # cluster — matches the atom numbers used in the QTAIM output.
    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1

    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    bcps     = read_bcps(qpath)

    for u in ureas:
        n_urea = [at for at in u.atoms if at.symbol == 'N']
        o_urea = next(at for at in u.atoms if at.symbol == 'O')
        for ch in cholines:
            n_pairs_total += 1

            nh2_hit = False
            for n in n_urea:
                for _, hs in find_xh_groups(ch, 'C', 3, neighbour_symbol='N'):
                    for h in hs:
                        p = bcp_props(bcps, n, h)
                        if p is not None:
                            nh2_ch3.append((distance(n, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                            nh2_hit = True
            if nh2_hit:
                n_pairs_nh2_ch3 += 1

            _, hN, _, hO = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            hits_N = [(h, bcp_props(bcps, o_urea, h)) for h in hN]
            hits_N = [(h, p) for h, p in hits_N if p is not None]
            hits_O = [(h, bcp_props(bcps, o_urea, h)) for h in hO]
            hits_O = [(h, p) for h, p in hits_O if p is not None]
            hits   = hits_N + hits_O
            if hits:
                n_pairs_o_any += 1
                if hits_N:
                    n_pairs_o_hch2_N += 1
                if hits_O:
                    n_pairs_o_hch2_O += 1
                for h, p in hits:
                    o_hch2_all.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                for h, p in hits_N:
                    o_hch2_N.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                for h, p in hits_O:
                    o_hch2_O.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                if hits_N and hits_O:
                    n_pairs_o_bridge += 1
                    for h, p in hits:
                        o_hch2_bridge.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                elif len(hits) == 1:
                    n_pairs_o_single += 1
                    h, p = hits[0]
                    o_hch2_single.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                else:
                    n_pairs_o_same_ch2 += 1

            hoh_hits = [(h, bcp_props(bcps, o_urea, h))
                        for _, hs in find_xh_groups(ch, 'O', 1)
                        for h in hs]
            hoh_hits = [(h, p) for h, p in hoh_hits if p is not None]
            if hoh_hits:
                n_pairs_o_hoh += 1
                for h, p in hoh_hits:
                    o_hoh.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))

            if hits and hoh_hits:
                n_pairs_o_ch2_oh_bridge += 1
                for h, p in hits + hoh_hits:
                    o_ch2_oh_bridge.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                if hits_N:
                    n_pairs_o_ch2N_oh += 1
                    for h, p in hits_N + hoh_hits:
                        o_ch2N_oh_bridge.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                if hits_O:
                    n_pairs_o_ch2O_oh += 1
                    for h, p in hits_O + hoh_hits:
                        o_ch2O_oh_bridge.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))

finish()

# ── report ────────────────────────────────────────────────────────────────
def summary(label, data):
    if not data:
        print(f"  {label}: n=0")
        return
    arr = np.array(data)
    ds, rs, gs, vs, ks = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    print(f"  {label}: n={len(arr)}")
    print(f"    d     (A) : mean={ds.mean():.3f}  std={ds.std():.3f}  "
          f"min={ds.min():.3f}  max={ds.max():.3f}")
    print(f"    rho  (au) : mean={rs.mean():.4f}  std={rs.std():.4f}  "
          f"min={rs.min():.4f}  max={rs.max():.4f}")
    print(f"    Gb   (au) : mean={gs.mean():.4f}  std={gs.std():.4f}  "
          f"min={gs.min():.4f}  max={gs.max():.4f}")
    print(f"    Vb   (au) : mean={vs.mean():.4f}  std={vs.std():.4f}  "
          f"min={vs.min():.4f}  max={vs.max():.4f}")
    print(f"    |Vb|/Gb   : mean={ks.mean():.4f}  std={ks.std():.4f}  "
          f"min={ks.min():.4f}  max={ks.max():.4f}")

print(f"systems with QTAIM data: {n_systems_used}")
print(f"urea-choline pairs:      {n_pairs_total}")
print()
print("N(urea NH2) - H(CH3)  [BCP only]")
print(f"  pairs with >=1 such BCP: {n_pairs_nh2_ch3}/{n_pairs_total}")
summary("N-H(CH3)", nh2_ch3)
print()
print("O(urea) - H(CH2 choline)  [BCP only]")
print(f"  pairs with >=1 O-H(CH2) BCP: {n_pairs_o_any}/{n_pairs_total}")
summary("all BCP                          ", o_hch2_all)
summary("CH2 near N+                      ", o_hch2_N)
summary("CH2 near O                       ", o_hch2_O)
summary(f"O bridges BOTH CH2  ({n_pairs_o_bridge:3d} pairs)", o_hch2_bridge)
summary(f"O sees only ONE H   ({n_pairs_o_single:3d} pairs)", o_hch2_single)
print(f"  >=2 BCP, same CH2   ({n_pairs_o_same_ch2:3d} pairs)")
print()
print("O(urea) - H(OH choline)  [BCP only]")
print(f"  pairs with >=1 O-H(OH) BCP:  {n_pairs_o_hoh}/{n_pairs_total}")
summary("H(OH)                            ", o_hoh)
print()
if n_pairs_o_ch2_oh_bridge:
    print("O(urea) bridges H(CH2) + H(OH)  [BCP only]")
    print(f"  pairs with both:              {n_pairs_o_ch2_oh_bridge}/{n_pairs_total}")
    summary("CH2-N + OH                       ", o_ch2N_oh_bridge)
    summary("CH2-O + OH                       ", o_ch2O_oh_bridge)
    print()
if n_pairs_total:
    def pct(n): return f"{n}/{n_pairs_total}  ({100*n/n_pairs_total:.1f}%)"
    print("-- occurrence comparison (per urea-choline pair) --")
    print(f"  N(urea)-H(CH3):                {pct(n_pairs_nh2_ch3)}")
    print(f"  O(urea)-H(CH2-N):              {pct(n_pairs_o_hch2_N)}")
    print(f"  O(urea)-H(CH2-O):              {pct(n_pairs_o_hch2_O)}")
    print(f"  O(urea)-H(CH2) [any]:          {pct(n_pairs_o_any)}")
    print(f"  O(urea)-H(OH):                 {pct(n_pairs_o_hoh)}")
    print(f"  O(urea)-H(CH2-N) & H(OH):      {pct(n_pairs_o_ch2N_oh)}")
    print(f"  O(urea)-H(CH2-O) & H(OH):      {pct(n_pairs_o_ch2O_oh)}")
    print(f"  O(urea)-H(CH2,any) & H(OH):    {pct(n_pairs_o_ch2_oh_bridge)}")

# ── plots: 2-column grid, conditional bridge panels ───────────────────────
PROPS = [
    (1, r"$\rho_{BCP}$ (au)", "steelblue"),
    (2, r"$G_b$ (au)",        "seagreen"),
    (3, r"$V_b$ (au)",        "crimson"),
]
BONDS = [
    (nh2_ch3,    "N(urea NH2) – H(CH3)"),
    (o_hch2_N,   "O(urea) – H(CH2-N)"),
    (o_hch2_O,   "O(urea) – H(CH2-O)"),
    (o_hoh,      "O(urea) – H(OH)"),
]
if n_pairs_o_bridge > 0:
    BONDS.append((o_hch2_bridge, "O(urea) – H(CH2-N)+H(CH2-O)"))
if n_pairs_o_ch2N_oh > 0:
    BONDS.append((o_ch2N_oh_bridge, "O(urea) – H(CH2-N)+H(OH)"))
if n_pairs_o_ch2O_oh > 0:
    BONDS.append((o_ch2O_oh_bridge, "O(urea) – H(CH2-O)+H(OH)"))

ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
axes = axes.flatten()

for ax, (data, title) in zip(axes, BONDS):
    if data:
        arr = np.array(data)
        for col, ylabel, color in PROPS:
            ax.scatter(arr[:, 0], arr[:, col], s=22, c=color, alpha=0.6,
                       edgecolor='black', linewidth=0.3, label=ylabel)
    ax.set_title(f"{title}  (n={len(data)})")
    ax.set_xlabel("distance (Å)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

for ax in axes[len(BONDS):]:
    ax.set_visible(False)

fig.tight_layout()
plt.show()

