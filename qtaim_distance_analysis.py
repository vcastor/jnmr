import os
import glob
import numpy as np
import matplotlib.pyplot as plt

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "amsoutput/qtaim"
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"

def find_ch3_groups(choline):
    """Return [(C, [H,H,H]), ...] - the three methyls bonded to N+."""
    out = []
    for n in choline.atoms:
        if n.symbol != 'N':
            continue
        for b in n.bonds:
            c = b.other_end(n)
            if c.symbol != 'C':
                continue
            hs = [b2.other_end(c) for b2 in c.bonds if b2.other_end(c).symbol == 'H']
            if len(hs) == 3:
                out.append((c, hs))
    return out

def find_ch2_pair(choline):
    """Return (C_N, [H,H], C_O, [H,H]) - the two adjacent CH2 groups."""
    ch2 = []
    for at in choline.atoms:
        if at.symbol != 'C':
            continue
        hs = [b.other_end(at) for b in at.bonds if b.other_end(at).symbol == 'H']
        if len(hs) == 2:
            ch2.append((at, hs))
    for i, (c1, h1) in enumerate(ch2):
        nbrs1 = [b.other_end(c1) for b in c1.bonds]
        for j, (c2, h2) in enumerate(ch2):
            if j <= i or c2 not in nbrs1:
                continue
            if any(a.symbol == 'N' for a in nbrs1):
                return c1, h1, c2, h2
            return c2, h2, c1, h1

def D(a, b):
    return float(np.linalg.norm(np.array(a.coords) - np.array(b.coords)))

def read_bcps(path):
    """Set of frozenset({atomA, atomB}) (1-based cluster indices) from QTAIM BOND PATHS."""
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if BP_HEADER in l) + 4
    bcps = set()
    for line in lines[start:]:
        if line.strip().startswith("---"):
            break
        parts = line.split()
        bcps.add(frozenset({int(parts[2]), int(parts[4])}))
    return bcps

def has_bcp(bcps, a, b):
    return frozenset({a.cluster_id, b.cluster_id}) in bcps

def stats(data):
    if not data:
        return "n=0"
    d = np.array(data)
    return f"n={len(d)}  mean={d.mean():.3f} A  std={d.std():.3f} A"

# ── accumulators ──────────────────────────────────────────────────────────
nh2_ch3       = []   # N(urea NH2)-H(CH3) distance, only where that N-H has a BCP
o_hch2_all    = []   # every O(urea)-H(CH2) distance at a BCP
o_hch2_bridge = []   # O-H distances where one choline bridges O with BOTH its CH2
o_hch2_single = []   # O-H distance where O has exactly one CH2 BCP

n_systems_used     = 0
n_pairs_total      = 0
n_pairs_nh2_ch3    = 0   # pairs with >=1 N(NH2)-H(CH3) BCP
n_pairs_o_any      = 0   # pairs with >=1 O-H(CH2) BCP
n_pairs_o_bridge   = 0   # pairs where O bridges BOTH CH2 of the choline
n_pairs_o_single   = 0   # pairs where O sees exactly one CH2 H
n_pairs_o_same_ch2 = 0   # pairs with >=2 BCP but all on the same CH2

# ── main ──────────────────────────────────────────────────────────────────
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
    cluster.guess_bonds()
    for i, at in enumerate(cluster.atoms):
        at.cluster_id = i + 1
    mols = cluster.separate()

    ureas    = [m for m in mols if m.get_formula() == 'CH4N2O']
    cholines = [m for m in mols if m.get_formula() == 'C5H14NO']
    bcps     = read_bcps(qpath)

    for u in ureas:
        n_urea = [at for at in u.atoms if at.symbol == 'N']
        o_urea = next(at for at in u.atoms if at.symbol == 'O')
        for ch in cholines:
            n_pairs_total += 1

            # N(urea NH2) - H(CH3): keep ONLY the H that has a BCP with the N,
            # never the rest of that CH3
            nh2_hit = False
            for n in n_urea:
                for _, hs in find_ch3_groups(ch):
                    for h in hs:
                        if has_bcp(bcps, n, h):
                            nh2_ch3.append(D(n, h))
                            nh2_hit = True
            if nh2_hit:
                n_pairs_nh2_ch3 += 1

            # O(urea) - H(CH2): inspect the two CH2 of THIS choline separately
            _, hN, _, hO = find_ch2_pair(ch)
            hits_N = [h for h in hN if has_bcp(bcps, o_urea, h)]
            hits_O = [h for h in hO if has_bcp(bcps, o_urea, h)]
            hits   = hits_N + hits_O
            if not hits:
                continue
            n_pairs_o_any += 1
            for h in hits:
                o_hch2_all.append(D(o_urea, h))
            if hits_N and hits_O:                 # both CH2 bridge the same O
                n_pairs_o_bridge += 1
                for h in hits:
                    o_hch2_bridge.append(D(o_urea, h))
            elif len(hits) == 1:                  # O sees a single H
                n_pairs_o_single += 1
                o_hch2_single.append(D(o_urea, hits[0]))
            else:                                 # >=2 BCP but on one CH2 only
                n_pairs_o_same_ch2 += 1

finish()

# ── report ────────────────────────────────────────────────────────────────
print(f"systems with QTAIM data: {n_systems_used}")
print(f"urea-choline pairs:      {n_pairs_total}")
print()
print("N(urea NH2) - H(CH3)  [only the H that has a BCP with the N]")
print(f"  pairs with >=1 such BCP: {n_pairs_nh2_ch3}/{n_pairs_total}")
print(f"  {stats(nh2_ch3)}")
print()
print("O(urea) - H(CH2 choline)  [only the H that has a BCP with the O]")
print(f"  pairs with >=1 O-H(CH2) BCP: {n_pairs_o_any}/{n_pairs_total}")
print(f"  all BCP distances:                {stats(o_hch2_all)}")
print(f"  O bridges BOTH CH2  ({n_pairs_o_bridge:3d} pairs): {stats(o_hch2_bridge)}")
print(f"  O sees only ONE H   ({n_pairs_o_single:3d} pairs): {stats(o_hch2_single)}")
print(f"  >=2 BCP, same CH2   ({n_pairs_o_same_ch2:3d} pairs)")

# ── distributions (display only) ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].hist(nh2_ch3, bins=30, color="seagreen", edgecolor="black", alpha=0.7)
axes[0].set_title("N(urea NH2) - H(CH3)  [BCP only]")
axes[0].set_xlabel("distance (A)")
axes[0].set_ylabel("count")

axes[1].hist(o_hch2_all, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
axes[1].set_title("O(urea) - H(CH2)  [all BCP]")
axes[1].set_xlabel("distance (A)")

axes[2].hist(o_hch2_bridge, bins=20, color="darkorange", edgecolor="black", alpha=0.6,
             label=f"both CH2 bridge (n={len(o_hch2_bridge)})")
axes[2].hist(o_hch2_single, bins=20, color="crimson", edgecolor="black", alpha=0.6,
             label=f"single H (n={len(o_hch2_single)})")
axes[2].set_title("O(urea) - H(CH2)  bridge vs single")
axes[2].set_xlabel("distance (A)")
axes[2].legend()

fig.tight_layout()
plt.show()
