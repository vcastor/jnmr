#!$AMSBIN/plams
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "amsoutput/qtaim"
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"

def find_ch3_groups(choline):
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
    """Return {frozenset({atomA, atomB}): rho} from QTAIM output."""
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

    # second pass: only read rho for the CPs we care about
    bcps = {}
    for i in range(bp_start):
        s = lines[i].strip()
        if not s.startswith("CP #"):
            continue
        cp_num = int(s.split()[2])
        if cp_num not in pair_by_cp:
            continue
        bcps[pair_by_cp[cp_num]] = float(lines[i + 24].split()[2])
    return bcps

def bcp_rho(bcps, a, b):
    return bcps.get(frozenset({a.cluster_id, b.cluster_id}))

# accumulators: each entry is (distance, rho) ------------------------------
nh2_ch3       = []
o_hch2_all    = []
o_hch2_bridge = []
o_hch2_single = []

n_systems_used     = 0
n_pairs_total      = 0
n_pairs_nh2_ch3    = 0
n_pairs_o_any      = 0
n_pairs_o_bridge   = 0
n_pairs_o_single   = 0
n_pairs_o_same_ch2 = 0

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

            nh2_hit = False
            for n in n_urea:
                for _, hs in find_ch3_groups(ch):
                    for h in hs:
                        rho = bcp_rho(bcps, n, h)
                        if rho is not None:
                            nh2_ch3.append((D(n, h), rho))
                            nh2_hit = True
            if nh2_hit:
                n_pairs_nh2_ch3 += 1

            _, hN, _, hO = find_ch2_pair(ch)
            hits_N = [(h, bcp_rho(bcps, o_urea, h)) for h in hN]
            hits_N = [(h, r) for h, r in hits_N if r is not None]
            hits_O = [(h, bcp_rho(bcps, o_urea, h)) for h in hO]
            hits_O = [(h, r) for h, r in hits_O if r is not None]
            hits   = hits_N + hits_O
            if not hits:
                continue
            n_pairs_o_any += 1
            for h, r in hits:
                o_hch2_all.append((D(o_urea, h), r))
            if hits_N and hits_O:
                n_pairs_o_bridge += 1
                for h, r in hits:
                    o_hch2_bridge.append((D(o_urea, h), r))
            elif len(hits) == 1:
                n_pairs_o_single += 1
                h, r = hits[0]
                o_hch2_single.append((D(o_urea, h), r))
            else:
                n_pairs_o_same_ch2 += 1

finish()

# ── report ────────────────────────────────────────────────────────────────
def summary(label, data):
    if not data:
        print(f"  {label}: n=0")
        return
    arr = np.array(data)
    ds, rs = arr[:, 0], arr[:, 1]
    print(f"  {label}: n={len(arr)}")
    print(f"    d   (A) : mean={ds.mean():.3f}  std={ds.std():.3f}  "
          f"min={ds.min():.3f}  max={ds.max():.3f}")
    print(f"    rho (au): mean={rs.mean():.4f}  std={rs.std():.4f}  "
          f"min={rs.min():.4f}  max={rs.max():.4f}")

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
summary(f"O bridges BOTH CH2  ({n_pairs_o_bridge:3d} pairs)", o_hch2_bridge)
summary(f"O sees only ONE H   ({n_pairs_o_single:3d} pairs)", o_hch2_single)
print(f"  >=2 BCP, same CH2   ({n_pairs_o_same_ch2:3d} pairs)")

# ── plots: scatter (top row) is what you want for picking a rho threshold ─
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

def scatter(ax, data, color, title):
    if not data:
        ax.set_title(title + "  (empty)")
        return
    arr = np.array(data)
    ax.scatter(arr[:, 0], arr[:, 1], s=22, c=color, alpha=0.6,
               edgecolor='black', linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("distance (A)")
    ax.set_ylabel(r"$\rho_{BCP}$ (au)")
    ax.grid(alpha=0.3)

scatter(axes[0, 0], nh2_ch3,       "seagreen",   "N(urea NH2) - H(CH3)")
scatter(axes[0, 1], o_hch2_all,    "steelblue",  "O(urea) - H(CH2)  all")
scatter(axes[0, 2], o_hch2_bridge, "darkorange", "O(urea) - H(CH2)  bridge")

def rho_hist(ax, data, color, title):
    if not data:
        ax.set_title(title + "  (empty)")
        return
    arr = np.array(data)
    ax.hist(arr[:, 1], bins=30, color=color, edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(r"$\rho_{BCP}$ (au)")
    ax.set_ylabel("count")

rho_hist(axes[1, 0], nh2_ch3,    "seagreen",  r"$\rho$  N(NH2) - H(CH3)")
rho_hist(axes[1, 1], o_hch2_all, "steelblue", r"$\rho$  O - H(CH2)  all")

axes[1, 2].hist(np.array(o_hch2_bridge)[:, 1] if o_hch2_bridge else [],
                bins=20, color="darkorange", edgecolor="black", alpha=0.6,
                label=f"bridge (n={len(o_hch2_bridge)})")
axes[1, 2].hist(np.array(o_hch2_single)[:, 1] if o_hch2_single else [],
                bins=20, color="crimson", edgecolor="black", alpha=0.6,
                label=f"single H (n={len(o_hch2_single)})")
axes[1, 2].set_title(r"$\rho$  bridge vs single")
axes[1, 2].set_xlabel(r"$\rho_{BCP}$ (au)")
axes[1, 2].legend()

fig.tight_layout()
plt.show()

