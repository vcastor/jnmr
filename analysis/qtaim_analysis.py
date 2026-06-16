#!$AMSBIN/plams
import os
import sys
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance
from hassan_functions.finders   import find_xh_groups, find_adjacent_xh_pair_anchored
from hassan_functions.ordering  import classify_sort, compute_offsets
from hassan_functions.io        import read_qtaim_charges
from hassan_functions.constants import FORMULAS
from hassan_functions.plotting  import PLOT_STYLES, style_axes

plt.rcParams['font.size']       = 16
plt.rcParams['axes.titlesize']  = 19
plt.rcParams['axes.labelsize']  = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "/Users/vcastor/Desktop/backup_qtaimcdft/qtaim" #tmp
# QTAIM_DIR    = "amsoutput/qtaim"
PLOT_DIR     = "plots"
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

def poincare_hopf_ok(path):
    with open(path) as f:
        return "Poincare-Hopf satisfied" in f.read()

def bcp_props(bcps, a, b):
    return bcps.get(frozenset({a.cluster_id, b.cluster_id}))

def carbon_tag(c):
    nbrs = [b.other_end(c) for b in c.bonds]
    if sum(a.symbol == 'H' for a in nbrs) == 3:
        return 'CH3'
    heavy = {a.symbol for a in nbrs if a.symbol != 'H'}
    return 'CH2N' if 'N' in heavy else 'CH2O'

def site_tag(at):
    if at.symbol != 'H':
        return at.symbol
    nb = at.bonds[0].other_end(at)
    if nb.symbol == 'C':
        return f"H({carbon_tag(nb)})"
    return f"H({nb.symbol})"

# QTAIM net charge of H from CH2 next to N+ / next to O
q_Nside = []
q_Oside = []

# BCP accumulators: each entry is (distance, rho, gb, vb, ratio) -----------
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

other_by_type      = defaultdict(list)   # label -> [(d, rho, gb, vb, ratio)]
n_pairs_other_type = defaultdict(int)    # label -> # interacting pairs showing it
sig_labels         = set()               # 'other' labels counted toward the 100% (one H + one N/O)

n_systems_used          = 0
n_pairs_total           = 0
n_pairs_interacting     = 0
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
n_pairs_other           = 0
n_other_bcp             = 0
n_ph_satisfied          = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    base  = os.path.splitext(os.path.basename(xf))[0]
    qpath = os.path.join(QTAIM_DIR, f"{base}.out")
    if not os.path.exists(qpath):
        continue
    n_systems_used += 1
    n_ph_satisfied += poincare_hopf_ok(qpath)

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs = compute_offsets(mol_data, SPECIES)

    # Tag atoms with the global 1-based index they have in the reordered
    # cluster (which is what coupling_generator.py wrote to the ADF input, and
    # therefore what the QTAIM output's atom numbers refer to).
    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1

    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    charges  = read_qtaim_charges(qpath)
    bcps     = read_bcps(qpath)

    # ── QTAIM net charge of the CH2 hydrogens (once per choline) ────────────
    for ch in cholines:
        _, hN, _, hO = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
        for h in hN:
            q = charges.get(h.cluster_id)
            if q is not None:
                q_Nside.append(q)
        for h in hO:
            q = charges.get(h.cluster_id)
            if q is not None:
                q_Oside.append(q)

    # ── BCP properties per urea-choline pair ────────────────────────────────
    for u in ureas:
        n_urea   = [at for at in u.atoms if at.symbol == 'N']
        o_urea   = next(at for at in u.atoms if at.symbol == 'O')
        urea_ids = {at.cluster_id for at in u.atoms}
        for ch in cholines:
            n_pairs_total += 1
            ch_ids    = {at.cluster_id for at in ch.atoms}
            pair_keys = [k for k in bcps if (k & urea_ids) and (k & ch_ids)]
            if pair_keys:
                n_pairs_interacting += 1
            tracked = set()

            nh2_hit = False
            for n in n_urea:
                for _, hs in find_xh_groups(ch, 'C', 3, neighbour='N'):
                    for h in hs:
                        p = bcp_props(bcps, n, h)
                        if p is not None:
                            nh2_ch3.append((distance(n, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                            tracked.add(frozenset({n.cluster_id, h.cluster_id}))
                            nh2_hit = True
            if nh2_hit:
                n_pairs_nh2_ch3 += 1

            _, hN, _, hO = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            hits_N = [(h, bcp_props(bcps, o_urea, h)) for h in hN]
            hits_N = [(h, p) for h, p in hits_N if p is not None]
            hits_O = [(h, bcp_props(bcps, o_urea, h)) for h in hO]
            hits_O = [(h, p) for h, p in hits_O if p is not None]
            hits   = hits_N + hits_O
            tracked |= {frozenset({o_urea.cluster_id, h.cluster_id}) for h, _ in hits}
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
            tracked |= {frozenset({o_urea.cluster_id, h.cluster_id}) for h, _ in hoh_hits}
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

            other_here = set()
            for k in pair_keys:
                if k in tracked:
                    continue
                ua    = next(a for a in u.atoms if a.cluster_id in k)
                ca    = next(a for a in ch.atoms if a.cluster_id in k)
                label = f"urea {site_tag(ua)} - choline {site_tag(ca)}"
                if (ua.symbol == 'H') ^ (ca.symbol == 'H'):
                    heavy = ca.symbol if ua.symbol == 'H' else ua.symbol
                    if heavy in ('N', 'O'):
                        sig_labels.add(label)
                p     = bcps[k]
                other_by_type[label].append((distance(ua, ca), p['rho'], p['gb'], p['vb'], p['ratio']))
                other_here.add(label)
                n_other_bcp += 1
            if other_here:
                n_pairs_other += 1
                for label in other_here:
                    n_pairs_other_type[label] += 1

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
print("Other urea-choline BCPs (not in the categories above)")
print(f"  pairs with >=1 'other' BCP:  {n_pairs_other}/{n_pairs_interacting}")
print(f"  total 'other' BCPs:          {n_other_bcp}")
for label in sorted(other_by_type, key=lambda L: -len(other_by_type[L])):
    summary(label, other_by_type[label])
print()
if n_pairs_interacting:
    def pct(n): return f"{n}/{n_pairs_interacting}  ({100*n/n_pairs_interacting:.1f}%)"
    print("-- occurrence comparison (per interacting urea-choline pair) --")
    print(f"  interacting pairs (>=1 BCP):   {n_pairs_interacting}/{n_pairs_total}  ({100*n_pairs_interacting/n_pairs_total:.1f}%)")
    print(f"  N(urea)-H(CH3):                {pct(n_pairs_nh2_ch3)}")
    print(f"  O(urea)-H(CH2-N):              {pct(n_pairs_o_hch2_N)}")
    print(f"  O(urea)-H(CH2-O):              {pct(n_pairs_o_hch2_O)}")
    print(f"  O(urea)-H(CH2) [any]:          {pct(n_pairs_o_any)}")
    print(f"  O(urea)-H(OH):                 {pct(n_pairs_o_hoh)}")
    print(f"  O(urea)-H(CH2-N) & H(OH):      {pct(n_pairs_o_ch2N_oh)}")
    print(f"  O(urea)-H(CH2-O) & H(OH):      {pct(n_pairs_o_ch2O_oh)}")
    print(f"  O(urea)-H(CH2,any) & H(OH):    {pct(n_pairs_o_ch2_oh_bridge)}")
    for label in sorted(n_pairs_other_type, key=lambda L: -n_pairs_other_type[L]):
        print(f"  {label+':':<31}{pct(n_pairs_other_type[label])}")

n_other_sig  = sum(len(other_by_type[L]) for L in sig_labels)
n_other_weak = n_other_bcp - n_other_sig
n_bcp        = len(nh2_ch3) + len(o_hch2_all) + len(o_hoh) + n_other_bcp
n_bcp_sig    = len(nh2_ch3) + len(o_hch2_all) + len(o_hoh) + n_other_sig
if n_bcp_sig:
    def pctb(n): return f"{n}/{n_bcp_sig}  ({100*n/n_bcp_sig:.1f}%)"
    print()
    print("-- BCP-level breakdown (per significant urea-choline BCP, sums to 100%) --")
    print(f"  significant urea-choline BCPs: {n_bcp_sig}  (of {n_bcp} total)")
    print(f"  N(urea)-H(CH3):                {pctb(len(nh2_ch3))}")
    print(f"  O(urea)-H(CH2-N):              {pctb(len(o_hch2_N))}")
    print(f"  O(urea)-H(CH2-O):              {pctb(len(o_hch2_O))}")
    print(f"  O(urea)-H(OH):                 {pctb(len(o_hoh))}")
    for label in sorted(sig_labels, key=lambda L: -len(other_by_type[L])):
        print(f"  {label+':':<31}{pctb(len(other_by_type[label]))}")
    if n_other_weak:
        print(f"  -- weak / numerically noisy (excluded; {n_other_weak}/{n_bcp} = {100*n_other_weak/n_bcp:.1f}% of all BCPs) --")
        for label in sorted((L for L in other_by_type if L not in sig_labels),
                            key=lambda L: -len(other_by_type[L])):
            n = len(other_by_type[label])
            print(f"  {label+':':<31}{n}/{n_bcp}  ({100*n/n_bcp:.1f}%)")

print()
q_N = np.array(q_Nside); q_O = np.array(q_Oside)
print(f"q (CH2 near N+): n={len(q_N)}, mean={q_N.mean():.6f}, std={q_N.std():.6f}")
print(f"q (CH2 near O ): n={len(q_O)}, mean={q_O.mean():.6f}, std={q_O.std():.6f}")
print()
print(f"Poincare-Hopf satisfied: {n_ph_satisfied}/{n_systems_used} systems")

# ── plots: distance vs BCP properties, 3 subfigures ───────────────────────
PROPS = [
    (1, r"$\rho_{BCP}$ (au)", "steelblue"),
    (2, r"$G_b$ (au)",        "seagreen"),
    (3, r"$V_b$ (au)",        "crimson"),
]
BONDS = [
    (nh2_ch3,    r"N(urea NH$_2$) – H(CH$_3$)"),
    (o_hch2_all, r"O(urea) – H(any CH$_2$)"),
    (o_hoh,      r"O(urea) – H(OH)"),
]

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (data, title) in zip(axes, BONDS):
        if data:
            arr = np.array(data)
            for col, ylabel, color in PROPS:
                ax.scatter(arr[:, 0], arr[:, col], s=22, c=color, alpha=0.6,
                           edgecolor='black', linewidth=0.3, label=ylabel)
        ax.set_title(title)
        ax.set_xlabel("distance (Å)")
        ax.legend(fontsize=14)
        ax.grid(alpha=0.3)
        style_axes(ax, LETTER_COLOUR)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/qtaim_distance{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

# ── plot: QTAIM net-charge histogram, CH2-N vs CH2-O ──────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.hist(q_Nside, bins=40, color="seagreen", edgecolor="white", alpha=0.5,
        density=True, label="CH2-N")
ax.hist(q_Oside, bins=40, color="mediumpurple", edgecolor="white", alpha=0.5,
        density=True, label="CH2-O")

sns.kdeplot(q_Nside, ax=ax, color="seagreen",     linewidth=2)
sns.kdeplot(q_Oside, ax=ax, color="mediumpurple", linewidth=2)

ax.set_xlim(-0.12, 0.12)
ax.set_title("QTAIM charge", color="white")
ax.set_xlabel("q (au)",      color="white")
ax.set_ylabel("density",     color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")
ax.legend(facecolor="none", edgecolor="white", labelcolor="white")
fig.tight_layout()
plt.savefig(f"{PLOT_DIR}/qtaim_charges.pdf", transparent=True)

