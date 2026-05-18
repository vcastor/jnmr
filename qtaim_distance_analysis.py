import os
import glob
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR     = "plots"
CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "amsoutput/qtaim"
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"
PLOT_STYLES  = [("black", False, ""), ("white", True, "_transparent")]

def find_nh2(urea):
    """Return [(N, H), ...] for every N-H bond in urea."""
    pairs = []
    for n in urea.atoms:
        if n.symbol != 'N':
            continue
        for b in n.bonds:
            h = b.other_end(n)
            if h.symbol == 'H':
                pairs.append((n, h))
    return pairs

def find_carbonyl(urea):
    """Return (C, O) of the urea C=O group."""
    for at in urea.atoms:
        if at.symbol != 'C':
            continue
        nbrs = [b.other_end(at) for b in at.bonds]
        syms = [a.symbol for a in nbrs]
        if syms.count('N') == 2 and syms.count('O') == 1:
            o = next(a for a in nbrs if a.symbol == 'O')
            return at, o

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

def dihedral(a, b, c, d):
    """Signed dihedral a-b-c-d in radians (range [-pi, pi])."""
    p0, p1, p2, p3 = (np.array(x.coords) for x in (a, b, c, d))
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2/np.linalg.norm(b2))
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

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

# ── accumulators 
inter_NH_CH3  = ([], [], [])
inter_Cu_HCH3 = ([], [], [])
inter_HH_NCH3 = ([], [], [])
inter_Cu_HCH2 = []
inter_Nu_HCH2 = []
inter_HN_HCH2 = []
bcp_O_HCH2    = []   # O(urea)-H(CH2) at confirmed BCPs
bcp_N_HCH3    = []   # N(urea)-H(CH3) at confirmed BCPs

n_systems_total   = 0
n_systems_used    = 0
n_pairs_total     = 0
n_pairs_O_CH2_BCP = 0
n_pairs_N_CH3_BCP = 0

# ── main ──────────────────────────────────────────────────────────────────
init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    n_systems_total += 1
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

    bcps = read_bcps(qpath)

    for u in ureas:
        c_urea, o_urea = find_carbonyl(u)
        n_urea_atoms   = [at for at in u.atoms if at.symbol == 'N']
        nh2_pairs      = find_nh2(u)
        for ch in cholines:
            n_pairs_total += 1
            ch3 = find_ch3_groups(ch)
            _, hN, _, hO = find_ch2_pair(ch)
            ch2_hs = hN + hO

            o_ch2_hits = [h for h in ch2_hs
                          if frozenset({o_urea.cluster_id, h.cluster_id}) in bcps]
            n_ch3_hits = [(n_u, h) for _, hs in ch3 for h in hs for n_u in n_urea_atoms
                          if frozenset({n_u.cluster_id, h.cluster_id}) in bcps]

            if o_ch2_hits:
                n_pairs_O_CH2_BCP += 1
                for h in o_ch2_hits:
                    bcp_O_HCH2.append(D(o_urea, h))
                for h in ch2_hs:
                    inter_Cu_HCH2.append(D(c_urea, h))
                    for n_u in n_urea_atoms:
                        inter_Nu_HCH2.append(D(n_u, h))
                    for _, h_nh2 in nh2_pairs:
                        inter_HN_HCH2.append(D(h_nh2, h))

            if n_ch3_hits:
                n_pairs_N_CH3_BCP += 1
                for n_u, h in n_ch3_hits:
                    bcp_N_HCH3.append(D(n_u, h))
                for n_nh2, h_nh2 in nh2_pairs:
                    ranked = sorted(range(len(ch3)),
                                    key=lambda gi: D(ch3[gi][0], n_nh2))
                    for rank, gi in enumerate(ranked):
                        c_ch3, hs = ch3[gi]
                        for h_ch3 in hs:
                            inter_NH_CH3[rank].append(D(n_nh2, h_ch3))
                            inter_Cu_HCH3[rank].append(D(c_urea, h_ch3))
                            inter_HH_NCH3[rank].append(D(h_nh2, h_ch3))

finish()

# ── stats ────────────────────────────────────────────────────────────────
def stats(data):
    if not data:
        return "n=0"
    d = np.array(data)
    return f"n={len(d)}  mean={d.mean():.3f} A  std={d.std():.3f} A"

# ── plotting ─────────────────────────────────────────────────────────────
def hist(ax, data, label, color, bins=40):
    ax.hist(data, bins=bins, alpha=0.5, label=label,
            color=color, edgecolor=LETTER_COLOUR)

def style_axes(ax):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(LETTER_COLOUR)
    ax.tick_params(colors=LETTER_COLOUR, which="both")
    ax.xaxis.label.set_color(LETTER_COLOUR)
    ax.yaxis.label.set_color(LETTER_COLOUR)
    ax.title.set_color(LETTER_COLOUR)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor(LETTER_COLOUR)
        for t in leg.get_texts():
            t.set_color(LETTER_COLOUR)

def style_cbar(cbar):
    cbar.ax.tick_params(colors=LETTER_COLOUR)
    cbar.ax.yaxis.label.set_color(LETTER_COLOUR)
    cbar.outline.set_edgecolor(LETTER_COLOUR)

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    # BCP-direct distances
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    hist(axes[0], bcp_O_HCH2, "O(urea)-H(CH2)", "steelblue")
    axes[0].set_title("QTAIM-confirmed O(urea) - H(CH2)")
    axes[0].set_xlabel("distance (A)")
    axes[0].legend()
    hist(axes[1], bcp_N_HCH3, "N(urea)-H(CH3)", "seagreen")
    axes[1].set_title("QTAIM-confirmed N(urea) - H(CH3)")
    axes[1].set_xlabel("distance (A)")
    axes[1].legend()
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_qtaim_bcp{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # inter NH2-CH3 (QTAIM-filtered)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = ["steelblue", "darkorange", "seagreen"]
    labels = ["closest CH3", "mid CH3", "far CH3"]
    for k, (c, l) in enumerate(zip(colors, labels)):
        hist(axes[0], inter_NH_CH3[k],  l, c)
        hist(axes[1], inter_Cu_HCH3[k], l, c)
        hist(axes[2], inter_HH_NCH3[k], l, c)
    axes[0].set_title("Inter · N(NH2) - H(CH3) [QTAIM-filtered]")
    axes[0].set_xlabel("distance (A)")
    axes[0].legend()
    axes[1].set_title("Inter · C(urea) - H(CH3) [QTAIM-filtered]")
    axes[1].set_xlabel("distance (A)")
    axes[1].legend()
    axes[2].set_title("Inter · H(NH2) - H(CH3) [QTAIM-filtered]")
    axes[2].set_xlabel("distance (A)")
    axes[2].legend()
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_qtaim_inter_NH2_CH3{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # inter urea-CH2 (QTAIM-filtered)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Cu_HCH2, "C(urea)-H(CH2)", "steelblue")
    axes[0].set_title("Inter · C(urea) - H(CH2) [QTAIM-filtered]")
    axes[0].set_xlabel("distance (A)")
    axes[0].legend()
    hist(axes[1], inter_Nu_HCH2, "N(NH2)-H(CH2)", "seagreen")
    axes[1].set_title("Inter · N(NH2) - H(CH2) [QTAIM-filtered]")
    axes[1].set_xlabel("distance (A)")
    axes[1].legend()
    hist(axes[2], inter_HN_HCH2, "H(NH2)-H(CH2)", "darkorange")
    axes[2].set_title("Inter · H(NH2) - H(CH2) [QTAIM-filtered]")
    axes[2].set_xlabel("distance (A)")
    axes[2].legend()
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_qtaim_inter_urea_CH2{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

print(f"systems: {n_systems_used}/{n_systems_total} with QTAIM data")
print(f"urea-choline pairs: {n_pairs_total} total · "
      f"{n_pairs_O_CH2_BCP} with O-CH2 BCP · "
      f"{n_pairs_N_CH3_BCP} with N-CH3 BCP")
print()
print("BCP-confirmed O(urea) - H(CH2):")
print(f"  {stats(bcp_O_HCH2)}")
print("BCP-confirmed N(urea) - H(CH3):")
print(f"  {stats(bcp_N_HCH3)}")
print()

print("Inter · N(NH2) - H(CH3) [QTAIM-filtered]:")
for k, l in enumerate(labels):
    print(f"  {l}: {stats(inter_NH_CH3[k])}")
print("Inter · C(urea) - H(CH3) [QTAIM-filtered]:")
for k, l in enumerate(labels):
    print(f"  {l}: {stats(inter_Cu_HCH3[k])}")
print("Inter · H(NH2) - H(CH3) [QTAIM-filtered]:")
for k, l in enumerate(labels):
    print(f"  {l}: {stats(inter_HH_NCH3[k])}")
print("Inter · C(urea) - H(CH2) [QTAIM-filtered]:")
print(f"  {stats(inter_Cu_HCH2)}")
print("Inter · N(NH2) - H(CH2) [QTAIM-filtered]:")
print(f"  {stats(inter_Nu_HCH2)}")
print("Inter · H(NH2) - H(CH2) [QTAIM-filtered]:")
print(f"  {stats(inter_HN_HCH2)}")

