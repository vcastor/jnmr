#!/usr/bin/python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

CLUSTERS_DIR = "clusters"
THR_NH2_CH3  = 5.0    # H(NH2)-H(CH3) cutoff, same as run_generator
THR_UREA_CH2 = 4.5    # C(urea)-H(CH2) cutoff

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

def find_carbonyl_C(urea):
    for at in urea.atoms:
        if at.symbol != 'C':
            continue
        nbrs = [b.other_end(at).symbol for b in at.bonds]
        if nbrs.count('N') == 2 and nbrs.count('O') == 1:
            return at

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

# ── accumulators ──────────────────────────────────────────────────────────
intra_NH      = []
intra_HH      = []
intra_HH_dih  = []                  # |H-C-C-H| dihedral (rad), paired 1:1 with intra_HH
intra_CH_N    = []
intra_CH_O    = []
inter_NH_CH3  = ([], [], [])        # ranked by C(CH3)-N(NH2): closest, mid, far
inter_Cu_HCH3 = ([], [], [])
inter_HH_NCH3 = ([], [], [])
inter_Cu_HCH2 = []
inter_Nu_HCH2 = []
inter_HN_HCH2 = []

# ── main ──────────────────────────────────────────────────────────────────
init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue

    cluster = Molecule(xf)
    cluster.guess_bonds()
    mols = cluster.separate()

    ureas    = [m for m in mols if m.get_formula() == 'CH4N2O']
    cholines = [m for m in mols if m.get_formula() == 'C5H14NO']

    for u in ureas:
        for n, h in find_nh2(u):
            intra_NH.append(D(n, h))

    for ch in cholines:
        c_N, hN, c_O, hO = find_ch2_pair(ch)
        for a in hN:
            for b in hO:
                intra_HH.append(D(a, b))
                intra_HH_dih.append(abs(dihedral(a, c_N, c_O, b)))
        for h in hN:
            intra_CH_N.append(D(c_N, h))
        for h in hO:
            intra_CH_O.append(D(c_O, h))

    for u in ureas:
        nh2_pairs = find_nh2(u)
        c_urea    = find_carbonyl_C(u)
        for ch in cholines:
            ch3 = find_ch3_groups(ch)
            for n_nh2, h_nh2 in nh2_pairs:
                ranked = sorted(range(len(ch3)),
                                key=lambda gi: D(ch3[gi][0], n_nh2))
                for rank, gi in enumerate(ranked):
                    c_ch3, hs = ch3[gi]
                    for h_ch3 in hs:
                        d_hh = D(h_nh2, h_ch3)
                        if d_hh <= THR_NH2_CH3:
                            inter_NH_CH3[rank].append(D(n_nh2, h_ch3))
                            inter_Cu_HCH3[rank].append(D(c_urea, h_ch3))
                            inter_HH_NCH3[rank].append(d_hh)

    for u in ureas:
        c_urea    = find_carbonyl_C(u)
        nh2_pairs = find_nh2(u)
        n_urea    = [at for at in u.atoms if at.symbol == 'N']
        for ch in cholines:
            _, hN, _, hO = find_ch2_pair(ch)
            for h in hN + hO:
                d = D(c_urea, h)
                if d <= THR_UREA_CH2:
                    inter_Cu_HCH2.append(d)
                    for n in n_urea:
                        inter_Nu_HCH2.append(D(n, h))
                    for _, h_nh2 in nh2_pairs:
                        inter_HN_HCH2.append(D(h_nh2, h))

finish()

# ── stats    ──────────────────────────────────────────────────────────────
def stats(data):
    if not data:
        return "n=0"
    d = np.array(data)
    return f"n={len(d)}  mean={d.mean():.2f} A  std={d.std():.2f} A"

# ── plotting ──────────────────────────────────────────────────────────────
def hist(ax, data, label, color, bins=40):
    ax.hist(data, bins=bins, alpha=0.5, label=label,
            color=color, edgecolor="white")

def style_axes(ax):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor("white")
        for t in leg.get_texts():
            t.set_color("white")

def style_cbar(cbar):
    cbar.ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")
    cbar.outline.set_edgecolor("white")

# intra
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
hist(axes[0], intra_NH, "N-H", "steelblue")
axes[0].set_title("Intra · NH2 of urea")
axes[0].set_xlabel("distance (A)")
axes[0].legend()

hb = axes[1].hexbin(intra_HH_dih, intra_HH, gridsize=40, cmap="Oranges", mincnt=1)
cbar = fig.colorbar(hb, ax=axes[1], label="count")
style_cbar(cbar)
axes[1].set_title("Intra · CH2-CH2")
axes[1].set_xlabel("|H-C-C-H| dihedral (rad)")
axes[1].set_ylabel("H-H distance (A)")
axes[1].set_xlim(0, np.pi)

hist(axes[2], intra_CH_N, "CH2 bonded to N", "seagreen")
hist(axes[2], intra_CH_O, "CH2 bonded to O", "purple")
axes[2].set_title("Intra · C-H of each CH2")
axes[2].set_xlabel("distance (A)")
axes[2].legend()

for ax in axes:
    style_axes(ax)
fig.tight_layout()
fig.savefig("distance_intra.pdf", transparent=True)
plt.close(fig)

# inter NH2-CH3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
colors = ["steelblue", "darkorange", "seagreen"]
labels = ["closest CH3", "mid CH3", "far CH3"]
for k, (c, l) in enumerate(zip(colors, labels)):
    hist(axes[0], inter_NH_CH3[k],  l, c)
    hist(axes[1], inter_Cu_HCH3[k], l, c)
    hist(axes[2], inter_HH_NCH3[k], l, c)
axes[0].set_title("Inter · N(NH2) - H(CH3)")
axes[0].set_xlabel("distance (A)")
axes[0].legend()
axes[1].set_title("Inter · C(urea) - H(CH3)")
axes[1].set_xlabel("distance (A)")
axes[1].legend()
axes[2].set_title(f"Inter · H(NH2) - H(CH3)  [cutoff {THR_NH2_CH3} A]")
axes[2].set_xlabel("distance (A)")
axes[2].legend()
for ax in axes:
    style_axes(ax)
fig.tight_layout()
fig.savefig("distance_inter_NH2_CH3.pdf", transparent=True)
plt.close(fig)

# inter urea-CH2
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
hist(axes[0], inter_Cu_HCH2, "C(urea)-H(CH2)", "steelblue")
axes[0].set_title(f"Inter · C(urea) - H(CH2)  [cutoff {THR_UREA_CH2} A]")
axes[0].set_xlabel("distance (A)")
axes[0].legend()
hist(axes[1], inter_Nu_HCH2, "N(NH2)-H(CH2)", "seagreen")
axes[1].set_title("Inter · N(NH2) - H(CH2)  [same H from CH2]")
axes[1].set_xlabel("distance (A)")
axes[1].legend()
hist(axes[2], inter_HN_HCH2, "H(NH2)-H(CH2)", "darkorange")
axes[2].set_title("Inter · H(NH2) - H(CH2)  [same H from CH2]")
axes[2].set_xlabel("distance (A)")
axes[2].legend()
for ax in axes:
    style_axes(ax)
fig.tight_layout()
fig.savefig("distance_inter_urea_CH2.pdf", transparent=True)
plt.close(fig)

print(f"intra:  N-H={len(intra_NH)}  H-H(CH2-CH2)={len(intra_HH)}  "
      f"C-H={len(intra_CH_N)}+{len(intra_CH_O)}"
      f"  H-H dihedral (CH2-CH2) n={len(intra_HH_dih)}")
print(f"inter NH2-CH3 (cutoff {THR_NH2_CH3} A): "
      f"N-H per rank={[len(x) for x in inter_NH_CH3]}  "
      f"C-H per rank={[len(x) for x in inter_Cu_HCH3]}  "
      f"H-H per rank={[len(x) for x in inter_HH_NCH3]}")
print(f"inter urea-CH2 (cutoff {THR_UREA_CH2} A): "
      f"C-H={len(inter_Cu_HCH2)}  N-H={len(inter_Nu_HCH2)}  "
      f"H-H={len(inter_HN_HCH2)}")

print("\nIntra · NH2 of urea:")
print(stats(intra_NH))
print("\nIntra · CH2-CH2 of choline:")
print(stats(intra_HH))
print(stats(intra_CH_N))
print(stats(intra_CH_O))
print("\nInter · N(NH2) - H(CH3):")
for k, l in enumerate(labels):
    print(f"  {l}: {stats(inter_NH_CH3[k])}")
print("\nInter · C(urea) - H(CH3):")
for k, l in enumerate(labels):
    print(f"  {l}: {stats(inter_Cu_HCH3[k])}")
print("\nInter · H(NH2) - H(CH3):")
for k, l in enumerate(labels):
    print(f"  {l}: {stats(inter_HH_NCH3[k])}")
print("\nInter · C(urea) - H(CH2):")
print(stats(inter_Cu_HCH2))
print("\nInter · N(NH2) - H(CH2):")
print(stats(inter_Nu_HCH2))
print("\nInter · H(NH2) - H(CH2):")
print(stats(inter_HN_HCH2))
print("wrote distance_histogram.pdf")
