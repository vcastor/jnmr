#!$AMSBIN/plams
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

PLOT_DIR     = "plots"
CLUSTERS_DIR = "clusters"
THR_NH2_CH3  = 4.0    # H(NH2)-H(CH3) cutoff
THR_UREA_CH2 = 3.0    # O(urea)-H(CH2) cutoff

PLOT_STYLES = [('black', False, ''), ('white', True, '_transparent')]

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

def find_nh2_groups(urea):
    """Return [(N, [H, H]), ...] - one entry per NH2 group."""
    out = []
    for n in urea.atoms:
        if n.symbol != 'N':
            continue
        hs = [b.other_end(n) for b in n.bonds if b.other_end(n).symbol == 'H']
        if hs:
            out.append((n, hs))
    return out

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
intra_NH      = []                  # N-H bonded pairs only
intra_NH_all  = []                  # every (N, H) pair within the urea (incl. non-bonded)
intra_CH_urea = []                  # carbonyl C to every H within the urea
intra_HH      = []
intra_HH_dih  = []                  # |H-C-C-H| dihedral (rad), paired 1:1 with intra_HH
intra_CH_N    = []
intra_CH_O    = []
inter_NH_CH3  = []                  # flat list (all CH3 groups, no ranking)
inter_Cu_HCH3 = []
inter_HH_NCH3 = []
inter_Ou_HCH2 = []                  # cutoff now on O(urea)-H(CH2)
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
        ns_u   = [at for at in u.atoms if at.symbol == 'N']
        hs_u   = [at for at in u.atoms if at.symbol == 'H']
        c_u    = [at for at in u.atoms if at.symbol == 'C'][0]
        for n, h in find_nh2(u):
            intra_NH.append(D(n, h))
        for n in ns_u:
            for h in hs_u:
                intra_NH_all.append(D(n, h))
        for h in hs_u:
            intra_CH_urea.append(D(c_u, h))

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
        nh2_groups = find_nh2_groups(u)
        c_urea     = next(at for at in u.atoms if at.symbol == 'C')
        for ch in cholines:
            ch3 = find_ch3_groups(ch)
            for n_nh2, h_nh2_list in nh2_groups:
                for c_ch3, h_ch3_list in ch3:
                    near = sorted(h_ch3_list, key=lambda h: D(n_nh2, h))[:2]
                    near = [h for h in near if D(n_nh2, h) <= THR_NH2_CH3]
                    for h_ch3 in near:
                        inter_NH_CH3.append(D(n_nh2, h_ch3))
                        inter_Cu_HCH3.append(D(c_urea, h_ch3))
                        for h_nh2 in h_nh2_list:
                            inter_HH_NCH3.append(D(h_nh2, h_ch3))

    for u in ureas:
        o_urea     = next(at for at in u.atoms if at.symbol == 'O')
        nh2_groups = find_nh2_groups(u)
        for ch in cholines:
            _, hN, _, hO = find_ch2_pair(ch)
            for h in hN+hO:
                d = D(o_urea, h)
                if d <= THR_UREA_CH2:
                    inter_Ou_HCH2.append(d)
                    n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: D(g[0], h))
                    inter_Nu_HCH2.append(D(n_nh2, h))
                    for h_nh2 in h_nh2_list:
                        inter_HN_HCH2.append(D(h_nh2, h))

finish()

# ── stats    ──────────────────────────────────────────────────────────────
def stats(data, unit="A"):
    if not data:
        return "n=0"
    d = np.array(data)
    u = "Å" if unit == "A" else "rad"
    return f"n={len(d)}  mean={d.mean():.2f} {u}  std={d.std():.2f} {u}"

def mlabel(prefix, data, unit="Å"):
    if not data:
        return f"{prefix} (n=0)"
    return f"{prefix}  ⟨d⟩={np.mean(data):.2f}±{np.std(data):.2f} {unit}"

# ── plotting ──────────────────────────────────────────────────────────────
def hist(ax, data, label, colour, bins=40):
    if not data:
        return
    sns.histplot(data, bins=bins, kde=True, ax=ax, label=label, color=colour,
                 alpha=0.5, edgecolor=LETTER_COLOUR, line_kws={"linewidth": 2})

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
    # intra
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    # shared bin edges so the bonded peak has the SAME height in both series
    bins_NH = np.linspace(min(intra_NH_all), max(intra_NH_all), 41)
    hist(axes[0], intra_NH,     mlabel("N-H (bonded)", intra_NH),     "steelblue", bins=bins_NH)
    hist(axes[0], intra_NH_all, mlabel("N to all H",   intra_NH_all), "crimson",   bins=bins_NH)
    axes[0].set_title("Intra · N - H of urea")
    axes[0].set_xlabel("distance (Å)")
    axes[0].legend()

    hb = axes[1].hexbin(intra_HH_dih, intra_HH, gridsize=40, cmap="Oranges", mincnt=1)
    cbar = fig.colorbar(hb, ax=axes[1], label="count")
    style_cbar(cbar)
    axes[1].set_title("Intra · CH2-CH2")
    axes[1].set_xlabel("|H-C-C-H| dihedral (rad)")
    axes[1].set_ylabel("H-H distance (A)")
    axes[1].set_xlim(0, np.pi)

    hist(axes[2], intra_CH_N, mlabel("CH2 - N\n", intra_CH_N), "seagreen")
    hist(axes[2], intra_CH_O, mlabel("CH2 - O\n", intra_CH_O), "purple")
    axes[2].set_title("Intra · C-H of each CH2")
    axes[2].set_xlabel("distance (A)")
    axes[2].legend()

    hist(axes[3], intra_CH_urea, mlabel("", intra_CH_urea), "indianred")
    axes[3].set_title("Intra · C - H of urea")
    axes[3].set_xlabel("distance (A)")
    axes[3].legend()

    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_intra{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # inter NH2-CH3
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_NH_CH3,  mlabel("N(NH2)-H(CH3)", inter_NH_CH3),  "steelblue")
    hist(axes[1], inter_Cu_HCH3, mlabel("C(urea)-H(CH3)", inter_Cu_HCH3), "darkorange")
    hist(axes[2], inter_HH_NCH3, mlabel("H(NH2)-H(CH3)", inter_HH_NCH3),  "seagreen")
    axes[0].set_title(f"Inter · N(NH2) - H(CH3)  [H-H cutoff {THR_NH2_CH3} A]")
    axes[0].set_xlabel("distance (A)")
    axes[0].legend()
    axes[1].set_title(f"Inter · C(urea) - H(CH3)  [H-H cutoff {THR_NH2_CH3} A]")
    axes[1].set_xlabel("distance (A)")
    axes[1].legend()
    axes[2].set_title(f"Inter · H(NH2) - H(CH3)  [cutoff {THR_NH2_CH3} A]")
    axes[2].set_xlabel("distance (A)")
    axes[2].legend()
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_NH2_CH3{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # inter urea-CH2
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Ou_HCH2, mlabel("O(urea)-H(CH2)", inter_Ou_HCH2), "steelblue")
    axes[0].set_title(f"Inter · O(urea) - H(CH2)  [cutoff {THR_UREA_CH2} A]")
    axes[0].set_xlabel("distance (A)")
    axes[0].legend()
    hist(axes[1], inter_Nu_HCH2, mlabel("N(urea)-H(CH2)", inter_Nu_HCH2), "seagreen")
    axes[1].set_title("Inter · N(urea) - H(CH2)")
    axes[1].set_xlabel("distance (A)")
    axes[1].legend()
    hist(axes[2], inter_HN_HCH2, mlabel("H(NH2)-H(CH2)", inter_HN_HCH2), "darkorange")
    axes[2].set_title("Inter · H(NH2) - H(CH2)")
    axes[2].set_xlabel("distance (A)")
    axes[2].legend()
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_urea_CH2{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

print(f"intra:  N-H={len(intra_NH)}  N-Hall={len(intra_NH_all)}  "
      f"C-H(urea)={len(intra_CH_urea)}  H-H(CH2-CH2)={len(intra_HH)}  "
      f"C-H(CH2)={len(intra_CH_N)}+{len(intra_CH_O)}"
      f"  H-H dihedral (CH2-CH2) n={len(intra_HH_dih)}")
print(f"inter NH2-CH3 (cutoff {THR_NH2_CH3} A): "
      f"N-H={len(inter_NH_CH3)}  C-H={len(inter_Cu_HCH3)}  H-H={len(inter_HH_NCH3)}")
print(f"inter urea-CH2 (cutoff {THR_UREA_CH2} A on O-H): "
      f"O-H={len(inter_Ou_HCH2)}  N-H={len(inter_Nu_HCH2)}  H-H={len(inter_HN_HCH2)}")

print("\nIntra · N - H of urea (bonded only):")
print(stats(intra_NH))
print("\nIntra · N - H of urea (all H):")
print(stats(intra_NH_all))
print("\nIntra · C - H of urea:")
print(stats(intra_CH_urea))
print("\nIntra · CH2-CH2 of choline:")
print(f"  H-H        {stats(intra_HH)}")
print(f"  C-H (N)    {stats(intra_CH_N)}")
print(f"  C-H (O)    {stats(intra_CH_O)}")
print()
print("\nInter · N(NH2) - H(CH3):")
print(stats(inter_NH_CH3))
print("\nInter · C(urea) - H(CH3):")
print(stats(inter_Cu_HCH3))
print("\nInter · H(NH2) - H(CH3):")
print(stats(inter_HH_NCH3))
print("\nInter · O(urea) - H(CH2):")
print(stats(inter_Ou_HCH2))
print("\nInter · N(urea) - H(CH2):")
print(stats(inter_Nu_HCH2))
print("\nInter · H(NH2) - H(CH2):")
print(stats(inter_HN_HCH2))

