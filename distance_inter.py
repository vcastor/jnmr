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

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac}\usepackage{amsmath}'

def find_nh2_groups(urea):
    out = []
    for n in urea.atoms:
        if n.symbol != 'N':
            continue
        hs = [b.other_end(n) for b in n.bonds if b.other_end(n).symbol == 'H']
        if hs:
            out.append((n, hs))
    return out

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
    return float(np.linalg.norm(np.array(a.coords)-np.array(b.coords)))

# NH2-CH3
inter_NH_CH3   = []
inter_Cu_HCH3  = []
inter_HH_NCH3  = []
# urea-CH2 HCH-O-HCH (one H per CH2 below cutoff)
inter_Ou_HCH2  = []
inter_Nu_HCH2  = []
inter_HN_HCH2  = []
# urea-CH2 double bridge (split by which CH2 the H belongs to)
inter_Ou_HCH2_dbl_N = []   # O-H, H comes from CH2 adjacent to N+
inter_Ou_HCH2_dbl_O = []   # O-H, H comes from CH2 adjacent to O
inter_Nu_HCH2_dbl   = []
inter_HN_HCH2_dbl   = []
n_dbl_events = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf))-2 < 1:
        continue

    cluster = Molecule(xf)
    cluster.guess_bonds()
    mols = cluster.separate()

    ureas    = [m for m in mols if m.get_formula() == 'CH4N2O']
    cholines = [m for m in mols if m.get_formula() == 'C5H14NO']

    for u in ureas:
        nh2_groups = find_nh2_groups(u)
        c_urea = next(at for at in u.atoms if at.symbol == 'C')
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

            close_N = [(h, D(o_urea, h)) for h in hN if D(o_urea, h) <= THR_UREA_CH2]
            close_O = [(h, D(o_urea, h)) for h in hO if D(o_urea, h) <= THR_UREA_CH2]
            if not (close_N and close_O):
                continue

            # HCH-O-HCH: one H per CH2 within cutoff, record each close H
            for h, d in close_N+close_O:
                inter_Ou_HCH2.append(d)
                n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: D(g[0], h))
                inter_Nu_HCH2.append(D(n_nh2, h))
                for h_nh2 in h_nh2_list:
                    inter_HN_HCH2.append(D(h_nh2, h))

            # double bridge stats split by which CH2 the H sits on
            n_dbl_events += 1
            for hNa, dN in close_N:
                for hOa, dO in close_O:
                    inter_Ou_HCH2_dbl_N.append(dN)
                    inter_Ou_HCH2_dbl_O.append(dO)
                    for hh in (hNa, hOa):
                        n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: D(g[0], hh))
                        inter_Nu_HCH2_dbl.append(D(n_nh2, hh))
                        for h_nh2 in h_nh2_list:
                            inter_HN_HCH2_dbl.append(D(h_nh2, hh))

finish()

def stats(data):
    if not data:
        return "n=0"
    d = np.array(data)
    return f"n={len(d)}  mean={d.mean():.2f} Å  std={d.std():.2f} Å"

def mlabel(prefix, data):
    if not data:
        return rf"{prefix} (n=0)"
    return rf"{prefix} $\langle d\rangle={np.mean(data):.2f}\pm{np.std(data):.2f}$ \AA"

def hist(ax, data, label, colour, bins=40):
    if not data:
        return
    sns.histplot(data, bins=bins, kde=True, ax=ax, label=label, color=colour,
                 alpha=0.5, edgecolor=LETTER_COLOUR, line_kws={"linewidth": 2},
                 stat='density')

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

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    # NH2-CH3
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_NH_CH3,  mlabel(r"N(NH$_2$)$-$H(CH$_3$)", inter_NH_CH3),  "steelblue")
    hist(axes[1], inter_Cu_HCH3, mlabel(r"C(urea)$-$H(CH$_3$)",   inter_Cu_HCH3), "darkorange")
    hist(axes[2], inter_HH_NCH3, mlabel(r"H(NH$_2$)$-$H(CH$_3$)", inter_HH_NCH3), "seagreen")
    axes[0].set_title(rf"Inter $\cdot$ N(NH$_2$)$-$H(CH$_3$)  [H-H cutoff {THR_NH2_CH3} \AA]")
    axes[1].set_title(rf"Inter $\cdot$ C(urea)$-$H(CH$_3$)  [H-H cutoff {THR_NH2_CH3} \AA]")
    axes[2].set_title(rf"Inter $\cdot$ H(NH$_2$)$-$H(CH$_3$)  [cutoff {THR_NH2_CH3} \AA]")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_NH2_CH3{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # urea-CH2 HCH-O-HCH
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Ou_HCH2, mlabel(r"O(urea)$-$H(CH$_2$)", inter_Ou_HCH2), "steelblue")
    hist(axes[1], inter_Nu_HCH2, mlabel(r"N(urea)$-$H(CH$_2$)", inter_Nu_HCH2), "seagreen")
    hist(axes[2], inter_HN_HCH2, mlabel(r"H(NH$_2$)$-$H(CH$_2$)", inter_HN_HCH2), "darkorange")
    axes[0].set_title(rf"Inter $\cdot$ O(urea)$-$H(CH$_2$)  [HCH-O-HCH, cutoff {THR_UREA_CH2} \AA]")
    axes[1].set_title(r"Inter $\cdot$ N(urea)$-$H(CH$_2$)")
    axes[2].set_title(r"Inter $\cdot$ H(NH$_2$)$-$H(CH$_2$)")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.1)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_urea_CH2{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # urea-CH2 double bridge
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Ou_HCH2_dbl_N,
         mlabel(r"O$-$H (CH$_2$ near N$^+$)", inter_Ou_HCH2_dbl_N), "steelblue")
    hist(axes[0], inter_Ou_HCH2_dbl_O,
         mlabel(r"O$-$H (CH$_2$ near O)",     inter_Ou_HCH2_dbl_O), "crimson")
    hist(axes[1], inter_Nu_HCH2_dbl,
         mlabel(r"N(urea)$-$H(CH$_2$)",       inter_Nu_HCH2_dbl),   "seagreen")
    hist(axes[2], inter_HN_HCH2_dbl,
         mlabel(r"H(NH$_2$)$-$H(CH$_2$)",     inter_HN_HCH2_dbl),   "darkorange")
    axes[0].set_title(rf"Inter $\cdot$ O(urea)$-$H(CH$_2$) double bridge  "
                      rf"[cutoff {THR_UREA_CH2} \AA]")
    axes[1].set_title(r"Inter $\cdot$ N(urea)$-$H(CH$_2$) double bridge")
    axes[2].set_title(r"Inter $\cdot$ H(NH$_2$)$-$H(CH$_2$) double bridge")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.1)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_urea_CH2_double{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

print(f"inter NH2-CH3 (cutoff {THR_NH2_CH3} Å): "
      f"N-H={len(inter_NH_CH3)}  C-H={len(inter_Cu_HCH3)}  H-H={len(inter_HH_NCH3)}")
print(f"inter urea-CH2 HCH-O-HCH (cutoff {THR_UREA_CH2} Å, one H per CH2): "
      f"O-H={len(inter_Ou_HCH2)}  N-H={len(inter_Nu_HCH2)}  H-H={len(inter_HN_HCH2)}")
print(f"inter urea-CH2 double bridge (both H within {THR_UREA_CH2} Å): "
      f"events={n_dbl_events}  "
      f"O-H pairs={len(inter_Ou_HCH2_dbl_N)}+{len(inter_Ou_HCH2_dbl_O)}  "
      f"N-H={len(inter_Nu_HCH2_dbl)}  H-H={len(inter_HN_HCH2_dbl)}")

print("\nInter · N(NH2)-H(CH3):")
print(stats(inter_NH_CH3))
print("\nInter · C(urea)-H(CH3):")
print(stats(inter_Cu_HCH3))
print("\nInter · H(NH2)-H(CH3):")
print(stats(inter_HH_NCH3))

print("\nInter · O(urea)-H(CH2) [HCH-O-HCH]:")
print(stats(inter_Ou_HCH2))
print("\nInter · N(urea)-H(CH2) [HCH-O-HCH]:")
print(stats(inter_Nu_HCH2))
print("\nInter · H(NH2)-H(CH2) [HCH-O-HCH]:")
print(stats(inter_HN_HCH2))

print("\nInter · O(urea)-H(CH2) [double bridge, CH2 near N+]:")
print(stats(inter_Ou_HCH2_dbl_N))
print("\nInter · O(urea)-H(CH2) [double bridge, CH2 near O]:")
print(stats(inter_Ou_HCH2_dbl_O))
print("\nInter · N(urea)-H(CH2) [double bridge]:")
print(stats(inter_Nu_HCH2_dbl))
print("\nInter · H(NH2)-H(CH2) [double bridge]:")
print(stats(inter_HN_HCH2_dbl))

