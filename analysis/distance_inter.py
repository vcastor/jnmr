#!$AMSBIN/plams
import os
import glob
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance
from hassan_functions.finders   import find_xh_groups, find_adjacent_xh_pair_anchored
from hassan_functions.plotting  import PLOT_STYLES, hist, mlabel, stats, style_axes
from hassan_functions.constants import FORMULAS

PLOT_DIR     = "plots"
CLUSTERS_DIR = "clusters"
THR_NH2_CH3  = 3.4    # H(NH2)-H(CH3) cutoff
THR_OU_HCH2O = 3.2    # O(urea)-H(CH2 near O) cutoff
THR_OU_HCH2N = 3.0    # O(urea)-H(CH2 near N) cutoff
THR_OU_HOH   = 2.8    # O(urea)-H(OH) cutoff

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac}\usepackage{amsmath}'
plt.rcParams['font.size']       = 16
plt.rcParams['axes.titlesize']  = 19
plt.rcParams['axes.labelsize']  = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# NH2-CH3
inter_NH_CH3   = []
inter_Cu_HCH3  = []
inter_HH_NCH3  = []
# urea-CH2 HCH-O-HCH (one H per CH2 below cutoff)
inter_Ou_HCH2  = []
inter_Nu_HCH2  = []
inter_HN_HCH2  = []
# urea-OH  O(urea)-H(OH)
inter_Ou_HOH   = []
inter_Nu_HOH   = []
inter_HN_HOH   = []
# double bridge O(urea)-[H(CH2-N) & H(OH)]
inter_Ou_HCH2N_dbl = []   # O-H, H from CH2 adjacent to N+
inter_Ou_HOH_dbl   = []   # O-H, H from OH
inter_Nu_dbl       = []
inter_HN_dbl       = []
n_dbl_events = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue

    cluster = Molecule(xf)
    cluster.guess_bonds()
    mols    = cluster.separate()

    ureas    = [m for m in mols if m.get_formula() == FORMULAS['urea']]
    cholines = [m for m in mols if m.get_formula() == FORMULAS['choline']]

    for u in ureas:
        nh2_groups = find_xh_groups(u, 'N', 2)
        c_urea = next(at for at in u.atoms if at.symbol == 'C')
        for ch in cholines:
            ch3 = find_xh_groups(ch, 'C', 3, neighbour='N')
            for n_nh2, h_nh2_list in nh2_groups:
                for c_ch3, h_ch3_list in ch3:
                    near = sorted(h_ch3_list, key=lambda h: distance(n_nh2, h))[:2]
                    near = [h for h in near if distance(n_nh2, h) <= THR_NH2_CH3]
                    for h_ch3 in near:
                        inter_NH_CH3.append(distance(n_nh2, h_ch3))
                        inter_Cu_HCH3.append(distance(c_urea, h_ch3))
                        for h_nh2 in h_nh2_list:
                            inter_HH_NCH3.append(distance(h_nh2, h_ch3))

    for u in ureas:
        o_urea     = next(at for at in u.atoms if at.symbol == 'O')
        nh2_groups = find_xh_groups(u, 'N', 2)
        for ch in cholines:
            _, hN, _, hO = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            _, hOH       = find_xh_groups(ch, 'O', 1)[0]

            close_N  = [(h, distance(o_urea, h)) for h in hN  if distance(o_urea, h) <= THR_OU_HCH2N]
            close_O  = [(h, distance(o_urea, h)) for h in hO  if distance(o_urea, h) <= THR_OU_HCH2O]
            close_OH = [(h, distance(o_urea, h)) for h in hOH if distance(o_urea, h) <= THR_OU_HOH]

            # O(urea)-H(OH): single interaction, record each close OH H
            for h, d in close_OH:
                inter_Ou_HOH.append(d)
                n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: distance(g[0], h))
                inter_Nu_HOH.append(distance(n_nh2, h))
                for h_nh2 in h_nh2_list:
                    inter_HN_HOH.append(distance(h_nh2, h))

            # HCH-O-HCH: one H per CH2 within cutoff (both CH2 groups required)
            if close_N and close_O:
                for h, d in close_N + close_O:
                    inter_Ou_HCH2.append(d)
                    n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: distance(g[0], h))
                    inter_Nu_HCH2.append(distance(n_nh2, h))
                    for h_nh2 in h_nh2_list:
                        inter_HN_HCH2.append(distance(h_nh2, h))

            # double bridge O(urea)-[H(CH2-N) & H(OH)]
            if close_N and close_OH:
                n_dbl_events += 1
                for hNa, dN in close_N:
                    for hOHa, dOH in close_OH:
                        inter_Ou_HCH2N_dbl.append(dN)
                        inter_Ou_HOH_dbl.append(dOH)
                        for hh in (hNa, hOHa):
                            n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: distance(g[0], hh))
                            inter_Nu_dbl.append(distance(n_nh2, hh))
                            for h_nh2 in h_nh2_list:
                                inter_HN_dbl.append(distance(h_nh2, hh))

finish()

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    # NH2-CH3
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_NH_CH3,  mlabel(r"N(NH$_2$)$-$H(CH$_3$)", inter_NH_CH3),  "steelblue",  LETTER_COLOUR)
    hist(axes[1], inter_Cu_HCH3, mlabel(r"C(urea)$-$H(CH$_3$)",   inter_Cu_HCH3), "darkorange", LETTER_COLOUR)
    hist(axes[2], inter_HH_NCH3, mlabel(r"H(NH$_2$)$-$H(CH$_3$)", inter_HH_NCH3), "seagreen",   LETTER_COLOUR)
    axes[0].set_title(rf"N(NH$_2$)$-$H(CH$_3$)") # [H-H cutoff {THR_NH2_CH3} \AA]")
    axes[1].set_title(rf"C(urea)$-$H(CH$_3$)") #  [H-H cutoff {THR_NH2_CH3} \AA]")
    axes[2].set_title(rf"H(NH$_2$)$-$H(CH$_3$)") #  [cutoff {THR_NH2_CH3} \AA]")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend(loc="upper left")
        style_axes(ax, LETTER_COLOUR)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax*1.2)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax*1.15)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.15)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_NH2_CH3{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # urea-CH2 HCH-O-HCH
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Ou_HCH2, mlabel(r"O(urea)$-$H(CH$_2$)",   inter_Ou_HCH2), "steelblue",  LETTER_COLOUR)
    hist(axes[1], inter_Nu_HCH2, mlabel(r"N(urea)$-$H(CH$_2$)",   inter_Nu_HCH2), "seagreen",   LETTER_COLOUR)
    hist(axes[2], inter_HN_HCH2, mlabel(r"H(NH$_2$)$-$H(CH$_2$)", inter_HN_HCH2), "darkorange", LETTER_COLOUR)
    axes[0].set_title(rf"O(urea)$-$H(CH$_2$)") #  [HCH-O-HCH, CH2N {THR_OU_HCH2N} \AA, CH2O {THR_OU_HCH2O} \AA]")
    axes[1].set_title(r"N(urea)$-$H(CH$_2$)")
    axes[2].set_title(r"H(NH$_2$)$-$H(CH$_2$)")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax, LETTER_COLOUR)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax*1.25)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax*1.2)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.15)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_urea_CH2{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # urea-OH  O(urea)-H(OH)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Ou_HOH, mlabel(r"O(urea)$-$H(OH)",   inter_Ou_HOH), "steelblue",  LETTER_COLOUR)
    hist(axes[1], inter_Nu_HOH, mlabel(r"N(urea)$-$H(OH)",   inter_Nu_HOH), "seagreen",   LETTER_COLOUR)
    hist(axes[2], inter_HN_HOH, mlabel(r"H(NH$_2$)$-$H(OH)", inter_HN_HOH), "darkorange", LETTER_COLOUR)
    axes[0].set_title(rf"O(urea)$-$H(OH)") #  [cutoff {THR_OU_HOH} \AA]")
    axes[1].set_title(r"N(urea)$-$H(OH)")
    axes[2].set_title(r"H(NH$_2$)$-$H(OH)")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax, LETTER_COLOUR)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax*1.15)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax*1.15)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.15)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_urea_OH{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # double bridge O(urea)-[H(CH2-N) & H(OH)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Ou_HCH2N_dbl,
         mlabel(r"O$-$H (CH$_2$ near N$^+$)", inter_Ou_HCH2N_dbl), "steelblue", LETTER_COLOUR)
    hist(axes[0], inter_Ou_HOH_dbl,
         mlabel(r"O$-$H (OH)",                inter_Ou_HOH_dbl),   "crimson",   LETTER_COLOUR)
    hist(axes[1], inter_Nu_dbl,
         mlabel(r"N(urea)$-$H",               inter_Nu_dbl),       "seagreen",  LETTER_COLOUR)
    hist(axes[2], inter_HN_dbl,
         mlabel(r"H(NH$_2$)$-$H",             inter_HN_dbl),       "darkorange", LETTER_COLOUR)
    axes[0].set_title(rf"O(urea)$-$[H(CH$_2$N) \& H(OH)]") # rf"[CH2N {THR_OU_HCH2N} \AA, OH {THR_OU_HOH} \AA]")
    axes[1].set_title(r"N(urea)$-$H")
    axes[2].set_title(r"H(NH$_2$)$-$H")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax, LETTER_COLOUR)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax*1.3)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax*1.2)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.15)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_inter_urea_double{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

print(f"inter NH2-CH3 (cutoff {THR_NH2_CH3} Å): "
      f"N-H={len(inter_NH_CH3)}  C-H={len(inter_Cu_HCH3)}  H-H={len(inter_HH_NCH3)}")
print(f"inter urea-CH2 HCH-O-HCH (CH2N {THR_OU_HCH2N} Å, CH2O {THR_OU_HCH2O} Å, one H per CH2): "
      f"O-H={len(inter_Ou_HCH2)}  N-H={len(inter_Nu_HCH2)}  H-H={len(inter_HN_HCH2)}")
print(f"inter urea-OH (cutoff {THR_OU_HOH} Å): "
      f"O-H={len(inter_Ou_HOH)}  N-H={len(inter_Nu_HOH)}  H-H={len(inter_HN_HOH)}")
print(f"inter double bridge O(urea)-[H(CH2N) & H(OH)] (CH2N {THR_OU_HCH2N} Å, OH {THR_OU_HOH} Å): "
      f"events={n_dbl_events}  "
      f"O-H pairs={len(inter_Ou_HCH2N_dbl)}+{len(inter_Ou_HOH_dbl)}  "
      f"N-H={len(inter_Nu_dbl)}  H-H={len(inter_HN_dbl)}")

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

print("\nInter · O(urea)-H(OH):")
print(stats(inter_Ou_HOH))
print("\nInter · N(urea)-H(OH):")
print(stats(inter_Nu_HOH))
print("\nInter · H(NH2)-H(OH):")
print(stats(inter_HN_HOH))

print("\nInter · O(urea)-H(CH2N) [double bridge]:")
print(stats(inter_Ou_HCH2N_dbl))
print("\nInter · O(urea)-H(OH) [double bridge]:")
print(stats(inter_Ou_HOH_dbl))
print("\nInter · N(urea)-H [double bridge]:")
print(stats(inter_Nu_dbl))
print("\nInter · H(NH2)-H [double bridge]:")
print(stats(inter_HN_dbl))

