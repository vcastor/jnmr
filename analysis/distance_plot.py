#!/usr/bin/python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.plotting import PLOT_STYLES, hist, mlabel, stats, style_axes, style_cbar, save_fig
from hassan_functions.cache    import load_cache
from hassan_functions.style    import apply_style

apply_style("default")

PLOT_DIR         = "plots"
GAUCHE_THRESHOLD = 2*np.pi/3
THR_NH2_CH3      = 3.4    # H(NH2)-H(CH3) cutoff
THR_OU_HCH2O     = 3.2    # O(urea)-H(CH2 near O) cutoff
THR_OU_HCH2N     = 3.0    # O(urea)-H(CH2 near N) cutoff
THR_OU_HOH       = 2.8    # O(urea)-H(OH) cutoff
THR_HCH2_UREA    = 4.0    # choline CH2 H "in contact" with a urea atom
THR_CU_OH        = 4.5    # C(urea)-H(OH) contact cutoff

os.makedirs(PLOT_DIR, exist_ok=True)

def two_peak_fit(data):
    arr = np.array(data).reshape(-1, 1)
    g = GaussianMixture(n_components=2, random_state=0).fit(arr)
    m = g.means_.flatten()
    s = np.sqrt(g.covariances_.flatten())
    w = g.weights_
    order = np.argsort(m)
    return [(m[i], s[i], w[i]) for i in order]

# ══════════════════════════ intramolecular ══════════════════════════════════
c             = load_cache("distance_intra")
intra_NH        = c["intra_NH"]
intra_Nu_Hother = c.get("intra_Nu_Hother", [])
intra_CH_urea = c["intra_CH_urea"]
intra_HH      = c["intra_HH"]
intra_HH_dih  = c["intra_HH_dih"]
intra_CH_N    = c["intra_CH_N"]
intra_CH_O    = c["intra_CH_O"]
dihedrals     = c["dihedrals"]

nh_arr   = np.array(intra_NH)
ch_arr   = np.array(intra_CH_urea)
d_arr    = np.array(dihedrals)
n_tot    = len(d_arr)
n_gauche = int(np.sum(d_arr <= GAUCHE_THRESHOLD))
n_anti   = n_tot - n_gauche

print(f"N H (bonded):  {nh_arr.mean():.3f} +/- {nh_arr.std():.3f} A  (n={len(nh_arr)})")
if intra_Nu_Hother:
    noh    = np.array(intra_Nu_Hother)
    nh_all = np.concatenate([nh_arr, noh])          # N to ALL urea H (bonded + other N)
    print(f"N H (all):     {nh_all.mean():.3f} +/- {nh_all.std():.3f} A  (n={len(nh_all)})")
    print(f"N..H (other N):{noh.mean():.3f} +/- {noh.std():.3f} A  (n={len(noh)})")
print(f"C H (urea):    {ch_arr.mean():.3f} +/- {ch_arr.std():.3f} A  (n={len(ch_arr)})")
print(f"C H (CH2 N):   {stats(intra_CH_N)}")
print(f"C H (CH2 O):   {stats(intra_CH_O)}")
print(f"H H (CH2-CH2): {stats(intra_HH)}")
print(f"H C C H dihedral (rad): {stats(intra_HH_dih, unit='rad')}")
print()
print(f"N-CH2-CH2-O dihedral  (n={n_tot})")
print(f"  gauche (phi <= {GAUCHE_THRESHOLD:.4f} rad): {n_gauche}  ({100*n_gauche/n_tot:.1f}%)")
print(f"  anti   (phi >  {GAUCHE_THRESHOLD:.4f} rad): {n_anti}  ({100*n_anti/n_tot:.1f}%)")

print("\nH H (CH2-CH2) two-Gaussian decomposition:")
for k, (m, s, w) in enumerate(two_peak_fit(intra_HH)):
    print(f"  peak {k + 1}: mean={m:.3f} A  std={s:.3f} A  weight={w:.2f}")

print("\nH C C H dihedral two-Gaussian decomposition:")
for k, (m, s, w) in enumerate(two_peak_fit(intra_HH_dih)):
    print(f"  peak {k + 1}: mean={m:.3f} rad  std={s:.3f} rad  weight={w:.2f}")

DIH_TICKS  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
DIH_LABELS = [r'$0$', r'$\sfrac{\pi}{4}$', r'$\sfrac{\pi}{2}$',
              r'$\sfrac{3\pi}{4}$', r'$\pi$']

NCCO_TICKS  = [0, np.pi/3, 2*np.pi/3, np.pi]
NCCO_LABELS = [r'$0$', r'$\sfrac{\pi}{3}$', r'$\sfrac{2\pi}{3}$', r'$\pi$']

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # ── panel 0: C H of each CH2 ──────────────────────────────────────────
    hist(axes[0], intra_CH_N, mlabel(r"H2", intra_CH_N), "seagreen", LETTER_COLOUR)
    hist(axes[0], intra_CH_O, mlabel(r"H3", intra_CH_O), "purple",   LETTER_COLOUR)
    axes[0].set_title(r"C$-$H of each CH$_2$")
    axes[0].set_xlabel(r"distance (\AA)")
    axes[0].set_ylabel("density")
    axes[0].legend(loc='upper right')
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax*1.2)

    # ── panel 1: |H-C-C-H| dihedral vs H-H distance ──────────────────────
    hb = axes[1].hexbin(intra_HH_dih, intra_HH, gridsize=40, cmap="Oranges", mincnt=1)
    cbar = fig.colorbar(hb, ax=axes[1], label="count")
    style_cbar(cbar, LETTER_COLOUR)
    axes[1].set_title(r"CH$_2$$-$CH$_2$")
    axes[1].set_xlabel(r"$|$H-C-C-H$|$ dihedral (rad)")
    axes[1].set_ylabel(r"H-H distance (\AA)")
    axes[1].set_xlim(0, np.pi)
    axes[1].set_xticks(DIH_TICKS)
    axes[1].set_xticklabels(DIH_LABELS)

    # ── panel 2: |N-CH2-CH2-O| dihedral (gauche / anti) ──────────────────
    axes[2].axvspan(0, GAUCHE_THRESHOLD, alpha=0.12, color='steelblue', label='gauche')
    axes[2].axvspan(GAUCHE_THRESHOLD, np.pi, alpha=0.12, color='tomato', label='anti')
    hist(axes[2], list(d_arr), None, 'steelblue', LETTER_COLOUR, bins=40)
    axes[2].set_title(r"N–CH$_2$–CH$_2$–O dihedral angle")
    axes[2].set_xlabel(r"$|\phi|$ (rad)")
    axes[2].set_ylabel("density")
    axes[2].set_xticks(NCCO_TICKS)
    axes[2].set_xticklabels(NCCO_LABELS)
    axes[2].set_xlim(0, np.pi)
    axes[2].legend(loc='upper right')
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.15)

    for ax in axes:
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/distance_intra{SUFFIX}", TRANSPARENT)

    # urea: N to its H atoms — the 2 bonded (N-H) and the 2 on the other N. The two
    # regimes (~1 Å bond vs the cross-molecule contacts) sit on very different scales,
    # so they get a panel each.
    if intra_NH and intra_Nu_Hother:
        figu, axu = plt.subplots(1, 2, figsize=(11, 4.5))
        hist(axu[0], intra_NH,        mlabel(r"N$-$H5 (bonded)", intra_NH),
             "seagreen",   LETTER_COLOUR)
        hist(axu[1], intra_Nu_Hother, mlabel(r"N$\cdots$H5 (other N)", intra_Nu_Hother),
             "darkorange", LETTER_COLOUR)
        axu[0].set_title(r"urea N $-$ H5 (2 bonded)")
        axu[1].set_title(r"urea N $\cdots$ H5 (2 on the other N)")
        for ax in axu:
            ax.set_xlabel(r"distance (\AA)")
            ax.set_ylabel("density")
            ax.legend()
            style_axes(ax, LETTER_COLOUR, TRANSPARENT)
            ymax = ax.get_ylim()[1]
            ax.set_ylim(0, ymax*1.15)
        figu.tight_layout()
        save_fig(figu, f"{PLOT_DIR}/distance_intra_urea_NH{SUFFIX}", TRANSPARENT)

# ══════════════════════════ intermolecular ══════════════════════════════════
c = load_cache("distance_inter")
inter_NH_CH3       = c["inter_NH_CH3"]
inter_Cu_HCH3      = c["inter_Cu_HCH3"]
inter_HH_NCH3      = c["inter_HH_NCH3"]
inter_Ou_HCH2      = c["inter_Ou_HCH2"]
inter_Nu_HCH2      = c["inter_Nu_HCH2"]
inter_HN_HCH2      = c["inter_HN_HCH2"]
inter_Ou_HOH       = c["inter_Ou_HOH"]
inter_Nu_HOH       = c["inter_Nu_HOH"]
inter_HN_HOH       = c["inter_HN_HOH"]
inter_Ou_HCH2N_dbl = c["inter_Ou_HCH2N_dbl"]
inter_Ou_HOH_dbl   = c["inter_Ou_HOH_dbl"]
inter_Nu_dbl       = c["inter_Nu_dbl"]
inter_HN_dbl       = c["inter_HN_dbl"]
n_dbl_events       = c["n_dbl_events"]
inter_HCH2N_HN     = c.get("inter_HCH2N_HN", [])
inter_HCH2N_Cu     = c.get("inter_HCH2N_Cu", [])
inter_HCH2N_Nu     = c.get("inter_HCH2N_Nu", [])
inter_HCH2O_HN     = c.get("inter_HCH2O_HN", [])
inter_HCH2O_Cu     = c.get("inter_HCH2O_Cu", [])
inter_HCH2O_Nu     = c.get("inter_HCH2O_Nu", [])
inter_Cu_HOH       = c.get("inter_Cu_HOH", [])
# O(urea) / C(urea) / N(urea) distance to each choline H type (paired on the same H)
oc_HCH2N_O = c.get("oc_HCH2N_O", []); oc_HCH2N_C = c.get("oc_HCH2N_C", []); oc_HCH2N_N = c.get("oc_HCH2N_N", [])
oc_HCH2O_O = c.get("oc_HCH2O_O", []); oc_HCH2O_C = c.get("oc_HCH2O_C", []); oc_HCH2O_N = c.get("oc_HCH2O_N", [])
oc_HOH_O   = c.get("oc_HOH_O", []);   oc_HOH_C   = c.get("oc_HOH_C", []);   oc_HOH_N   = c.get("oc_HOH_N", [])
OC_PANELS  = [(r"H2", oc_HCH2N_O, oc_HCH2N_C, oc_HCH2N_N),
              (r"H3", oc_HCH2O_O, oc_HCH2O_C, oc_HCH2O_N),
              (r"H4", oc_HOH_O,   oc_HOH_C,   oc_HOH_N)]

def closest_pct(dO, dC, dN):
    """Percent of H's for which O / C / N is the closest urea site."""
    if not dO:
        return 0.0, 0.0, 0.0
    who = np.argmin(np.vstack([dO, dC, dN]), axis=0)
    return tuple(100*np.mean(who == k) for k in (0, 1, 2))

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    # NH2-CH3
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_NH_CH3,  mlabel(r"N(urea)$-$H1", inter_NH_CH3),  "steelblue",  LETTER_COLOUR)
    hist(axes[1], inter_Cu_HCH3, mlabel(r"C(urea)$-$H1", inter_Cu_HCH3), "darkorange", LETTER_COLOUR)
    hist(axes[2], inter_HH_NCH3, mlabel(r"H5$-$H1",      inter_HH_NCH3), "seagreen",   LETTER_COLOUR)
    axes[0].set_title(rf"N(urea)$-$H1") # [H-H cutoff {THR_NH2_CH3} \AA]")
    axes[1].set_title(rf"C(urea)$-$H1") #  [H-H cutoff {THR_NH2_CH3} \AA]")
    axes[2].set_title(rf"H5$-$H1") #  [cutoff {THR_NH2_CH3} \AA]")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend(loc="upper left")
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax*1.2)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax*1.15)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.15)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/distance_inter_NH2_CH3{SUFFIX}", TRANSPARENT)

    # urea-CH2 HCH-O-HCH
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Ou_HCH2, mlabel(r"O(urea)$-$H2/H3", inter_Ou_HCH2), "steelblue",  LETTER_COLOUR)
    hist(axes[1], inter_Nu_HCH2, mlabel(r"N(urea)$-$H2/H3", inter_Nu_HCH2), "seagreen",   LETTER_COLOUR)
    hist(axes[2], inter_HN_HCH2, mlabel(r"H5$-$H2/H3",      inter_HN_HCH2), "darkorange", LETTER_COLOUR)
    axes[0].set_title(rf"O(urea)$-$H2/H3") #  [HCH-O-HCH, CH2N {THR_OU_HCH2N} \AA, CH2O {THR_OU_HCH2O} \AA]")
    axes[1].set_title(r"N(urea)$-$H2/H3")
    axes[2].set_title(r"H5$-$H2/H3")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax*1.25)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax*1.2)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.15)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/distance_inter_urea_CH2{SUFFIX}", TRANSPARENT)

    # urea-OH  [O/N/H(NH2)/C](urea)-H(OH)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    hist(axes[0], inter_Ou_HOH, mlabel(r"O(urea)$-$H4", inter_Ou_HOH), "steelblue",  LETTER_COLOUR)
    hist(axes[1], inter_Nu_HOH, mlabel(r"N(urea)$-$H4", inter_Nu_HOH), "seagreen",   LETTER_COLOUR)
    hist(axes[2], inter_HN_HOH, mlabel(r"H5$-$H4",      inter_HN_HOH), "darkorange", LETTER_COLOUR)
    hist(axes[3], inter_Cu_HOH, mlabel(r"C(urea)$-$H4", inter_Cu_HOH), "purple",     LETTER_COLOUR)
    axes[0].set_title(rf"O(urea)$-$H4") #  [cutoff {THR_OU_HOH} \AA]")
    axes[1].set_title(r"N(urea)$-$H4")
    axes[2].set_title(r"H5$-$H4")
    axes[3].set_title(rf"C(urea)$-$H4") #  [cutoff {THR_CU_OH} \AA]")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax*1.15)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/distance_inter_urea_OH{SUFFIX}", TRANSPARENT)

    # double bridge O(urea)-[H(CH2-N) & H(OH)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_Ou_HCH2N_dbl,
         mlabel(r"O$-$H2", inter_Ou_HCH2N_dbl), "steelblue", LETTER_COLOUR)
    hist(axes[0], inter_Ou_HOH_dbl,
         mlabel(r"O$-$H4", inter_Ou_HOH_dbl),   "crimson",   LETTER_COLOUR)
    hist(axes[1], inter_Nu_dbl,
         mlabel(r"N(urea)$-$H", inter_Nu_dbl),  "seagreen",  LETTER_COLOUR)
    hist(axes[2], inter_HN_dbl,
         mlabel(r"H5$-$H",      inter_HN_dbl),   "darkorange", LETTER_COLOUR)
    axes[0].set_title(rf"O(urea)$-$[H2 \& H4]") # rf"[CH2N {THR_OU_HCH2N} \AA, OH {THR_OU_HOH} \AA]")
    axes[1].set_title(r"N(urea)$-$H (H2 \& H4)")
    axes[2].set_title(r"H5$-$H (H2 \& H4)")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax*1.3)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax*1.2)
    ymax = axes[2].get_ylim()[1]
    axes[2].set_ylim(0, ymax*1.15)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/distance_inter_urea_double{SUFFIX}", TRANSPARENT)

    # choline CH2-N H vs urea H(NH2) / C / N
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_HCH2N_HN, mlabel(r"H2$-$H5",      inter_HCH2N_HN), "steelblue",  LETTER_COLOUR)
    hist(axes[1], inter_HCH2N_Cu, mlabel(r"H2$-$C(urea)", inter_HCH2N_Cu), "darkorange", LETTER_COLOUR)
    hist(axes[2], inter_HCH2N_Nu, mlabel(r"H2$-$N(urea)", inter_HCH2N_Nu), "seagreen",   LETTER_COLOUR)
    axes[0].set_title(r"H2$-$H5")
    axes[1].set_title(r"H2$-$C(urea)")
    axes[2].set_title(r"H2$-$N(urea)")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax*1.15)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/distance_inter_CH2N_urea{SUFFIX}", TRANSPARENT)

    # choline CH2-O H vs urea H(NH2) / C / N
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist(axes[0], inter_HCH2O_HN, mlabel(r"H3$-$H5",      inter_HCH2O_HN), "steelblue",  LETTER_COLOUR)
    hist(axes[1], inter_HCH2O_Cu, mlabel(r"H3$-$C(urea)", inter_HCH2O_Cu), "darkorange", LETTER_COLOUR)
    hist(axes[2], inter_HCH2O_Nu, mlabel(r"H3$-$N(urea)", inter_HCH2O_Nu), "seagreen",   LETTER_COLOUR)
    axes[0].set_title(r"H3$-$H5")
    axes[1].set_title(r"H3$-$C(urea)")
    axes[2].set_title(r"H3$-$N(urea)")
    for ax in axes:
        ax.set_xlabel(r"distance (\AA)")
        ax.set_ylabel("density")
        ax.legend()
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax*1.15)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/distance_inter_CH2O_urea{SUFFIX}", TRANSPARENT)

    # O(urea) vs C(urea) vs N(urea): which urea site is closest to each choline H type
    if any(len(dO) for _, dO, _, _ in OC_PANELS):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, (name, dO, dC, dN) in zip(axes, OC_PANELS):
            hist(ax, dO, mlabel(r"O(urea)", dO), "steelblue",  LETTER_COLOUR)
            hist(ax, dC, mlabel(r"C(urea)", dC), "darkorange", LETTER_COLOUR)
            hist(ax, dN, mlabel(r"N(urea)", dN), "seagreen",   LETTER_COLOUR)
            pO, pC, pN = closest_pct(dO, dC, dN)
            ax.set_title(rf"urea $-$ {name}" + "\n"
                         + rf"closest O/C/N: {pO:.0f}/{pC:.0f}/{pN:.0f}")
            ax.set_xlabel(r"distance (\AA)")
            ax.set_ylabel("density")
            ax.legend()
            style_axes(ax, LETTER_COLOUR, TRANSPARENT)
            ymax = ax.get_ylim()[1]
            ax.set_ylim(0, ymax*1.15)
        fig.tight_layout()
        save_fig(fig, f"{PLOT_DIR}/distance_inter_urea_OCN{SUFFIX}", TRANSPARENT)

print("\nO(urea) vs C(urea) vs N(urea) distance to the choline H's (nearest urea, paired):")
for name, dO, dC, dN in [("H(CH2-N)", oc_HCH2N_O, oc_HCH2N_C, oc_HCH2N_N),
                         ("H(CH2-O)", oc_HCH2O_O, oc_HCH2O_C, oc_HCH2O_N),
                         ("H(OH)",    oc_HOH_O,   oc_HOH_C,   oc_HOH_N)]:
    if not dO:
        print(f"  {name:9}: n=0")
        continue
    aO, aC, aN = np.array(dO), np.array(dC), np.array(dN)
    pO, pC, pN = closest_pct(dO, dC, dN)
    print(f"  {name:9}: n={len(aO):5d}  <d(O-H)>={aO.mean():.2f}  <d(C-H)>={aC.mean():.2f}  "
          f"<d(N-H)>={aN.mean():.2f} A  -> closest O/C/N = {pO:.0f}%/{pC:.0f}%/{pN:.0f}%")

print(f"\ninter NH2-CH3 (cutoff {THR_NH2_CH3} Å): "
      f"N-H={len(inter_NH_CH3)}  C-H={len(inter_Cu_HCH3)}  H-H={len(inter_HH_NCH3)}")
print(f"inter urea-CH2 HCH-O-HCH (CH2N {THR_OU_HCH2N} Å, CH2O {THR_OU_HCH2O} Å, one H per CH2): "
      f"O-H={len(inter_Ou_HCH2)}  N-H={len(inter_Nu_HCH2)}  H-H={len(inter_HN_HCH2)}")
print(f"inter urea-OH (cutoff {THR_OU_HOH} Å): "
      f"O-H={len(inter_Ou_HOH)}  N-H={len(inter_Nu_HOH)}  H-H={len(inter_HN_HOH)}")
print(f"inter double bridge O(urea)-[H(CH2N) & H(OH)] (CH2N {THR_OU_HCH2N} Å, OH {THR_OU_HOH} Å): "
      f"events={n_dbl_events}  "
      f"O-H pairs={len(inter_Ou_HCH2N_dbl)}+{len(inter_Ou_HOH_dbl)}  "
      f"N-H={len(inter_Nu_dbl)}  H-H={len(inter_HN_dbl)}")
print(f"inter CH2 H (contact <= {THR_HCH2_UREA} Å of a urea atom) vs urea H/C/N: "
      f"CH2-N n={len(inter_HCH2N_Cu)}  CH2-O n={len(inter_HCH2O_Cu)}")
print(f"inter C(urea)-H(OH) (cutoff {THR_CU_OH} Å): n={len(inter_Cu_HOH)}")

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

print("\nInter · H(CH2-N)-H(NH2):")
print(stats(inter_HCH2N_HN))
print("\nInter · H(CH2-N)-C(urea):")
print(stats(inter_HCH2N_Cu))
print("\nInter · H(CH2-N)-N(urea):")
print(stats(inter_HCH2N_Nu))
print("\nInter · H(CH2-O)-H(NH2):")
print(stats(inter_HCH2O_HN))
print("\nInter · H(CH2-O)-C(urea):")
print(stats(inter_HCH2O_Cu))
print("\nInter · H(CH2-O)-N(urea):")
print(stats(inter_HCH2O_Nu))
print("\nInter · C(urea)-H(OH):")
print(stats(inter_Cu_HOH))
