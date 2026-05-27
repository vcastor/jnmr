#!$AMSBIN/plams
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from hassan_functions.geometry import distance, dihedral
from hassan_functions.finders import find_xh_bonds, find_adjacent_xh_pair_anchored
from hassan_functions.plotting import (PLOT_STYLES, hist, mlabel, stats,
                                       style_axes, style_cbar)
from hassan_functions.constants import FORMULAS

PLOT_DIR     = "plots"
CLUSTERS_DIR = "clusters"

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac}\usepackage{amsmath}'

intra_NH      = []
intra_NH_all  = []
intra_CH_urea = []
intra_HH      = []
intra_HH_dih  = []
intra_CH_N    = []
intra_CH_O    = []

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue

    cluster = Molecule(xf)
    cluster.guess_bonds()
    mols = cluster.separate()

    ureas    = [m for m in mols if m.get_formula() == FORMULAS['urea']]
    cholines = [m for m in mols if m.get_formula() == FORMULAS['choline']]

    for u in ureas:
        ns_u = [at for at in u.atoms if at.symbol == 'N']
        hs_u = [at for at in u.atoms if at.symbol == 'H']
        c_u  = [at for at in u.atoms if at.symbol == 'C'][0]
        for n, h in find_xh_bonds(u, 'N'):
            intra_NH.append(distance(n, h))
        for n in ns_u:
            for h in hs_u:
                intra_NH_all.append(distance(n, h))
        for h in hs_u:
            intra_CH_urea.append(distance(c_u, h))

    for ch in cholines:
        c_N, hN, c_O, hO = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
        for a in hN:
            for b in hO:
                intra_HH.append(distance(a, b))
                intra_HH_dih.append(abs(dihedral(a, c_N, c_O, b)))
        for h in hN:
            intra_CH_N.append(distance(c_N, h))
        for h in hO:
            intra_CH_O.append(distance(c_O, h))

finish()

DIH_TICKS  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
DIH_LABELS = [r'$0$', r'$\sfrac{\pi}{4}$', r'$\sfrac{\pi}{2}$',
              r'$\sfrac{3\pi}{4}$', r'$\pi$']

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    hist(axes[0,0], intra_NH, mlabel(r"N$-$H (bonded)", intra_NH), "steelblue", LETTER_COLOUR)
    axes[0,0].set_title(r"Intra $\cdot$ N$-$H of urea")
    axes[0,0].set_xlabel(r"distance (\AA)")
    axes[0,0].set_ylabel("density")
    axes[0,0].legend(loc="upper right")
    ymax = axes[0,0].get_ylim()[1]
    axes[0,0].set_ylim(0, ymax*1.1)

    hb = axes[0,1].hexbin(intra_HH_dih, intra_HH, gridsize=40, cmap="Oranges", mincnt=1)
    cbar = fig.colorbar(hb, ax=axes[0,1], label="count")
    style_cbar(cbar, LETTER_COLOUR)
    axes[0,1].set_title(r"Intra $\cdot$ CH$_2$$-$CH$_2$")
    axes[0,1].set_xlabel(r"$\lvert$H-C-C-H$\rvert$ dihedral (rad)")
    axes[0,1].set_ylabel(r"H-H distance (\AA)")
    axes[0,1].set_xlim(0, np.pi)
    axes[0,1].set_xticks(DIH_TICKS)
    axes[0,1].set_xticklabels(DIH_LABELS)

    hist(axes[1,0], intra_CH_N, mlabel(r"CH$_2$$-$N", intra_CH_N), "seagreen", LETTER_COLOUR)
    hist(axes[1,0], intra_CH_O, mlabel(r"CH$_2$$-$O", intra_CH_O), "purple", LETTER_COLOUR)
    axes[1,0].set_title(r"Intra $\cdot$ C$-$H of each CH$_2$")
    axes[1,0].set_xlabel(r"distance (\AA)")
    axes[1,0].set_ylabel("density")
    axes[1,0].legend(loc='upper right')
    ymax = axes[1,0].get_ylim()[1]
    axes[1,0].set_ylim(0, ymax*1.15)

    hist(axes[1,1], intra_CH_urea, mlabel(r"C$-$H", intra_CH_urea), "indianred", LETTER_COLOUR)
    axes[1,1].set_title(r"Intra $\cdot$ C$-$H of urea")
    axes[1,1].set_xlabel(r"distance (\AA)")
    axes[1,1].set_ylabel("density")
    axes[1,1].legend(loc='upper right')
    ymax = axes[1,1].get_ylim()[1]
    axes[1,1].set_ylim(0, ymax*1.05)

    for ax in axes.ravel():
        style_axes(ax, LETTER_COLOUR)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_intra{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # extra: N-H bonded vs all
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins_NH = np.linspace(min(intra_NH_all), max(intra_NH_all), 41)
    hist(ax, intra_NH,     mlabel(r"N$-$H (bonded)", intra_NH),     "steelblue", LETTER_COLOUR, bins=bins_NH)
    hist(ax, intra_NH_all, mlabel(r"N$-$H (all)",    intra_NH_all), "crimson",   LETTER_COLOUR, bins=bins_NH)
    ax.set_title(r"Intra $\cdot$ N$-$H of urea: bonded vs all")
    ax.set_xlabel(r"distance (\AA)")
    ax.set_ylabel("density")
    ax.legend()
    style_axes(ax, LETTER_COLOUR)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_intra_NH_compare{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

def two_peak_fit(data):
    arr = np.array(data).reshape(-1, 1)
    g = GaussianMixture(n_components=2, random_state=0).fit(arr)
    m = g.means_.flatten()
    s = np.sqrt(g.covariances_.flatten())
    w = g.weights_
    order = np.argsort(m)
    return [(m[i], s[i], w[i]) for i in order]

HH_peaks  = two_peak_fit(intra_HH)
dih_peaks = two_peak_fit(intra_HH_dih)

print(f"intra: N-H={len(intra_NH)}  N-Hall={len(intra_NH_all)}  "
      f"C-H(urea)={len(intra_CH_urea)}  H-H(CH2-CH2)={len(intra_HH)}  "
      f"C-H(CH2)={len(intra_CH_N)}+{len(intra_CH_O)}"
      f"  dihedral n={len(intra_HH_dih)}")

print("\nIntra · N-H of urea (bonded only):")
print(stats(intra_NH))
print("\nIntra · N-H of urea (all H):")
print(stats(intra_NH_all))
print("\nIntra · C-H of urea:")
print(stats(intra_CH_urea))
print("\nIntra · CH2-CH2 of choline:")
print(f"  H-H        {stats(intra_HH)}")
print(f"  C-H (N)    {stats(intra_CH_N)}")
print(f"  C-H (O)    {stats(intra_CH_O)}")
print(f"  H-H (ang)  {stats(intra_HH_dih, unit='rad')}")

print("\nIntra · H-H (CH2-CH2) two-Gaussian decomposition:")
for k, (m, s, w) in enumerate(HH_peaks):
    print(f"  peak {k + 1}: mean={m:.2f} Å  std={s:.2f} Å  weight={w:.2f}")

print("\nIntra · |H-C-C-H| dihedral (rad) two-Gaussian decomposition:")
for k, (m, s, w) in enumerate(dih_peaks):
    print(f"  peak {k + 1}: mean={m:.2f} rad  std={s:.2f} rad  weight={w:.2f}")
