#!$AMSBIN/plams
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance, dihedral
from hassan_functions.finders   import find_xh_bonds, find_adjacent_xh_pair_anchored
from hassan_functions.plotting  import PLOT_STYLES, hist, mlabel, stats, style_axes, style_cbar
from hassan_functions.constants import FORMULAS

PLOT_DIR         = "plots"
CLUSTERS_DIR     = "clusters"
GAUCHE_THRESHOLD = 2*np.pi/3

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac}\usepackage{amsmath}'
plt.rcParams['font.size']       = 16
plt.rcParams['axes.titlesize']  = 19
plt.rcParams['axes.labelsize']  = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

intra_NH      = []
intra_CH_urea = []
intra_HH      = []
intra_HH_dih  = []
intra_CH_N    = []
intra_CH_O    = []
dihedrals     = []

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
        c_u  = next(at for at in u.atoms if at.symbol == 'C')
        hs_u = [at for at in u.atoms if at.symbol == 'H']
        for n, h in find_xh_bonds(u, 'N'):
            intra_NH.append(distance(n, h))
        for h in hs_u:
            intra_CH_urea.append(distance(c_u, h))

    for ch in cholines:
        pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
        if pair is None:
            continue
        c_N, hN, c_O, hO = pair
        for a in hN:
            for b in hO:
                intra_HH.append(distance(a, b))
                intra_HH_dih.append(abs(dihedral(a, c_N, c_O, b)))
        for h in hN:
            intra_CH_N.append(distance(c_N, h))
        for h in hO:
            intra_CH_O.append(distance(c_O, h))
        n_at = next((b.other_end(c_N) for b in c_N.bonds if b.other_end(c_N).symbol == 'N'), None)
        o_at = next((b.other_end(c_O) for b in c_O.bonds if b.other_end(c_O).symbol == 'O'), None)
        if n_at is not None and o_at is not None:
            dihedrals.append(abs(dihedral(n_at, c_N, c_O, o_at)))

finish()

# ── stats ─────────────────────────────────────────────────────────────────
nh_arr   = np.array(intra_NH)
ch_arr   = np.array(intra_CH_urea)
d_arr    = np.array(dihedrals)
n_tot    = len(d_arr)
n_gauche = int(np.sum(d_arr <= GAUCHE_THRESHOLD))
n_anti   = n_tot - n_gauche

print(f"N H (bonded):  {nh_arr.mean():.3f} +/- {nh_arr.std():.3f} A  (n={len(nh_arr)})")
print(f"C H (urea):    {ch_arr.mean():.3f} +/- {ch_arr.std():.3f} A  (n={len(ch_arr)})")
print(f"C H (CH2 N):   {stats(intra_CH_N)}")
print(f"C H (CH2 O):   {stats(intra_CH_O)}")
print(f"H H (CH2-CH2): {stats(intra_HH)}")
print(f"H C C H dihedral (rad): {stats(intra_HH_dih, unit='rad')}")
print()
print(f"N-CH2-CH2-O dihedral  (n={n_tot})")
print(f"  gauche (phi <= {GAUCHE_THRESHOLD:.4f} rad): {n_gauche}  ({100*n_gauche/n_tot:.1f}%)")
print(f"  anti   (phi >  {GAUCHE_THRESHOLD:.4f} rad): {n_anti}  ({100*n_anti/n_tot:.1f}%)")

def two_peak_fit(data):
    arr = np.array(data).reshape(-1, 1)
    g = GaussianMixture(n_components=2, random_state=0).fit(arr)
    m = g.means_.flatten()
    s = np.sqrt(g.covariances_.flatten())
    w = g.weights_
    order = np.argsort(m)
    return [(m[i], s[i], w[i]) for i in order]

print("\nH H (CH2-CH2) two-Gaussian decomposition:")
for k, (m, s, w) in enumerate(two_peak_fit(intra_HH)):
    print(f"  peak {k + 1}: mean={m:.3f} A  std={s:.3f} A  weight={w:.2f}")

print("\nH C C H dihedral two-Gaussian decomposition:")
for k, (m, s, w) in enumerate(two_peak_fit(intra_HH_dih)):
    print(f"  peak {k + 1}: mean={m:.3f} rad  std={s:.3f} rad  weight={w:.2f}")

# ── plots ─────────────────────────────────────────────────────────────────
DIH_TICKS  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
DIH_LABELS = [r'$0$', r'$\sfrac{\pi}{4}$', r'$\sfrac{\pi}{2}$',
              r'$\sfrac{3\pi}{4}$', r'$\pi$']

NCCO_TICKS  = [0, np.pi/3, 2*np.pi/3, np.pi]
NCCO_LABELS = [r'$0$', r'$\sfrac{\pi}{3}$', r'$\sfrac{2\pi}{3}$', r'$\pi$']

os.makedirs(PLOT_DIR, exist_ok=True)

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # ── panel 0: C H of each CH2 ──────────────────────────────────────────
    hist(axes[0], intra_CH_N, mlabel(r"CH$_2$ N", intra_CH_N), "seagreen", LETTER_COLOUR)
    hist(axes[0], intra_CH_O, mlabel(r"CH$_2$ O", intra_CH_O), "purple",   LETTER_COLOUR)
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
        style_axes(ax, LETTER_COLOUR)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_intra{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

