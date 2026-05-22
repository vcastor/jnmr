#!$AMSBIN/plams
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

PLOT_DIR     = "plots"
CLUSTERS_DIR = "clusters"

PLOT_STYLES = [('black', False, ''), ('white', True, '_transparent')]

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac}\usepackage{amsmath}'

def find_nh2(urea):
    pairs = []
    for n in urea.atoms:
        if n.symbol != 'N':
            continue
        for b in n.bonds:
            h = b.other_end(n)
            if h.symbol == 'H':
                pairs.append((n, h))
    return pairs

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

def dihedral(a, b, c, d):
    p0, p1, p2, p3 = (np.array(x.coords) for x in (a, b, c, d))
    b1, b2, b3 = p1-p0, p2-p1, p3-p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2/np.linalg.norm(b2))
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

intra_NH      = []
intra_NH_all  = []
intra_CH_urea = []
intra_HH      = []
intra_HH_dih  = []
intra_CH_N    = []
intra_CH_O    = []

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
        ns_u = [at for at in u.atoms if at.symbol == 'N']
        hs_u = [at for at in u.atoms if at.symbol == 'H']
        c_u  = [at for at in u.atoms if at.symbol == 'C'][0]
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

finish()

def stats(data, units="Å"):
    if not data:
        return "n=0"
    d = np.array(data)
    if units == 'rad':
        return f"n={len(d)}  mean={d.mean():.2f} rad  std={d.std():.2f} rad"
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

def style_cbar(cbar):
    cbar.ax.tick_params(colors=LETTER_COLOUR)
    cbar.ax.yaxis.label.set_color(LETTER_COLOUR)
    cbar.outline.set_edgecolor(LETTER_COLOUR)

DIH_TICKS  = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
DIH_LABELS = [r'$0$', r'$\sfrac{\pi}{4}$', r'$\sfrac{\pi}{2}$',
              r'$\sfrac{3\pi}{4}$', r'$\pi$']

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    hist(axes[0,0], intra_NH, mlabel(r"N$-$H (bonded)", intra_NH), "steelblue")
    axes[0,0].set_title(r"Intra $\cdot$ N$-$H of urea")
    axes[0,0].set_xlabel(r"distance (\AA)")
    axes[0,0].set_ylabel("density")
    axes[0,0].legend()

    hb = axes[0,1].hexbin(intra_HH_dih, intra_HH, gridsize=40, cmap="Oranges", mincnt=1)
    cbar = fig.colorbar(hb, ax=axes[0,1], label="count")
    style_cbar(cbar)
    axes[0,1].set_title(r"Intra $\cdot$ CH$_2$$-$CH$_2$")
    axes[0,1].set_xlabel(r"$\lvert$H-C-C-H$\rvert$ dihedral (rad)")
    axes[0,1].set_ylabel(r"H-H distance (\AA)")
    axes[0,1].set_xlim(0, np.pi)
    axes[0,1].set_xticks(DIH_TICKS)
    axes[0,1].set_xticklabels(DIH_LABELS)

    hist(axes[1,0], intra_CH_N, mlabel(r"CH$_2$$-$N", intra_CH_N), "seagreen")
    hist(axes[1,0], intra_CH_O, mlabel(r"CH$_2$$-$O", intra_CH_O), "purple")
    axes[1,0].set_title(r"Intra $\cdot$ C$-$H of each CH$_2$")
    axes[1,0].set_xlabel(r"distance (\AA)")
    axes[1,0].set_ylabel("density")
    axes[1,0].legend(loc='upper right')
    ymax = axes[1,0].get_ylim()[1]
    axes[1,0].set_ylim(0, ymax*1.45)

    hist(axes[1,1], intra_CH_urea, mlabel(r"C$-$H", intra_CH_urea), "indianred")
    axes[1,1].set_title(r"Intra $\cdot$ C$-$H of urea")
    axes[1,1].set_xlabel(r"distance (\AA)")
    axes[1,1].set_ylabel("density")
    axes[1,1].legend()

    for ax in axes.ravel():
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/distance_intra{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

    # extra: N-H bonded vs all
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins_NH = np.linspace(min(intra_NH_all), max(intra_NH_all), 41)
    hist(ax, intra_NH,     mlabel(r"N$-$H (bonded)", intra_NH),     "steelblue", bins=bins_NH)
    hist(ax, intra_NH_all, mlabel(r"N$-$H (all)",    intra_NH_all), "crimson",   bins=bins_NH)
    ax.set_title(r"Intra $\cdot$ N$-$H of urea: bonded vs all")
    ax.set_xlabel(r"distance (\AA)")
    ax.set_ylabel("density")
    ax.legend()
    style_axes(ax)
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
print(f"  H-H (ang)  {stats(intra_HH_dih, units='rad')}")

print("\nIntra · H-H (CH2-CH2) two-Gaussian decomposition:")
for k, (m, s, w) in enumerate(HH_peaks):
    print(f"  peak {k+1}: mean={m:.2f} Å  std={s:.2f} Å  weight={w:.2f}")

print("\nIntra · |H-C-C-H| dihedral (rad) two-Gaussian decomposition:")
for k, (m, s, w) in enumerate(dih_peaks):
    print(f"  peak {k+1}: mean={m:.2f} rad  std={s:.2f} rad  weight={w:.2f}")

