#!$AMSBIN/plams
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from hassan_functions.geometry import dihedral
from hassan_functions.finders import find_adjacent_xh_pair_anchored
from hassan_functions.ordering import classify_sort, compute_offsets
from hassan_functions.constants import FORMULAS
from hassan_functions.plotting import PLOT_STYLES, hist, style_axes

CLUSTERS_DIR     = "clusters"
PLOT_DIR         = "plots"
SPECIES          = ['urea', 'choline', 'chloride']
GAUCHE_THRESHOLD = 2*np.pi / 3   # rad  (~120°)


def ncco_dihedral(ch):
    pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
    if pair is None:
        return None
    c1, _, c2, _ = pair
    n_at = next((b.other_end(c1) for b in c1.bonds if b.other_end(c1).symbol == 'N'), None)
    o_at = next((b.other_end(c2) for b in c2.bonds if b.other_end(c2).symbol == 'O'), None)
    if n_at is None or o_at is None:
        return None
    return dihedral(n_at, c1, c2, o_at)


dihedrals      = []
n_systems_used = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    n_systems_used += 1

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})

    for ch in mol_data['choline']:
        d = ncco_dihedral(ch)
        if d is not None:
            dihedrals.append(abs(d))

finish()

d_arr    = np.array(dihedrals)
n_tot    = len(d_arr)
n_gauche = int(np.sum(d_arr <= GAUCHE_THRESHOLD))
n_anti   = n_tot - n_gauche

print(f"snapshots:    {n_systems_used}")
print(f"dihedrals:    {n_tot}")
print(f"  gauche (phi <= {GAUCHE_THRESHOLD:.4f} rad):  "
      f"{n_gauche}  ({100*n_gauche/n_tot:.1f}%)")
print(f"  anti   (phi >  {GAUCHE_THRESHOLD:.4f} rad):  "
      f"{n_anti}  ({100*n_anti/n_tot:.1f}%)")

os.makedirs(PLOT_DIR, exist_ok=True)

ticks      = [0, np.pi/3, 2*np.pi/3, np.pi]
ticklabels = [r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$']

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.axvspan(0, GAUCHE_THRESHOLD, alpha=0.12, color='steelblue', label='gauche')
    ax.axvspan(GAUCHE_THRESHOLD, np.pi, alpha=0.12, color='tomato', label='anti')

    hist(ax, list(d_arr), None, 'steelblue', LETTER_COLOUR, bins=40)

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_xlim(0, np.pi)
    ax.set_xlabel(r'$|\phi|$ (rad)')
    ax.set_ylabel('density')
    ax.set_title(r'N–CH$_2$–CH$_2$–O dihedral angle')
    ax.legend(fontsize=14)
    style_axes(ax, LETTER_COLOUR)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/gauche_anti{SUFFIX}.pdf", transparent=TRANSPARENT)
    plt.close(fig)

