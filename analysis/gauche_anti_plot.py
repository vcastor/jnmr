#!/usr/bin/python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.plotting import PLOT_STYLES, hist, style_axes, save_fig
from hassan_functions.cache    import load_cache
from hassan_functions.style    import apply_style

apply_style("notex")

PLOT_DIR         = "plots"
GAUCHE_THRESHOLD = 2*np.pi/3

c              = load_cache("gauche_anti")
dihedrals      = c["dihedrals"]
n_systems_used = c["n_systems_used"]

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
    # colour tomato 🍅
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
    style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/gauche_anti{SUFFIX}", TRANSPARENT)
