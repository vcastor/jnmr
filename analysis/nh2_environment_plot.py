#!/usr/bin/python3
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.plotting import PLOT_STYLES, style_axes, save_fig, hist, mlabel
from hassan_functions.style    import apply_style
from hassan_functions.cache    import load_cache

apply_style("default")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

c       = load_cache("nh2_environment")
d_acc   = c["d_acc"]
bcp     = c["bcp"]
contact = c["contact"]

COLOUR = {'Cl': "seagreen", 'Och': "mediumpurple", 'Ou': "steelblue",
          'H1': "darkorange", 'H2': "crimson", 'H3': "tomato", 'H4': "gray",
          'H5': "black"}
LABEL  = {'Cl': r"Cl$^-$", 'Och': r"O(choline)", 'Ou': r"O(urea)", 'H1': r"H1"}

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for k in ('Cl', 'Och', 'Ou'):
        hist(axes[0], d_acc[k], mlabel(LABEL[k], d_acc[k]), COLOUR[k], LETTER_COLOUR)
    axes[0].axvline(contact, color=LETTER_COLOUR, ls=':', linewidth=1.2)
    axes[0].set_title(r"H5$\cdots$acceptor (nearest)")
    axes[0].set_xlabel(r"distance (\AA)")
    axes[0].set_ylabel("density")
    axes[0].legend(loc='upper right')

    hist(axes[1], d_acc['H1'], mlabel(LABEL['H1'], d_acc['H1']), COLOUR['H1'], LETTER_COLOUR)
    axes[1].set_title(r"H5$\cdots$H1 (nearest, the NH$_2$-CH$_3$ contact)")
    axes[1].set_xlabel(r"distance (\AA)")
    axes[1].set_ylabel("density")
    axes[1].legend(loc='upper right')

    for ax in axes:
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax*1.15)
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/nh2_distances{SUFFIX}", TRANSPARENT)

    if not bcp:
        continue
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, rows in sorted(bcp.items(), key=lambda x: -len(x[1])):
        d   = [r[0] for r in rows]
        rho = [r[1] for r in rows]
        ax.scatter(d, rho, s=18, alpha=0.6,
                   color=COLOUR.get(tag, "gray"), label=rf"{tag} (n={len(rows)})")
    ax.set_xlabel(r"distance (\AA)")
    ax.set_ylabel(r"$\rho_{BCP}$ (au)")
    ax.set_title(r"H5 bond critical points")
    ax.legend(loc='upper right')
    style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/nh2_bcp{SUFFIX}", TRANSPARENT)
