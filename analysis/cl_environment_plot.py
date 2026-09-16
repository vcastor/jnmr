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

c       = load_cache("cl_environment")
d_site  = c["d_site"]
d_ncho  = c["d_ncho"]
d_clcl  = c["d_clcl"]
bcp     = c["bcp"]
contact = c["contact"]

BCP_COLOUR = {'H5': "seagreen", 'H4': "mediumpurple", 'H1': "steelblue",
              'H2': "darkorange", 'H3': "crimson"}

for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    hist(axes[0], d_site['H5'], mlabel(r"H$^{5}$", d_site['H5']), "seagreen",     LETTER_COLOUR)
    hist(axes[0], d_site['H4'], mlabel(r"H$^{4}$", d_site['H4']), "mediumpurple", LETTER_COLOUR)
    axes[0].axvline(contact, color=LETTER_COLOUR, ls=':', linewidth=1.2)
    axes[0].set_title(r"Cl$^-\cdots$H donors (nearest)")
    axes[0].set_xlabel(r"distance (\AA)")
    axes[0].set_ylabel("density")
    axes[0].legend(loc='upper right')

    hist(axes[1], d_site['H1'], mlabel(r"H$^{1}$", d_site['H1']), "steelblue",  LETTER_COLOUR)
    hist(axes[1], d_site['H2'], mlabel(r"H$^{2}$", d_site['H2']), "darkorange", LETTER_COLOUR)
    hist(axes[1], d_site['H3'], mlabel(r"H$^{3}$", d_site['H3']), "crimson",    LETTER_COLOUR)
    axes[1].set_title(r"Cl$^-\cdots$H choline C$-$H (nearest)")
    axes[1].set_xlabel(r"distance (\AA)")
    axes[1].set_ylabel("density")
    axes[1].legend(loc='upper right')

    hist(axes[2], d_ncho, mlabel(r"N$^+$", d_ncho), "steelblue", LETTER_COLOUR)
    hist(axes[2], d_clcl, mlabel(r"Cl$^-$", d_clcl), "tomato",   LETTER_COLOUR)
    axes[2].set_title(r"Cl$^-\cdots$N$^+$ and Cl$^-\cdots$Cl$^-$ (nearest)")
    axes[2].set_xlabel(r"distance (\AA)")
    axes[2].set_ylabel("density")
    axes[2].legend(loc='upper right')

    for ax in axes:
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax*1.15)
        style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/cl_distances{SUFFIX}", TRANSPARENT)

    if not bcp:
        continue
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, rows in sorted(bcp.items(), key=lambda x: -len(x[1])):
        d   = [r[0] for r in rows]
        rho = [r[1] for r in rows]
        ax.scatter(d, rho, s=18, alpha=0.6,
                   color=BCP_COLOUR.get(tag, "gray"), label=rf"{tag} (n={len(rows)})")
    ax.set_xlabel(r"distance (\AA)")
    ax.set_ylabel(r"$\rho_{BCP}$ (au)")
    ax.set_title(r"Cl$^-$ bond critical points")
    ax.legend(loc='upper right')
    style_axes(ax, LETTER_COLOUR, TRANSPARENT)
    fig.tight_layout()
    save_fig(fig, f"{PLOT_DIR}/cl_bcp{SUFFIX}", TRANSPARENT)
