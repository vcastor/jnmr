import numpy as np
import seaborn as sns

PLOT_STYLES = [('black', False, ''), ('white', True, '_transparent')]

def stats(data, unit='A'):
    if not data:
        return "n=0"
    d = np.array(data)
    u = 'Å' if unit == 'A' else 'rad'
    return f"n={len(d)}  mean={d.mean():.2f} {u}  std={d.std():.2f} {u}"

def mlabel(prefix, data):
    if not data:
        return rf"{prefix} (n=0)"
    return rf"{prefix} $\langle d\rangle={np.mean(data):.2f}\pm{np.std(data):.2f}$ \AA"

def hist(ax, data, label, colour, letter_colour, bins=40, stat='density'):
    if not data:
        return
    sns.histplot(data, bins=bins, kde=True, ax=ax, label=label, color=colour,
                 alpha=0.5, edgecolor=letter_colour,
                 line_kws={"linewidth": 2}, stat=stat)

def style_axes(ax, letter_colour):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(letter_colour)
    ax.tick_params(colors=letter_colour, which="both")
    ax.xaxis.label.set_color(letter_colour)
    ax.yaxis.label.set_color(letter_colour)
    ax.title.set_color(letter_colour)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor(letter_colour)
        for t in leg.get_texts():
            t.set_color(letter_colour)

def style_cbar(cbar, letter_colour):
    cbar.ax.tick_params(colors=letter_colour)
    cbar.ax.yaxis.label.set_color(letter_colour)
    cbar.outline.set_edgecolor(letter_colour)

