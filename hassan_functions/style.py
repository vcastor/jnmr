import matplotlib.pyplot as plt

PREAMBLE_FULL  = r'\usepackage{xfrac}\usepackage{amsmath}'
PREAMBLE_XFRAC = r'\usepackage{xfrac}'

PRESETS = {
    # usetex, LaTeX labels (distance/dihedral/CH plots)
    "default": {"font": 18, "title": 21, "label": 19, "tick": 17, "legend": 15,
                "usetex": True,  "preamble": PREAMBLE_FULL},
    # slightly larger, single-figure fits (karplus)
    "large":   {"font": 20, "title": 22, "label": 21, "tick": 19, "legend": 17,
                "usetex": True,  "preamble": PREAMBLE_XFRAC},
    # mathtext only, unicode labels such as Å (qtaim, gauche/anti)
    "notex":   {"font": 18, "title": 21, "label": 19, "tick": 17, "legend": 15,
                "usetex": False, "preamble": PREAMBLE_FULL},
}

def apply_style(preset="default", fonts=True):
    """Set matplotlib rcParams from a preset. fonts=False keeps only the usetex /
    LaTeX-preamble part and leaves matplotlib's default font sizes untouched (for
    dense multi-panel figures that would overflow at the preset sizes)."""
    s = PRESETS[preset]
    plt.rcParams["text.usetex"]         = s["usetex"]
    plt.rcParams["text.latex.preamble"] = s["preamble"]
    if fonts:
        plt.rcParams["font.size"]       = s["font"]
        plt.rcParams["axes.titlesize"]  = s["title"]
        plt.rcParams["axes.labelsize"]  = s["label"]
        plt.rcParams["xtick.labelsize"] = s["tick"]
        plt.rcParams["ytick.labelsize"] = s["tick"]
        plt.rcParams["legend.fontsize"] = s["legend"]
