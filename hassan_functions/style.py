import matplotlib.pyplot as plt

# Every plot script pulls its font sizes / LaTeX setup from here so the look is
# defined in one place. Pick a preset with apply_style("name").
PREAMBLE_FULL  = r'\usepackage{xfrac}\usepackage{amsmath}'
PREAMBLE_XFRAC = r'\usepackage{xfrac}'

PRESETS = {
    # usetex, LaTeX labels (distance/dihedral/CH plots)
    "default": {"font": 16, "title": 19, "label": 16, "tick": 14, "legend": 14,
                "usetex": True,  "preamble": PREAMBLE_FULL},
    # slightly larger, single-figure fits (karplus)
    "large":   {"font": 18, "title": 20, "label": 18, "tick": 16, "legend": 16,
                "usetex": True,  "preamble": PREAMBLE_XFRAC},
    # mathtext only, unicode labels such as Å (qtaim, gauche/anti)
    "notex":   {"font": 16, "title": 19, "label": 16, "tick": 14, "legend": 14,
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
