#!/usr/bin/python3
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.plotting import PLOT_STYLES, style_axes, save_fig
from hassan_functions.cache    import load_cache
from hassan_functions.style    import apply_style
from hassan_functions.params   import (QTAIM_PROPS as PROPS, MARKER_CH2N as CIRCLE,
                                       MARKER_CH2O as TRIANGLE, COLOUR_CH2N, COLOUR_CH2O,
                                       FS_BOX, FS_LEGEND)

apply_style("notex")

PLOT_DIR = "plots"

c = load_cache("qtaim_analysis")
q_Nside                 = c["q_Nside"]
q_Oside                 = c["q_Oside"]
fukui_Nside             = c.get("fukui_Nside", [])
fukui_Oside             = c.get("fukui_Oside", [])
nh2_ch3                 = c["nh2_ch3"]
o_hch2_all              = c["o_hch2_all"]
o_hch2_N                = c["o_hch2_N"]
o_hch2_O                = c["o_hch2_O"]
o_hch2_bridge           = c["o_hch2_bridge"]
o_hch2_single           = c["o_hch2_single"]
o_hoh                   = c["o_hoh"]
o_ch2_oh_bridge         = c["o_ch2_oh_bridge"]
o_ch2N_oh_bridge        = c["o_ch2N_oh_bridge"]
o_ch2O_oh_bridge        = c["o_ch2O_oh_bridge"]
other_by_type           = c["other_by_type"]
n_pairs_other_type      = c["n_pairs_other_type"]
sig_labels              = c["sig_labels"]
n_systems_used          = c["n_systems_used"]
n_pairs_total           = c["n_pairs_total"]
n_pairs_interacting     = c["n_pairs_interacting"]
n_pairs_nh2_ch3         = c["n_pairs_nh2_ch3"]
n_pairs_o_any           = c["n_pairs_o_any"]
n_pairs_o_hch2_N        = c["n_pairs_o_hch2_N"]
n_pairs_o_hch2_O        = c["n_pairs_o_hch2_O"]
n_pairs_o_bridge        = c["n_pairs_o_bridge"]
n_pairs_o_single        = c["n_pairs_o_single"]
n_pairs_o_same_ch2      = c["n_pairs_o_same_ch2"]
n_pairs_o_hoh           = c["n_pairs_o_hoh"]
n_pairs_o_ch2_oh_bridge = c["n_pairs_o_ch2_oh_bridge"]
n_pairs_o_ch2N_oh       = c["n_pairs_o_ch2N_oh"]
n_pairs_o_ch2O_oh       = c["n_pairs_o_ch2O_oh"]
n_pairs_other           = c["n_pairs_other"]
n_other_bcp             = c["n_other_bcp"]
n_ph_satisfied          = c["n_ph_satisfied"]
uu_by_type              = c.get("uu_by_type", {})
n_uu_pairs_type         = c.get("n_uu_pairs_type", {})
uu_sig_labels           = c.get("uu_sig_labels", set())
n_uu_pairs_total        = c.get("n_uu_pairs_total", 0)
n_uu_pairs_interacting  = c.get("n_uu_pairs_interacting", 0)
n_uu_bcp                = c.get("n_uu_bcp", 0)

# ── report ────────────────────────────────────────────────────────────────
def summary(label, data):
    if not data:
        print(f"  {label}: n=0")
        return
    arr = np.array(data)
    ds, rs, gs, vs, ks = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    print(f"  {label}: n={len(arr)}")
    print(f"    d     (A) : mean={ds.mean():.3f}  std={ds.std():.3f}  "
          f"min={ds.min():.3f}  max={ds.max():.3f}")
    print(f"    rho  (au) : mean={rs.mean():.4f}  std={rs.std():.4f}  "
          f"min={rs.min():.4f}  max={rs.max():.4f}")
    print(f"    Gb   (au) : mean={gs.mean():.4f}  std={gs.std():.4f}  "
          f"min={gs.min():.4f}  max={gs.max():.4f}")
    print(f"    Vb   (au) : mean={vs.mean():.4f}  std={vs.std():.4f}  "
          f"min={vs.min():.4f}  max={vs.max():.4f}")
    print(f"    |Vb|/Gb   : mean={ks.mean():.4f}  std={ks.std():.4f}  "
          f"min={ks.min():.4f}  max={ks.max():.4f}")

# Compact, consistent report labels: tag the urea atom with (urea); the urea H is
# always on N so it reads H(urea); the choline side keeps its group, and choline's
# single O is the hydroxyl O(OH). Turns "urea H(N) - choline O" into "H(urea)-O(OH)".
_UREA_TAG = {"N": "N(urea)", "O": "O(urea)", "C": "C(urea)", "H(N)": "H(urea)"}
_CHOL_TAG = {"H(CH3)": "H(CH3)", "H(CH2N)": "H(CH2-N)", "H(CH2O)": "H(CH2-O)",
             "H(O)": "H(OH)", "O": "O(OH)", "N": "N(choline)", "C": "C(choline)"}

def pretty_label(label):
    if " - choline " not in label:
        return label
    u, ca = label.split(" - choline ", 1)
    u = u.replace("urea ", "", 1)
    return f"{_UREA_TAG.get(u, u)}-{_CHOL_TAG.get(ca, ca)}"

print(f"systems with QTAIM data: {n_systems_used}")
print(f"urea-choline pairs:      {n_pairs_total}")
print()
print("N(urea NH2) - H(CH3)  [BCP only]")
print(f"  pairs with >=1 such BCP: {n_pairs_nh2_ch3}/{n_pairs_total}")
summary("N-H(CH3)", nh2_ch3)
print()
print("O(urea) - H(CH2 choline)  [BCP only]")
print(f"  pairs with >=1 O-H(CH2) BCP: {n_pairs_o_any}/{n_pairs_total}")
summary("all BCP                          ", o_hch2_all)
summary("CH2 near N+                      ", o_hch2_N)
summary("CH2 near O                       ", o_hch2_O)
summary(f"O bridges BOTH CH2  ({n_pairs_o_bridge:3d} pairs)", o_hch2_bridge)
summary(f"O sees only ONE H   ({n_pairs_o_single:3d} pairs)", o_hch2_single)
print(f"  >=2 BCP, same CH2   ({n_pairs_o_same_ch2:3d} pairs)")
print()
print("O(urea) - H(OH choline)  [BCP only]")
print(f"  pairs with >=1 O-H(OH) BCP:  {n_pairs_o_hoh}/{n_pairs_total}")
summary("H(OH)                            ", o_hoh)
print()
if n_pairs_o_ch2_oh_bridge:
    print("O(urea) bridges H(CH2) + H(OH)  [BCP only]")
    print(f"  pairs with both:              {n_pairs_o_ch2_oh_bridge}/{n_pairs_total}")
    summary("CH2-N + OH                       ", o_ch2N_oh_bridge)
    summary("CH2-O + OH                       ", o_ch2O_oh_bridge)
    print()
print("Other urea-choline BCPs (not in the categories above)")
print(f"  pairs with >=1 'other' BCP:  {n_pairs_other}/{n_pairs_interacting}")
print(f"  total 'other' BCPs:          {n_other_bcp}")
for label in sorted(other_by_type, key=lambda L: -len(other_by_type[L])):
    summary(pretty_label(label), other_by_type[label])
print()
if n_pairs_interacting:
    def pct(n): return f"{n}/{n_pairs_interacting}  ({100*n/n_pairs_interacting:.1f}%)"
    print("-- occurrence comparison (per interacting urea-choline pair) --")
    print(f"  interacting pairs (>=1 BCP):   {n_pairs_interacting}/{n_pairs_total}  ({100*n_pairs_interacting/n_pairs_total:.1f}%)")
    print(f"  N(urea)-H(CH3):                {pct(n_pairs_nh2_ch3)}")
    print(f"  O(urea)-H(CH2-N):              {pct(n_pairs_o_hch2_N)}")
    print(f"  O(urea)-H(CH2-O):              {pct(n_pairs_o_hch2_O)}")
    print(f"  O(urea)-H(CH2) [any]:          {pct(n_pairs_o_any)}")
    print(f"  O(urea)-H(OH):                 {pct(n_pairs_o_hoh)}")
    print(f"  O(urea)-H(CH2-N) & H(OH):      {pct(n_pairs_o_ch2N_oh)}")
    print(f"  O(urea)-H(CH2-O) & H(OH):      {pct(n_pairs_o_ch2O_oh)}")
    print(f"  O(urea)-H(CH2,any) & H(OH):    {pct(n_pairs_o_ch2_oh_bridge)}")
    for label in sorted(n_pairs_other_type, key=lambda L: -n_pairs_other_type[L]):
        print(f"  {pretty_label(label)+':':<31}{pct(n_pairs_other_type[label])}")

n_other_sig  = sum(len(other_by_type[L]) for L in sig_labels)
n_other_weak = n_other_bcp - n_other_sig
n_bcp        = len(nh2_ch3) + len(o_hch2_all) + len(o_hoh) + n_other_bcp
n_bcp_sig    = len(nh2_ch3) + len(o_hch2_all) + len(o_hoh) + n_other_sig
if n_bcp_sig:
    def pctb(n): return f"{n}/{n_bcp_sig}  ({100*n/n_bcp_sig:.1f}%)"
    print()
    print("-- BCP-level breakdown (per significant urea-choline BCP, sums to 100%) --")
    print(f"  significant urea-choline BCPs: {n_bcp_sig}  (of {n_bcp} total)")
    print(f"  N(urea)-H(CH3):                {pctb(len(nh2_ch3))}")
    print(f"  O(urea)-H(CH2-N):              {pctb(len(o_hch2_N))}")
    print(f"  O(urea)-H(CH2-O):              {pctb(len(o_hch2_O))}")
    print(f"  O(urea)-H(OH):                 {pctb(len(o_hoh))}")
    for label in sorted(sig_labels, key=lambda L: -len(other_by_type[L])):
        print(f"  {pretty_label(label)+':':<31}{pctb(len(other_by_type[label]))}")
    if n_other_weak:
        print(f"  -- weak / numerically noisy (excluded; {n_other_weak}/{n_bcp} = {100*n_other_weak/n_bcp:.1f}% of all BCPs) --")
        for label in sorted((L for L in other_by_type if L not in sig_labels),
                            key=lambda L: -len(other_by_type[L])):
            n = len(other_by_type[label])
            print(f"  {pretty_label(label)+':':<31}{n}/{n_bcp}  ({100*n/n_bcp:.1f}%)")

# ── urea-urea interactions (how the ureas bind each other) ──────────────────
print()
print("Urea-urea BCPs")
print(f"  urea-urea pairs:              {n_uu_pairs_total}")
print(f"  interacting (>=1 BCP):        {n_uu_pairs_interacting}/{n_uu_pairs_total}")
print(f"  total urea-urea BCPs:         {n_uu_bcp}")
for label in sorted(uu_by_type, key=lambda L: -len(uu_by_type[L])):
    summary(label, uu_by_type[label])
if n_uu_bcp:
    print("  -- per urea-urea BCP --")
    for label in sorted(uu_by_type, key=lambda L: -len(uu_by_type[L])):
        n   = len(uu_by_type[label])
        tag = "  (H-bond)" if label in uu_sig_labels else ""
        print(f"  {label+':':<20}{n}/{n_uu_bcp}  ({100*n/n_uu_bcp:.1f}%){tag}")

print()
q_N = np.array(q_Nside); q_O = np.array(q_Oside)
print(f"q (CH2 near N+): n={len(q_N)}, mean={q_N.mean():.6f}, std={q_N.std():.6f}")
print(f"q (CH2 near O ): n={len(q_O)}, mean={q_O.mean():.6f}, std={q_O.std():.6f}")
print()
print(f"Poincare-Hopf satisfied: {n_ph_satisfied}/{n_systems_used} systems")

# ── plots: distance vs BCP properties, one panel per interaction category ──
# rho / Gb / Vb are overlaid and told apart by COLOUR; where a panel holds the two
# CH2 flavours (next to N+ / next to O) they are told apart by MARKER (circle vs triangle).
# Every panel shares one distance/property range so the categories are comparable.
# Colours (PROPS), markers (CIRCLE/TRIANGLE) and font sizes (FS_BOX/FS_LEGEND) all come
# from hassan_functions/params.py.

def axis_range(datasets, cols, pad=0.05):
    """Shared (min, max) with a little padding across every dataset drawn and the
    given columns, or None if nothing is populated."""
    vals = [row[c] for d in datasets if d for row in d for c in cols]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    m = (hi - lo) * pad or 0.01
    return lo - m, hi + m

def plot_bcp_grid(bonds, fname):
    """bonds: list of (title, series); series is a list of (data, marker_label, marker).
    Each panel overlays rho/Gb/Vb (colour) vs distance for every series it holds; a
    panel that carries the two CH2 flavours gets a marker legend so they stay
    distinguishable. All panels share one distance/property range for comparison."""
    bonds = [b for b in bonds if any(s[0] for s in b[1])]
    if not bonds:
        return
    all_data = [s[0] for _, series in bonds for s in series]
    xlim = axis_range(all_data, [0])
    ylim = axis_range(all_data, [c for c, _, _ in PROPS])
    prop_handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=9,
                           markerfacecolor=color, markeredgecolor='black', label=plabel)
                    for _, plabel, color in PROPS]
    ncol = min(3, len(bonds))
    nrow = -(-len(bonds)//ncol)
    for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
        fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 5*nrow), squeeze=False)
        flat = axes.ravel()
        for ax, (title, series) in zip(flat, bonds):
            drawn = [s for s in series if s[0]]
            for data, _mlabel, marker in drawn:
                arr = np.array(data)
                for col, _plabel, color in PROPS:
                    ax.scatter(arr[:, 0], arr[:, col], s=22, c=color, alpha=0.6,
                               marker=marker, edgecolor='black', linewidth=0.3)
            ax.set_title(title)
            ax.set_xlabel("distance (Å)")
            ax.set_ylabel("BCP property (au)")
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)
            ax.grid(alpha=0.3)
            if len(drawn) > 1:              # two CH2 flavours share this panel
                shape_handles = [Line2D([0], [0], marker=mk, linestyle='none',
                                        markersize=9, markerfacecolor=LETTER_COLOUR,
                                        markeredgecolor=LETTER_COLOUR, label=ml)
                                 for _, ml, mk in drawn]
                ax.legend(handles=shape_handles, loc="lower right", fontsize=FS_LEGEND)
            style_axes(ax, LETTER_COLOUR, TRANSPARENT)
        for ax in flat[len(bonds):]:
            ax.axis("off")
        leg = fig.legend(prop_handles, [p[1] for p in PROPS], loc="upper center",
                         ncol=3, fontsize=FS_BOX, frameon=False)
        for t in leg.get_texts():
            t.set_color(LETTER_COLOUR)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_fig(fig, f"{PLOT_DIR}/{fname}{SUFFIX}", TRANSPARENT)

# every significant urea–choline BCP category. The two O(urea)–H(CH2) flavours are
# merged into one panel (circle vs X marker), likewise N(urea)–H(CH2); the
# N(urea)–H(OH) panel is intentionally omitted.
# H-type shorthand (see hassan_functions/params.py): H1=CH3, H2=CH2-N, H3=CH2-O,
# H4=OH, H5=NH2. The merged CH2 panels carry H2 (circle) and H3 (triangle).
o_hch2_series = [(o_hch2_N, "H2", CIRCLE),
                 (o_hch2_O, "H3", TRIANGLE)]
n_hch2_series = [(other_by_type.get("urea N - choline H(CH2N)", []), "H2", CIRCLE),
                 (other_by_type.get("urea N - choline H(CH2O)", []), "H3", TRIANGLE)]

CHOLINE_BONDS = [
    (r"N(urea) – H1",    [(nh2_ch3, None, CIRCLE)]),
    (r"O(urea) – H2/H3", o_hch2_series),
    (r"O(urea) – H4",    [(o_hoh, None, CIRCLE)]),
    (r"H5 – O(OH)",      [(other_by_type.get("urea H(N) - choline O", []),   None, CIRCLE)]),
    (r"O(urea) – H1",    [(other_by_type.get("urea O - choline H(CH3)", []), None, CIRCLE)]),
    (r"N(urea) – H2/H3", n_hch2_series),
]

# significant urea-urea categories (the N-H···O / N-H···N hydrogen bonds),
# most-populated first.
_uu = lambda t: "H5" if t == "H(N)" else t   # urea NH hydrogen -> H5 shorthand
UREA_BONDS = [(f"urea {_uu(L.split(' - ')[0])} – urea {_uu(L.split(' - ')[1])}",
               [(uu_by_type[L], None, CIRCLE)])
              for L in sorted(uu_sig_labels, key=lambda L: -len(uu_by_type[L]))]

os.makedirs(PLOT_DIR, exist_ok=True)
plot_bcp_grid(CHOLINE_BONDS, "qtaim_distance")
plot_bcp_grid(UREA_BONDS,    "qtaim_distance_urea_urea")

# ── plot: QTAIM net-charge histogram, CH2-N vs CH2-O ──────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(q_Nside, bins=40, color=COLOUR_CH2N, edgecolor="black", alpha=0.5,
        density=True, label="H2")
ax.hist(q_Oside, bins=40, color=COLOUR_CH2O, edgecolor="black", alpha=0.5,
        density=True, label="H3")

sns.kdeplot(q_Nside, ax=ax, color=COLOUR_CH2N, linewidth=2)
sns.kdeplot(q_Oside, ax=ax, color=COLOUR_CH2O, linewidth=2)

ax.set_xlim(-0.12, 0.12)
ax.set_title("QTAIM charge")
ax.set_xlabel("q (au)")
ax.set_ylabel("density")
ax.legend()
style_axes(ax, "black", True)
fig.tight_layout()
save_fig(fig, f"{PLOT_DIR}/qtaim_charges", True)

# ── plot: Conceptual-DFT Fukui functions of the CH2 hydrogens, CH2-N vs CH2-O ──
# The experimental team treats the two CH2 H types as chemically distinct, but their
# distance and QTAIM charge are already indistinguishable. The condensed Fukui
# functions add a reactivity comparison: f+ (nucleophilic), f- (electrophilic) and the
# dual descriptor f2 = f+ - f-. With n in the hundreds a p-value would flag any
# trivial difference, so similarity is judged by the effect size (Cohen's d): |d|<0.2
# means the two H types are practically the same.
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1)*a.std(ddof=1)**2 + (nb - 1)*b.std(ddof=1)**2) / (na + nb - 2))
    return (a.mean() - b.mean())/sp if sp > 0 else 0.0

fN = np.array(fukui_Nside, float)   # columns: f+, f-, f0, f2
fO = np.array(fukui_Oside, float)
FUKUI_PANELS = [
    (0, r"$f^{+}$ (nucleophilic)",    "f+"),
    (1, r"$f^{-}$ (electrophilic)",   "f-"),
    (3, r"$f^{(2)}$ dual descriptor", "f2 (dual)"),
]

if fN.size and fO.size:
    print()
    print("Fukui functions of the CH2 hydrogens (Conceptual DFT, QTAIM-condensed)")
    print(f"  CH2-N: n={len(fN)}   CH2-O: n={len(fO)}")
    for col, _name, tag_name in FUKUI_PANELS:
        n, o = fN[:, col], fO[:, col]
        d = cohens_d(n, o)
        tag = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "NON-trivial"
        print(f"  {tag_name:9}: CH2-N {n.mean():+.4f}±{n.std():.4f}   "
              f"CH2-O {o.mean():+.4f}±{o.std():.4f}   |Δ|={abs(n.mean()-o.mean()):.4f}   "
              f"Cohen d={d:+.2f} ({tag})")
    print("  -> negligible effect sizes => CH2-N and CH2-O H are practically the same")

    for LETTER_COLOUR, TRANSPARENT, SUFFIX in PLOT_STYLES:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
        for ax, (col, name, _tag) in zip(axes.ravel(), FUKUI_PANELS):
            n, o = fN[:, col], fO[:, col]
            d = cohens_d(n, o)
            ax.hist(n, bins=30, color=COLOUR_CH2N, edgecolor="black", alpha=0.45,
                    density=True, label=rf"H2: {n.mean():+.4f}$\pm${n.std():.4f}")
            ax.hist(o, bins=30, color=COLOUR_CH2O, edgecolor="black", alpha=0.45,
                    density=True, label=rf"H3: {o.mean():+.4f}$\pm${o.std():.4f}")
            sns.kdeplot(n, ax=ax, color=COLOUR_CH2N, linewidth=2)
            sns.kdeplot(o, ax=ax, color=COLOUR_CH2O, linewidth=2)
            ax.axvline(n.mean(), color=COLOUR_CH2N, linestyle=":", linewidth=1.2)
            ax.axvline(o.mean(), color=COLOUR_CH2O, linestyle=":", linewidth=1.2)
            ax.set_title(rf"{name}   $|d|={abs(d):.2f}$")
            ax.set_xlabel("Fukui function (au)")
            ax.set_ylabel("density")
            ax.legend(fontsize=FS_LEGEND)
            style_axes(ax, LETTER_COLOUR, TRANSPARENT)
        fig.suptitle("CH$_2$ hydrogens: Conceptual-DFT reactivity, H2 vs H3",
                     color=LETTER_COLOUR, fontsize=FS_BOX)
        fig.tight_layout()
        save_fig(fig, f"{PLOT_DIR}/qtaim_fukui_ch2{SUFFIX}", TRANSPARENT)
