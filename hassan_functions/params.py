# Shared plotting / analysis parameters, so the look and the analysis choices live
# in one place instead of being hard-coded across the individual analysis scripts.
# (Global matplotlib font/usetex presets live alongside this, in style.py.)

# ── font sizes (per-call overrides on top of the style.py rcParams defaults) ──
# The "label box" at the top of a multi-panel figure (e.g. the rho/Gb/Vb key on the
# QTAIM distance plots) is enlarged to sit at the panel-title weight, so it reads the
# same as the rest of the figure text — matching the other plots.
FS_BOX    = 18   # figure-level label-box / key legends
FS_LEGEND = 14   # in-axes legends

# ── QTAIM distance plots ──────────────────────────────────────────────────────
# rho / Gb / Vb are encoded by COLOUR; the two CH2 flavours by MARKER (circle vs X).
# Each entry is (column of the property in a (d, rho, Gb, Vb, ratio) row, LaTeX
# label, colour).
QTAIM_PROPS = [
    (1, r"$\rho_{BCP}$ (au)", "steelblue"),
    (2, r"$G_b$ (au)",        "seagreen"),
    (3, r"$V_b$ (au)",        "crimson"),
]
MARKER_CH2N = 'o'   # CH2 next to N+  (circle)
MARKER_CH2O = '^'   # CH2 next to O   (triangle; an X was near-invisible in print)

# CH2-type colours (QTAIM charge histogram)
COLOUR_CH2N = "seagreen"
COLOUR_CH2O = "mediumpurple"

# ── hydrogen-type shorthand used in the distance / QTAIM plot labels ──────────
# Numbered per the molecular scheme so the figures stay compact:
#   H1 = H(CH3)    choline methyl
#   H2 = H(CH2-N)  choline CH2 next to N+
#   H3 = H(CH2-O)  choline CH2 next to O
#   H4 = H(OH)     choline hydroxyl
#   H5 = H(NH2)    urea amino
H_LABEL = {"H(CH3)": "H1", "H(CH2N)": "H2", "H(CH2O)": "H3", "H(OH)": "H4", "H(NH2)": "H5"}

# ── numerical-sanity cutoff on |J| ───────────────────────────────────────────
# Unconverged / ill-behaved SCF can emit couplings of hundreds to >1000 Hz for these
# through-space contacts, which are physically meaningless. The reader already flags
# SCF-warned steps (comment_{variant}); this is a second line of defence for the C-H
# and N-H analyses — a converged-but-garbage step can still slip a |J| well over
# 100 Hz past the flag, so anything above this ceiling is dropped from the stats/plots.
J_PHYSICAL_MAX_HZ = 100.0

