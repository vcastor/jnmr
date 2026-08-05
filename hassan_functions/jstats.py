import numpy as np

# "Cubic" (power-mean) average used to collapse an ensemble of |J| couplings into a
# single effective value comparable to experiment. The exponent up-weights the larger
# couplings, which dominate the measured signal.
#
# p = 3 — a genuine cubic mean, matching the name. The two couplings pick it out
# independently: on the well-sampled TZ2P_FC, the exponent reproducing experiment is
# p*=2.98 for intra (CH2-CH2, 5.90 Hz) and p*=3.08 for inter (NH2-CH3, 1.104 Hz). At
# p=3 exactly they give 5.917 Hz (+0.3%) and 1.065 Hz (-3.5%). Minimising RMS relative
# deviation over the four intra variants plus inter TZ2P_FC puts the optimum at p=3.10
# (9.4%), with p=3 at 9.8%, against 17.9% at p=2.5 and 29.4% at p=2.
#
# That agreement only appears on CORRECT data. The previous value, 2.25, was fitted to
# the inter couplings while ~33% of the stored inter J values were wrong — including 79%
# of those above 1 Hz, which are exactly the ones a power mean weights. Those inflated
# values pulled the apparent inter optimum down to 2.25 and made the two couplings
# disagree. With the DB rebuilt from the ADF outputs the disagreement disappears.
#
# Note this is still a calibration, not a derivation. In fast exchange the NMR observable
# is the arithmetic mean of the SIGNED coupling (p=1), which gives 3.45 Hz intra against
# 5.90 Hz measured; a Karplus fit over the full 3309-snapshot geometry ensemble returns
# the same 3.44 Hz, so that shortfall is a level-of-theory deficiency, not undersampling.
# p>1 absorbs it. What changed is that one exponent now absorbs it consistently for two
# chemically unrelated couplings, which it did not before.
CUBIC_P = 3.0

def cubic_mean(vals, p=CUBIC_P):
    return float((np.mean(vals**p))**(1./p))

def cubic_dispersion(vals, p=CUBIC_P):
    cm = cubic_mean(vals, p=p)
    return float((np.mean(np.abs(vals - cm)**p))**(1./p))

def effective_n(vals, groups, p=CUBIC_P):
    """Kish effective sample size of the p-th moment, counted in snapshots.

    A power mean concentrates weight on the few snapshots holding the largest |J|, so the
    nominal snapshot count overstates how much independent sampling backs the number.
    For the inter couplings this collapses to ~3-5 of 46 snapshots, which is why the
    poorly sampled variants cannot be compared with the well sampled one."""
    groups = np.asarray(groups)
    w = np.array([np.sum(np.asarray(vals)[groups == g]**p) for g in np.unique(groups)])
    tot = w.sum()
    return float(tot**2 / np.sum(w**2)) if tot > 0 else 0.0

def cubic_mean_ci(vals, groups, p=CUBIC_P, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI resampling whole snapshots — the independent unit here.
    Resampling individual H-H pairs would treat the 4-16 pairs of one snapshot as
    independent draws and understate the interval several-fold."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ug = np.unique(groups)
    if ug.size < 2:
        return (float("nan"), float("nan"))
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=ug.size, replace=True)
        draws.append(cubic_mean(vals[np.concatenate([idx[g] for g in pick])], p=p))
    return (float(np.quantile(draws, alpha/2)), float(np.quantile(draws, 1-alpha/2)))
