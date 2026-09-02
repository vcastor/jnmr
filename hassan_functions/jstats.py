import numpy as np

CUBIC_P = 3.0

def cubic_mean(vals, p=CUBIC_P):
    return float((np.mean(vals**p))**(1./p))

def cubic_dispersion(vals, p=CUBIC_P):
    cm = cubic_mean(vals, p=p)
    return float((np.mean(np.abs(vals - cm)**p))**(1./p))

def effective_n(vals, groups, p=CUBIC_P):
    groups = np.asarray(groups)
    w = np.array([np.sum(np.asarray(vals)[groups == g]**p) for g in np.unique(groups)])
    tot = w.sum()
    return float(tot**2 / np.sum(w**2)) if tot > 0 else 0.0

def cubic_mean_ci(vals, groups, p=CUBIC_P, n_boot=2000, alpha=0.05, seed=0):
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

