import os

# ── shared env-var flags for the generator scripts ───────────────────────────
# MIN_STEP=<n>       skip clusters before this MD step (each script sets its default)
# ALLOW_MEDIUM=1     also admit the (4,2,2) tier
# NO_SIZE_LIMIT=1    lift the size restriction entirely
# <NAME>_LIMIT=<n>   cap how many new run/sl pairs a branch writes (0 = no cap),
#                    e.g. SMALL_LIMIT, CH_LIMIT, NH_LIMIT, NH_INTRA_LIMIT, SITE_LIMIT,
#                    LIMIT (property_generator)
# PARTITION=<p>      submit this batch to CRIANN partition p (walltime = that
#                    partition's cap) instead of each variant's default
# COMPOSITION=u,c,cl exact (urea, choline, chloride) tiers to admit, ';'-separated
#                    for several (e.g. COMPOSITION="6,3,3" or "2,1,1;6,3,3");
#                    overrides the SMALL/MEDIUM tiers
# VARIANTS=a,b       main HH workflow only: generate just these variants
#                    (e.g. VARIANTS="TZ2P_FC,TZ2P_all"); unset = all four
# TAPE=SAVE          sl brings back the whole work dir (rkf, TAPE10, ...);
#                    default: only the .out file
# VERBOSE=1 / -v     progress prints (silent by default)

# Only submit clusters with these exact (urea, choline, chloride) counts. The carve
# ratio is 2 urea : 1 choline chloride, so (2,1,1) is one formula unit (38 atoms) and
# (4,2,2) is two (76 atoms). Anything larger is far more expensive on CRIANN for a
# coupling that is dominated by the nearest contacts anyway.
SMALL_COMPOSITION  = (2, 1, 1)
MEDIUM_COMPOSITION = (4, 2, 2)

# Atom counts per species, to read a cluster's composition from its element tally
# alone — far cheaper than a PLAMS classify on every candidate file.
SPECIES_ATOMS = {'urea': 8, 'choline': 21, 'chloride': 1}

def env_int(name, default=0):
    """Integer env-var flag (limits, MIN_STEP)."""
    return int(os.environ.get(name, default))

def env_list(name):
    """Comma-separated env-var flag as a list, or None if unset."""
    v = os.environ.get(name)
    return v.split(",") if v else None

def partition_override():
    """CRIANN partition from PARTITION=<p>, or None to keep each variant's default."""
    return os.environ.get("PARTITION")

def verbose():
    """True only with -v on the command line or VERBOSE=1 — progress prints are
    opt-in, silent by default."""
    import sys
    return "-v" in sys.argv or bool(os.environ.get("VERBOSE"))

def vprint(*a, **k):
    if verbose():
        print(*a, **k)

def allowed_compositions():
    """Size restriction from the env: [SMALL] by default, +MEDIUM with
    ALLOW_MEDIUM=1, None (no restriction) with NO_SIZE_LIMIT=1, or the exact
    tiers given as COMPOSITION="u,c,cl[;u,c,cl...]" (which overrides the rest)."""
    if os.environ.get("COMPOSITION"):
        return [tuple(int(n) for n in tier.split(","))
                for tier in os.environ["COMPOSITION"].split(";")]
    if os.environ.get("NO_SIZE_LIMIT"):
        return None
    if os.environ.get("ALLOW_MEDIUM"):
        return [SMALL_COMPOSITION, MEDIUM_COMPOSITION]
    return [SMALL_COMPOSITION]

def cluster_composition(xyz_file):
    """(n_urea, n_choline, n_chloride) inferred from the element counts of an xyz,
    or None if the tally is not a whole number of the three species.

    urea CH4N2O, choline C5H14NO, chloride Cl, so per cluster:
        n_Cl = n_chloride,  n_O = n_urea + n_choline,  n_N = 2*n_urea + n_choline
    which inverts to n_urea = n_N - n_O and n_choline = n_O - n_urea. Carbon and
    hydrogen then have to match, which is what rejects an off-ratio carve."""
    counts = {}
    with open(xyz_file) as f:
        lines = f.readlines()
    if len(lines) < 3:
        return None
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            counts[parts[0]] = counts.get(parts[0], 0) + 1
    n_o, n_n, n_cl = counts.get('O', 0), counts.get('N', 0), counts.get('Cl', 0)
    n_urea    = n_n - n_o
    n_choline = n_o - n_urea
    if n_urea < 0 or n_choline < 0:
        return None
    expect_c = n_urea + 5*n_choline
    expect_h = 4*n_urea + 14*n_choline
    if counts.get('C', 0) != expect_c or counts.get('H', 0) != expect_h:
        return None
    return (n_urea, n_choline, n_cl)

def composition_allowed(xyz_file, allowed):
    """True if the cluster may be submitted under the current size restriction.
    `allowed` of None lifts the restriction entirely."""
    if allowed is None:
        return True
    return cluster_composition(xyz_file) in allowed
