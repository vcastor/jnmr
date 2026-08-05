#!$AMSBIN/plams
import os
import sys
import glob
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance
from hassan_functions.finders   import find_xh_groups, find_adjacent_xh_pair_anchored
from hassan_functions.ordering  import classify_sort, compute_offsets
from hassan_functions.io        import read_qtaim_charges, read_cdft_fukui
from hassan_functions.constants import FORMULAS
from hassan_functions.cache     import save_cache

CLUSTERS_DIR = "clusters"
QTAIM_DIR    = "/Users/vcastor/Desktop/backup_qtaimcdft/qtaim" #tmp
# QTAIM_DIR    = "amsoutput/qtaim"
CDFT_DIR     = "amsoutput/cdft"   # Conceptual-DFT outputs (Fukui functions)
BP_HEADER    = "BOND PATHS (BP) AND PROPERTIES ALONG THEM ARE WRITTEN TO TAPE21"
SPECIES      = ['urea', 'choline', 'chloride']

def read_bcps(path):
    """Return {frozenset({atomA, atomB}): {'rho', 'gb', 'vb', 'ratio'}} from QTAIM output."""
    with open(path) as f:
        lines = f.readlines()

    bp_start = next(i for i, l in enumerate(lines) if BP_HEADER in l)

    # first pass: BP summary -> {cp_num: frozenset({atomA, atomB})}
    pair_by_cp = {}
    for line in lines[bp_start + 4:]:
        if line.strip().startswith("---"):
            break
        parts = line.split()
        pair_by_cp[int(parts[1])] = frozenset({int(parts[2]), int(parts[4])})

    # second pass: rho, Gb, Vb for each BCP
    bcps = {}
    for i in range(bp_start):
        s = lines[i].strip()
        if not s.startswith("CP #"):
            continue
        cp_num = int(s.split()[2])
        if cp_num not in pair_by_cp:
            continue
        rho   = float(lines[i + 24].split()[2])
        gb    = float(lines[i + 45].split()[2])
        vb    = float(lines[i + 46].split()[2])
        ratio = np.abs(vb)/gb if gb != 0 else 0.0
        bcps[pair_by_cp[cp_num]] = {'rho': rho, 'gb': gb, 'vb': vb, 'ratio': ratio}
    return bcps

def poincare_hopf_ok(path):
    with open(path) as f:
        return "Poincare-Hopf satisfied" in f.read()

def bcp_props(bcps, a, b):
    return bcps.get(frozenset({a.cluster_id, b.cluster_id}))

def carbon_tag(c):
    nbrs = [b.other_end(c) for b in c.bonds]
    if sum(a.symbol == 'H' for a in nbrs) == 3:
        return 'CH3'
    heavy = {a.symbol for a in nbrs if a.symbol != 'H'}
    return 'CH2N' if 'N' in heavy else 'CH2O'

def site_tag(at):
    if at.symbol != 'H':
        return at.symbol
    nb = at.bonds[0].other_end(at)
    if nb.symbol == 'C':
        return f"H({carbon_tag(nb)})"
    return f"H({nb.symbol})"

def uu_label(a, b):
    """Canonical urea-urea BCP label; the H-bond acceptor (heavy site) is written
    first, so O-H(N)/N-H(N) read as acceptor - donor-H."""
    ta, tb = site_tag(a), site_tag(b)
    if (a.symbol == 'H') ^ (b.symbol == 'H'):
        return f"{tb} - {ta}" if a.symbol == 'H' else f"{ta} - {tb}"
    return " - ".join(sorted((ta, tb)))

# QTAIM net charge of H from CH2 next to N+ / next to O
q_Nside = []
q_Oside = []

# Conceptual-DFT Fukui functions (f+, f-, f0, f2) of the same CH2 hydrogens, so the
# two H types can be compared on reactivity as well as on charge/distance.
fukui_Nside = []   # H from CH2 next to N+
fukui_Oside = []   # H from CH2 next to O

# BCP accumulators: each entry is (distance, rho, gb, vb, ratio) -----------
nh2_ch3       = []
o_hch2_all    = []
o_hch2_N      = []   # H from CH2 next to N+
o_hch2_O      = []   # H from CH2 next to O
o_hch2_bridge = []
o_hch2_single = []
o_hoh              = []   # H from OH of choline
o_ch2_oh_bridge    = []   # O sees H(CH2, any) and H(OH)
o_ch2N_oh_bridge   = []   # O sees H(CH2-N) and H(OH)
o_ch2O_oh_bridge   = []   # O sees H(CH2-O) and H(OH)

other_by_type      = defaultdict(list)   # label -> [(d, rho, gb, vb, ratio)]
n_pairs_other_type = defaultdict(int)    # label -> # interacting pairs showing it
sig_labels         = set()               # 'other' labels counted toward the 100% (one H + one N/O)

# urea-urea BCP accumulators (how the ureas bind each other) -----------------
uu_by_type      = defaultdict(list)      # canonical label -> [(d, rho, gb, vb, ratio)]
n_uu_pairs_type = defaultdict(int)       # label -> # urea-urea pairs showing it
uu_sig_labels   = set()                  # significant labels (one H + one N/O)
n_uu_pairs_total       = 0
n_uu_pairs_interacting = 0
n_uu_bcp               = 0

n_systems_used          = 0
n_pairs_total           = 0
n_pairs_interacting     = 0
n_pairs_nh2_ch3         = 0
n_pairs_o_any           = 0
n_pairs_o_hch2_N        = 0
n_pairs_o_hch2_O        = 0
n_pairs_o_bridge        = 0
n_pairs_o_single        = 0
n_pairs_o_same_ch2      = 0
n_pairs_o_hoh           = 0
n_pairs_o_ch2_oh_bridge = 0
n_pairs_o_ch2N_oh       = 0
n_pairs_o_ch2O_oh       = 0
n_pairs_other           = 0
n_other_bcp             = 0
n_ph_satisfied          = 0

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue
    base  = os.path.splitext(os.path.basename(xf))[0]
    qpath = os.path.join(QTAIM_DIR, f"{base}.out")
    if not os.path.exists(qpath):
        continue
    n_systems_used += 1
    n_ph_satisfied += poincare_hopf_ok(qpath)

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs = compute_offsets(mol_data, SPECIES)

    # Tag atoms with the global 1-based index they have in the reordered
    # cluster (which is what coupling_generator.py wrote to the ADF input, and
    # therefore what the QTAIM output's atom numbers refer to).
    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1

    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    charges  = read_qtaim_charges(qpath)
    bcps     = read_bcps(qpath)
    cpath    = os.path.join(CDFT_DIR, f"{base}.out")
    fukui    = read_cdft_fukui(cpath) if os.path.exists(cpath) else {}

    # ── QTAIM charge + CDFT Fukui of the CH2 hydrogens (once per choline) ────
    for ch in cholines:
        pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
        if pair is None:                     # distorted choline: no CH2-CH2 pair
            continue
        _, hN, _, hO = pair
        for h in hN:
            q = charges.get(h.cluster_id)
            if q is not None:
                q_Nside.append(q)
            fk = fukui.get(h.cluster_id)
            if fk is not None:
                fukui_Nside.append(fk)
        for h in hO:
            q = charges.get(h.cluster_id)
            if q is not None:
                q_Oside.append(q)
            fk = fukui.get(h.cluster_id)
            if fk is not None:
                fukui_Oside.append(fk)

    # ── BCP properties per urea-choline pair ────────────────────────────────
    for u in ureas:
        n_urea   = [at for at in u.atoms if at.symbol == 'N']
        o_urea   = next(at for at in u.atoms if at.symbol == 'O')
        urea_ids = {at.cluster_id for at in u.atoms}
        for ch in cholines:
            n_pairs_total += 1
            ch_ids    = {at.cluster_id for at in ch.atoms}
            pair_keys = [k for k in bcps if (k & urea_ids) and (k & ch_ids)]
            if pair_keys:
                n_pairs_interacting += 1
            tracked = set()

            nh2_hit = False
            for n in n_urea:
                for _, hs in find_xh_groups(ch, 'C', 3, neighbour='N'):
                    for h in hs:
                        p = bcp_props(bcps, n, h)
                        if p is not None:
                            nh2_ch3.append((distance(n, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                            tracked.add(frozenset({n.cluster_id, h.cluster_id}))
                            nh2_hit = True
            if nh2_hit:
                n_pairs_nh2_ch3 += 1

            pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            hN = hO = []                     # distorted choline w/o CH2-CH2 pair -> no CH2 H's
            if pair is not None:
                _, hN, _, hO = pair
            hits_N = [(h, bcp_props(bcps, o_urea, h)) for h in hN]
            hits_N = [(h, p) for h, p in hits_N if p is not None]
            hits_O = [(h, bcp_props(bcps, o_urea, h)) for h in hO]
            hits_O = [(h, p) for h, p in hits_O if p is not None]
            hits   = hits_N + hits_O
            tracked |= {frozenset({o_urea.cluster_id, h.cluster_id}) for h, _ in hits}
            if hits:
                n_pairs_o_any += 1
                if hits_N:
                    n_pairs_o_hch2_N += 1
                if hits_O:
                    n_pairs_o_hch2_O += 1
                for h, p in hits:
                    o_hch2_all.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                for h, p in hits_N:
                    o_hch2_N.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                for h, p in hits_O:
                    o_hch2_O.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                if hits_N and hits_O:
                    n_pairs_o_bridge += 1
                    for h, p in hits:
                        o_hch2_bridge.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                elif len(hits) == 1:
                    n_pairs_o_single += 1
                    h, p = hits[0]
                    o_hch2_single.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                else:
                    n_pairs_o_same_ch2 += 1

            hoh_hits = [(h, bcp_props(bcps, o_urea, h))
                        for _, hs in find_xh_groups(ch, 'O', 1)
                        for h in hs]
            hoh_hits = [(h, p) for h, p in hoh_hits if p is not None]
            tracked |= {frozenset({o_urea.cluster_id, h.cluster_id}) for h, _ in hoh_hits}
            if hoh_hits:
                n_pairs_o_hoh += 1
                for h, p in hoh_hits:
                    o_hoh.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))

            if hits and hoh_hits:
                n_pairs_o_ch2_oh_bridge += 1
                for h, p in hits + hoh_hits:
                    o_ch2_oh_bridge.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                if hits_N:
                    n_pairs_o_ch2N_oh += 1
                    for h, p in hits_N + hoh_hits:
                        o_ch2N_oh_bridge.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))
                if hits_O:
                    n_pairs_o_ch2O_oh += 1
                    for h, p in hits_O + hoh_hits:
                        o_ch2O_oh_bridge.append((distance(o_urea, h), p['rho'], p['gb'], p['vb'], p['ratio']))

            other_here = set()
            for k in pair_keys:
                if k in tracked:
                    continue
                ua    = next(a for a in u.atoms if a.cluster_id in k)
                ca    = next(a for a in ch.atoms if a.cluster_id in k)
                label = f"urea {site_tag(ua)} - choline {site_tag(ca)}"
                if (ua.symbol == 'H') ^ (ca.symbol == 'H'):
                    heavy = ca.symbol if ua.symbol == 'H' else ua.symbol
                    if heavy in ('N', 'O'):
                        sig_labels.add(label)
                p     = bcps[k]
                other_by_type[label].append((distance(ua, ca), p['rho'], p['gb'], p['vb'], p['ratio']))
                other_here.add(label)
                n_other_bcp += 1
            if other_here:
                n_pairs_other += 1
                for label in other_here:
                    n_pairs_other_type[label] += 1

    # ── urea-urea BCP properties (how the ureas bind each other) ─────────────
    # Every unordered pair of distinct ureas in the cluster; classify each
    # urea-urea BCP by the canonical site-pair label (O-H(N) is the H-bond).
    for iu in range(len(ureas)):
        u1   = ureas[iu]
        ids1 = {at.cluster_id for at in u1.atoms}
        for u2 in ureas[iu + 1:]:
            ids2 = {at.cluster_id for at in u2.atoms}
            n_uu_pairs_total += 1
            pair_keys = [k for k in bcps if (k & ids1) and (k & ids2)]
            if not pair_keys:
                continue
            n_uu_pairs_interacting += 1
            here = set()
            for k in pair_keys:
                a1    = next(a for a in u1.atoms if a.cluster_id in k)
                a2    = next(a for a in u2.atoms if a.cluster_id in k)
                label = uu_label(a1, a2)
                if (a1.symbol == 'H') ^ (a2.symbol == 'H'):
                    heavy = a2.symbol if a1.symbol == 'H' else a1.symbol
                    if heavy in ('N', 'O'):
                        uu_sig_labels.add(label)
                p = bcps[k]
                uu_by_type[label].append((distance(a1, a2), p['rho'], p['gb'], p['vb'], p['ratio']))
                here.add(label)
                n_uu_bcp += 1
            for label in here:
                n_uu_pairs_type[label] += 1

finish()

save_cache("qtaim_analysis", {
    "q_Nside":                 q_Nside,
    "q_Oside":                 q_Oside,
    "fukui_Nside":             fukui_Nside,
    "fukui_Oside":             fukui_Oside,
    "nh2_ch3":                 nh2_ch3,
    "o_hch2_all":              o_hch2_all,
    "o_hch2_N":                o_hch2_N,
    "o_hch2_O":                o_hch2_O,
    "o_hch2_bridge":           o_hch2_bridge,
    "o_hch2_single":           o_hch2_single,
    "o_hoh":                   o_hoh,
    "o_ch2_oh_bridge":         o_ch2_oh_bridge,
    "o_ch2N_oh_bridge":        o_ch2N_oh_bridge,
    "o_ch2O_oh_bridge":        o_ch2O_oh_bridge,
    "other_by_type":           dict(other_by_type),
    "n_pairs_other_type":      dict(n_pairs_other_type),
    "sig_labels":              sig_labels,
    "uu_by_type":              dict(uu_by_type),
    "n_uu_pairs_type":         dict(n_uu_pairs_type),
    "uu_sig_labels":           uu_sig_labels,
    "n_uu_pairs_total":        n_uu_pairs_total,
    "n_uu_pairs_interacting":  n_uu_pairs_interacting,
    "n_uu_bcp":                n_uu_bcp,
    "n_systems_used":          n_systems_used,
    "n_pairs_total":           n_pairs_total,
    "n_pairs_interacting":     n_pairs_interacting,
    "n_pairs_nh2_ch3":         n_pairs_nh2_ch3,
    "n_pairs_o_any":           n_pairs_o_any,
    "n_pairs_o_hch2_N":        n_pairs_o_hch2_N,
    "n_pairs_o_hch2_O":        n_pairs_o_hch2_O,
    "n_pairs_o_bridge":        n_pairs_o_bridge,
    "n_pairs_o_single":        n_pairs_o_single,
    "n_pairs_o_same_ch2":      n_pairs_o_same_ch2,
    "n_pairs_o_hoh":           n_pairs_o_hoh,
    "n_pairs_o_ch2_oh_bridge": n_pairs_o_ch2_oh_bridge,
    "n_pairs_o_ch2N_oh":       n_pairs_o_ch2N_oh,
    "n_pairs_o_ch2O_oh":       n_pairs_o_ch2O_oh,
    "n_pairs_other":           n_pairs_other,
    "n_other_bcp":             n_other_bcp,
    "n_ph_satisfied":          n_ph_satisfied,
})
