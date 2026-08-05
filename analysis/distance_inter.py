#!$AMSBIN/plams
import os
import sys
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hassan_functions.geometry  import distance
from hassan_functions.finders   import find_xh_groups, find_adjacent_xh_pair_anchored
from hassan_functions.constants import FORMULAS
from hassan_functions.cache     import save_cache

CLUSTERS_DIR = "clusters"
THR_NH2_CH3   = 3.4    # H(NH2)-H(CH3) cutoff
THR_OU_HCH2O  = 3.2    # O(urea)-H(CH2 near O) cutoff
THR_OU_HCH2N  = 3.0    # O(urea)-H(CH2 near N) cutoff
THR_OU_HOH    = 2.8    # O(urea)-H(OH) cutoff
THR_HCH2_UREA = 4.0    # choline CH2 H "in contact" with a urea atom (C is buried -> larger)
THR_CU_OH     = 4.5    # C(urea)-H(OH) contact cutoff
THR_UREA_H    = 5.0    # a choline H "sees" a urea when its nearest O/C is within this (O-vs-C compare)

# NH2-CH3
inter_NH_CH3   = []
inter_Cu_HCH3  = []
inter_HH_NCH3  = []
# urea-CH2 HCH-O-HCH (one H per CH2 below cutoff)
inter_Ou_HCH2  = []
inter_Nu_HCH2  = []
inter_HN_HCH2  = []
# urea-OH  O(urea)-H(OH)
inter_Ou_HOH   = []
inter_Nu_HOH   = []
inter_HN_HOH   = []
# double bridge O(urea)-[H(CH2-N) & H(OH)]
inter_Ou_HCH2N_dbl = []   # O-H, H from CH2 adjacent to N+
inter_Ou_HOH_dbl   = []   # O-H, H from OH
inter_Nu_dbl       = []
inter_HN_dbl       = []
n_dbl_events = 0
# choline CH2 H (split N+/O side) vs urea H(NH2) / C / N
inter_HCH2N_HN = []   # H(CH2-N) - H(NH2 urea)
inter_HCH2N_Cu = []   # H(CH2-N) - C(urea)
inter_HCH2N_Nu = []   # H(CH2-N) - N(urea)
inter_HCH2O_HN = []   # H(CH2-O) - H(NH2 urea)
inter_HCH2O_Cu = []   # H(CH2-O) - C(urea)
inter_HCH2O_Nu = []   # H(CH2-O) - N(urea)
# urea C to choline OH
inter_Cu_HOH   = []   # C(urea) - H(OH)
# O(urea) / C(urea) / N(urea) distance to each choline H type, paired on the SAME H
# (nearest urea) so the three urea sites can be compared: which one is closer to the
# choline H's. d_N is to the nearest of the urea's two amino N's.
oc_HCH2N_O = []; oc_HCH2N_C = []; oc_HCH2N_N = []   # H(CH2 near N+)
oc_HCH2O_O = []; oc_HCH2O_C = []; oc_HCH2O_N = []   # H(CH2 near O )
oc_HOH_O   = []; oc_HOH_C   = []; oc_HOH_N   = []   # H(OH)

def urea_site_dists(sites, h):
    """(d_O, d_C, d_N) from a urea's (O, C, [N,...]) sites to H; d_N is the nearest N."""
    o, c, ns = sites
    return distance(o, h), distance(c, h), min(distance(n, h) for n in ns)

init()

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue

    cluster = Molecule(xf)
    cluster.guess_bonds()
    mols    = cluster.separate()

    ureas    = [m for m in mols if m.get_formula() == FORMULAS['urea']]
    cholines = [m for m in mols if m.get_formula() == FORMULAS['choline']]

    for u in ureas:
        nh2_groups = find_xh_groups(u, 'N', 2)
        c_urea = next(at for at in u.atoms if at.symbol == 'C')
        for ch in cholines:
            ch3 = find_xh_groups(ch, 'C', 3, neighbour='N')
            for n_nh2, h_nh2_list in nh2_groups:
                for c_ch3, h_ch3_list in ch3:
                    near = sorted(h_ch3_list, key=lambda h: distance(n_nh2, h))[:2]
                    near = [h for h in near if distance(n_nh2, h) <= THR_NH2_CH3]
                    for h_ch3 in near:
                        inter_NH_CH3.append(distance(n_nh2, h_ch3))
                        inter_Cu_HCH3.append(distance(c_urea, h_ch3))
                        for h_nh2 in h_nh2_list:
                            inter_HH_NCH3.append(distance(h_nh2, h_ch3))

    for u in ureas:
        o_urea     = next(at for at in u.atoms if at.symbol == 'O')
        nh2_groups = find_xh_groups(u, 'N', 2)
        for ch in cholines:
            pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            if pair is None:
                continue
            _, hN, _, hO = pair
            _, hOH       = find_xh_groups(ch, 'O', 1)[0]

            close_N  = [(h, distance(o_urea, h)) for h in hN  if distance(o_urea, h) <= THR_OU_HCH2N]
            close_O  = [(h, distance(o_urea, h)) for h in hO  if distance(o_urea, h) <= THR_OU_HCH2O]
            close_OH = [(h, distance(o_urea, h)) for h in hOH if distance(o_urea, h) <= THR_OU_HOH]

            # O(urea)-H(OH): single interaction, record each close OH H
            for h, d in close_OH:
                inter_Ou_HOH.append(d)
                n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: distance(g[0], h))
                inter_Nu_HOH.append(distance(n_nh2, h))
                for h_nh2 in h_nh2_list:
                    inter_HN_HOH.append(distance(h_nh2, h))

            # HCH-O-HCH: one H per CH2 within cutoff (both CH2 groups required)
            if close_N and close_O:
                for h, d in close_N + close_O:
                    inter_Ou_HCH2.append(d)
                    n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: distance(g[0], h))
                    inter_Nu_HCH2.append(distance(n_nh2, h))
                    for h_nh2 in h_nh2_list:
                        inter_HN_HCH2.append(distance(h_nh2, h))

            # double bridge O(urea)-[H(CH2-N) & H(OH)]
            if close_N and close_OH:
                n_dbl_events += 1
                for hNa, dN in close_N:
                    for hOHa, dOH in close_OH:
                        inter_Ou_HCH2N_dbl.append(dN)
                        inter_Ou_HOH_dbl.append(dOH)
                        for hh in (hNa, hOHa):
                            n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: distance(g[0], hh))
                            inter_Nu_dbl.append(distance(n_nh2, hh))
                            for h_nh2 in h_nh2_list:
                                inter_HN_dbl.append(distance(h_nh2, hh))

    # choline CH2 H's (split N+/O side) vs urea H(NH2) / C / N, and C(urea)-H(OH).
    # A CH2 H counts as "in contact" when it is within THR_HCH2_UREA of any urea heavy
    # atom (C or N); its distances to C, the nearest N and the nearest NH2 H are recorded.
    for u in ureas:
        c_urea     = next(at for at in u.atoms if at.symbol == 'C')
        n_ureas    = [at for at in u.atoms if at.symbol == 'N']
        nh2_groups = find_xh_groups(u, 'N', 2)
        heavy      = [c_urea] + n_ureas
        for ch in cholines:
            pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            if pair is None:
                continue
            _, hN, _, hO = pair
            _, hOH       = find_xh_groups(ch, 'O', 1)[0]

            for h_list, (L_H, L_C, L_N) in (
                    (hN, (inter_HCH2N_HN, inter_HCH2N_Cu, inter_HCH2N_Nu)),
                    (hO, (inter_HCH2O_HN, inter_HCH2O_Cu, inter_HCH2O_Nu))):
                for h in h_list:
                    if min(distance(a, h) for a in heavy) > THR_HCH2_UREA:
                        continue
                    n_nh2, h_nh2_list = min(nh2_groups, key=lambda g: distance(g[0], h))
                    L_C.append(distance(c_urea, h))
                    L_N.append(distance(n_nh2, h))
                    for h_nh2 in h_nh2_list:
                        L_H.append(distance(h_nh2, h))

            for h in hOH:
                if distance(c_urea, h) <= THR_CU_OH:
                    inter_Cu_HOH.append(distance(c_urea, h))

    # ── O(urea) vs C(urea) vs N(urea): which urea site is closest to each choline H ──
    # For every choline H (CH2-N, CH2-O, OH) take the nearest urea (by its closest of
    # O / C / N) and record the O-H, C-H and (nearest) N-H distance, so the three urea
    # sites are compared on the SAME hydrogen.
    urea_sites = [(next(a for a in u.atoms if a.symbol == 'O'),
                   next(a for a in u.atoms if a.symbol == 'C'),
                   [a for a in u.atoms if a.symbol == 'N']) for u in ureas]
    if urea_sites:
        for ch in cholines:
            pair = find_adjacent_xh_pair_anchored(ch, 'C', 2, 'N')
            if pair is None:
                continue
            _, hN, _, hO = pair
            _, hOH       = find_xh_groups(ch, 'O', 1)[0]
            for h_list, LO, LC, LN in ((hN,  oc_HCH2N_O, oc_HCH2N_C, oc_HCH2N_N),
                                       (hO,  oc_HCH2O_O, oc_HCH2O_C, oc_HCH2O_N),
                                       (hOH, oc_HOH_O,   oc_HOH_C,   oc_HOH_N)):
                for h in h_list:
                    best = min(urea_sites, key=lambda s: min(urea_site_dists(s, h)))
                    dO, dC, dN = urea_site_dists(best, h)
                    if min(dO, dC, dN) <= THR_UREA_H:
                        LO.append(dO)
                        LC.append(dC)
                        LN.append(dN)

finish()

save_cache("distance_inter", {
    "inter_NH_CH3":       inter_NH_CH3,
    "inter_Cu_HCH3":      inter_Cu_HCH3,
    "inter_HH_NCH3":      inter_HH_NCH3,
    "inter_Ou_HCH2":      inter_Ou_HCH2,
    "inter_Nu_HCH2":      inter_Nu_HCH2,
    "inter_HN_HCH2":      inter_HN_HCH2,
    "inter_Ou_HOH":       inter_Ou_HOH,
    "inter_Nu_HOH":       inter_Nu_HOH,
    "inter_HN_HOH":       inter_HN_HOH,
    "inter_Ou_HCH2N_dbl": inter_Ou_HCH2N_dbl,
    "inter_Ou_HOH_dbl":   inter_Ou_HOH_dbl,
    "inter_Nu_dbl":       inter_Nu_dbl,
    "inter_HN_dbl":       inter_HN_dbl,
    "n_dbl_events":       n_dbl_events,
    "inter_HCH2N_HN":     inter_HCH2N_HN,
    "inter_HCH2N_Cu":     inter_HCH2N_Cu,
    "inter_HCH2N_Nu":     inter_HCH2N_Nu,
    "inter_HCH2O_HN":     inter_HCH2O_HN,
    "inter_HCH2O_Cu":     inter_HCH2O_Cu,
    "inter_HCH2O_Nu":     inter_HCH2O_Nu,
    "inter_Cu_HOH":       inter_Cu_HOH,
    "oc_HCH2N_O":         oc_HCH2N_O,
    "oc_HCH2N_C":         oc_HCH2N_C,
    "oc_HCH2N_N":         oc_HCH2N_N,
    "oc_HCH2O_O":         oc_HCH2O_O,
    "oc_HCH2O_C":         oc_HCH2O_C,
    "oc_HCH2O_N":         oc_HCH2O_N,
    "oc_HOH_O":           oc_HOH_O,
    "oc_HOH_C":           oc_HOH_C,
    "oc_HOH_N":           oc_HOH_N,
})
