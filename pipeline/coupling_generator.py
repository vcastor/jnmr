#!$AMSBIN/plams
import os
import glob
import sqlite3
import numpy as np
from itertools import groupby
from typing import List, Tuple, Dict
from hassan_functions import (distance, classify_sort, compute_offsets,
                              find_xh_bonds, find_xh_groups, find_adjacent_xh_pairs,
                              table_exists, column_exists, get_step_from_filename,
                              slurm_script, VARIANT_SLURM, SCF_WARNINGS, FORMULAS,
                              PARTITION_WALLTIME, choline_sites, urea_sites,
                              env_int, vprint, partition_override, env_list,
                              allowed_compositions, composition_allowed)

SPECIES = ['urea', 'choline', 'chloride']
DB_PATH = "nmr_jcoupling.db"
CLUSTERS_DIR = "clusters"

# ── temporary CRIANN budget restrictions ─────────────────────────────────────
# Both are deliberate, temporary limits on what gets submitted. To lift them, set
# ALLOWED_COMPOSITIONS = None and/or MIN_STEP = 0 (or override per-run via the env
# vars noted below). They apply to EVERY branch, so a restricted batch stays
# restricted whether it is the main HH workflow or one of the opt-in extensions.

# Size restriction (tiers + env flags now live in hassan_functions.flags):
# default is the smallest (2,1,1) tier only; ALLOW_MEDIUM=1 adds (4,2,2);
# NO_SIZE_LIMIT=1 lifts it.

# Don't start before this MD step. The earlier clusters were computed first and at a
# different point of the trajectory, so restricting new batches to a later window
# avoids inheriting whatever bias that sampling carried. Override with MIN_STEP=<n>.
MIN_STEP = 60_000_000

# MD integration timestep, fs per step. Snapshots are named MDStep<N> where N is the MD
# step, so time[ns] = N * MD_TIMESTEP_FS / 1e6. Used only to turn IGNORE_TIME (ns, an
# env var) into a minimum step. >>> SET THIS TO YOUR ACTUAL MD TIMESTEP <<<
MD_TIMESTEP_FS = 1.0

# main NMR workflow: (variant, dso/pso/sd contributions, TZ2P-J basis)
VARIANTS = [
    ("TZ2P_FC",   False, False),
    ("TZ2P_all",  True,  False),
    ("TZ2PJ_FC",  False, True),
    ("TZ2PJ_all", True,  True),
]
DISTANCE_THRESHOLD = 5.0

# C(urea)-H(choline) coupling for small clusters → court partition
CH_OUT_DIR         = "run_scripts/ch"
CH_PARTITION       = "long"
CH_DIST_THRESHOLD  = 5.0   # Å from the urea C; keep every choline H within this
CH_REQUIRE_VARIANT = "TZ2P_FC"  # only clusters whose H-H (this variant) is already computed

# N(urea)-H1 coupling: like the CH block but the urea N is the perturber and the only
# responders are the choline methyl H's [H(CH3) = H1]. Set NH_LIMIT to generate only the
# N smallest pending clusters first (faster J approximations).
NH_OUT_DIR         = "run_scripts/nh"
NH_PARTITION       = "long"
NH_DIST_THRESHOLD  = 5.0   # Å from the urea N; keep every choline methyl H (H1) within this
NH_REQUIRE_VARIANT = "TZ2P_FC"  # only clusters whose H-H (this variant) is already computed

# J_NH intra-urea: the urea N coupled to the urea's OWN H's (its two bonded H's and the
# two on the other N). Opt in with NH_INTRA=1; NH_INTRA_LIMIT caps the batch size.
NH_INTRA_OUT_DIR         = "run_scripts/nh_intra"
NH_INTRA_PARTITION       = "long"
NH_INTRA_REQUIRE_VARIANT = "TZ2P_FC"

# Named-site couplings (H1-H5, Nurea) requested by the experimental team. Opt in with
# SITE=1; SITE_LIMIT caps the batch size. Unlike the CH/NH extensions this does NOT
# require the H-H run to exist first — these are a fresh set of clusters chosen by
# MIN_STEP, so waiting on the older H-H batch would defeat the point.
SITE_OUT_DIR        = "run_scripts/site"
SITE_PARTITION      = "long"
SITE_DIST_THRESHOLD = 5.0   # Å; inter pairs kept within this of the perturber

def classify_cluster(xyz):
    """Classify a cluster into SPECIES, tag each atom with its global 1-based
    cluster_id, and return (mol_data, offsets, sorted_mols, counts)."""
    cluster = Molecule(xyz)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    mol_data = classify_sort(cluster.separate(), centre,
                             {k: FORMULAS[k] for k in SPECIES})
    offs = compute_offsets(mol_data, SPECIES)
    for name in SPECIES:
        for mi, mol in enumerate(mol_data[name]):
            off = offs[name][mi]
            for ai, at in enumerate(mol.atoms):
                at.cluster_id = off + ai + 1
    sorted_mols = [m for name in SPECIES for m in mol_data[name]]
    counts      = {name: len(mol_data[name]) for name in SPECIES}
    return mol_data, offs, sorted_mols, counts

def write_system_engine(f, sorted_mols, basename, j_basis=False):
    """System/Atoms + ADF engine block (saves TAPE10 for the cpl runs)."""
    f.write("#!/bin/sh\n\n")
    f.write(f"export AMS_JOBNAME={basename}\n")
    f.write("$AMSBIN/ams <<eor\n")
    f.write("System\n")
    f.write("  Atoms\n")
    for mol in sorted_mols:
        for atom in mol.atoms:
            f.write(f"    {atom.symbol:>4s} {atom.x:>14.8f} {atom.y:>14.8f} {atom.z:>14.8f}\n")
    f.write("  End\n")
    f.write("End\n\n")
    f.write("Task SinglePoint\n\n")
    f.write("Engine ADF\n")
    f.write(f"  title {basename}\n")
    f.write("  NumericalQuality Excellent\n")
    f.write("  Basis\n")
    f.write(f"    Type {'TZ2P-J' if j_basis else 'TZ2P'}\n")
    f.write("    core None\n")
    f.write("  End\n")
    f.write("  save TAPE10\n")
    f.write("  symmetry NOSYM\n")
    f.write("  XC\n")
    f.write("    GGA PBE\n")
    f.write("  End\n")
    f.write("  Relativity\n")
    f.write("    Level None\n")
    f.write("  End\n")
    f.write("EndEngine\n")
    f.write("eor\n")

def write_cpl(f, basename, pert, resp, contributions=False):
    """One $AMSBIN/cpl nmrcoupling block: pert against the resp list."""
    resp_str = " ".join(str(r) for r in resp)
    f.write("$AMSBIN/cpl << eor\n")
    f.write(f"  adffile {basename}.results/adf.rkf\n")
    f.write(f"  tape10file {basename}.results/TAPE10\n")
    f.write("  nmrcoupling\n")
    if contributions:
        f.write("    dso\n")
        f.write("    pso\n")
        f.write("    sd\n")
    f.write(f"    atompert {pert}\n")
    f.write(f"    atomresp {resp_str}\n")
    f.write("  end\n")
    f.write("eor\n")

def intra_choline_interactions(mol_data, choline_offsets):
    """CH2-CH2 intra interactions per choline:
    {choline_index: [(C1_global, H1_global, C2_global, [H2_globals]), ...]}."""
    intra = {}
    for ci, choline in enumerate(mol_data['choline']):
        offset = choline_offsets[ci]
        interactions = []
        for c1, h1_list, c2, h2_list in find_adjacent_xh_pairs(choline, 'C', 2, return_indices=True):
            global_c1 = offset + c1 + 1
            global_c2 = offset + c2 + 1
            for h1 in h1_list:
                global_h1  = offset + h1 + 1
                global_h2s = [offset + h2 + 1 for h2 in h2_list]
                interactions.append((global_c1, global_h1, global_c2, global_h2s))
        intra[ci] = interactions
    return intra

def inter_nh2_ch3_interactions(mol_data, urea_offsets, choline_offsets,
                               distance_threshold=5.0):
    """NH2-CH3 inter interactions (threshold between H atoms). Rows of the same
    NH2-(CH3)3 contact share a contact_id; the closest H-H pair gets is_main=1."""
    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    contacts: Dict[Tuple[int, int], List[Dict]] = {}

    for ui, urea in enumerate(ureas):
        nh2_bonds = find_xh_bonds(urea, 'N', return_indices=True)
        for ci, choline in enumerate(cholines):
            ch3_groups = find_xh_groups(choline, 'C', 3, neighbour='N', return_indices=True)
            if not ch3_groups:
                continue
            choline_h_indices = [h for _, hs in ch3_groups for h in hs]

            for _, urea_h_idx in nh2_bonds:
                urea_h_coord = np.array(urea.atoms[urea_h_idx].coords)
                for ch3_h_idx in choline_h_indices:
                    ch3_h_coord = np.array(choline.atoms[ch3_h_idx].coords)
                    dist = np.linalg.norm(urea_h_coord - ch3_h_coord)
                    if dist > distance_threshold:
                        continue
                    contacts.setdefault((ui, ci), []).append({
                        'urea_idx':    ui,
                        'choline_idx': ci,
                        'H_urea':      urea_offsets[ui] + urea_h_idx + 1,
                        'H_choline':   choline_offsets[ci] + ch3_h_idx + 1,
                        'distance':    dist,
                    })

    interactions = []
    for contact_id, ((ui, ci), rows) in enumerate(contacts.items()):
        min_dist = min(r['distance'] for r in rows)
        for r in rows:
            r['contact_id'] = contact_id
            r['is_main'] = 1 if r['distance'] == min_dist else 0
        interactions.extend(rows)
    return interactions

def write_main_run(sorted_mols, filename, intra_interactions, inter_interactions,
                   contributions=False, j_basis=False):
    """ADF input with intra CH2-CH2 and inter NH2-CH3 NMR coupling blocks."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, 'w') as f:
        write_system_engine(f, sorted_mols, basename, j_basis)
        f.write("\n#\n# NMR J-coupling calculation\n#\n")

        f.write("\n# Intra-molecular J coupling (CH2-CH2 in choline)\n")
        for ci in sorted(intra_interactions.keys()):
            f.write(f"# Choline {ci + 1}\n")
            for _c1, h1, _c2, h2_list in intra_interactions[ci]:
                write_cpl(f, basename, h1, h2_list, contributions)
                f.write("\n")

        f.write("# Inter-molecular J coupling (NH2-CH3)\n")
        sorted_inter = sorted(inter_interactions,
                              key=lambda x: (x['urea_idx'], x['choline_idx'], x['H_urea']))
        current_pair = None
        for h_urea, group in groupby(sorted_inter, key=lambda x: x['H_urea']):
            rows = list(group)
            ui, ci = rows[0]['urea_idx'], rows[0]['choline_idx']
            if (ui, ci) != current_pair:
                current_pair = (ui, ci)
                f.write(f"# Urea {ui + 1} - Choline {ci + 1}\n")
            write_cpl(f, basename, h_urea, [r['H_choline'] for r in rows], contributions)
            f.write("\n")

def add_snapshot_to_db(db_path, n_step, n_choline, intra_interactions, inter_interactions):
    """Seed step_{n}_intra / step_{n}_inter (geometry only; J columns stay NULL).
    Only populated on first creation — re-running must not wipe reader values."""
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO snapshots (n_step, n_choline, n_inter) VALUES (?, ?, ?)",
        (n_step, n_choline, len(inter_interactions)))

    # C_pert / C_resp are the CH2 carbons that H_pert / H_resp sit on; the reader
    # uses them to pull the HC and CC DI/chi terms from the QTAIM/CDFT matrices.
    intra_table = f"step_{n_step}_intra"
    if not table_exists(cursor, intra_table):
        cursor.execute(f'''
            CREATE TABLE {intra_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                H_pert      INTEGER,
                H_resp      INTEGER,
                C_pert      INTEGER,
                C_resp      INTEGER,
                J_TZ2P_FC   REAL,
                J_TZ2P_all  REAL,
                J_TZ2PJ_FC  REAL,
                J_TZ2PJ_all REAL
            )
        ''')
        for ci in sorted(intra_interactions.keys()):
            for c1, h1, c2, h2_list in intra_interactions[ci]:
                for h2 in h2_list:
                    cursor.execute(
                        f"INSERT INTO {intra_table} (H_pert, H_resp, C_pert, C_resp) "
                        f"VALUES (?, ?, ?, ?)", (h1, h2, c1, c2))
    else:
        # backfill the carbon columns on pre-existing tables (geometry only)
        for col in ("C_pert", "C_resp"):
            if not column_exists(cursor, intra_table, col):
                cursor.execute(f"ALTER TABLE {intra_table} ADD COLUMN {col} INTEGER")
        for ci in sorted(intra_interactions.keys()):
            for c1, h1, c2, h2_list in intra_interactions[ci]:
                for h2 in h2_list:
                    cursor.execute(
                        f"UPDATE {intra_table} SET C_pert = ?, C_resp = ? "
                        f"WHERE H_pert = ? AND H_resp = ?", (c1, c2, h1, h2))

    inter_table = f"step_{n_step}_inter"
    if not table_exists(cursor, inter_table):
        cursor.execute(f'''
            CREATE TABLE {inter_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                H_pert      INTEGER,
                H_resp      INTEGER,
                distance    REAL,
                is_main     INTEGER DEFAULT 0,
                J_TZ2P_FC   REAL,
                J_TZ2P_all  REAL,
                J_TZ2PJ_FC  REAL,
                J_TZ2PJ_all REAL
            )
        ''')
        for inter in inter_interactions:
            cursor.execute(
                f"INSERT INTO {inter_table} (H_pert, H_resp, distance, is_main) "
                f"VALUES (?, ?, ?, ?)",
                (inter['H_urea'], inter['H_choline'], inter['distance'], inter['is_main']))

    conn.commit()
    conn.close()

def add_snapshot_to_db_error(db_path, n_step):
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO snapshots (n_step, n_choline, n_inter, comment)
        VALUES (?, 0, 0, 'Error processing snapshot')
        ON CONFLICT(n_step) DO UPDATE SET comment = 'Error processing snapshot'
    ''', (n_step,))
    conn.commit()
    conn.close()

def ch_interactions(mol_data, dist_threshold=CH_DIST_THRESHOLD):
    """C(urea)-H(choline) blocks: every choline H (CH2, CH3 and OH) within
    dist_threshold of the urea carbon, keyed by cluster_id."""
    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    blocks = []
    for ui, u in enumerate(ureas):
        c_urea = next(at for at in u.atoms if at.symbol == 'C')
        for ci, ch in enumerate(cholines):
            resp = [h for h in ch.atoms
                    if h.symbol == 'H' and distance(c_urea, h) <= dist_threshold]
            if not resp:
                continue
            blocks.append({
                'label': f"Urea {ui + 1} - Choline {ci + 1}  [C(urea)-H within {dist_threshold} A]",
                'pert':  c_urea.cluster_id,
                'resp':  [h.cluster_id for h in resp],
            })
    return blocks

def nh_interactions(mol_data, dist_threshold=NH_DIST_THRESHOLD):
    """N(urea)-H1 coupling blocks: for each urea N, every choline methyl H [H1] within
    dist_threshold, keyed by cluster_id (pert = urea N, resp = choline H(CH3); one cpl
    block per N-choline pair). Earlier runs also coupled N to the CH2 H's and CH2
    carbons; those already-computed clusters are kept, and the reader reads whatever
    responders each .out holds, so the new runs are the leaner N-H1 only."""
    ureas    = mol_data['urea']
    cholines = mol_data['choline']
    blocks = []
    for ui, u in enumerate(ureas):
        n_ureas = [at for at in u.atoms if at.symbol == 'N']
        for ni, n_urea in enumerate(n_ureas):
            for ci, ch in enumerate(cholines):
                ch3_h = [h for _c, hs in find_xh_groups(ch, 'C', 3, neighbour='N')
                         for h in hs if distance(n_urea, h) <= dist_threshold]
                if not ch3_h:
                    continue
                blocks.append({
                    'label': f"Urea {ui + 1} N{ni + 1} - Choline {ci + 1}  "
                             f"[N(urea)-H(CH3) within {dist_threshold} A]",
                    'pert':  n_urea.cluster_id,
                    'resp':  [h.cluster_id for h in ch3_h],
                })
    return blocks

def nh_intra_interactions(mol_data):
    """Intra-urea N-H coupling blocks: for each urea N, all H's in the SAME urea (its two
    bonded H's plus the two on the other N), keyed by cluster_id (pert = urea N, resp =
    urea H's; one cpl block per urea N). No distance threshold — the whole urea is small."""
    blocks = []
    for ui, u in enumerate(mol_data['urea']):
        n_ureas = [at for at in u.atoms if at.symbol == 'N']
        h_ureas = [at for at in u.atoms if at.symbol == 'H']
        if not h_ureas:
            continue
        for ni, n_urea in enumerate(n_ureas):
            blocks.append({
                'label': f"Urea {ui + 1} N{ni + 1} - intra  [N(urea)-H(urea)]",
                'pert':  n_urea.cluster_id,
                'resp':  [h.cluster_id for h in h_ureas],
            })
    return blocks

def cluster_natoms(xyz_file):
    """Atom count from an xyz's first line (0 if empty) — to process small clusters first."""
    with open(xyz_file) as f:
        line = f.readline().strip()
    return int(line) if line.isdigit() else 0

def site_interactions(mol_data, dist_threshold=SITE_DIST_THRESHOLD):
    """cpl blocks for the named-site couplings (H1-H5, Nurea; see
    hassan_functions.constants.SITE_COUPLINGS).

    Responders are merged per perturber so one cpl block covers several requested
    couplings — cpl takes one atompert against many atomresp, and the reader assigns
    each (pert, resp) pair its own pair_type from the geometry. That keeps a 2:1
    cluster down to ~9 blocks instead of one per coupling:

      pert = choline H2  -> H1 of the same choline          [H1-H2 intra]
                            + H5 of nearby ureas            [H5-H2 inter]
      pert = choline H3  -> H1 of the same choline          [H1-H3 intra]
                            + H1 of other nearby cholines   [H1-H3 inter]
                            + H5 of nearby ureas            [H5-H3 inter]
      pert = choline H4  -> same three groups               [H1-H4 intra/inter,
                                                             H5-H4 inter]
      pert = urea N      -> H1/H2/H3 of nearby cholines     [Nurea-H1/H2/H3 inter]

    The perturber is always the sparser site (H2/H3/H4 are 2/2/1 atoms against nine
    H1's), so the block count follows the small site, not the methyl multiplicity."""
    cholines = mol_data['choline']
    ureas    = mol_data['urea']
    ch_sites = [choline_sites(ch) for ch in cholines]
    u_sites  = [urea_sites(u) for u in ureas]

    def near(pert_at, candidates):
        return [a for a in candidates if distance(pert_at, a) <= dist_threshold]

    blocks = []
    for ci, sites in enumerate(ch_sites):
        own_h1   = sites['H1']
        other_h1 = [h for cj, s in enumerate(ch_sites) if cj != ci for h in s['H1']]
        urea_h5  = [h for s in u_sites for h in s['H5']]

        for site_name, perts, with_other_cholines in (
                ('H2', sites['H2'], False),   # H1-H2 is intra only
                ('H3', sites['H3'], True),
                ('H4', sites['H4'], True)):
            for p in perts:
                resp = list(own_h1)
                if with_other_cholines:
                    resp += near(p, other_h1)
                resp += near(p, urea_h5)
                if not resp:
                    continue
                blocks.append({
                    'label': f"Choline {ci + 1} {site_name} - H1 (intra"
                             f"{'/inter' if with_other_cholines else ''}) + urea H5 "
                             f"within {dist_threshold} A",
                    'pert':  p.cluster_id,
                    'resp':  [a.cluster_id for a in resp],
                })

    for ui, s in enumerate(u_sites):
        choline_h = [h for cs in ch_sites for k in ('H1', 'H2', 'H3') for h in cs[k]]
        for ni, n_at in enumerate(s['Nurea']):
            resp = near(n_at, choline_h)
            if not resp:
                continue
            blocks.append({
                'label': f"Urea {ui + 1} N{ni + 1} - choline H1/H2/H3 "
                         f"within {dist_threshold} A",
                'pert':  n_at.cluster_id,
                'resp':  [a.cluster_id for a in resp],
            })
    return blocks

def write_cpl_run(filename, basename, sorted_mols, cpl_blocks):
    """One System/engine + a cpl block per (pert, resp) entry — shared by the CH and
    NH single-perturber runs."""
    with open(filename, "w") as f:
        write_system_engine(f, sorted_mols, basename)
        for block in cpl_blocks:
            f.write(f"\n# {block['label']}\n")
            write_cpl(f, basename, block['pert'], block['resp'])

def warning_steps(cursor, variant):
    """Steps whose SCF warned for this variant (excluded from submission)."""
    placeholders = ",".join("?" for _ in SCF_WARNINGS)
    cursor.execute(
        f"SELECT n_step FROM snapshots WHERE comment_{variant} IN ({placeholders})",
        tuple(SCF_WARNINGS))
    return {r[0] for r in cursor.fetchall()}

# ── main ──────────────────────────────────────────────────────────────────
# Default: the 4-variant HH NMR workflow (seed DB + write .run/.sl). SMALL_LIMIT / SMALL_MAX
# temporarily restrict it to the N smallest clusters (by atom count) for a fast batch, e.g.
#   SMALL_LIMIT=5 $AMSBIN/plams pipeline/coupling_generator.py     # 5 smallest pending
#   SMALL_MAX=80  $AMSBIN/plams pipeline/coupling_generator.py     # only <=80-atom clusters
# Opt-in extensions (single perturber): CH=1 -> C(urea)-H, NH=1 -> N(urea)-H1,
# NH_INTRA=1 -> intra-urea N-H (own H's), SITE=1 -> the named-site couplings
# (H1-H5, Nurea). CH_LIMIT / NH_LIMIT / NH_INTRA_LIMIT / SITE_LIMIT cap those batches:
#   CH=1 $AMSBIN/plams pipeline/coupling_generator.py
#   NH=1 $AMSBIN/plams pipeline/coupling_generator.py
#   NH_INTRA=1 NH_INTRA_LIMIT=5 $AMSBIN/plams pipeline/coupling_generator.py
#   SITE=1 SITE_LIMIT=10 $AMSBIN/plams pipeline/coupling_generator.py
# IGNORE_TIME=<ns> drops the first <ns> nanoseconds of the trajectory (over-sampled MD
# start) — applies to every branch above. E.g. IGNORE_TIME=4 skips steps before 4 ns.
#
# Two restrictions apply to EVERY branch and are on by default (see the constants at
# the top of the file): MIN_STEP=60000000 and the (2,1,1)-only size limit. Per run:
#   MIN_STEP=0 ...        start from the beginning of the trajectory again
#   ALLOW_MEDIUM=1 ...    also submit (4 urea, 2 choline, 2 chloride) clusters
#   NO_SIZE_LIMIT=1 ...   submit any cluster size

init()

xyz_files = sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz")))

# IGNORE_TIME (ns): don't over-populate the equilibration start of the MD — keep only
# clusters whose MD step is at/after this many ns (step*MD_TIMESTEP_FS/1e6 >= IGNORE_TIME).
ignore_ns = float(os.environ.get("IGNORE_TIME", 0))
if ignore_ns > 0:
    n_before  = len(xyz_files)
    xyz_files = [xf for xf in xyz_files
                 if get_step_from_filename(xf) * MD_TIMESTEP_FS >= ignore_ns * 1e6]
    vprint(f"IGNORE_TIME={ignore_ns} ns (dt={MD_TIMESTEP_FS} fs): kept {len(xyz_files)}/"
          f"{n_before} clusters (>= step {int(ignore_ns * 1e6 / MD_TIMESTEP_FS)})")

# MIN_STEP: start the new batches later in the trajectory (see the constant above).
min_step = env_int("MIN_STEP", MIN_STEP)
if min_step > 0:
    n_before  = len(xyz_files)
    xyz_files = [xf for xf in xyz_files if get_step_from_filename(xf) >= min_step]
    vprint(f"MIN_STEP={min_step}: kept {len(xyz_files)}/{n_before} clusters")

# Cluster-size restriction (hassan_functions.flags): (2,1,1) by default,
# ALLOW_MEDIUM=1 adds the (4,2,2) tier, NO_SIZE_LIMIT=1 lifts it for this run.
allowed = allowed_compositions()
if allowed is not None:
    n_before  = len(xyz_files)
    xyz_files = [xf for xf in xyz_files if composition_allowed(xf, allowed)]
    vprint(f"size restriction {allowed}: kept {len(xyz_files)}/{n_before} clusters")

if os.environ.get("CH"):
    os.makedirs(CH_OUT_DIR, exist_ok=True)
    # CH_LIMIT caps how many new run files to write in this pass (0 = no cap).
    ch_limit = env_int("CH_LIMIT")
    written  = 0
    for xyz_file in xyz_files:
        if ch_limit and written >= ch_limit:
            break
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        n_step   = get_step_from_filename(xyz_file)

        if sum(1 for _ in open(xyz_file)) - 2 < 1:
            continue

        # only clusters whose H-H is already computed (smallest systems first)
        if not os.path.exists(os.path.join("amsoutput", CH_REQUIRE_VARIANT,
                                           f"{basename}.out")):
            continue

        ch_run = os.path.join(CH_OUT_DIR, f"{basename}.run")
        ch_sl  = os.path.join(CH_OUT_DIR, f"{basename}.sl")
        ch_out = os.path.join("amsoutput", "ch", f"{basename}.out")
        if os.path.exists(ch_out) or os.path.exists(ch_run) or os.path.exists(ch_sl):
            continue

        mol_data, offs, sorted_mols, counts = classify_cluster(xyz_file)
        cpl_blocks = ch_interactions(mol_data)
        if cpl_blocks:
            write_cpl_run(ch_run, basename, sorted_mols, cpl_blocks)
            with open(os.path.join(CH_OUT_DIR, f"{basename}.sl"), "w") as f:
                f.write(slurm_script(f"ch{n_step}", basename, CH_PARTITION))
            written += 1

elif os.environ.get("NH"):
    os.makedirs(NH_OUT_DIR, exist_ok=True)
    # Smallest clusters first (fast J approximations); already-computed / already-scripted
    # clusters are skipped, so this only ever writes the NEXT pending clusters. NH_LIMIT
    # caps how many new run files to write in this pass (0 = no cap).
    nh_limit = env_int("NH_LIMIT")
    written  = 0
    for xyz_file in sorted(xyz_files, key=cluster_natoms):
        if nh_limit and written >= nh_limit:
            break
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        n_step   = get_step_from_filename(xyz_file)

        if sum(1 for _ in open(xyz_file)) - 2 < 1:
            continue

        # only clusters whose H-H is already computed
        if not os.path.exists(os.path.join("amsoutput", NH_REQUIRE_VARIANT,
                                           f"{basename}.out")):
            continue

        nh_run = os.path.join(NH_OUT_DIR, f"{basename}.run")
        nh_sl  = os.path.join(NH_OUT_DIR, f"{basename}.sl")
        nh_out = os.path.join("amsoutput", "nh", f"{basename}.out")
        if os.path.exists(nh_out) or os.path.exists(nh_run) or os.path.exists(nh_sl):
            continue

        mol_data, offs, sorted_mols, counts = classify_cluster(xyz_file)
        cpl_blocks = nh_interactions(mol_data)
        if cpl_blocks:
            write_cpl_run(nh_run, basename, sorted_mols, cpl_blocks)
            with open(os.path.join(NH_OUT_DIR, f"{basename}.sl"), "w") as f:
                f.write(slurm_script(f"nh{n_step}", basename, NH_PARTITION))
            written += 1

elif os.environ.get("SITE"):
    os.makedirs(SITE_OUT_DIR, exist_ok=True)
    limit   = env_int("SITE_LIMIT")   # 0 = no cap
    written = 0
    for xyz_file in sorted(xyz_files, key=cluster_natoms):
        if limit and written >= limit:
            break
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        n_step   = get_step_from_filename(xyz_file)

        if sum(1 for _ in open(xyz_file)) - 2 < 1:
            continue

        site_run = os.path.join(SITE_OUT_DIR, f"{basename}.run")
        site_sl  = os.path.join(SITE_OUT_DIR, f"{basename}.sl")
        site_out = os.path.join("amsoutput", "site", f"{basename}.out")
        if (os.path.exists(site_out) or os.path.exists(site_run)
                or os.path.exists(site_sl)):
            continue

        mol_data, offs, sorted_mols, counts = classify_cluster(xyz_file)
        cpl_blocks = site_interactions(mol_data)
        if cpl_blocks:
            write_cpl_run(site_run, basename, sorted_mols, cpl_blocks)
            with open(site_sl, "w") as f:
                f.write(slurm_script(f"site{n_step}", basename, SITE_PARTITION))
            written += 1
    vprint(f"SITE: wrote {written} run/sl pairs into {SITE_OUT_DIR}")

elif os.environ.get("NH_INTRA"):
    os.makedirs(NH_INTRA_OUT_DIR, exist_ok=True)
    limit   = env_int("NH_INTRA_LIMIT")   # 0 = no cap
    written = 0
    for xyz_file in sorted(xyz_files, key=cluster_natoms):
        if limit and written >= limit:
            break
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        n_step   = get_step_from_filename(xyz_file)

        if sum(1 for _ in open(xyz_file)) - 2 < 1:
            continue

        # only clusters whose H-H is already computed
        if not os.path.exists(os.path.join("amsoutput", NH_INTRA_REQUIRE_VARIANT,
                                           f"{basename}.out")):
            continue

        nhi_run = os.path.join(NH_INTRA_OUT_DIR, f"{basename}.run")
        nhi_sl  = os.path.join(NH_INTRA_OUT_DIR, f"{basename}.sl")
        nhi_out = os.path.join("amsoutput", "nh_intra", f"{basename}.out")
        if os.path.exists(nhi_out) or os.path.exists(nhi_run) or os.path.exists(nhi_sl):
            continue

        mol_data, offs, sorted_mols, counts = classify_cluster(xyz_file)
        cpl_blocks = nh_intra_interactions(mol_data)
        if cpl_blocks:
            write_cpl_run(nhi_run, basename, sorted_mols, cpl_blocks)
            with open(nhi_sl, "w") as f:
                f.write(slurm_script(f"nhi{n_step}", basename, NH_INTRA_PARTITION))
            written += 1

else:
    conn = sqlite3.connect(DB_PATH)
    warned = {variant: warning_steps(conn.cursor(), variant) for variant, _, _ in VARIANTS}
    conn.close()

    # SMALL_LIMIT / SMALL_MAX: temporarily send only the N smallest clusters (by atom
    # count) to CRIANN for a fast batch; unset -> original order and no cap.
    small_limit = env_int("SMALL_LIMIT")
    small_max   = env_int("SMALL_MAX")
    # VARIANTS="a,b" generates only those variants this pass (unset = all four)
    only_variants = env_list("VARIANTS")
    order   = sorted(xyz_files, key=cluster_natoms) if (small_limit or small_max) else xyz_files
    written = 0

    for xyz_file in order:
        if small_limit and written >= small_limit:
            break
        if small_max and cluster_natoms(xyz_file) > small_max:
            break                          # sorted ascending -> the rest are all bigger
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        n_step   = get_step_from_filename(xyz_file)

        if sum(1 for _ in open(xyz_file)) - 2 < 1:
            add_snapshot_to_db_error(DB_PATH, n_step)
            continue

        # check the files first: only variants without out/run/sl (and not
        # SCF-warned) need generating — skip the whole PLAMS step if none do
        todo = []
        for variant, contributions, j_basis in VARIANTS:
            if only_variants and variant not in only_variants:
                continue
            out_file   = os.path.join("amsoutput", variant, f"{basename}.out")
            run_script = os.path.join("run_scripts", variant, f"{basename}.run")
            sl_script  = os.path.join("run_scripts", variant, f"{basename}.sl")
            if (os.path.exists(out_file) or os.path.exists(run_script)
                    or os.path.exists(sl_script) or n_step in warned[variant]):
                continue
            todo.append((variant, contributions, j_basis, run_script, sl_script))

        if not todo:
            continue

        mol_data, offs, sorted_mols, counts = classify_cluster(xyz_file)
        intra = intra_choline_interactions(mol_data, offs['choline'])
        inter = inter_nh2_ch3_interactions(mol_data, offs['urea'], offs['choline'],
                                           DISTANCE_THRESHOLD)
        add_snapshot_to_db(DB_PATH, n_step, counts['choline'], intra, inter)

        for variant, contributions, j_basis, run_script, sl_script in todo:
            write_main_run(sorted_mols, run_script, intra, inter, contributions, j_basis)
            cfg  = VARIANT_SLURM[variant]
            part = partition_override() or cfg["partition"]
            wall = PARTITION_WALLTIME[part] if partition_override() else cfg["walltime"]
            with open(sl_script, "w") as f:
                f.write(slurm_script(str(n_step), basename, part, wall))
        written += 1                       # one cluster's run/sl written (for SMALL_LIMIT)

finish()

