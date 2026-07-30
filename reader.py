#!/usr/bin/python3
import os
import re
import glob
import sqlite3
import subprocess
import numpy as np
from typing import Optional
from hassan_functions.db import table_exists, column_exists
from hassan_functions.io import get_step_from_filename, read_labeled_matrix
from hassan_functions.criann import SCF_WARNINGS

DB_PATH    = "nmr_jcoupling.db"
VARIANTS   = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]
QTAIM_DIR  = "amsoutput/qtaim"
CDFT_DIR   = "amsoutput/cdft"
CH_DIR     = "amsoutput/ch"
NH_DIR     = "amsoutput/nh"
DI_HEADER  = "LOCALIZATION AND DELOCALIZATION INDEXES (MATRIX ELEMENTS)"
CHI_HEADER = "CONDENSED LINEAR RESPONSE FUNCTION (MATRIX ELEMENTS)"
GRID_WARNING = "WARNING: Nr of shared arrays in use at the end"

INTRA_TERMS = ["HpHr", "HpCp", "HpCr", "CpCr", "HrCp", "HrCr"]
INTER_TERMS = ["HpHr"]

# Coupling-block header shared by every cpl parser (HH main runs, CH, NH):
# "<sym>(<num>) perturbing <sym>(<num>)".
FC_HEADER_RE = re.compile(
    r"Atom input numbers in the ADF calculation:\s+"
    r"(\w+)\((\d+)\) perturbing\s+(\w+)\((\d+)\)")
# Isotropic-J section markers inside a block. An FC-only run exposes only the
# 'fermi-contact contribution'; a dso/pso/sd run (…_all variants) adds a 'total
# calculated spin-spin coupling' section, which is the physical coupling to read.
FC_MARKER    = "fermi-contact contribution"
TOTAL_MARKER = "total calculated spin-spin coupling"

# ── SCF convergence warnings ────────────────────────────────────────────────

def mark_warnings(cursor):
    """Set comment_{variant} for steps whose SCF did not (fully) converge."""
    for variant in VARIANTS:
        output_dir  = f"amsoutput/{variant}"
        comment_col = f"comment_{variant}"

        # Clear stale warnings first: a previously-failing calc may have been
        # rerun and now converges, in which case the old comment must not stick
        # around (otherwise the step is silently excluded downstream).
        cursor.execute(f"UPDATE snapshots SET {comment_col} = NULL")

        for pattern in SCF_WARNINGS:
            result = subprocess.run(
                f"grep -rl '{pattern}' {output_dir}",
                shell=True, capture_output=True, text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                continue
            for filepath in result.stdout.strip().split('\n'):
                filename = filepath.split('/')[-1]
                n_step = int(filename.replace('MDStep', '').replace('_cluster.out', ''))
                cursor.execute(
                    f"UPDATE snapshots SET {comment_col} = ? WHERE n_step = ?",
                    (pattern, n_step),
                )

# ── J-coupling values ───────────────────────────────────────────────────────

def get_pending_steps(cursor):
    cursor.execute("SELECT n_step FROM snapshots WHERE comment IS NULL")
    return [row[0] for row in cursor.fetchall()]

def check_jcolumn_filled(cursor, table_name, j_col) -> bool:
    """Check if a specific J column is fully filled in a table."""
    if not table_exists(cursor, table_name):
        return False
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {j_col} IS NOT NULL")
    filled = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total = cursor.fetchone()[0]
    return filled == total and total > 0

def _iso_j(line):
    """Isotropic j (last field) of an 'isotropic k= .. j= ..' line, or None."""
    try:
        return float(line.split()[-1])
    except (ValueError, IndexError):
        return None

def parse_all_couplings(filepath):
    """{(pert_num, resp_num): isotropic J (Hz)} for every coupling block in an ADF cpl
    output. Reads the 'total calculated spin-spin coupling' isotropic j when the run has
    the dso/pso/sd contributions (…_all variants), else the 'fermi-contact contribution'
    isotropic j (FC-only). This is the fix for the …_all values previously read from the
    wrong (diamagnetic) contribution because of a fixed line offset. The isotropic j is
    the line right after each section marker; blocks are split on the header regex, and
    atoms are matched exactly (no substring collisions)."""
    with open(filepath) as f:
        lines = f.readlines()
    out = {}
    i, n = 0, len(lines)
    while i < n:
        m = FC_HEADER_RE.search(lines[i])
        if not m:
            i += 1
            continue
        pert, resp = int(m.group(2)), int(m.group(4))
        fc = total = None
        j = i + 1
        while j < n and not FC_HEADER_RE.search(lines[j]):
            s = lines[j].strip()
            if s.startswith(TOTAL_MARKER) and j + 1 < n:
                total = _iso_j(lines[j + 1])
            elif s.startswith(FC_MARKER) and j + 1 < n:
                fc = _iso_j(lines[j + 1])
            j += 1
        val = total if total is not None else fc
        if val is not None:
            out[(pert, resp)] = val
        i = j
    return out

def fill_j(conn, cursor):
    """Fill J_{variant} in step_{n}_intra / step_{n}_inter from ADF output."""
    for n_step in get_pending_steps(cursor):
        filename = f"MDStep{n_step}_cluster.out"

        for variant in VARIANTS:
            filepath = os.path.join(f"amsoutput/{variant}", filename)
            j_col    = f"J_{variant}"

            if (check_jcolumn_filled(cursor, f"step_{n_step}_intra", j_col)
                    and check_jcolumn_filled(cursor, f"step_{n_step}_inter", j_col)):
                continue
            if not os.path.exists(filepath):
                continue

            couplings = parse_all_couplings(filepath)   # parse the file once
            for suffix in ("intra", "inter"):
                table = f"step_{n_step}_{suffix}"
                if not table_exists(cursor, table):
                    continue
                cursor.execute(
                    f"SELECT H_pert, H_resp FROM {table} WHERE {j_col} IS NULL")
                for h_pert, h_resp in cursor.fetchall():
                    j_value = couplings.get((h_pert, h_resp))
                    if j_value is not None:
                        cursor.execute(
                            f"UPDATE {table} SET {j_col} = ? WHERE H_pert = ? AND H_resp = ?",
                            (j_value, h_pert, h_resp))
            conn.commit()

# ── QTAIM (DI) and CDFT (\chi) ───────────────────────────────────────────────────

def step_tables(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name GLOB 'step_*_intra' OR name GLOB 'step_*_inter')")
    return [row[0] for row in cursor.fetchall()]

def table_exists_conn(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,)).fetchone() is not None

def update_matrix_for_step(conn, nstep, atom_labels, matrix, prop):
    idx = {atom: i for i, atom in enumerate(atom_labels)}
    def v(a, b):
        return float(matrix[idx[a], idx[b]])

    inter = f"step_{nstep}_inter"
    if table_exists_conn(conn, inter):
        for rid, hp, hr in conn.execute(
                f"SELECT id, H_pert, H_resp FROM {inter}").fetchall():
            conn.execute(
                f"UPDATE {inter} SET {prop}_HpHr = ? WHERE id = ?", (v(hp, hr), rid))

    intra = f"step_{nstep}_intra"
    if table_exists_conn(conn, intra):
        for rid, hp, hr, cp, cr in conn.execute(
                f"SELECT id, H_pert, H_resp, C_pert, C_resp FROM {intra}").fetchall():
            conn.execute(
                f"UPDATE {intra} SET {prop}_HpHr = ?, {prop}_HpCp = ?, {prop}_HpCr = ?, "
                f"{prop}_CpCr = ?, {prop}_HrCp = ?, {prop}_HrCr = ? WHERE id = ?",
                (v(hp, hr), v(hp, cp), v(hp, cr), v(cp, cr), v(hr, cp), v(hr, cr), rid))

def check_output_ok(path):
    if subprocess.run(["grep", "-qF", GRID_WARNING, path]).returncode == 0:
        return False
    return subprocess.run(["grep", "-qF", "NORMAL TERMINATION", path]).returncode == 0

def _step_needs_prop(cursor, nstep, prop):
    for suffix in ("intra", "inter"):
        table = f"step_{nstep}_{suffix}"
        if not table_exists(cursor, table):
            continue
        terms = INTRA_TERMS if suffix == "intra" else INTER_TERMS
        for col in [f"{prop}_{t}" for t in terms]:
            if not column_exists(cursor, table, col):
                return True
            cursor.execute(f"SELECT 1 FROM {table} WHERE {col} IS NULL LIMIT 1")
            if cursor.fetchone() is not None:
                return True
    return False

def fill_di_chi(conn, cursor):
    for table in step_tables(cursor):
        terms = INTRA_TERMS if table.endswith("_intra") else INTER_TERMS
        for col in [f"{p}_{t}" for p in ("DI", "chi") for t in terms]:
            if not column_exists(cursor, table, col):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
    conn.commit()

    for path in sorted(glob.glob(os.path.join(QTAIM_DIR, "*.out"))):
        nstep = get_step_from_filename(path)
        if not _step_needs_prop(cursor, nstep, "DI"):
            continue
        if not check_output_ok(path):
            continue
        atom_labels, di = read_labeled_matrix(path, DI_HEADER)
        update_matrix_for_step(conn, nstep, atom_labels, di, "DI")

    for path in sorted(glob.glob(os.path.join(CDFT_DIR, "*.out"))):
        nstep = get_step_from_filename(path)
        if not _step_needs_prop(cursor, nstep, "chi"):
            continue
        if not check_output_ok(path):
            continue
        atom_labels, chi = read_labeled_matrix(path, CHI_HEADER)
        update_matrix_for_step(conn, nstep, atom_labels, chi, "chi")

    conn.commit()

# ── C(urea)-H(choline) couplings ─────────────────────────────────────────────

CH_VARIANT     = "TZ2P_FC"          # only CH variant computed so far
CH_J_COL       = f"J_{CH_VARIANT}"
CH_COMMENT_COL = f"comment_{CH_VARIANT}"

def read_out_atoms(out_path):
    """{atom_num: (symbol, xyz)} from the 'Geometry / Atoms' echo of an ADF output.
    The Index column is the ADF atom input number, i.e. the coupling atom number /
    cluster_id (Å, input orientation). Read from the .out so no .run/.xyz is needed."""
    with open(out_path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith("Index Symbol")) + 1
    atoms = {}
    for l in lines[start:]:
        p = l.split()
        if len(p) != 5:
            break
        atoms[int(p[0])] = (p[1], np.array([float(p[2]), float(p[3]), float(p[4])]))
    return atoms

# choline H-type tags for the CH coupling: which group the responding H sits on.
# From the .out geometry alone — the H's nearest heavy atom is its bond partner, and
# a carbon partner is CH3 / CH2-N / CH2-O by its own neighbours (mirrors the QTAIM
# site_tag/carbon_tag). Every CH resp H is a choline H, so C- or O-bound.
CH_XH_BOND = 1.3   # X-H covalent bond ceiling (Å)
CH_CC_BOND = 1.8   # C-C / C-N / C-O covalent bond ceiling (Å)

def classify_h(h_num, atoms):
    hx    = atoms[h_num][1]
    heavy = min((n for n, (s, _) in atoms.items() if s != 'H'),
                key=lambda n: np.linalg.norm(atoms[n][1] - hx))
    hsym, cx = atoms[heavy]
    if hsym != 'C':
        return f"H({hsym})"                     # choline OH -> H(O)
    nb = [(s, np.linalg.norm(x - cx)) for n, (s, x) in atoms.items() if n != heavy]
    if sum(1 for s, d in nb if s == 'H' and d < CH_XH_BOND) == 3:
        return "H(CH3)"
    heavy_nb = {s for s, d in nb if s in ('N', 'O') and d < CH_CC_BOND}
    return "H(CH2N)" if 'N' in heavy_nb else "H(CH2O)"

def parse_fc_couplings(out_path):
    """Yield (pert_num, resp_num, j) for each FC coupling block in a single-perturber
    cpl output (CH: pert = urea C; NH: pert = urea N). The isotropic FC j value sits
    9 lines below the 'Atom input numbers' header."""
    with open(out_path) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        m = FC_HEADER_RE.search(l)
        if not m:
            continue
        yield int(m.group(2)), int(m.group(4)), float(lines[i + 9].split()[-1])

def scf_warning(path):
    """The SCF-convergence warning found in the output, or None. Unconverged SCF is
    what produces the nonphysical ~500 Hz C-H values, so the coupling is still stored
    but flagged via comment_{variant} (mirroring the HH snapshots) so the analysis can
    exclude it rather than silently dropping the whole step."""
    for w in SCF_WARNINGS:
        if subprocess.run(["grep", "-qF", w, path]).returncode == 0:
            return w
    return None

def fill_ch_j(conn, cursor):
    """Fill ch_coupling.J_TZ2P_FC, distance, comment_TZ2P_FC and h_type from
    amsoutput/ch/*.out. h_type is the choline group the responding H sits on
    (H(CH3) / H(CH2N) / H(CH2O) / H(O)). SCF-warned steps are kept, with the
    warning recorded in comment_TZ2P_FC."""
    for col in (CH_COMMENT_COL, "h_type"):
        if not column_exists(cursor, "ch_coupling", col):
            conn.execute(f"ALTER TABLE ch_coupling ADD COLUMN {col} TEXT")
    conn.commit()

    for out_path in sorted(glob.glob(os.path.join(CH_DIR, "*.out"))):
        if not check_output_ok(out_path):
            continue
        warning = scf_warning(out_path)
        n_step = get_step_from_filename(out_path)
        atoms  = read_out_atoms(out_path)
        for c_num, h_num, j in parse_fc_couplings(out_path):
            d  = float(np.linalg.norm(atoms[c_num][1] - atoms[h_num][1]))
            ht = classify_h(h_num, atoms)
            cursor.execute(
                f"INSERT INTO ch_coupling "
                f"(n_step, C_pert, H_resp, distance, {CH_J_COL}, {CH_COMMENT_COL}, h_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(n_step, C_pert, H_resp) DO UPDATE SET "
                f"distance = excluded.distance, {CH_J_COL} = excluded.{CH_J_COL}, "
                f"{CH_COMMENT_COL} = excluded.{CH_COMMENT_COL}, h_type = excluded.h_type",
                (n_step, c_num, h_num, d, j, warning, ht))
        conn.commit()

# ── N(urea)-choline couplings ─────────────────────────────────────────────────

NH_VARIANT     = "TZ2P_FC"          # only NH variant computed so far
NH_J_COL       = f"J_{NH_VARIANT}"
NH_COMMENT_COL = f"comment_{NH_VARIANT}"

def classify_nh_resp(num, atoms):
    """Responder group for an N(urea) coupling. The nh run couples each urea N to the
    choline methyl H's, the choline CH2 hydrogens and the choline CH2 carbons, so the
    responder is either an H (methyl -> H(CH3), CH2 -> H(CH2N)/H(CH2O), both via
    classify_h) or a CH2 carbon, tagged C(CH2N)/C(CH2O) by the heavy atom (N+/O) it is
    bonded to."""
    sym, cx = atoms[num]
    if sym == 'H':
        return classify_h(num, atoms)
    if sym == 'C':
        heavy_nb = {s for n, (s, x) in atoms.items()
                    if n != num and s in ('N', 'O')
                    and np.linalg.norm(x - cx) < CH_CC_BOND}
        if 'N' in heavy_nb:
            return "C(CH2N)"
        if 'O' in heavy_nb:
            return "C(CH2O)"
        return "C(CH3)"
    return f"{sym}({sym})"

def fill_nh_j(conn, cursor):
    """Fill nh_coupling.J_TZ2P_FC, distance, comment_TZ2P_FC and resp_type from
    amsoutput/nh/*.out. Perturber is a urea N; responders are choline methyl H's
    [H(CH3)], CH2 hydrogens [H(CH2N)/H(CH2O)] and CH2 carbons [C(CH2N)/C(CH2O)].
    resp_type records which group the responder sits on so the analysis can split by
    it, mirroring ch_coupling.h_type. SCF-warned steps are kept, with the warning
    recorded in comment_TZ2P_FC."""
    # nh_coupling is created by init_db.init_nh_table (as ch_coupling is by
    # init_ch_table). Until that has been run there is nothing to fill — skip rather
    # than crash on the ALTERs below (mirrors the guard in analysis/nh_coupling.py).
    if not table_exists(cursor, "nh_coupling"):
        print("nh_coupling table absent; run init_nh_table (init_db.py) first "
              "-- skipping NH couplings.")
        return
    for col in (NH_COMMENT_COL, "resp_type"):
        if not column_exists(cursor, "nh_coupling", col):
            conn.execute(f"ALTER TABLE nh_coupling ADD COLUMN {col} TEXT")
    conn.commit()

    for out_path in sorted(glob.glob(os.path.join(NH_DIR, "*.out"))):
        if not check_output_ok(out_path):
            continue
        warning = scf_warning(out_path)
        n_step  = get_step_from_filename(out_path)
        atoms   = read_out_atoms(out_path)
        for n_num, r_num, j in parse_fc_couplings(out_path):
            d  = float(np.linalg.norm(atoms[n_num][1] - atoms[r_num][1]))
            rt = classify_nh_resp(r_num, atoms)
            cursor.execute(
                f"INSERT INTO nh_coupling "
                f"(n_step, N_pert, resp, distance, {NH_J_COL}, {NH_COMMENT_COL}, resp_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(n_step, N_pert, resp) DO UPDATE SET "
                f"distance = excluded.distance, {NH_J_COL} = excluded.{NH_J_COL}, "
                f"{NH_COMMENT_COL} = excluded.{NH_COMMENT_COL}, resp_type = excluded.resp_type",
                (n_step, n_num, r_num, d, j, warning, rt))
        conn.commit()

if __name__ == "__main__":
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    mark_warnings(cursor)
    conn.commit()
    fill_j(conn, cursor)
    fill_di_chi(conn, cursor)
    fill_ch_j(conn, cursor)
    fill_nh_j(conn, cursor)

    conn.close()

