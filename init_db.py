#!/usr/bin/python3
import sqlite3

VARIANTS = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]
CH_VARIANTS = ["TZ2P_FC"]   # C(urea)-H(choline)    coupling variants computed so far
NH_VARIANTS = ["TZ2P_FC"]   # N(urea)-choline       coupling variants computed so far
NH_INTRA_VARIANTS = ["TZ2P_FC"]   # intra-urea N-H  coupling variants computed so far
SITE_VARIANTS = ["TZ2P_FC"]       # named-site (H1-H5, Nurea) coupling variants so far

def init_database(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    comment_cols = ", ".join(f"comment_{v} TEXT" for v in VARIANTS)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS snapshots (
            n_step    INTEGER PRIMARY KEY,
            n_choline INTEGER,
            n_inter   INTEGER,
            comment   TEXT,
            {comment_cols}
        )
    ''')

    conn.commit()
    conn.close()

def add_comment_columns(db_path: str) -> None:
    """Add variant comment columns if they don't exist yet."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(snapshots)")}

    for v in VARIANTS:
        col = f"comment_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE snapshots ADD COLUMN {col} TEXT")

    conn.commit()
    conn.close()

def init_ch_table(db_path: str) -> None:
    """Flat table for C(urea)-H(choline) couplings: one row per (step, C, H) pair."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    j_cols       = ", ".join(f"J_{v} REAL" for v in CH_VARIANTS)
    comment_cols = ", ".join(f"comment_{v} TEXT" for v in CH_VARIANTS)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS ch_coupling (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            n_step   INTEGER,
            C_pert   INTEGER,
            H_resp   INTEGER,
            distance REAL,
            {j_cols},
            {comment_cols},
            UNIQUE(n_step, C_pert, H_resp)
        )
    ''')

    conn.commit()
    conn.close()

def add_ch_jcolumns(db_path: str) -> None:
    """Add CH J_{variant} columns if they don't exist yet (safe to re-run)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(ch_coupling)")}
    for v in CH_VARIANTS:
        col = f"J_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE ch_coupling ADD COLUMN {col} REAL")

    conn.commit()
    conn.close()

def add_ch_comment_columns(db_path: str) -> None:
    """Add CH comment_{variant} columns if they don't exist yet (safe to re-run).
    They hold the SCF-convergence warning of the step, mirroring comment_{variant}
    on snapshots, so numerically suspect C-H couplings can be flagged downstream."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(ch_coupling)")}
    for v in CH_VARIANTS:
        col = f"comment_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE ch_coupling ADD COLUMN {col} TEXT")

    conn.commit()
    conn.close()

def init_nh_table(db_path: str) -> None:
    """Flat table for N(urea)-choline couplings: one row per (step, N, responder) pair.
    The perturber is a urea N; the responder is a choline methyl H or a CH2 carbon.
    resp_type (filled by the reader) records which group the responder sits on."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    j_cols       = ", ".join(f"J_{v} REAL" for v in NH_VARIANTS)
    comment_cols = ", ".join(f"comment_{v} TEXT" for v in NH_VARIANTS)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS nh_coupling (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            n_step   INTEGER,
            N_pert   INTEGER,
            resp     INTEGER,
            distance REAL,
            {j_cols},
            {comment_cols},
            UNIQUE(n_step, N_pert, resp)
        )
    ''')

    conn.commit()
    conn.close()

def add_nh_jcolumns(db_path: str) -> None:
    """Add NH J_{variant} columns if they don't exist yet (safe to re-run)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(nh_coupling)")}
    for v in NH_VARIANTS:
        col = f"J_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE nh_coupling ADD COLUMN {col} REAL")

    conn.commit()
    conn.close()

def add_nh_comment_columns(db_path: str) -> None:
    """Add NH comment_{variant} columns if they don't exist yet (safe to re-run).
    They hold the SCF-convergence warning of the step, mirroring comment_{variant}
    on snapshots, so numerically suspect N couplings can be flagged downstream."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(nh_coupling)")}
    for v in NH_VARIANTS:
        col = f"comment_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE nh_coupling ADD COLUMN {col} TEXT")

    conn.commit()
    conn.close()

def init_nh_intra_table(db_path: str) -> None:
    """Flat table for intra-urea N-H couplings: one row per (step, N, urea-H) pair.
    Separate from nh_coupling because this is a within-urea coupling (the urea N to its
    own H's), physically distinct from the through-space N-choline couplings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    j_cols       = ", ".join(f"J_{v} REAL" for v in NH_INTRA_VARIANTS)
    comment_cols = ", ".join(f"comment_{v} TEXT" for v in NH_INTRA_VARIANTS)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS nh_intra_coupling (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            n_step   INTEGER,
            N_pert   INTEGER,
            H_resp   INTEGER,
            distance REAL,
            {j_cols},
            {comment_cols},
            UNIQUE(n_step, N_pert, H_resp)
        )
    ''')

    conn.commit()
    conn.close()

def add_nh_intra_jcolumns(db_path: str) -> None:
    """Add intra-urea NH J_{variant} columns if they don't exist yet (safe to re-run)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(nh_intra_coupling)")}
    for v in NH_INTRA_VARIANTS:
        col = f"J_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE nh_intra_coupling ADD COLUMN {col} REAL")

    conn.commit()
    conn.close()

def add_nh_intra_comment_columns(db_path: str) -> None:
    """Add intra-urea NH comment_{variant} columns if they don't exist yet (safe to
    re-run). They hold the SCF-convergence warning, so numerically suspect couplings can
    be flagged downstream (as for every other J table)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(nh_intra_coupling)")}
    for v in NH_INTRA_VARIANTS:
        col = f"comment_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE nh_intra_coupling ADD COLUMN {col} TEXT")

    conn.commit()
    conn.close()

def init_site_table(db_path: str) -> None:
    """Flat table for the named-site couplings requested by the experimental team:
    one row per (step, perturber, responder) pair.

    pair_type is a key of constants.SITE_COUPLINGS ('H1-H2', 'H5-H3', 'Nurea-H1', ...)
    and scope is 'intra' (both atoms in the same molecule) or 'inter' (different
    molecules); together they say exactly which coupling the row holds. See the
    "Named-site J couplings" table in README.md for what each label means.

    One table rather than one per label: the nine couplings share every column, and a
    single table lets the analysis group by pair_type instead of UNIONing thirteen
    near-identical tables. pert/resp are cluster atom numbers, so a row is unique on
    (n_step, pert, resp) — a given atom pair belongs to exactly one pair_type."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    j_cols       = ", ".join(f"J_{v} REAL" for v in SITE_VARIANTS)
    comment_cols = ", ".join(f"comment_{v} TEXT" for v in SITE_VARIANTS)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS site_coupling (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            n_step    INTEGER,
            pair_type TEXT,
            scope     TEXT,
            pert      INTEGER,
            resp      INTEGER,
            pert_site TEXT,
            resp_site TEXT,
            distance  REAL,
            {j_cols},
            {comment_cols},
            UNIQUE(n_step, pert, resp)
        )
    ''')
    # grouping by coupling is the common query, so index it
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_site_pair "
                   "ON site_coupling (pair_type, scope)")

    conn.commit()
    conn.close()

def add_site_jcolumns(db_path: str) -> None:
    """Add named-site J_{variant} columns if they don't exist yet (safe to re-run).
    Extend SITE_VARIANTS and re-run to add a basis/contribution variant later."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(site_coupling)")}
    for v in SITE_VARIANTS:
        col = f"J_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE site_coupling ADD COLUMN {col} REAL")

    conn.commit()
    conn.close()

def add_site_comment_columns(db_path: str) -> None:
    """Add named-site comment_{variant} columns if they don't exist yet (safe to
    re-run). They hold the SCF-convergence warning of the step, so numerically
    suspect couplings can be flagged downstream (as for every other J table)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(site_coupling)")}
    for v in SITE_VARIANTS:
        col = f"comment_{v}"
        if col not in existing:
            cursor.execute(f"ALTER TABLE site_coupling ADD COLUMN {col} TEXT")

    conn.commit()
    conn.close()

db_path = "nmr_jcoupling.db"
# init_database(db_path)      # uncomment to create from scratch
# add_comment_columns(db_path)   # safe to re-run (skips existing columns)
# init_ch_table(db_path)      # create ch_coupling (safe: CREATE IF NOT EXISTS)
# add_ch_jcolumns(db_path)    # add new CH J_{variant} columns later (safe to re-run)
# add_ch_comment_columns(db_path)  # add CH comment_{variant} columns (safe to re-run)
# init_nh_table(db_path)      # create nh_coupling (safe: CREATE IF NOT EXISTS)
# add_nh_jcolumns(db_path)    # add new NH J_{variant} columns later (safe to re-run)
# add_nh_comment_columns(db_path)  # add NH comment_{variant} columns (safe to re-run)
# init_nh_intra_table(db_path)     # create nh_intra_coupling (safe: CREATE IF NOT EXISTS)
# add_nh_intra_jcolumns(db_path)   # add new intra-NH J_{variant} columns (safe to re-run)
# add_nh_intra_comment_columns(db_path)  # add intra-NH comment_{variant} columns (safe to re-run)
# init_site_table(db_path)          # create site_coupling (safe: CREATE IF NOT EXISTS)
# add_site_jcolumns(db_path)        # add new named-site J_{variant} columns (safe to re-run)
# add_site_comment_columns(db_path) # add named-site comment_{variant} columns (safe to re-run)

