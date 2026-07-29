#!/usr/bin/python3
import sqlite3

VARIANTS = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]
CH_VARIANTS = ["TZ2P_FC"]   # C(urea)-H(choline)    coupling variants computed so far
NH_VARIANTS = ["TZ2P_FC"]   # N(urea)-choline       coupling variants computed so far
NH_INTRA_VARIANTS = ["TZ2P_FC"]   # intra-urea N-H  coupling variants computed so far

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

