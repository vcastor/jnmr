#!/usr/bin/python3
import sqlite3

VARIANTS = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]

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

db_path = "nmr_jcoupling.db"
# init_database(db_path)      # uncomment to create from scratch
# add_comment_columns(db_path)   # safe to re-run (skips existing columns)

