#!/usr/bin/python3
import sys
import sqlite3

DB_PATH  = "nmr_jcoupling.db"
VARIANTS = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]

REFERENCE = "TZ2PJ_all"
TARGET    = "TZ2P_FC"
OTHERS    = [v for v in VARIANTS if v not in (REFERENCE, TARGET)]

def check_step(cursor, n_step: int, kind: str) -> dict:
    table = f"step_{n_step}_{kind}"
    cols  = [f"J_{REFERENCE}", f"J_{TARGET}"] + [f"J_{v}" for v in OTHERS]
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if cursor.fetchone() is None:
        return {c: None for c in cols}
    col_list = ", ".join(cols)
    cursor.execute(f"SELECT {col_list} FROM {table}")
    rows = cursor.fetchall()
    if not rows:
        return {c: None for c in cols}
    return {cols[i]: all(row[i] is not None for row in rows) for i in range(len(cols))}

def comment_excludes(cursor, n_step: int, variant: str) -> bool:
    cursor.execute(
        f"SELECT comment_{variant} FROM snapshots WHERE n_step = ?",
        (n_step,),
    )
    c = cursor.fetchone()[0]
    return c is not None


conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT n_step FROM snapshots ORDER BY n_step")
steps = [r[0] for r in cursor.fetchall()]

header_variants = [REFERENCE, TARGET] + OTHERS
print('step | ' + ' | '.join([f"{v:>12}" for v in header_variants]))
for n_step in steps:
    if comment_excludes(cursor, n_step, TARGET):
        continue
    status = check_step(cursor, n_step, 'intra')
    vals = []
    for v in header_variants:
        col = f"J_{v}"
        s = status[col]
        vals.append("OK" if s is True else ("MISSING" if s is False else "NO TABLE"))
    if all(vals[i] == "MISSING" for i in range(1, len(vals))):
        continue
    print(f"step {n_step:>4} | " + ' | '.join([f"{v:>12}" for v in vals]))

conn.close()

