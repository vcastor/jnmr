
def table_exists(cursor, name):
    cur = cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cur.fetchone() is not None

def column_exists(cursor, table, column):
    cur = cursor.execute(f"PRAGMA table_info({table})")
    return any(r[1] == column for r in cur.fetchall())

