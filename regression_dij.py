#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = "nmr_jcoupling.db"
JCOL    = "J_TZ2P_FC"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'step_%_intra'")
tables = [r[0] for r in cur.fetchall()]

di, j = [], []
for table in tables:
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}
    if "DI" not in cols or JCOL not in cols:
        continue
    cur.execute(f"SELECT DI, {JCOL} FROM {table} WHERE DI IS NOT NULL AND {JCOL} IS NOT NULL")
    for d, x in cur.fetchall():
        di.append(d)
        j.append(x)

conn.close()

di = np.array(di); j = np.array(j)

print(f"Number of (DI, J) points: {len(di)}")

plt.figure(figsize=(7,5))
plt.scatter(di, np.abs(j), s=10, alpha=0.6)
plt.xlabel("DI")
plt.ylabel("|J| (Hz)")
plt.title(f"DI vs |J| ({JCOL})")
plt.tight_layout()
plt.show()

