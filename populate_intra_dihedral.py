#!$AMSBIN/plams
import os
import re
import glob
import sqlite3
import numpy as np

DB_PATH      = "nmr_jcoupling.db"
CLUSTERS_DIR = "clusters"

def get_step(filename):
    m = re.search(r'(\d+)', os.path.splitext(os.path.basename(filename))[0])
    return int(m.group(1)) if m else 0

def canonical_order(mol):
    """BFS canonical order, identical to run_generator.py."""
    n = len(mol.atoms)
    if mol.get_formula() == 'Cl':
        return [0]
    adj = {i: [] for i in range(n)}
    for bond in mol.bonds:
        i = mol.atoms.index(bond.atom1)
        j = mol.atoms.index(bond.atom2)
        adj[i].append(j)
        adj[j].append(i)
    counts = {}
    for at in mol.atoms:
        counts[at.symbol] = counts.get(at.symbol, 0) + 1
    start = 0
    pri = (counts[mol.atoms[0].symbol], mol.atoms[0].symbol)
    for i, at in enumerate(mol.atoms):
        p = (counts[at.symbol], at.symbol)
        if p < pri:
            pri, start = p, i
    visited = [False]*n
    order = []
    queue = [start]
    visited[start] = True
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        nbrs = sorted([k for k in adj[cur] if not visited[k]],
                      key=lambda k: (mol.atoms[k].symbol, len(adj[k])))
        for k in nbrs:
            visited[k] = True
            queue.append(k)
    return order

def reorder(mol):
    order = canonical_order(mol)
    new   = Molecule()
    new.properties = mol.properties.copy()
    old_to_new = {old: i for i, old in enumerate(order)}
    for old in order:
        at = mol.atoms[old]
        new.add_atom(Atom(atnum=at.atnum, coords=at.coords))
    for bond in mol.bonds:
        i = mol.atoms.index(bond.atom1)
        j = mol.atoms.index(bond.atom2)
        new.add_bond(new.atoms[old_to_new[i]], new.atoms[old_to_new[j]], bond.order)
    return new


def classify_sort(molecules, centre):
    urea, choline, chloride = [], [], []
    for mol in molecules:
        f = mol.get_formula()
        d = np.linalg.norm(np.array(mol.get_center_of_mass()) - centre)
        rmol = reorder(mol)
        if f == 'CH4N2O':
            urea.append((rmol, d))
        elif f == 'C5H14NO':
            choline.append((rmol, d))
        elif f == 'Cl':
            chloride.append((rmol, d))
    urea.sort(key=lambda x: x[1])
    choline.sort(key=lambda x: x[1])
    chloride.sort(key=lambda x: x[1])
    return ([m for m, _ in urea],
            [m for m, _ in choline],
            [m for m, _ in chloride])

def offsets(ureas, cholines, chlorides):
    off = 0
    cho_off = []
    for u in ureas:
        off += len(u)
    for c in cholines:
        cho_off.append(off)
        off += len(c)
    return cho_off

def ch2_pairs_with_C(choline):
    """[(C_a, [Ha, Ha'], C_b, [Hb, Hb']), ...] - same iteration order as run_generator.get_ch2_pairs."""
    ch2 = []
    for at in choline.atoms:
        if at.symbol != 'C':
            continue
        hs = [b.other_end(at) for b in at.bonds if b.other_end(at).symbol == 'H']
        if len(hs) == 2:
            ch2.append((at, hs))
    pairs = []
    for i, (c1, h1s) in enumerate(ch2):
        nbrs = [b.other_end(c1) for b in c1.bonds]
        for j, (c2, h2s) in enumerate(ch2):
            if j > i and c2 in nbrs:
                pairs.append((c1, h1s, c2, h2s))
    return pairs

def dihedral(a, b, c, d):
    p0, p1, p2, p3 = (np.array(x.coords) for x in (a, b, c, d))
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2/np.linalg.norm(b2))
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

def distance(a, b):
    p0, p1 = (np.array(x.coords) for x in (a, b))
    return float(np.linalg.norm(p1 - p0))

def column_exists(cursor, table, col):
    cursor.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cursor.fetchall())

# ── main ──────────────────────────────────────────────────────────────────
init()

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

n_updates = 0
n_tables  = 0

for xf in sorted(glob.glob(os.path.join(CLUSTERS_DIR, "*.xyz"))):
    n_step = get_step(xf)
    table  = f"step_{n_step}_intra"

    cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if cursor.fetchone() is None:
        continue
    if sum(1 for _ in open(xf)) - 2 < 1:
        continue

    if not column_exists(cursor, table, "dihedral"):
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN dihedral REAL")
    if not column_exists(cursor, table, "distance"):
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN distance REAL")

    cluster = Molecule(xf)
    centre  = np.mean(cluster.as_array(), axis=0)
    cluster.guess_bonds()
    ureas, cholines, chlorides = classify_sort(cluster.separate(), centre)
    cho_off = offsets(ureas, cholines, chlorides)

    for ci, choline in enumerate(cholines):
        off = cho_off[ci]
        for c1, h1s, c2, h2s in ch2_pairs_with_C(choline):
            for h1 in h1s:
                gi1 = off + choline.atoms.index(h1) + 1
                for h2 in h2s:
                    gi2 = off + choline.atoms.index(h2) + 1
                    theta = abs(dihedral(h1, c1, c2, h2))
                    dis   = distance(h1, h2)
                    cursor.execute(
                        f"UPDATE {table} SET dihedral = ?, distance = ? "
                        f"WHERE H_pert = ? AND H_resp = ?",
                        (theta, dis, gi1, gi2),
                    )
                    n_updates += cursor.rowcount
    n_tables += 1

conn.commit()
conn.close()
finish()

