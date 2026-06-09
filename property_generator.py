#!/usr/bin/python3
import os
import sqlite3
from hassan_functions import slurm_script

DB_PATH  = "nmr_jcoupling.db"
RUN_BASE = "run_scripts"
VARIANTS = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]

# QTAIM and CDFT share everything but the ADF engine block, the I/O dirs and
# the partition they run on.
ANALYSES = {
    "qtaim": {
        "run_dir":   "run_scripts/qtaim",
        "out_dir":   "amsoutput/qtaim",
        "partition": "long",
        "engine":    "QTAIM",
        "extra":     ["Spacing 0.1"],
    },
    "cdft": {
        "run_dir":   "run_scripts/cdft",
        "out_dir":   "amsoutput/cdft",
        "partition": "court",
        "engine":    "ConceptualDFT",
        "extra":     [],
    },
}

def find_successful_steps(cursor):
    where = " OR ".join(f"comment_{v} IS NULL" for v in VARIANTS)
    cursor.execute(f"SELECT n_step FROM snapshots WHERE {where}")
    steps = sorted(r[0] for r in cursor.fetchall())

    j_cols = [f"J_{v}" for v in VARIANTS]
    out = []
    for s in steps:
        for kind in ("intra", "inter"):
            table = f"step_{s}_{kind}"
            cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                (table,),
            )
            if cursor.fetchone() is None:
                continue
            cursor.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cursor.fetchall()]
            existing = [j for j in j_cols if j in cols]
            if not existing:
                continue
            cond = " OR ".join(f"{j} IS NOT NULL" for j in existing)
            cursor.execute(f"SELECT 1 FROM {table} WHERE {cond} LIMIT 1")
            if cursor.fetchone() is not None:
                out.append(s)
                break
    return out

def find_run_file(basename):
    for v in VARIANTS:
        path = os.path.join(RUN_BASE, v, f"{basename}.run")
        if os.path.exists(path):
            return path
    return None

def extract_atoms_block(run_file):
    """Return raw atom lines (with original indentation) from System/Atoms."""
    lines = []
    in_atoms = False
    with open(run_file) as f:
        for line in f:
            if not in_atoms:
                if line.strip() == "Atoms":
                    in_atoms = True
                continue
            if line.strip() == "End":
                break
            lines.append(line.rstrip("\n"))
    return lines

def get_h_indices(cursor, n_step):
    hs = set()
    for kind in ("intra", "inter"):
        table = f"step_{n_step}_{kind}"
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        )
        if cursor.fetchone() is None:
            continue
        for col in ("H_pert", "H_resp"):
            cursor.execute(f"SELECT {col} FROM {table}")
            for r in cursor.fetchall():
                if r[0] is not None:
                    hs.add(r[0])
    return sorted(hs)

def write_property_run(out_path, basename, atoms, h_indices, cfg):
    with open(out_path, "w") as f:
        f.write("#!/bin/sh\n\n")
        f.write(f"export AMS_JOBNAME={basename}\n")
        f.write("$AMSBIN/ams <<eor\n")
        f.write("System\n")
        f.write("  Atoms\n")
        for a in atoms:
            f.write(a + "\n")
        f.write("  End\n")
        f.write("End\n\n")
        f.write("Task SinglePoint\n\n")
        f.write("Engine ADF\n")
        f.write(f"  title {basename}_{cfg['name']}\n")
        f.write("  NumericalQuality Excellent\n")
        f.write("  Basis\n")
        f.write("    Type TZ2P\n")
        f.write("    core None\n")
        f.write("  End\n")
        f.write("  symmetry NOSYM\n")
        f.write("  XC\n")
        f.write("    GGA PBE\n")
        f.write("  End\n")
        f.write("  Relativity\n")
        f.write("    Level None\n")
        f.write("  End\n")
        f.write(f"  {cfg['engine']}\n")
        f.write("    AnalysisLevel Full\n")
        for line in cfg["extra"]:
            f.write(f"    {line}\n")
        f.write("    AtomsToDo " + " ".join(str(h) for h in h_indices) + "\n")
        f.write("  End\n")
        f.write("EndEngine\n")
        f.write("eor\n")

if __name__ == "__main__":
    for cfg in ANALYSES.values():
        os.makedirs(cfg["run_dir"], exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    steps = find_successful_steps(cursor)

    for n_step in steps:
        basename = f"MDStep{n_step}_cluster"

        src = find_run_file(basename)
        if src is None:
            continue
        h_indices = get_h_indices(cursor, n_step)
        if not h_indices:
            continue
        atoms = extract_atoms_block(src)

        for name, cfg in ANALYSES.items():
            cfg = {**cfg, "name": name}
            if os.path.exists(os.path.join(cfg["out_dir"], f"{basename}.out")):
                continue
            run_path = os.path.join(cfg["run_dir"], f"{basename}.run")
            sl_path  = os.path.join(cfg["run_dir"], f"{basename}.sl")
            write_property_run(run_path, basename, atoms, h_indices, cfg)
            with open(sl_path, "w") as f:
                f.write(slurm_script(f"{name}{n_step}", basename, cfg["partition"]))

    conn.close()

