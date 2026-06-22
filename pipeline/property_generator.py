#!/usr/bin/python3
import os
import sys
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        "partition": "long",
        "engine":    "ConceptualDFT",
        "extra":     [],
    },
}

def find_successful_steps(cursor):
    cursor.execute(
        "SELECT n_step FROM snapshots "
        "WHERE comment IS NULL AND comment_TZ2P_FC IS NULL"
    )
    steps = sorted(r[0] for r in cursor.fetchall())

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
            if "J_TZ2P_FC" not in {r[1] for r in cursor.fetchall()}:
                continue
            cursor.execute(
                f"SELECT 1 FROM {table} WHERE J_TZ2P_FC IS NOT NULL LIMIT 1"
            )
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

def analysis_done(basename, cfg):
    """True if this analysis already has the step's out, run or sl file."""
    return (os.path.exists(os.path.join(cfg["out_dir"], f"{basename}.out"))
            or os.path.exists(os.path.join(cfg["run_dir"], f"{basename}.run"))
            or os.path.exists(os.path.join(cfg["run_dir"], f"{basename}.sl")))

def extract_atoms_block(run_file):
    """Return raw atom lines (with original indentation) from System/Atoms."""
    lines = []; in_atoms = False
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

def get_atoms_todo(cursor, n_step):
    """AtomsToDo for the analysis run: the H atoms plus, for the intra
    CH2-CH2 table, the two carbons (C_pert / C_resp)."""
    atoms = set()
    for kind in ("intra", "inter"):
        table = f"step_{n_step}_{kind}"
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        )
        if cursor.fetchone() is None:
            continue
        cursor.execute(f"PRAGMA table_info({table})")
        present = {r[1] for r in cursor.fetchall()}
        cols = ["H_pert", "H_resp"]
        if kind == "intra":
            cols += ["C_pert", "C_resp"]
        for col in cols:
            if col not in present:
                continue
            cursor.execute(f"SELECT {col} FROM {table}")
            for r in cursor.fetchall():
                if r[0] is not None:
                    atoms.add(r[0])
    return sorted(atoms)

def write_property_run(out_path, basename, atoms, atoms_todo, cfg):
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
        f.write("    AtomsToDo " + " ".join(str(a) for a in atoms_todo) + "\n")
        f.write("  End\n")
        f.write("EndEngine\n")
        f.write("eor\n")

if __name__ == "__main__":
    for cfg in ANALYSES.values():
        os.makedirs(cfg["run_dir"], exist_ok=True)

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    steps  = find_successful_steps(cursor)

    for n_step in steps:
        basename = f"MDStep{n_step}_cluster"

        if all(analysis_done(basename, cfg) for cfg in ANALYSES.values()):
            continue

        src = find_run_file(basename)
        if src is None:
            continue
        atoms_todo = get_atoms_todo(cursor, n_step)
        if not atoms_todo:
            continue
        atoms = extract_atoms_block(src)

        for name, cfg in ANALYSES.items():
            if analysis_done(basename, cfg):
                continue
            cfg = {**cfg, "name": name}
            run_path = os.path.join(cfg["run_dir"], f"{basename}.run")
            sl_path  = os.path.join(cfg["run_dir"], f"{basename}.sl")
            write_property_run(run_path, basename, atoms, atoms_todo, cfg)
            with open(sl_path, "w") as f:
                f.write(slurm_script(f"{name}{n_step}", basename, cfg["partition"]))

    conn.close()

