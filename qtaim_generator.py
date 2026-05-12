#!/usr/bin/python3
import os
import sqlite3

DB_PATH   = "nmr_jcoupling.db"
RUN_BASE  = "run_scripts"
QTAIM_DIR = "run_scripts/qtaim"
VARIANTS  = ["TZ2P_FC", "TZ2P_all", "TZ2PJ_FC", "TZ2PJ_all"]
PARTITION = "court"
WALLTIME  = "46:00:00"

SL_TEMPLATE = """\
#!/bin/bash

#SBATCH --exclusive
#SBATCH -J {jobname}
#SBATCH --output {jobname}.o%J
#SBATCH --error  {jobname}.e%J
#SBATCH --partition {partition}
#SBATCH --time {walltime}

#SBATCH --ntasks 12
#SBATCH --cpus-per-task 16

module purge
module load atomic_simu/cobra-ams/2025.106_amd
module list

env | grep SCMLICENSE

CASE={case}
INP=${{CASE}}.run
OUT=${{CASE}}.out

cp $INP $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR
echo Working directory : $PWD

export SCM_GPUENABLED="FALSE"

set -x
sh $INP > $OUT
set +x

mkdir -p $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
mv * $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
"""

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

def write_qtaim_run(out_path, basename, atoms, h_indices):
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
        f.write(f"  title {basename}_qtaim\n")
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
        f.write("  QTAIM\n")
        f.write("    AnalysisLevel Full\n")
        f.write("    Spacing 0.1\n")
        f.write("    AtomsToDo " + " ".join(str(h) for h in h_indices) + "\n")
        f.write("  End\n")
        f.write("EndEngine\n")
        f.write("eor\n")

def write_sl(out_path, basename, n_step):
    jobname = f"qtaim{n_step}"
    with open(out_path, "w") as f:
        f.write(SL_TEMPLATE.format(
            jobname=jobname,
            case=basename,
            partition=PARTITION,
            walltime=WALLTIME,
        ))

# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(QTAIM_DIR, exist_ok=True)

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    steps = find_successful_steps(cursor)

    n_run_new     = 0
    n_run_kept    = 0
    n_sl_written  = 0
    n_skip_no_run = 0
    n_skip_no_h   = 0

    for n_step in steps:
        basename = f"MDStep{n_step}_cluster"
        src = find_run_file(basename)
        if src is None:
            n_skip_no_run += 1
            continue

        h_indices = get_h_indices(cursor, n_step)
        if not h_indices:
            n_skip_no_h += 1
            continue

        run_path = os.path.join(QTAIM_DIR, f"{basename}.run")
        sl_path  = os.path.join(QTAIM_DIR, f"{basename}.sl")

        if os.path.exists(run_path):
            n_run_kept += 1
        else:
            atoms = extract_atoms_block(src)
            write_qtaim_run(run_path, basename, atoms, h_indices)
            n_run_new += 1

        write_sl(sl_path, basename, n_step)
        n_sl_written += 1

    conn.close()

    print(f"qtaim: {n_run_new} new run files, {n_run_kept} kept · "
          f"{n_sl_written} sl (re)written · "
          f"{n_skip_no_run} skipped (no source) · "
          f"{n_skip_no_h} skipped (no analysed H)")

