#!/usr/bin/python3
import re
import sqlite3
from pathlib import Path

CONFIG = {
    "db_path": "nmr_jcoupling.db",
    "run_root": Path("run_scripts"),
    "output_root": Path("amsoutput"),
    "table": "snapshots",
    "warning_messages": {
        "SCF MODERATELY CONVERGED",
        "SCF NOT CONVERGED",
    },
    "variants": {
        # Adjust these two values as you want.
        # On CRIANN Austral:
        #   court <= 48h
        #   long  <= 100h
        #   tlong <= 300h
        "TZ2P_FC": {
            "partition": "long",
            "walltime": "99:00:00",
            "ntasks": 12,
            "cpus_per_task": 16,
        },
        "TZ2P_all": {
            "partition": "tlong",
            "walltime": "199:00:00",
            "ntasks": 12,
            "cpus_per_task": 16,
        },
        "TZ2PJ_FC": {
            "partition": "tlong",
            "walltime": "199:00:00",
            "ntasks": 12,
            "cpus_per_task": 16,
        },
        "TZ2PJ_all": {
            "partition": "tlong",
            "walltime": "199:00:00",
            "ntasks": 12,
            "cpus_per_task": 16,
        },
    },
    "module_load": "atomic_simu/cobra-ams/2025.106_amd",
    "job_name_suffix_to_strip": "_cluster",
}

STEP_RE = re.compile(r"MDStep(\d+)")
WARNING_PATTERNS = (
    "SCF MODERATELY CONVERGED",
    "SCF NOT CONVERGED",
)

def extract_n_step(name: str) -> int:
    m = STEP_RE.search(name)
    return int(m.group(1))

def get_warning_steps(conn: sqlite3.Connection, variant: str,
                      table: str, warning_messages: set[str]) -> set[int]:
    comment_col = f"comment_{variant}"
    placeholders = ",".join("?" for _ in warning_messages)
    query = f"""
        SELECT n_step
        FROM {table}
        WHERE {comment_col} IN ({placeholders})
    """
    rows = conn.execute(query, tuple(warning_messages)).fetchall()
    return {row[0] for row in rows}

def slurm_text(case_name: str, variant: str, variant_cfg: dict,
               module_load: str, suffix_to_strip: str) -> str:
    job_base = case_name
    if suffix_to_strip and job_base.endswith(suffix_to_strip):
        job_base = job_base[:-len(suffix_to_strip)]

    ntasks = variant_cfg["ntasks"]
    cpus_per_task = variant_cfg["cpus_per_task"]
    partition = variant_cfg["partition"]
    walltime = variant_cfg["walltime"]

    return f"""#!/bin/bash

#SBATCH --exclusive
#SBATCH -J {job_base}
#SBATCH --output {job_base}.o%J
#SBATCH --error  {job_base}.e%J
#SBATCH --partition {partition}
#SBATCH --time {walltime}

#SBATCH --ntasks {ntasks}
#SBATCH --cpus-per-task {cpus_per_task}

module purge
module load {module_load}
module list

env | grep SCMLICENSE

CASE={case_name}
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


conn = sqlite3.connect(CONFIG["db_path"])

created = []
removed = []
skipped_output = []
skipped_warning = []
skipped_error = []
skipped_unknown = []

try:
    for variant, variant_cfg in CONFIG["variants"].items():
        run_dir    = CONFIG["run_root"]/variant
        output_dir = CONFIG["output_root"]/variant

        warning_steps = get_warning_steps(
            conn,
            variant,
            CONFIG["table"],
            CONFIG["warning_messages"],
        )

        for run_file in sorted(run_dir.glob("*.run")):
            base       = run_file.with_suffix("")
            slurm_file = base.with_suffix(".sl")
            out_file = output_dir/f"{base.name}.out"
            n_step   = extract_n_step(run_file.name)

            if out_file.exists():
                if slurm_file.exists():
                    slurm_file.unlink()
                    removed.append(str(slurm_file))
                skipped_output.append(str(run_file))
                continue

            if n_step is None:
                skipped_unknown.append(str(run_file))
                continue

            if n_step in warning_steps:
                if slurm_file.exists():
                    slurm_file.unlink()
                    removed.append(str(slurm_file))
                skipped_warning.append(str(run_file))
                continue

            content = slurm_text(
                case_name=base.name,
                variant=variant,
                variant_cfg=variant_cfg,
                module_load=CONFIG["module_load"],
                suffix_to_strip=CONFIG["job_name_suffix_to_strip"],
            )
            slurm_file.write_text(content)
            created.append(str(slurm_file))

finally:
    conn.close()

