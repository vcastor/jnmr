import os

MODULE        = "atomic_simu/cobra-ams/2025.106_amd"
NTASKS        = 12
CPUS_PER_TASK = 16

# CRIANN Austral partitions and the walltime we request on each.
# Caps:  court <= 48h,  long <= 100h,  tlong <= 300h.
# tcourt is a placeholder — set the cap you actually want.
PARTITION_WALLTIME = {
    "court":  "47:00:00",
    "tcourt": "23:00:00",
    "long":   "96:00:00",
    "tlong":  "199:00:00",
}

# Main NMR variants → CRIANN partition / requested walltime.
VARIANT_SLURM = {
    "TZ2P_FC":   {"partition": "tlong", "walltime": "199:00:00"},
    "TZ2P_all":  {"partition": "tlong", "walltime": "199:00:00"},
    "TZ2PJ_FC":  {"partition": "tlong", "walltime": "199:00:00"},
    "TZ2PJ_all": {"partition": "tlong", "walltime": "299:00:00"},
}

# ADF SCF outcomes that mark a step unusable — excluded from submission & reads.
SCF_WARNINGS = ("SCF MODERATELY CONVERGED", "SCF NOT CONVERGED")

SL_TEMPLATE = """\
#!/bin/bash

#SBATCH --exclusive
#SBATCH -J {jobname}
#SBATCH --output {jobname}.o%J
#SBATCH --error  {jobname}.e%J
#SBATCH --partition {partition}
#SBATCH --time {walltime}

#SBATCH --ntasks {ntasks}
#SBATCH --cpus-per-task {cpus_per_task}

module purge
module load {module}
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

{epilogue}
"""

# What comes back from $LOCAL_WORK_DIR: just the .out by default (CRIANN storage);
# TAPE=SAVE at generation time keeps everything (rkf, TAPE10, ...).
EPILOGUE_OUT  = "cp $OUT $SLURM_SUBMIT_DIR/"
EPILOGUE_SAVE = ("mkdir -p $SLURM_SUBMIT_DIR/$SLURM_JOB_ID\n"
                 "mv * $SLURM_SUBMIT_DIR/$SLURM_JOB_ID")

def slurm_script(jobname, case, partition, walltime=None,
                 ntasks=NTASKS, cpus_per_task=CPUS_PER_TASK, module=MODULE):
    return SL_TEMPLATE.format(
        jobname=jobname,
        case=case,
        partition=partition,
        walltime=walltime or PARTITION_WALLTIME[partition],
        ntasks=ntasks,
        cpus_per_task=cpus_per_task,
        module=module,
        epilogue=EPILOGUE_SAVE if os.environ.get("TAPE") == "SAVE" else EPILOGUE_OUT,
    )
