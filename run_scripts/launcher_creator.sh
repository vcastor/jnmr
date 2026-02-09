#!/bin/bash
#
# Generate SLURM .sl launchers for every .run file found
# inside the TZ2P_FC, TZ2P_all, TZ2PJ_FC, TZ2PJ_all subdirectories.
#
# TZ2P_FC  → court partition (48 h)  — Fermi-contact only, fastest
# others   → long  partition (99 h)  — contributions / larger basis
#

DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)

for dir in "${DIRS[@]}"; do
  [ -d "$dir" ] || continue

  # Choose partition & walltime per variant
  if [ "$dir" = "TZ2P_FC" ]; then
    PARTITION="court"
    WALLTIME="47:00:00"
  else
    PARTITION="long"
    WALLTIME="99:00:00"
  fi

  for run in "$dir"/*.run; do
    [ -f "$run" ] || continue

    base=${run%.run}
    sl=${base}.sl
    name=$(basename "$base")

    [ -f "$sl" ] && continue

    cat > "$sl" <<EOF
#!/bin/bash

# Slurm submission script – ${dir}
# AMS 2025.106 parallel job · CRIANN Austral (Genoa)

#SBATCH --exclusive
#SBATCH -J ${name%_cluster}
#SBATCH --output ${name%_cluster}.o%J
#SBATCH --error  ${name%_cluster}.e%J
#SBATCH --partition ${PARTITION}
#SBATCH --time ${WALLTIME}

#SBATCH --ntasks 12
#SBATCH --cpus-per-task 16

module purge
module load atomic_simu/cobra-ams/2025.106_amd
module list

env | grep SCMLICENSE

CASE=${name}
INP=\${CASE}.run
OUT=\${CASE}.out

cp \$INP \$LOCAL_WORK_DIR
cd \$LOCAL_WORK_DIR
echo Working directory : \$PWD

export SCM_GPUENABLED="FALSE"

set -x
sh \$INP > \$OUT
set +x

mkdir \$SLURM_SUBMIT_DIR/\$SLURM_JOB_ID
mv * \$SLURM_SUBMIT_DIR/\$SLURM_JOB_ID
EOF

  done
done
