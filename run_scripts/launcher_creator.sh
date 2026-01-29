#!/bin/bash

for run in *.run; do
   [ -f "$run" ] || continue

   base=${run%.run}
   sl=${base}.sl

   if [ -f "$sl" ]; then
      continue
   fi

   cat > "$sl" <<EOF
#!/bin/bash

# Slurm submission script
# AMS 2023.204 parallel job
# CRIANN v 1.00 - Dec 2023

#SBATCH --exclusive
#SBATCH -J "${base%_cluster}"
#SBATCH --output ${base%_cluster}.o%J
#SBATCH --error  ${base%_cluster}.e%J
#SBATCH --partition court
#SBATCH --time 45:00:00

#SBATCH --ntasks 10
#SBATCH --mem 80000

module purge
module load atomic_simu/cobra-ams/2025.106_amd
module list

env | grep SCMLICENSE

CASE=${base}
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

