#!/bin/bash

# Slurm submission script – TZ2PJ_all
# AMS 2025.106 parallel job · CRIANN Austral (Genoa)

#SBATCH --exclusive
#SBATCH -J MDStep4460000
#SBATCH --output MDStep4460000.o%J
#SBATCH --error  MDStep4460000.e%J
#SBATCH --partition long
#SBATCH --time 99:00:00

#SBATCH --ntasks 12
#SBATCH --cpus-per-task 16

module purge
module load atomic_simu/cobra-ams/2025.106_amd
module list

env | grep SCMLICENSE

CASE=MDStep4460000_cluster
INP=${CASE}.run
OUT=${CASE}.out

cp $INP $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR
echo Working directory : $PWD

export SCM_GPUENABLED="FALSE"

set -x
sh $INP > $OUT
set +x

mkdir $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
mv * $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
