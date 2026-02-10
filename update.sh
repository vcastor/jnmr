#!/bin/bash

# Update MD snapshots from aragorn
(
  cd mdStepsrkf
  ./update_mdsteps.sh
)

# PLAMS scripting
$AMSBIN/plams rkf_to_xyz.py 2> /dev/null
$AMSBIN/plams region_selector.py 2> /dev/null
$AMSBIN/plams run_generator.py 2> /dev/null
rm -rf plams_workdir* # clean plams workdir

# Generate SLURM launchers and upload to CRIANN
(
  cd run_scripts
  ./launcher_creator.sh
  ./to_criann.sh
)

# Download outputs from CRIANN
(
  cd amsoutput
  ./download_outputs.sh
)

# Clean finished jobs on CRIANN (job dirs + run/sl)
ssh criann 'bash -s' < cleaner.sh

# Update the database with J values
./output_reader.py
./output_warning.py

