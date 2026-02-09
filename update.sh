#!/bin/bash

# Update MD snapshots from aragorn
(
  cd mdStepsrkf
  ./update_mdsteps.sh
)

# Convert RKF to XYZ
$AMSBIN/plams rkf_to_xyz.py 2> /dev/null

# Select regions
$AMSBIN/plams region_selector.py 2> /dev/null

# Generate run files (4 variants) and update database
$AMSBIN/plams run_generator.py 2> /dev/null

# Generate SLURM launchers
(
  cd run_scripts
  ./launcher_creator.sh
)

# Upload .run/.sl to CRIANN
(
  cd run_scripts
  ./to_criann.sh
)

# Cleanup PLAMS temp dirs
rm -rf plams_workdir*

# Download outputs from CRIANN
(
  cd amsoutput
  ./download_outputs.sh
)

# Clean finished jobs on CRIANN (job dirs + run/sl)
ssh criann 'bash -s' < cleaner.sh

# Update the database with J values
./output_reader.py

# Flag SCF warnings
./output_warning.py
