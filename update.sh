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
rm -rf plams_workdir*

# Generate SLURM launchers and upload to CRIANN
./new_launchers.py
(
  cd run_scripts
  ./to_criann.sh 2> /dev/null
)

# CRIANN
(
  cd amsoutput
  ./download_outputs.sh 2> /dev/null
)
ssh criann 'bash -s' < clean_criann.sh
ssh criann 'bash -s' < run_criann.sh

# Update the database with J values
echo "Updating database with J values..."
./output_reader.py
./output_warning.py

