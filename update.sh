#!/bin/bash

# Update MD snapshots
(
   cd mdStepsrkf
   ./update_mdsteps.sh
)

# Convert RKF to XYZ
$AMSBIN/plams rkf_to_xyz.py 2> /dev/null

# Select regions
$AMSBIN/plams region_selector.py 2> /dev/null

# Generate run files and update database
$AMSBIN/plams run_generator.py 2> /dev/null

# Send data to Rameau
(
  cd run_scripts
  ./to_rameau.sh
)

# Cleanup
rm -rf plams_workdir*

(
  cd amsoutput
  ./download_outputs.sh
)

# update de database
./output_reader.py

# warnings
./output_warning.py

