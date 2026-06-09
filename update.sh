#!/bin/bash

# Update MD snapshots from aragorn
(
  cd mdStepsrkf
  ./update_mdsteps.sh
)

# PLAMS scripting
$AMSBIN/plams rkf_to_xyz.py 2> /dev/null
$AMSBIN/plams region_selector.py 2> /dev/null
$AMSBIN/plams coupling_generator.py 2> /dev/null
$AMSBIN/plams populate_geometry.py 2> /dev/null
rm -rf plams_workdir*

# CRIANN
(
  cd run_scripts
  ./to_criann.sh 2> /dev/null
)
(
  cd amsoutput
  ./download_outputs.sh 2> /dev/null
)
ssh criann 'bash -s' < clean_criann.sh
ssh criann 'bash -s' < run_criann.sh

echo "Updating database with J values..."
./reader.py

