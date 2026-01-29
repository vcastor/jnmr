#!/bin/bash

# Update MD snapshots
(
   cd mdStepsrkf
   ./update_mdStepsrkf.sh
)

# Convert RKF to XYZ
$AMSBIN/plams rkf_to_xyz.py

# Select regions
$AMSBIN/plams region_selector.py

# Generate run files and update database
$AMSBIN/plams run_generator.py

# Send data to Rameau
./to_rameau.sh

