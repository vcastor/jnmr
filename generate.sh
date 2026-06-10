#!/bin/bash

# MD snapshots from aragorn
( cd mdSteps/rkf && ./update_mdsteps.sh )

# rkf -> xyz -> clusters -> ADF .run/.sl + DB rows
$AMSBIN/plams pipeline/rkf_to_xyz.py 2> /dev/null
$AMSBIN/plams pipeline/region_selector.py 2> /dev/null
$AMSBIN/plams pipeline/coupling_generator.py 2> /dev/null
$AMSBIN/plams pipeline/populate_geometry.py 2> /dev/null
./pipeline/property_generator.py
rm -rf plams_workdir*

