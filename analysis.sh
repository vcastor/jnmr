#!/bin/bash

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# PLAMS analyses
$AMSBIN/plams analysis/distance_intra.py 2> /dev/null
$AMSBIN/plams analysis/distance_inter.py 2> /dev/null
$AMSBIN/plams analysis/gauche_anti.py    2> /dev/null
$AMSBIN/plams analysis/qtaim_analysis.py 2> /dev/null
rm -rf plams_workdir*

./analysis/visualiser_data.py
./analysis/karplus_fit.py

