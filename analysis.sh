#!/bin/bash
#
# Run every analysis over the data already in the DB / clusters.
#
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# PLAMS analyses
$AMSBIN/plams analysis/distance_intra.py 2> /dev/null
$AMSBIN/plams analysis/distance_inter.py 2> /dev/null
$AMSBIN/plams analysis/gauche_anti.py 2> /dev/null
$AMSBIN/plams analysis/qtaim_analysis.py 2> /dev/null
rm -rf plams_workdir*

# python3 analyses
python3 analysis/visualiser_data.py
python3 analysis/direct_dij.py
python3 analysis/karplus_fit.py

# karplus_ml.py is memory-heavy (overflows in batch) — run it manually:
#   PYTHONPATH=. python3 analysis/karplus_ml.py
