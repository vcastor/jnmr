#!/bin/bash

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CACHE_DIR=analysis/cache

# Regenerate a plams cache only when its inputs changed
# Force a rebuild with:  rm analysis/cache/<name>.pkl
run_plams() {   # run_plams <cache_name> <script> <dep>...
  local name="$1" script="$2"; shift 2
  local cache="$CACHE_DIR/$name.pkl"
  local stale=0
  if [ ! -f "$cache" ]; then
     stale=1
  else
    for dep in "$script" "$@"; do
      [ -e "$dep" ] || continue
      if [ -n "$(find "$dep" -newer "$cache" 2>/dev/null | head -n 1)" ]; then
        stale=1
        break
      fi
    done
  fi
  if [ "$stale" -eq 1 ]; then
    echo "[plams] $name: regenerating cache"
    $AMSBIN/plams "$script" 2> /dev/null
  else
    echo "[plams] $name: cache up to date, skipping"
  fi
}

run_plams distance_intra analysis/distance_intra.py clusters
run_plams distance_inter analysis/distance_inter.py clusters
run_plams gauche_anti    analysis/gauche_anti.py    clusters
run_plams qtaim_analysis analysis/qtaim_analysis.py clusters
run_plams choline_fold   analysis/choline_fold.py   clusters
rm -rf plams_workdir*

./analysis/distance_plot.py
./analysis/gauche_anti_plot.py
./analysis/qtaim_analysis_plot.py
./analysis/visualiser_data.py
./analysis/variant_stats.py
./analysis/karplus_fit.py
./analysis/ch_coupling.py
./analysis/nh_coupling.py
./analysis/nh_intra_coupling.py

