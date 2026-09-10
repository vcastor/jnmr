#!/bin/bash

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CACHE_DIR=analysis/cache
RKF_DIR=mdSteps/rkf
CLUSTERS_DIR=clusters

# Snapshot bookkeeping: the rkf files are the source (rsync'ed by generate.sh),
# clusters are derived from them, the pkl caches are derived from clusters.
newest_of() { ls -t "$1"/$2 2>/dev/null | head -n 1; }
count_of()  { ls "$1"/$2 2>/dev/null | wc -l | tr -d ' '; }

nrkf=$(count_of  "$RKF_DIR"      '*.rkf')
nclu=$(count_of  "$CLUSTERS_DIR" '*_cluster.xyz')
lrkf=$(newest_of "$RKF_DIR"      '*.rkf')
lclu=$(newest_of "$CLUSTERS_DIR" '*_cluster.xyz')

echo "[md] rkf $nrkf  clusters $nclu  last rkf $(basename "${lrkf:-none}") $(date -r "$lrkf" '+%Y-%m-%d %H:%M' 2>/dev/null)"
for pkl in "$CACHE_DIR"/*.pkl; do
  [ -e "$pkl" ] || continue
  if [ -n "$lrkf" ] && [ "$lrkf" -nt "$pkl" ]; then state="older than last rkf"; else state="current"; fi
  echo "[md] $(basename "$pkl" .pkl): $(date -r "$pkl" '+%Y-%m-%d %H:%M')  $state"
done

# New MD steps arrived but never went through the pipeline: build the clusters
# first, otherwise the caches would just be rebuilt from the same old geometries.
if [ "$nrkf" -gt "$nclu" ] || { [ -n "$lrkf" ] && [ -n "$lclu" ] && [ "$lrkf" -nt "$lclu" ]; }; then
  echo "[md] $((nrkf - nclu)) new snapshot(s): running generate.sh"
  ./generate.sh
  nclu=$(count_of "$CLUSTERS_DIR" '*_cluster.xyz')
fi

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

run_plams distance_intra analysis/distance_intra.py clusters "$RKF_DIR"
run_plams distance_inter analysis/distance_inter.py clusters "$RKF_DIR"
run_plams gauche_anti    analysis/gauche_anti.py    clusters "$RKF_DIR"
run_plams qtaim_analysis analysis/qtaim_analysis.py clusters "$RKF_DIR"
run_plams choline_fold   analysis/choline_fold.py   clusters amsoutput/qtaim
run_plams cl_environment analysis/cl_environment.py clusters amsoutput/qtaim
run_plams nh2_environment analysis/nh2_environment.py clusters amsoutput/qtaim
run_plams representative_cluster analysis/representative_cluster.py clusters
run_plams interaction_matrix analysis/interaction_matrix.py clusters amsoutput/qtaim
rm -rf plams_workdir*

./analysis/distance_plot.py
./analysis/gauche_anti_plot.py
./analysis/qtaim_analysis_plot.py
./analysis/cl_environment_plot.py
./analysis/nh2_environment_plot.py
./analysis/visualiser_data.py
./analysis/variant_stats.py
./analysis/karplus_fit.py
./analysis/ch_coupling.py
./analysis/nh_coupling.py
./analysis/nh_intra_coupling.py

