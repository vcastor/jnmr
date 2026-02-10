#!/bin/bash
#
# Clean finished jobs on CRIANN.
#
# For each variant subdirectory:
#   1. Remove SLURM job directories (numeric names like 123456/)
#      — these contain ams.results/, TAPE files, etc.
#   2. Remove .run and .sl files whose .out has already been collected
#      inside a job directory (i.e. the calculation finished).
#
# Run this ON CRIANN after downloading outputs.
#

BASE=/home/2026014/vcasto03/nmr_main
DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)

cleaned=0

for dir in "${DIRS[@]}"; do
  target="$BASE/$dir"
  [ -d "$target" ] || continue

  # --- Remove finished .run/.sl whose output exists in a job dir ---
  for run in "$target"/*.run; do
    [ -f "$run" ] || continue
    name=$(basename "$run" .run)

    # Check if any numeric job directory contains the .out
    found=false
    for jobdir in "$target"/[0-9]*/; do
      [ -d "$jobdir" ] || continue
      if [ -f "$jobdir/${name}.out" ]; then
        found=true
        break
      fi
    done

    if $found; then
      rm -f "$run"
      rm -f "$target/${name}.sl"
      cleaned=$((cleaned + 1))
    fi
  done

  # --- Remove all numeric job directories ---
  for jobdir in "$target"/[0-9]*/; do
    [ -d "$jobdir" ] || continue
    rm -rf "$jobdir"
  done

done

