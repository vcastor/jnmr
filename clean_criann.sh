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

set -euo pipefail

BASE=/home/2026014/vcasto03/nmr_main
DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)

AGE_DAYS=1

for dir in "${DIRS[@]}"; do
  target="$BASE/$dir"
  [ -d "$target" ] || continue

  # --- Remove finished .run/.sl whose output exists in a job dir ---
  shopt -s nullglob
  for run in "$target"/*.run; do
    name=$(basename "$run" .run)

    found=false
    for jobdir in "$target"/[0-9]*/; do
      [ -d "$jobdir" ] || continue
      [ -f "$jobdir/${name}.out" ] && found=true && break
    done

    if $found; then
      rm -f "$run" "$target/${name}.sl"
    fi
  done
  shopt -u nullglob

  # --- Remove numeric job directories older than AGE_DAYS ---
  find "$target" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
    -regex '.*/[0-9]+' -mtime +"$AGE_DAYS" -exec rm -rf {} +
done

