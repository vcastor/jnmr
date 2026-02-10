#!/bin/bash
#
# Submit all .sl files found in each variant subdirectory.
# Run this ON CRIANN after uploading new run/sl files.
#

BASE=/home/2026014/vcasto03/nmr_main
DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)
TOTAL=0

for dir in "${DIRS[@]}"; do
  target="$BASE/$dir"
  [ -d "$target" ] || continue

  checkpoint="$target/.checkpoint"
  if [ -f $checkpoint ]; then
    COUNT=$(cat $checkpoint)
  else
    COUNT=20
  fi

  MAX_PERCALL=10
  call=0
  for sl in "$dir"/*.sl; do
    [ -f "$sl" ] || continue
    stepnum=$(basename $sl)
    stepnum=${stepnum%%_cluster.sl}
    stepnum=${stepnum##_MDStep}
    [ "$COUNT" -gt $stepnum ] || continue
    sbatch "$sl"
    call=$((call + 1))
    [ "$call" -gt "$MAX_PERCALL" ] || break
  done

  # [ "$COUNT" -gt 0 ] && echo "$dir: submitted $COUNT jobs"
  # TOTAL=$((TOTAL + COUNT))
done

