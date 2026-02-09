#!/bin/bash
#
# Submit all .sl files found in each variant subdirectory.
# Run this ON CRIANN after uploading new run/sl files.
#

DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)
TOTAL=0

for dir in "${DIRS[@]}"; do
  [ -d "$dir" ] || continue
  COUNT=0

  for sl in "$dir"/*.sl; do
    [ -f "$sl" ] || continue
    sbatch "$sl"
    COUNT=$((COUNT + 1))
  done

  [ "$COUNT" -gt 0 ] && echo "$dir: submitted $COUNT jobs"
  TOTAL=$((TOTAL + COUNT))
done

echo "Total submitted: $TOTAL"
