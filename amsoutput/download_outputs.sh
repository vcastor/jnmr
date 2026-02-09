#!/usr/bin/env bash
#
# Download .out files from CRIANN, preserving the subdirectory layout.
# The CRIANN job dirs (numeric SLURM IDs) live under each variant.
#
REMOTE=criann:/home/2026014/vcasto03/snapshots
DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)

for dir in "${DIRS[@]}"; do
  [ -d "$dir" ] || mkdir -p "$dir"

  rsync -av \
    --ignore-existing \
    --prune-empty-dirs \
    --include='*/' \
    --include='*.out' \
    --exclude='*' \
    "$REMOTE/$dir/1*/" \
    "$dir/"
done

