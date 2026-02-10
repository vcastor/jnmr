#!/usr/bin/env bash
#
# Upload .run and .sl files to CRIANN, mirroring the subdirectory layout.
#
REMOTE=criann:/home/2026014/vcasto03/nmr_main
DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)

for dir in "${DIRS[@]}"; do
  [ -d "$dir" ] || continue

  rsync -av --ignore-existing \
    --include='*.run' \
    --include='*.sl' \
    --exclude='*' \
    "$dir/" "$REMOTE/$dir/"
done
