#!/bin/bash

upload_runs() {
  local remote_base="$1"
  shift

  local dir
  for dir in "$@"; do
    echo ""
    [ -d "$dir" ] || continue
    echo "Uploading .run/.sl files for $dir..."

    rsync -av --ignore-existing \
      --include='*.run' \
      --include='*.sl' \
      --exclude='*' \
      "$dir/" "$remote_base/$dir/"
  done
}

upload_runs "criann:/home/2026014/vcasto03/nmr_main" \
  TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all

upload_runs "criann:/home/2026014/vcasto03" \
  qtaim cdft ch nh nh_intra site

