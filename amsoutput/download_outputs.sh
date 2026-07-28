#!/bin/bash

download_outs() {
  local remote_base="$1"
  shift
  local host="${remote_base%%:*}"
  local path="${remote_base#*:}"

  local dir
  for dir in "$@"; do
    echo ""
    [ -d "$dir" ] || mkdir -p "$dir"
    echo "Downloading .out files for $dir..."

    ssh "$host" "cd '$path/$dir' && find . -mindepth 2 -maxdepth 2 -name '*.out'" | \
      rsync -av --ignore-existing --no-relative --files-from=- \
        "$remote_base/$dir/" "$dir/"
  done
}

download_outs "criann:/home/2026014/vcasto03/nmr_main" \
  TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all ch nh

download_outs "criann:/home/2026014/vcasto03" \
  qtaim cdft

