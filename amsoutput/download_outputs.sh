#!/usr/bin/env bash

download_outs() {
  local remote_base="$1"
  shift

  local dir
  for dir in "$@"; do
    echo ""
    [ -d "$dir" ] || mkdir -p "$dir"
    echo "Downloading .out files for $dir..."

    rsync -avm \
      --ignore-existing \
      --include='*.out' \
      --exclude='*/' \
      --exclude='*' \
      "$remote_base/$dir/*/" \
      "$dir/"
  done
}

download_outs "criann:/home/2026014/vcasto03/nmr_main" \
  TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all

download_outs "criann:/home/2026014/vcasto03" \
  qtaim

