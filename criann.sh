#!/bin/bash
#
# CRIANN round-trip (run from the repo root):
#   1. upload .run/.sl, download .out files
#   2. ON CRIANN: clean finished jobs, then submit new ones
#

( cd run_scripts && ./to_criann.sh 2> /dev/null )
( cd amsoutput && ./download_outputs.sh 2> /dev/null )

ssh criann 'bash -s' <<'REMOTE'
set -euo pipefail

BASE=/home/2026014/vcasto03/nmr_main
NMR_DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)

N=20
PATTERN='MDStep*_cluster.sl'
CHECKPOINT_NAME='.checkpoint'

# ── clean ───────────────────────────────────────────────────────────────────
clean_dirs() {
  local base="$1"
  shift

  local dir target
  for dir in "$@"; do
    target="$base/$dir"
    [ -d "$target" ] || continue

    # Remove finished .run/.sl whose output exists in a job dir
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

    # Remove numeric job directories
    find "$target" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
      -regex '.*/[0-9]+' -exec rm -rf {} +
  done
}

clean_dirs "$BASE" "${NMR_DIRS[@]}"
clean_dirs /home/2026014/vcasto03 qtaim cdft ch nh nh_intra site

# ── submit ──────────────────────────────────────────────────────────────────
step_of() {
  local b
  b=$(basename "$1")
  if [[ $b =~ ([0-9]+) ]]; then
    printf '%d\n' "${BASH_REMATCH[1]}"
  else
    printf '0\n'
  fi
}

for dir in "${NMR_DIRS[@]}"; do
  target="$BASE/$dir"
  [ -d "$target" ] || continue

  cd "$target"

  if [ ! -f "$CHECKPOINT_NAME" ]; then
    echo 20000 >"$CHECKPOINT_NAME"
  fi

  LAST=$(<"$CHECKPOINT_NAME")

  shopt -s nullglob
  files=( $PATTERN )
  shopt -u nullglob
  [ ${#files[@]} -gt 0 ] || continue

  mapfile -t sorted < <(
    for f in "${files[@]}"; do
      printf '%s\t%s\n' "$(step_of "$f")" "$f"
    done | sort -n | cut -f2-
  )

  COUNT=0
  NEXT="$LAST"

  for sl in "${sorted[@]}"; do
    step=$(step_of "$sl")

    [ "$step" -le "$LAST" ] && continue
    [ "$COUNT" -ge "$N" ] && break

    sbatch "$sl"
    COUNT=$((COUNT + 1))
    [ "$step" -gt "$NEXT" ] && NEXT="$step"
  done

  [ "$COUNT" -gt 0 ] && echo "$NEXT" >"$CHECKPOINT_NAME"
done
REMOTE
