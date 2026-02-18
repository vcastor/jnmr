#!/bin/bash
# master.sh
# Submit up to N new SLURM scripts per variant directory, resuming via per-dir checkpoint.

set -euo pipefail

BASE=/home/2026014/vcasto03/nmr_main
DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)

N=20
PATTERN='MDStep*_cluster.sl'
CHECKPOINT_NAME='.checkpoint'

step_of() {
  local b
  b=$(basename "$1")
  if [[ $b =~ ([0-9]+) ]]; then
    printf '%d\n' "${BASH_REMATCH[1]}"
  else
    printf '0\n'
  fi
}

for dir in "${DIRS[@]}"; do
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

