#!/bin/bash
#
# Usage:
#   bash relaunch_scf.sh              # default: both SCF warnings
#   bash relaunch_scf.sh --all        # patch ALL .run files
#   bash relaunch_scf.sh --dry-run    # show what would be changed
#

set -euo pipefail

DB="nmr_jcoupling.db"
RUN_DIR="run_scripts"
OUTPUT_DIR="amsoutput"
DIRS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)
DRY_RUN=false
PATCH_ALL=false

for arg in "$@"; do
  case "$arg" in
    --all)     PATCH_ALL=true ;;
    --dry-run) DRY_RUN=true ;;
    *)         echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# ── Collect step numbers to patch ──────────────────────────────────────

if $PATCH_ALL; then
  echo "Mode: patching ALL .run files"
  steps=()
  for dir in "${DIRS[@]}"; do
    for f in "$RUN_DIR/$dir"/MDStep*_cluster.run; do
      [ -f "$f" ] || continue
      base=$(basename "$f" .run)
      n=${base#MDStep}
      n=${n%_cluster}
      steps+=("$n")
    done
  done
  # Remove duplicates (same step appears in multiple dirs)
  IFS=$'\n' steps=($(printf '%s\n' "${steps[@]}" | sort -un)); unset IFS
else
  echo "Mode: patching only SCF-warning snapshots"
  if ! command -v sqlite3 &>/dev/null; then
    echo "ERROR: sqlite3 not found, cannot query database." >&2
    exit 1
  fi
  steps=()
  while IFS= read -r s; do
    steps+=("$s")
  done < <(sqlite3 "$DB" "SELECT n_step FROM snapshots WHERE comment LIKE '%SCF%'")
fi

echo "Found ${#steps[@]} steps to process"
echo ""

# ── Patch each .run file across all subdirectories ─────────────────────

patched=0
skipped=0

for n in "${steps[@]}"; do
  step_patched=false

  for dir in "${DIRS[@]}"; do
    run_file="$RUN_DIR/$dir/MDStep${n}_cluster.run"
    sl_file="$RUN_DIR/$dir/MDStep${n}_cluster.sl"
    out_file="$OUTPUT_DIR/$dir/MDStep${n}_cluster.out"

    [ -f "$run_file" ] || continue

    # Check if already patched
    if grep -q "NumericalQuality" "$run_file" 2>/dev/null; then
      continue
    fi

    if ! $DRY_RUN; then
      sed -i '' '/^  beckegrid$/,/^  End$/{
/^  beckegrid$/c\
  NumericalQuality Excellent
/^    quality good$/d
/^  End$/d
}' "$run_file"

      # Remove old launcher so it gets regenerated
      [ -f "$sl_file" ] && rm "$sl_file"

      # Remove old output so it gets re-downloaded/re-processed
      [ -f "$out_file" ] && rm "$out_file"
    fi

    step_patched=true
  done

  if $step_patched; then
    if ! $DRY_RUN; then
      sqlite3 "$DB" "UPDATE snapshots SET comment = NULL WHERE n_step = $n"
    fi
    echo "  DONE step $n"
    patched=$((patched + 1))
  else
    skipped=$((skipped + 1))
  fi
done

echo ""
echo "──────────────────────────────────"
echo "  Patched: $patched"
echo "  Skipped: $skipped"
echo "──────────────────────────────────"
