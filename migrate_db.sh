#!/bin/bash
#
# Usage:
#   bash migrate_db.sh              # execute
#   bash migrate_db.sh --dry-run    # preview only
#

DB="nmr_jcoupling.db"
DRY_RUN=false
[ "$1" = "--dry-run" ] && DRY_RUN=true

if [ ! -f "$DB" ]; then
  echo "ERROR: $DB not found"
  exit 1
fi

VARIANTS=(TZ2P_FC TZ2P_all TZ2PJ_FC TZ2PJ_all)

echo "Database: $DB"
echo ""

for v in "${VARIANTS[@]}"; do
  col="comment_${v}"

  # Check if column already exists
  exists=$(sqlite3 "$DB" "PRAGMA table_info(snapshots)" | grep -c "$col")
  if [ "$exists" -gt 0 ]; then
    continue
  fi

  if $DRY_RUN; then
    echo "  [dry-run] Would add column: $col TEXT"
  else
    sqlite3 "$DB" "ALTER TABLE snapshots ADD COLUMN $col TEXT"
  fi
done

echo ""
echo "Current schema:"
sqlite3 "$DB" ".schema snapshots"

