#!/bin/bash
#
# Reset the database: drop all step_* tables and clear the snapshots table.
# The snapshots table structure is preserved (columns kept, rows deleted).
#
# Usage:
#   bash reset_db.sh              # execute the reset
#   bash reset_db.sh --dry-run    # show what would be dropped
#

DB="nmr_jcoupling.db"
DRY_RUN=false

[ "$1" = "--dry-run" ] && DRY_RUN=true

if [ ! -f "$DB" ]; then
  echo "ERROR: $DB not found"
  exit 1
fi

# Collect all step_* table names
tables=$(sqlite3 "$DB" "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'step_%'")
count=$(echo "$tables" | grep -c '^step_')
snap_rows=$(sqlite3 "$DB" "SELECT COUNT(*) FROM snapshots")

echo "Database: $DB"
echo "Step tables found: $count"
echo "Snapshot rows:     $snap_rows"
echo ""

if $DRY_RUN; then
  echo "[dry-run] Would drop $count tables and delete $snap_rows rows from snapshots"
  exit 0
fi

# Build a single SQL transaction
sql="BEGIN TRANSACTION;\n"
for t in $tables; do
  sql+="DROP TABLE IF EXISTS \"$t\";\n"
done
sql+="DELETE FROM snapshots;\n"
sql+="DELETE FROM sqlite_sequence WHERE name LIKE 'step_%';\n"
sql+="COMMIT;\n"
sql+="VACUUM;\n"

echo -e "$sql" | sqlite3 "$DB"

remaining=$(sqlite3 "$DB" "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'step_%'")
echo "Done. Step tables remaining: $remaining"
echo "Snapshots rows remaining:    $(sqlite3 "$DB" "SELECT COUNT(*) FROM snapshots")"
