#!/bin/bash

N=10
CHECKPOINT=.checkpoint
FILES=(MDStep*_cluster.sl)

# extract numeric step
step_of() {
   basename "$1" | sed -E 's/[^0-9]*([0-9]+).*/\1/'
}

# initialise checkpoint
if [ ! -f "$CHECKPOINT" ]; then
   echo 20000 > "$CHECKPOINT"
fi

LAST=$(cat "$CHECKPOINT")
COUNT=0
NEXT=$LAST

for sl in "${FILES[@]}"; do
   step=$(step_of "$sl")

   [ "$step" -le "$LAST" ] && continue
   [ "$COUNT" -ge "$N" ] && break

   sbatch "$sl"
   NEXT="$step"
   COUNT=$((COUNT+1))
done

[ "$COUNT" -gt 0 ] && echo "$NEXT" > "$CHECKPOINT"

