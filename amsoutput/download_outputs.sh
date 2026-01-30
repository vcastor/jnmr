#!/usr/bin/env bash

rsync -av \
   --ignore-existing \
   --prune-empty-dirs \
   --include='*.out' \
   --exclude='*' \
   criann:/home/2026014/vcasto03/snapshots/1*/ \
   .

