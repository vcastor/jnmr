#!/bin/sh

REMOTE_HOST=rameau
REMOTE_DIR=/home/victoria/Documents/postdoc/snapshots
LOCAL_DIR=.

rsync -a --ignore-existing \
   "$LOCAL_DIR/"*"run" \
   "$REMOTE_HOST:$REMOTE_DIR/"

