#!/bin/sh
REMOTE_HOST=aragorn
REMOTE_DIR=/home/vcastor/Documents/postdoc/production/ams.results
LOCAL_DIR=.

rsync -a --ignore-existing \
   "$REMOTE_HOST:$REMOTE_DIR/MDStep"*".rkf" \
   "$LOCAL_DIR/"

