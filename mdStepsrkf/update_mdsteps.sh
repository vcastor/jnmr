#!/bin/sh
REMOTE_IP=10.196.48.15
REMOTE_HOST=aragorn
REMOTE_DIR=/home/vcastor/Documents/postdoc/production/ams.results
LOCAL_DIR=.

if ping -c 1 -W 1 $REMOTE_IP > /dev/null 2>&1; then
  rsync -av --ignore-existing \
    "$REMOTE_HOST:$REMOTE_DIR/MDStep"*".rkf" \
    "$LOCAL_DIR/"
fi

