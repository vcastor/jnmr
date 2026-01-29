#!/bin/bash

BASE=/home/2026014/vcasto03/snapshots

find "$BASE" -maxdepth 1 -type d \
   \( -mtime +7 -o -name '1*' \) \
   -exec rm -rf {} +

