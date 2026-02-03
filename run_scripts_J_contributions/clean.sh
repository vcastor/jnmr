#!/bin/bash

command ls -1 *run | sed 's/run$//' | sort | tail -n +21 | while read -r base; do
   rm -f "${base}run" "${base}sl"
done
