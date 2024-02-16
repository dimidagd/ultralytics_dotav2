#!/bin/bash

set -e
if [ $# -eq 0 ]
  then
    echo "No arguments supplied, exiting"
    exit 1
fi
echo "Cleaning up now"
temp_dir=$1
unlink $temp_dir/notebooks/logs
unlink $temp_dir/notebooks/runs
unlink $temp_dir/scripts/hpc_logs
echo "All softlinks unlinked"
rm -rf $temp_dir
