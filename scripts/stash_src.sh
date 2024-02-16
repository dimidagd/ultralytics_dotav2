#!/bin/bash

temp_dir=/work3/dimda/stash_dir`mktemp -d`
mkdir -p $temp_dir
current_dir=$PWD/..
# Archive the source code directory (this goes in submit.sh script)
echo "Creating archive of $current_dir at $temp_dir/source_code.tar.gz"
tar \
--exclude="./notebooks/outputs" \
--exclude="scripts/hpc_logs" \
--exclude="runs" \
--exclude="wandb" \
--exclude="examples/datasets" \
--exclude="./.idea" \
--exclude="./.ipynb_checkpoints" \
--exclude="./.mypy_cache" \
--exclude="./notebooks/runs" \
--exclude="./notebooks/logs" \
--exclude="./notebooks/wandb" \
--exclude="./tmp" \
-cf "$temp_dir/source_code.tar.gz" \
-C $current_dir .

export temp_dir=$temp_dir
echo "Created stash"
