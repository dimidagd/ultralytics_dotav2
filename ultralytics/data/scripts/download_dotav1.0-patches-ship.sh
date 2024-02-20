#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Example usage: bash data/scripts/get_coco128.sh
# parent
# ├── ultralytics
# └── datasets
#     └── DOTAv1-patches-ship  ← downloads here

# Save git root path to a variable
set -e
GIT_ROOT=$(git rev-parse --show-toplevel)
bash $GIT_ROOT/ultralytics/data/scripts/download_dotav1.0.sh # XXX: ultralytics will have to be stripped for package distribution

parent_dataset_name=DOTAv1
dataset_name=$parent_dataset_name-patches-ship # Must match dataset definition
d=$GIT_ROOT/examples/datasets # unzip directory XXX: Might have to be removed somewhere else for ultralytics to actually find them
dataset_dir=$d/$dataset_name
rm -rf $dataset_dir


python3 - <<EOF
from ultralytics.data.split_dota import split_trainval, split_test
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # There is a security issue related to this.
# split train and val set, with labels.
split_trainval(
    data_root='$d/$parent_dataset_name',
    save_dir='$d/$dataset_name',
    rates=[0.5, 1.0, 1.5],    # multi-scale
    gap=500,
    mapping={1:0} # map ships from DOTAv1.0 to 0 class idx, removes all other classes
)
EOF

echo "Cleaning up.." && \
echo "Removing parent dir $d/$parent_dataset_name" && rm -rf $d/$parent_dataset_name