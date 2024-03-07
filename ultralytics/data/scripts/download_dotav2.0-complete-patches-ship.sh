#!/bin/bash

# Download/unzip images and labels
#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Download COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
# Example usage: bash data/scripts/get_coco128.sh
# parent
# ├── ultralytics
# └── datasets
#     └── DOTA-v2.0  ← downloads here


url=$1 # Base dataset url
if [ -z "$url" ]; then
    url=https://github.com/dimidagd/ultralytics_dotav2/releases/download/dota-v2.0/
fi
# Save git root path to a variable
GIT_ROOT=$(git rev-parse --show-toplevel)
parent_dataset_name=DOTA-v2.0
dataset_name=$parent_dataset_name-patches-ship
ULTRALYTICS_DS_DIR=$GIT_ROOT/ultralytics/datasets
d=/tmp/datasets # unzip directory XXX: Might have to be removed somewhere else for ultralytics to actually find them
mkdir -p $d
input_dir=$d/$parent_dataset_name
output_dir=$ULTRALYTICS_DS_DIR/$dataset_name
rm -rf $output_dir && mkdir -p $output_dir

bash $GIT_ROOT/ultralytics/data/scripts/download_dotav2.0-complete.sh $url $parent_dataset_name $input_dir # XXX: ultralytics will have to be stripped for package distribution


python3 - <<EOF
from ultralytics.data.split_dota import split_trainval, split_test
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # There is a security issue related to this.
# split train and val set, with labels.
split_trainval(
    data_root='$input_dir',
    save_dir='$output_dir',
    rates=[0.5, 1.0, 1.5],    # multi-scale
    gap=500,
    mapping={1:0} # map ships from DOTAv2.0 to 0 class idx
)

EOF
