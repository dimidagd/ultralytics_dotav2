# Download/unzip images and labels
#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Example usage: bash data/scripts/get_coco128.sh
# parent
# ├── ultralytics
# └── datasets
#     └── DOTA-v2.0  ← downloads here

# Save git root path to a variable
url=$1 # Base dataset url
if [ -z "$url" ]; then
    url="https://github.com/dimidagd/ultralytics_dotav2/releases/download/dota-v2.0/"
fi
GIT_ROOT=$(git rev-parse --show-toplevel)
parent_dataset_name=DOTA-v2.0
bash $GIT_ROOT/ultralytics/data/scripts/download_dotav2.0-complete.sh $url # XXX: ultralytics will have to be stripped for package distribution
dataset_name=$parent_dataset_name-patches
# Download/unzip images and labels

d=$GIT_ROOT/datasets # unzip directory XXX: Might have to be removed somewhere else for ultralytics to actually find them
dataset_dir=$d/$dataset_name
rm -rf $dataset_dir

python3 - << EOF
from ultralytics.data.split_dota import split_trainval, split_test
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # There is a security issue related to this.
# split train and val set, with labels.
split_trainval(
    data_root='$d/$parent_dataset_name',
    save_dir='$d/$dataset_name',
    rates=[0.5, 1.0, 1.5],    # multi-scale
    gap=500
)
EOF
