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

# Save git root path to a variable
set -p

url=$1 # Base dataset url
dataset_dir=$2 # unzip directory
dataset_name=$3 # dataset name

if [ -z "$url" ]; then
    echo "No url provided, exiting"
    return 1
fi
if [ -z "$dataset_name" ]; then
    echo "No dataset provided, exiting"
    return 1
fi
if [ -z "$dataset_dir" ]; then
    GIT_ROOT=$(git rev-parse --show-toplevel)
    d=$GIT_ROOT/datasets # unzip directory # unzip directory XXX: Might have to be removed somewhere else for ultralytics to actually find them
    dataset_dir=$d/$dataset_name
fi

MD5SUM_DATASET_HASH="a09649a06b3e52a7d3d54e9d2f765b2c"


# Download/unzip images and labels


DATA_DIR=/tmp/downloads/$dataset_name
mkdir -p $DATA_DIR
ZIPFILEBASENAME=dl-$dataset_name.tar
md5file=$DATA_DIR/$ZIPFILEBASENAME.md5list
#Check if url is filepath
if [ -d $url ]; then
    url=file://$url
    echo "URL is a file, reading from file $url"
else
    echo "URL $url is not a dir, using as is"
fi

python3 -c "
import os
n_cpu = os.cpu_count()
from ultralytics.utils import downloads
downloads.download('${url}md5list', dir='${DATA_DIR}')
md5_hashes, file_list = downloads.read_md5list('${DATA_DIR}/md5list')
file_list = ['${url}' + f for f in file_list]
downloads.download(url=file_list, dir='${DATA_DIR}', md5_hashes=md5_hashes,threads=n_cpu, delete=True)
"

zipfiles=$(cat ${DATA_DIR}/md5list | awk -v prefix="$DATA_DIR/" '{print prefix $2} ' | sed 's/\.tgz$//')
ZIPFILE=$DATA_DIR/$ZIPFILEBASENAME
# Combine all chunks into the original file, use pv to show progress, and du+awk to get file size sum and printf to print it as an integer.
echo "Combining files $zipfiles into $ZIPFILE" && \
cat $zipfiles | pv -s $(printf "%.0f" $(du -sb $zipfiles | awk '{total += $1} END {print total}')) > $ZIPFILE && \
echo "Combined all chunks into $ZIPFILE"

rm -rf $dataset_dir && \
mkdir -p $dataset_dir && \
echo "Untaring $ZIPFILE into $dataset_dir" && \
pv "$ZIPFILE" | tar xf - --directory "$dataset_dir" && \
echo "Untared $ZIPFILE into $dataset_dir" && \
rm $ZIPFILE && \
echo "Removed $ZIPFILE"
