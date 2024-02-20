# Download/unzip images and labels
#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Example usage: bash data/scripts/get_coco128.sh
# parent
# ├── ultralytics
# └── datasets
#     └── DOTAv1  ← downloads here

set -e
download_dataset() {
    GIT_ROOT=$(git rev-parse --show-toplevel)
    d=$GIT_ROOT/examples/datasets # unzip directory # unzip directory XXX: Might have to be removed somewhere else for ultralytics to actually find them

    dataset_name=DOTAv1
    dataset_dir=$d/$dataset_name
    DATA_DIR=/tmp
    ZIPFILEBASENAME=DOTAv1.zip
    ZIPFILE=$DATA_DIR/$ZIPFILEBASENAME
    # curl dataset .zip from 
    link=https://github.com/ultralytics/yolov5/releases/download/v1.0/

    outfile=$ZIPFILE
    if [ ! -f "$outfile" ]; then
        echo "$outfile does not exist, downloading..."
        echo "Downloading $link to $outfile ..."
        curl -L $link -o $outfile -# && echo "Download $outfile successful."
    else
        echo "File $outfile exists already, skipped download."
    fi
    # Remove dataset if exists already
    
    dataset_dir=$d/$dataset_name
    echo "Removing $dataset_dir in case it exists already" && rm -rf $dataset_dir
    # unzip to one directory up because zip file includes name
    echo "Unzipping dataset to $d" && unzip -o -d $d $ZIPFILE | pv -l -s $(unzip -Z -1 $ZIPFILE | wc -l) > /dev/null && \
    echo "Unzip to $dataset_dir successful."
}
download_dataset