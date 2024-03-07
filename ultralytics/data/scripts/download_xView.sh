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
    url=https://github.com/dimidagd/ultralytics_dotav2/releases/download/xView/
fi
if [ -z "$dataset_name" ]; then
    dataset_name=xView
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
ZIPFILEBASENAME=xView.zip
md5file=$DATA_DIR/$ZIPFILEBASENAME.md5list
#Check if url is filepath
if [ -d $url ]; then
    url=file://$url
    echo "URL is a file, reading from file $url"
else
    echo "URL $url is not a dir, using as is"
fi
curl -L $url$md5list -o $md5file -# && echo "Download $md5file successful."
zipfiles=
while read -r line; do
    md5sum=$(echo $line | awk '{print $1}')
    filename=$(echo $line | awk '{print $2}')
    link=$url$(basename $filename)
    outfile=$DATA_DIR/$(basename $filename)
    echo "Found filename in artifacts: $filename ($md5sum), "
    if [ ! -f "$outfile" ] || [ "$(md5sum "$outfile" | awk '{print $1}')" != "$md5sum" ]; then
        echo "Downloading $link to $outfile ..."
        #curl -L $link -o $outfile -# && echo "Download $outfile successful."
        curl -L $link -o $outfile -# > "$outfile.log" 2>&1 &
        echo "Download $outfile started in the background. See $outfile.log for progress."
    else
        echo "File $outfile exists already and has the same MD5 sum, skipped download."
    fi
    zipfiles="$zipfiles $outfile"
done < $md5file
wait

ZIPFILE=$DATA_DIR/$ZIPFILEBASENAME
# Combine all chunks into the original file, use pv to show progress, and du+awk to get file size sum and printf to print it as an integer.
echo "Combining files into $ZIPFILE" && \
cat $zipfiles | pv -s $(printf "%.0f" $(du -sb $zipfiles | awk '{total += $1} END {print total}')) > $ZIPFILE && \
echo "Combined all chunks into $ZIPFILE"

echo "Checking MD5sum of $ZIPFILE" && \
md5sum $ZIPFILE | awk '{print $1}' | grep -q $MD5SUM_DATASET_HASH && \
echo "MD5sum match" || echo "MD5sum mismatch, redownload the files."

# && \
# rm $ZIPFILE.?? && \
# echo "Removed all chunks" \
# || echo "MD5sum mismatch, redownload the files?"

rm -rf $dataset_dir && \
mkdir -p $dataset_dir && \
echo "Unzipping $ZIPFILE into $dataset_dir" && \
unzip -o -d $dataset_dir $ZIPFILE | pv -l -s $(unzip -Z -1 $ZIPFILE | wc -l) > /dev/null && \
echo "Unzipped $ZIPFILE into $dataset_dir" && \
rm $ZIPFILE && \
echo "Removed $ZIPFILE"
