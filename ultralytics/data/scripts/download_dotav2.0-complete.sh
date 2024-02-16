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
MD5SUM_DATASET_HASH="a83b375523f0951fb269f94b49b2d31b"
GIT_ROOT=$(git rev-parse --show-toplevel)
dataset_name=DOTA-v2.0
# Download/unzip images and labels

d=$GIT_ROOT/examples/datasets # unzip directory # unzip directory XXX: Might have to be removed somewhere else for ultralytics to actually find them
dataset_dir=$d/$dataset_name
DATA_DIR=/tmp
url=https://github.com/dimidagd/ultralytics_dotav2/releases/download/dota-v2.0-dataset-complete/
ZIPFILEBASENAME=dota-v2.0-complete.zip

# Download all files in the list

files=(".aa" ".ab" ".ac" ".ad" ".ae" ".af" ".ag" ".ah" ".ai" ".aj" ".ak")

for file in "${files[@]}"
do
    link=$url$ZIPFILEBASENAME$file
    outfile=$DATA_DIR/$ZIPFILEBASENAME$file
    if [ ! -f "$outfile" ]; then
    echo "Downloading $link to $outfile ..."
    curl -L $link -o $outfile -# && echo "Download $outfile successful."
    else
        echo "File $outfile exists already, skipped download."
    fi

done

ZIPFILE=$DATA_DIR/$ZIPFILEBASENAME
# Combine all chunks into the original file, use pv to show progress, and du+awk to get file size sum and printf to print it as an integer.
echo "Combining files into $ZIPFILE" && \
cat $ZIPFILE.?? | pv -s $(printf "%.0f" $(du -sb $ZIPFILE.?? | awk '{total += $1} END {print total}')) > $ZIPFILE && \
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
