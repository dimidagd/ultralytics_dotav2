#!/bin/bash

# Check if filenames are provided as arguments
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <file1> <file2> ..."
    exit 1
fi

# Loop through provided filenames
for filename in "$@"; do
    # Check if the filename length is greater than 7
    if [ ${#filename} -gt 7 ]; then
        # Extract the substring starting from the 8th character
        new_filename="${filename:7}"
        
        # Rename the file
        mv "$filename" "$new_filename"
        echo "Renamed $filename to $new_filename"
    fi
done
