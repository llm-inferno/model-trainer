#!/bin/bash

# Script to copy HTML benchmark data files organized as multiple directories,
# one per experiment, into a single data directory, and changing names 
# of files to correspond to the experiment names.
# Usage: ./copy-data.sh <input_dir> <output_dir>
# Example: ./copy-data.sh experiments/exp2/raw experiments/exp2/data

set -e  # Exit on error

# Check if input directory is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    echo "Example: $0 experiments/exp2/raw experiments/exp2/data"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2" 

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Count total HTML files
total_files=$(find "$INPUT_DIR" -maxdepth 2 -name "*.html" -type f | wc -l | tr -d ' ')

if [ "$total_files" -eq 0 ]; then
    echo "No HTML files found in $INPUT_DIR"
    exit 1
fi

echo "Found $total_files HTML file(s) to process"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "-----------------------------------"

# Counter for progress
count=0

for dir_name in "$INPUT_DIR"/*/; do
    # Skip if no directories match (handles case where glob doesn't expand)
    [ -d "$dir_name" ] || continue

    # Get the base directory name without path
    base_dir=$(basename "$dir_name")

    for html_file in "$dir_name"/*.html; do
        # Skip if no files match (handles case where glob doesn't expand)
        [ -e "$html_file" ] || continue

        # Increment counter
        count=$((count + 1))

        # Define output file path
        output_file="$OUTPUT_DIR/${base_dir}.html"

        echo "[$count/$total_files] Copying: $filename"
        echo "  Input:  $html_file"
        echo "  Output: $output_file"

        # Copy the HTML file to the output directory with .txt extension
        cp "$html_file" "$output_file"
    done
done

echo "-----------------------------------"
echo "Processing complete! Processed $count file(s)"
echo "Results saved in: $OUTPUT_DIR"
