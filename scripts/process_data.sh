#!/bin/bash

# Script to HTML benchmark data files through the guidellm trainer
# Usage: ./process_data.sh <input_dir> <output_dir>
# Example: ./process_data.sh experiments/exp1/data experiments/exp1/results

set -e  # Exit on error

# Check if input directory is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    echo "Example: $0 experiments/exp1/data experiments/exp1/results"
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

# Count total files
total_json_files=$(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f | wc -l | tr -d ' ')
total_html_files=$(find "$INPUT_DIR" -maxdepth 1 -name "*.html" -type f | wc -l | tr -d ' ')

if [ "$total_json_files" -eq 0 ] && [ "$total_html_files" -eq 0 ]; then
    echo "No JSON or HTML files found in $INPUT_DIR"
    exit 1
fi

if [ "$total_json_files" -gt 0 ]; then
    total_files=$total_json_files
    ftype="json"
elif [ "$total_html_files" -gt 0 ]; then
    total_files=$total_html_files
    ftype="html"
fi

echo "Found $total_files $ftype file(s) to process"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "-----------------------------------"

# Counter for progress
count=0

# Process each JSON or HTML file in the input directory
summary=""
new_line=$'\n'
for input_file in "$INPUT_DIR"/*.$ftype; do
    # Skip if no files match (handles case where glob doesn't expand)
    [ -e "$input_file" ] || continue

    # Increment counter
    count=$((count + 1))

    # Get the base filename without path
    filename=$(basename "$input_file")

    # Get the filename without extension
    name_without_ext="${filename%.$ftype}"

    # Define output file path
    output_file="$OUTPUT_DIR/${name_without_ext}.txt"

    echo "[$count/$total_files] Processing: $filename"
    echo "  Input:  $input_file"
    echo "  Output: $output_file"

    # Run the Go program and redirect output to text file
    if [ "$ftype" == "json" ]; then
        program="guidellm"
    else
        program="guidellm-html"
    fi
    
    if go run "../demos/${program}/main.go" "$input_file" > "$output_file" 2>&1; then
        sum=$(grep "Summary:" "$output_file")
        file_summary=$(echo ${name_without_ext} " " ${sum})
        summary+=$file_summary"\n"
        echo "  ✓ Success"
    else
        echo "  ✗ Failed (see $output_file for details)"
    fi

    echo ""
done

echo -e $summary

echo "-----------------------------------"
echo "Processing complete! Processed $count file(s)"
echo "Results saved in: $OUTPUT_DIR"
