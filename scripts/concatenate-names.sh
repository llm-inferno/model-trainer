#!/bin/bash

# Initialize an empty string
file_names_string=""
separator="$"

# Set directory path from the first argument
directory_path="$1"
# Check if the directory exists
if [ ! -d "$directory_path" ]; then
    echo "Error: Directory '$directory_path' does not exist."
    exit 1

# Loop through all files in the current directory
for file in "$directory_path"/*; do
    # Check if it is a regular file (optional)
    if [ -f "$file" ]; then
        # Concatenate the filename followed by a space
        file_names_string+="$file$separator"
    fi
done

# Print the final result
echo "$file_names_string"
