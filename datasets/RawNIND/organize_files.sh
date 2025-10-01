#!/bin/bash

# --- Script to organize downloaded RawNIND files ---
#
# This script reorganizes files from a flat structure into a nested
# directory structure based on their filenames.
#
# Example Filename: Bayer_7D-1_GT_ISO100_sha1=...cr2
#   - Type: Bayer
#   - Scene: 7D-1
#   - Subdir: gt (because of "_GT_")
#
# Final Path: <output_dir>/{Bayer or X-Trans}/<scene_name>/[gt/]<filename>

# --- Configuration and Argument Handling ---

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_base_directory>"
    echo "Example: $0 ./src/flat ."
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_BASE_DIR="$2"
TARGET_DIR="$OUTPUT_BASE_DIR/src"

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found."
    exit 1
fi

echo "Input directory:  $INPUT_DIR"
echo "Output directory: $TARGET_DIR"
echo "--------------------------------------------------"

# --- Main Logic ---

# Create the base target directory
mkdir -p "$TARGET_DIR"

# Loop through every file in the input directory
for filepath in "$INPUT_DIR"/*; do
    # Skip if it's not a regular file (e.g., a subdirectory)
    if [ ! -f "$filepath" ]; then
        continue
    fi

    filename=$(basename "$filepath")

    # --- Parse the filename ---
    # Get the type (Bayer or X-Trans) - the part before the first '_'
    type=$(echo "$filename" | cut -d'_' -f1)

    # Get the scene name - the part between the first and second '_'
    scene=$(echo "$filename" | cut -d'_' -f2)

    # Check if parsing was successful
    if [ -z "$type" ] || [ -z "$scene" ]; then
        echo "Warning: Could not parse '$filename'. Skipping."
        continue
    fi

    # --- Determine the final destination directory ---
    # Default destination
    dest_dir="$TARGET_DIR/$type/$scene"

    # If the filename contains "_GT_", add a "gt" subdirectory
    if [[ "$filename" == *"_GT_"* ]]; then
        dest_dir="$dest_dir/gt"
    fi

    # --- Create directory and move file ---
    echo "Processing: $filename"
    
    # Create the destination directory tree (-p creates parent dirs as needed)
    mkdir -p "$dest_dir"

    # Move the file into its new home
    mv -v "$filepath" "$dest_dir/"
done

echo "--------------------------------------------------"
echo "Organization complete."
echo "All files moved to subdirectories within '$TARGET_DIR'."
