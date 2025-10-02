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

    # --- Parse the filename using a Regular Expression ---
    # This regex is designed to handle scene names with multiple underscores.
    # It captures three parts:
    # 1. The type (e.g., "Bayer")
    # 2. The scene name (e.g., "UNK_LucieB_stick_figurine")
    # 3. The metadata part (everything from "_GT_ISO" or "_ISO" to the end)
    regex="^([^_]+)_(.*)_((GT_)?ISO.*)$"

    if [[ "$filename" =~ $regex ]]; then
        # Parsing was successful, extract the captured groups
        type="${BASH_REMATCH[1]}"
        scene="${BASH_REMATCH[2]}"

        # --- FIX: Clean the scene name ---
        # The regex's middle group (.*) is "greedy" and might capture a trailing "_GT".
        # This line removes the "_GT" suffix from the scene name if it exists.
        scene=${scene%_GT}

    else
        # If the regex doesn't match, the filename has an unexpected format.
        echo "Warning: Could not parse '$filename'. Skipping."
        continue
    fi

    # --- Determine the final destination directory ---
    # Base destination
    dest_dir="$TARGET_DIR/$type/$scene"

    # If the filename contains "_GT_", add a "gt" subdirectory
    if [[ "$filename" == *"_GT_"* ]]; then
        dest_dir="$dest_dir/gt"
    fi

    # --- Create directory and move file ---
    echo "Processing: $filename"
    echo "  -> To: $dest_dir"

    # Create the destination directory tree (-p creates parent dirs as needed)
    mkdir -p "$dest_dir"

    # Move the file into its new home
    # The -n flag prevents overwriting existing files, just in case.
    mv -vn "$filepath" "$dest_dir/"
done

echo "--------------------------------------------------"
echo "Organization complete."
echo "All files moved to subdirectories within '$TARGET_DIR'."
