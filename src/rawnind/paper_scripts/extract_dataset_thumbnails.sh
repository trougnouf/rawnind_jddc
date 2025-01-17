#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default mosaic format
MOSAICFORMAT="X-Trans"  # X-Trans, Bayer

# Parse command-line arguments
while getopts "m:" opt; do
  case $opt in
    m)
      MOSAICFORMAT=$OPTARG
      ;;
    *)
      echo "Usage: $0 [-m MOSAICFORMAT]"
      exit 1
      ;;
  esac
done

# Define source and destination directories
SRC_DIR="/orb/benoit_phd/datasets/RawNIND/src/${MOSAICFORMAT}"
DEST_DIR="/orb/benoit_phd/datasets/RawNIND/Thumbnails/${MOSAICFORMAT}"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Function to extract PreviewImage or fallback to ThumbnailImage
extract_preview_or_thumbnail() {
    local image_file="$1"
    local output_file="$2"

    # Attempt to extract PreviewImage
    if exiftool -b -PreviewImage "$image_file" > "$output_file" 2>/dev/null; then
        echo "PreviewImage extracted."
    # Fallback to ThumbnailImage if PreviewImage is not available
    elif exiftool -b -ThumbnailImage "$image_file" > "$output_file" 2>/dev/null; then
        echo "ThumbnailImage extracted as fallback."
    else
        echo "No PreviewImage or ThumbnailImage found."
        rm -f "$output_file"  # Remove empty or failed output file
        return 1
    fi
    return 0
}

# Find all directories named "gt" within the source directory
find "$SRC_DIR" -type d -name "gt" | while IFS= read -r gt_dir; do
    # Get the name of the parent directory of "gt" to use in the thumbnail filename
    parent_dir=$(basename "$(dirname "$gt_dir")")

    # Find one file within the "gt" directory (no extension check)
    image_file=$(find "$gt_dir" -type f | head -n 1)

    if [[ -n "$image_file" ]]; then
        # Define the output thumbnail filename
        # Using .jpg extension; adjust if necessary
        thumbnail_file="$DEST_DIR/${parent_dir}_preview.jpg"

        # Extract the preview or thumbnail using the function
        if extract_preview_or_thumbnail "$image_file" "$thumbnail_file"; then
            echo "Thumbnail extracted for '$parent_dir' and saved as '$thumbnail_file'."
        else
            echo "Failed to extract any image from '$image_file'. Skipping..."
        fi
    else
        echo "No image files found in '$gt_dir'. Skipping..."
    fi
done

echo "Thumbnail extraction completed."
