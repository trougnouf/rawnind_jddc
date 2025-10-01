#!/bin/bash

# Download a small sample of the RawNIND dataset for testing pairing logic
# Based on the command from README.md

echo "Creating dataset directory structure..."
mkdir -p src/rawnind/datasets/RawNIND/src/Bayer
mkdir -p src/rawnind/datasets/RawNIND/proc/lin_rec2020

echo "Fetching dataset file list..."
curl -s "https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM" > dataset_info.json

echo "Available files:"
jq -r '.data.latestVersion.files[] | "\(.dataFile.filename) - \(.dataFile.description // "No description")"' dataset_info.json | head -20

echo ""
echo "Downloading first few files for testing..."

# Download just the first few files to test the pairing logic
jq -r '.data.latestVersion.files[] | "wget -c -O \"\(.dataFile.filename)\" https://dataverse.uclouvain.be/api/access/datafile/\(.dataFile.id)"' dataset_info.json | head -5 | bash

echo "Downloaded files:"
ls -la *.tar.gz *.zip *.dng *.tiff 2>/dev/null || echo "No image files downloaded yet"

echo ""
echo "Note: The dataset appears to be packaged in archives. You may need to extract them to get individual image files."
echo "Check the downloaded files and extract as needed."