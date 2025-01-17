import os
import subprocess
import json
import csv
import shutil
import logging
from collections import defaultdict
from tqdm import tqdm

# Configure logging for error handling
logging.basicConfig(
    filename="error_log.txt",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Define datasets, base root, and root directories
DATASETS = ["Bayer", "X-Trans", "Bostitch"]
BASE_ROOT = "/orb/benoit_phd/datasets/"
# BASE_ROOT = (
#     "/run/media/trougnouf/7c3f52ec-9162-4a30-baa0-eab17dad3080/benoit_phd_datasets"
# )

ROOT_DIRS = {
    "Bayer": os.path.join(BASE_ROOT, "RawNIND/src/Bayer"),
    "X-Trans": os.path.join(BASE_ROOT, "RawNIND/src/X-Trans"),
    "Bostitch": os.path.join(BASE_ROOT, "RawNIND_Bostitch/src/Bayer"),
}

# Define substrings that identify test sets per dataset
TEST_RESERVES = {
    "Bayer": [
        "7D-2",
        "Vaxt-i-trad",
        "Pen-pile",
        "MuseeL-vases-A7C",
        "D60-1",
        "MuseeL-Saint-Pierre-C500D",
        "TitusToys",
        "boardgames_top",
        "Laura_Lemons_platformer",
        "MuseeL-bluebirds-A7C",
    ],
    "Bostitch": [
        "LucieB_bw_drawing1",
        "LucieB_bw_drawing2",
        "LucieB_board",
        "LucieB_painted_wallpaper",
        "LucieB_painted_plants",
        "LucieB_groceries",
    ],
    "X-Trans": [
        "ursulines-red",
        "stefantiek",
        "ursulines-building",
        "MuseeL-Bobo",
        "CourtineDeVillersDebris",
        "MuseeL-Bobo-C500D",
        "Pen-pile",
    ],
}

# Define raw image extensions (including 'raf') and handle case-insensitivity
RAW_EXTENSIONS = {
    ".raw",
    ".nef",
    ".cr2",
    ".arw",
    ".dng",
    ".rw2",
    ".orf",
    ".sr2",
    ".raf",
    ".crw",
}

# Initialize a dictionary to collect unknown extensions
unknown_extensions = defaultdict(int)

# Define the path to exiftool executable
EXIFTOOL_CMD = "exiftool"


def get_exif_data(file_path):
    """
    Extracts EXIF data using exiftool and returns a dictionary.
    """
    try:
        # Run exiftool and get JSON output for Model and ISO
        result = subprocess.run(
            [EXIFTOOL_CMD, "-j", "-Model", "-ISO", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        exif_json = json.loads(result.stdout)
        if exif_json:
            return exif_json[0]  # exiftool returns a list
    except subprocess.CalledProcessError as e:
        logging.error(f"Error reading EXIF data from {file_path}: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing EXIF data from {file_path}: {e}")
    return {}


def is_test_set(set_name, test_substrings):
    """
    Determines if the image set is a test set based on its name and test substrings.
    """
    for substring in test_substrings:
        if substring in set_name:
            return True
    return False


def process_image_set(image_set_path, set_type, data):
    """
    Processes a single image set directory.
    """
    # To track whether we've incremented the 'sets' count for a camera
    sets_counted_for_camera = set()
    this_set_count = {"clean": 0, "noisy": 0}
    camera = None

    # Walk through the image set directory
    for root, dirs, files in os.walk(image_set_path):
        # Determine if we're in the 'gt' directory
        relative_path = os.path.relpath(root, image_set_path)
        if relative_path == "gt":
            image_type = "clean"
        elif relative_path.startswith("gt" + os.sep):
            # In case 'gt' has subdirectories
            image_type = "clean"
        else:
            image_type = "noisy"

        for file in files:
            file_lower = file.lower()
            _, ext = os.path.splitext(file_lower)
            if ext in RAW_EXTENSIONS:
                file_path = os.path.join(root, file)
                exif_data = get_exif_data(file_path)
                camera = exif_data.get("Model", "UnknownCamera").strip()
                iso = exif_data.get("ISO", None)

                # Handle missing or invalid ISO
                if iso is None or not str(iso).isdigit():
                    iso = "UnknownISO"
                    print(f"Warning: ISO not found or invalid for file: {file_path}")
                else:
                    iso = str(iso)

                # Handle missing camera information
                if not camera:
                    camera = "UnknownCamera"

                # Increment 'sets' count if not already done for this camera in this set
                if camera not in sets_counted_for_camera:
                    data[camera][set_type]["sets"] += 1
                    sets_counted_for_camera.add(camera)

                # Update noisy or clean counts
                data[camera][set_type][image_type][iso] += 1
                this_set_count[image_type] += 1

            else:
                if ext:  # If the file has an extension
                    unknown_extensions[ext] += 1
    print(f"Processed {image_set_path} ({set_type}, {this_set_count}, {camera})")


def collect_unique_isos(data):
    """
    Collects all unique ISO values across all cameras and set types.
    Returns a sorted list of ISO values as strings.
    """
    iso_set = set()
    for camera, sets_data in data.items():
        for set_type in ["train", "test"]:
            for image_type in ["noisy", "clean"]:
                for iso in sets_data[set_type][image_type]:
                    iso_set.add(iso)
    # Remove 'UnknownISO' if present and sort numerically
    known_isos = sorted([iso for iso in iso_set if iso.isdigit()], key=lambda x: int(x))
    unknown_present = "UnknownISO" in iso_set
    if unknown_present:
        known_isos.append("UnknownISO")
    return known_isos


def write_csv(dataset_name, set_type, isos, data, output_dir):
    """
    Writes a CSV file for the given dataset and set type ('train' or 'test').
    Includes a 'Total' row at the end.
    """
    filename = os.path.join(output_dir, f"{dataset_name}_summary_{set_type}.csv")
    header = ["Camera"] + [f"ISO{iso}" for iso in isos] + ["#Scenes", "#images"]

    # Initialize totals
    total_iso_counts = {iso: 0 for iso in isos}
    total_sets = 0
    total_images = 0

    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        cameras = list(data.keys())
        with tqdm(
            cameras, desc=f"Writing {set_type} CSV for {dataset_name}", unit="camera"
        ) as pbar:
            for camera in cameras:
                sets_data = data[camera][set_type]
                row = [camera]
                camera_total_images = 0

                for iso in isos:
                    # Sum 'noisy' and 'clean' counts for the ISO
                    noisy_count = sets_data["noisy"].get(iso, 0)
                    clean_count = sets_data["clean"].get(iso, 0)
                    iso_total = noisy_count + clean_count
                    row.append(iso_total)
                    camera_total_images += iso_total

                    # Update totals
                    total_iso_counts[iso] += iso_total

                row.append(sets_data["sets"])
                row.append(camera_total_images)

                # Update total sets and images
                total_sets += sets_data["sets"]
                total_images += camera_total_images

                writer.writerow(row)
                pbar.update(1)

        # Write the Total row
        total_row = ["Total"]
        for iso in isos:
            total_row.append(total_iso_counts[iso])
        total_row.append(total_sets)
        total_row.append(total_images)

        writer.writerow(total_row)

    print(
        f"CSV file for '{set_type}' set of dataset '{dataset_name}' written to: {filename}"
    )


def main():
    # Check if exiftool is available
    if not shutil.which(EXIFTOOL_CMD):
        print(
            f"Error: '{EXIFTOOL_CMD}' is not found. Please install exiftool and ensure it's in your PATH."
        )
        return

    # Iterate over each dataset
    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")

        root_dir = ROOT_DIRS.get(dataset)
        if not root_dir:
            print(f"Warning: Root directory for dataset '{dataset}' is not defined.")
            continue

        test_substrings = TEST_RESERVES.get(dataset, [])

        # Initialize data structure for the current dataset
        # Structure:
        # {
        #   camera: {
        #       'train': {
        #           'sets': count,
        #           'noisy': {iso: count},
        #           'clean': {iso: count}
        #       },
        #       'test': {
        #           'sets': count,
        #           'noisy': {iso: count},
        #           'clean': {iso: count}
        #       }
        #   }
        # }
        data = defaultdict(
            lambda: {
                "train": {
                    "sets": 0,
                    "noisy": defaultdict(int),
                    "clean": defaultdict(int),
                },
                "test": {
                    "sets": 0,
                    "noisy": defaultdict(int),
                    "clean": defaultdict(int),
                },
            }
        )

        # Get list of image sets (directories) in root_dir
        try:
            image_set_names = [
                ds
                for ds in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, ds))
            ]
        except FileNotFoundError:
            print(
                f"Error: Root directory '{root_dir}' for dataset '{dataset}' does not exist."
            )
            continue

        # Process each image set with a progress bar
        for image_set in tqdm(
            image_set_names, desc=f"Processing Image Sets for {dataset}", unit="set"
        ):
            image_set_path = os.path.join(root_dir, image_set)
            set_type = "test" if is_test_set(image_set, test_substrings) else "train"
            process_image_set(image_set_path, set_type, data)

        # Collect all unique ISO values for the current dataset
        isos = collect_unique_isos(data)

        # Define output directory (can be customized)
        output_dir = root_dir  # Saving CSVs in the root_dir; modify if needed

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Write CSVs for both 'train' and 'test' sets
        for set_type in ["train", "test"]:
            write_csv(dataset, set_type, isos, data, output_dir)

    # After processing all datasets, report unknown file extensions
    if unknown_extensions:
        print("\nWarning: Unknown file extensions encountered:")
        for ext, count in unknown_extensions.items():
            print(f"  Extension '{ext}': {count} file(s)")
    else:
        print("\nNo unknown file extensions encountered.")


if __name__ == "__main__":
    main()
