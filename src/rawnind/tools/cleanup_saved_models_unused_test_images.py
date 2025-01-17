#!/usr/bin/env python3

import os
import yaml
import argparse
from pathlib import Path
from typing import Set, List, Tuple

# Import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Will handle if tqdm is not installed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify and report non-important models by removing their .tif and .exr images."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete the files. By default, the script performs a dry run.",
    )
    return parser.parse_args()


def load_trained_models(yaml_path: Path) -> Set[str]:
    """
    Load the trained models from a YAML file.

    Args:
        yaml_path (Path): Path to the YAML file.

    Returns:
        Set[str]: A set of model directory names.
    """
    if not yaml_path.exists():
        print(
            f"Warning: YAML file {yaml_path} does not exist. No important models loaded from this file."
        )
        return set()

    with open(yaml_path, "r") as f:
        try:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return set(data.keys())
            else:
                print(
                    f"Warning: Unexpected YAML format in {yaml_path}. Expected a dictionary at the top level."
                )
                return set()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            return set()


def find_nonimportant_models(
    base_dir: Path, important_models: Set[str], exclude_substring: str = None
) -> List[Path]:
    """
    Find model directories that are not in the important_models set and do not contain the exclude_substring.

    Args:
        base_dir (Path): Base directory containing model directories.
        important_models (Set[str]): Set of important model directory names.
        exclude_substring (str, optional): Substring to exclude models containing it. Defaults to None.

    Returns:
        List[Path]: List of non-important model directory paths.
    """
    nonimportant = []
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name in important_models:
            continue
        if exclude_substring and exclude_substring.lower() in model_dir.name.lower():
            print(
                f"Excluding model due to '{exclude_substring}' in name: {model_dir.name}"
            )
            continue
        nonimportant.append(model_dir)
    return nonimportant


def calculate_space_and_list_models(
    models: List[Path], delete: bool = False
) -> Tuple[List[str], int]:
    """
    Calculate the total space of .tif and .exr files in the given models.

    Args:
        models (List[Path]): List of model directory paths.
        delete (bool): If True, delete the files. Otherwise, perform a dry run.

    Returns:
        Tuple[List[str], int]: Tuple containing the list of model names to be cleaned and the total bytes saved.
    """
    models_to_clean = []
    total_bytes_saved = 0

    for model in models:
        # Find all .tif and .exr files recursively
        image_files = list(model.rglob("*.tif")) + list(model.rglob("*.exr"))
        if not image_files:
            continue  # No images to clean in this model

        # Calculate total size for this model
        model_size = sum(f.stat().st_size for f in image_files)
        total_bytes_saved += model_size
        models_to_clean.append(str(model))

    if delete and models_to_clean:
        print("\nStarting deletion of .tif and .exr images...")
        # Gather all files to delete
        all_files_to_delete = []
        for model in models:
            image_files = list(model.rglob("*.tif")) + list(model.rglob("*.exr"))
            all_files_to_delete.extend(image_files)

        if tqdm:
            # Use tqdm progress bar if available
            for file in tqdm(all_files_to_delete, desc="Deleting files", unit="file"):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
        else:
            # Fallback to simple progress messages if tqdm is not installed
            for idx, file in enumerate(all_files_to_delete, start=1):
                try:
                    file.unlink()
                    print(f"Deleted ({idx}/{len(all_files_to_delete)}): {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")

    return models_to_clean, total_bytes_saved


def main():
    args = parse_args()

    # Check if --delete is used and tqdm is not installed
    if args.delete and tqdm is None:
        print(
            "Note: 'tqdm' library is not installed. Progress bars will not be displayed."
        )
        print("To install tqdm and enable progress bars, run: pip install tqdm")

    # Paths to the YAML configuration files
    trained_dc_models_yaml = Path(
        "/orb/benoit_phd/src/rawnind/config/trained_dc_models.yaml"
    )
    trained_denoise_models_yaml = Path(
        "/orb/benoit_phd/src/rawnind/config/trained_denoise_models.yaml"
    )

    # Base directories for compression and denoising models
    dc_base = Path("/orb/benoit_phd/models/rawnind_dc")
    denoise_base = Path("/orb/benoit_phd/models/rawnind_denoise")

    # Load important models from YAML files
    important_dc_models = load_trained_models(trained_dc_models_yaml)
    important_denoise_models = load_trained_models(trained_denoise_models_yaml)

    # Find non-important models, excluding those with "bm3d" in their names
    exclude_substring = "bm3d"
    nonimportant_dc_models = find_nonimportant_models(
        dc_base, important_dc_models, exclude_substring=exclude_substring
    )
    nonimportant_denoise_models = find_nonimportant_models(
        denoise_base, important_denoise_models, exclude_substring=exclude_substring
    )

    # Combine non-important models from both directories
    all_nonimportant_models = nonimportant_dc_models + nonimportant_denoise_models

    if not all_nonimportant_models:
        print(
            "No non-important models found (excluding models with 'bm3d' in their names). Nothing to clean."
        )
        return

    # Calculate space and list models to clean
    models_to_clean, total_bytes_saved = calculate_space_and_list_models(
        all_nonimportant_models, delete=args.delete
    )

    if not models_to_clean:
        print(
            "No .tif or .exr images found in non-important models (excluding 'bm3d' models). Nothing to clean."
        )
        return

    # Convert bytes to gigabytes
    total_gb_saved = total_bytes_saved / (1024**3)

    # Print the results
    print("\nModels to be cleaned up:")
    for model in models_to_clean:
        print(f"- {model}")

    print(f"\nTotal space that would be saved: {total_gb_saved:.2f} GB.")

    if args.delete:
        print("\nDeletion process completed.")
    else:
        print("\nDry run completed. No files were deleted.")


if __name__ == "__main__":
    main()
