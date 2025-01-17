#!/usr/bin/env python3

import os
import yaml
import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleanup saved_models directories by removing unnecessary model files."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete the files. By default, the script performs a dry run.",
    )
    return parser.parse_args()


def load_best_steps(yaml_path, model_type):
    """
    Load the best_step values from the YAML file based on the model type.

    Args:
        yaml_path (Path): Path to the trainres.yaml file.
        model_type (str): 'compression' or 'denoising'

    Returns:
        Set[int]: A set of step numbers to keep.
    """
    with open(yaml_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            return set()

    best_step = data.get("best_step", {})
    steps = set()

    if model_type == "compression":
        # Collect steps for keys starting with 'val_combined'
        for key, step in best_step.items():
            if key.startswith("val_combined"):
                steps.add(step)
    elif model_type == "denoising":
        # Collect steps for keys starting with 'val_msssim_loss'
        for key, step in best_step.items():
            if key.startswith("val_msssim_loss"):
                steps.add(step)
    else:
        print(f"Unknown model type: {model_type}")

    return steps


def get_model_type(model_dir, dc_path, denoise_path):
    """
    Determine whether the model directory is a compression or denoising model.

    Args:
        model_dir (Path): Path to the model directory.
        dc_path (Path): Path to compression models base directory.
        denoise_path (Path): Path to denoising models base directory.

    Returns:
        str: 'compression' or 'denoising' or 'unknown'
    """
    try:
        relative = model_dir.relative_to(dc_path)
        return "compression"
    except ValueError:
        try:
            relative = model_dir.relative_to(denoise_path)
            return "denoising"
        except ValueError:
            return "unknown"


def cleanup_saved_models(base_dir, model_type, delete=False):
    """
    Cleanup the saved_models directories within the base directory.

    Args:
        base_dir (Path): Base directory containing model directories.
        model_type (str): 'compression' or 'denoising'
        delete (bool): If True, perform actual deletion. Otherwise, perform a dry run.

    Returns:
        int: Total bytes that would be or have been saved.
    """
    total_bytes_saved = 0
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue

        yaml_path = model_dir / "trainres.yaml"
        saved_models_dir = model_dir / "saved_models"

        if not yaml_path.exists():
            print(f"Skipping {model_dir}: trainres.yaml not found.")
            continue

        if not saved_models_dir.exists():
            print(f"Skipping {model_dir}: saved_models directory not found.")
            continue

        steps_to_keep = load_best_steps(yaml_path, model_type)
        if not steps_to_keep:
            print(f"No steps to keep for {model_dir}. Skipping.")
            continue

        # Check if at least one pair of 'to keep' files exists
        keep_files_exist = False
        for step in steps_to_keep:
            iter_pt = saved_models_dir / f"iter_{step}.pt"
            iter_pt_opt = saved_models_dir / f"iter_{step}.pt.opt"
            if iter_pt.exists() and iter_pt_opt.exists():
                keep_files_exist = True
                break

        if not keep_files_exist:
            print(
                f"Warning: None of the 'to keep' files exist for {model_dir}. Skipping deletion for this model."
            )
            continue

        # Compile regex to match files like iter_{number}.pt and iter_{number}.pt.opt
        pattern = re.compile(r"^iter_(\d+)\.pt(?:\.opt)?$")

        for file in saved_models_dir.iterdir():
            if not file.is_file():
                continue

            match = pattern.match(file.name)
            if match:
                step = int(match.group(1))
                if step not in steps_to_keep:
                    file_size = file.stat().st_size
                    total_bytes_saved += file_size
                    if delete:
                        try:
                            file.unlink()
                            print(f"Deleted: {file}")
                        except Exception as e:
                            print(f"Error deleting {file}: {e}")
                    else:
                        print(f"Would delete: {file}")
            else:
                # Optionally handle files that don't match the pattern
                print(f"Skipping unrecognized file: {file}")

    return total_bytes_saved


def main():
    args = parse_args()

    dc_base = Path("/orb/benoit_phd/models/rawnind_dc")
    denoise_base = Path("/orb/benoit_phd/models/rawnind_denoise")

    total_bytes = 0

    print("Processing Compression Models...")
    total_bytes += cleanup_saved_models(dc_base, "compression", delete=args.delete)

    print("\nProcessing Denoising Models...")
    total_bytes += cleanup_saved_models(denoise_base, "denoising", delete=args.delete)

    total_gb = total_bytes / (1024**3)

    if not args.delete:
        print(
            f"\nDry run completed. Total memory that would be saved: {total_gb:.2f} GB."
        )
    else:
        print(f"\nDeletion completed. Total memory saved: {total_gb:.2f} GB.")


if __name__ == "__main__":
    main()
