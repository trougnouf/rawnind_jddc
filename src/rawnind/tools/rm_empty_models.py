"""
This script removes empty models from ROOT_MODELS_DPATHS.

Empty models are defined as those which don't have a trainres.yaml file after 15 minutes or anything in saved_models after 1 hour,
or those which have no trainres.yaml file, an empty saved_models directory, and have a train.log file that has not been modified in 2 minutes.
"""

import os
import time
import argparse
import shutil
import yaml

ROOT_MODELS_DPATHS = ["../../models/rawnind_denoise/", "../../models/rawnind_dc/"]
TIME_LIMIT_START = 60
TIME_LIMIT_TRAINLOG = 5 * 60  # 5 minutes
TIME_LIMIT_TRAINRES = 15 * 60  # 15 minutes
TIME_LIMIT_SAVED_MODELS = 60 * 60  # 1 hour
TIME_LIMIT_TRAINRES_EMPTY = 60 * 60  # 1 hour

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove empty model directories")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which directories would be removed, but do not actually remove them.",
    )
    args = parser.parse_args()

    # Define the time duration in seconds for skipping recently created directories

    for root_models_dpath in ROOT_MODELS_DPATHS:
        # Get a list of subdirectories in the model path
        models_dpaths = [f.path for f in os.scandir(root_models_dpath) if f.is_dir()]

        # Loop through each subdirectory
        for model_dpath in models_dpaths:
            # Skip backups and old directories
            if os.path.basename(model_dpath) in ["backups", "old"]:
                continue

            # Get the last modified time of the model directory
            mod_time = os.stat(model_dpath).st_mtime

            # Spare if there are saved models
            if os.path.exists(os.path.join(model_dpath, "saved_models")):
                if len(os.listdir(os.path.join(model_dpath, "saved_models"))) > 0:
                    # and trainres.yaml's best_step key is not an empty dictionary (or it was created les than TIME_LIMIT_TRAINRES_EMPTY seconds ago)
                    if os.path.exists(os.path.join(model_dpath, "trainres.yaml")):
                        with open(os.path.join(model_dpath, "trainres.yaml")) as f:
                            trainres = yaml.load(f, Loader=yaml.FullLoader)
                        if (
                            (  # check that at least one key in trainres["best_step"] starts with "val"
                                trainres["best_step"] != {}
                                and any(
                                    akey.startswith("val")
                                    for akey in trainres["best_step"].keys()
                                )
                            )
                            or time.time() - mod_time < TIME_LIMIT_TRAINRES_EMPTY
                        ):
                            continue
            # Spare if the model was just created
            if time.time() - mod_time < TIME_LIMIT_START:
                continue
            # Spare if the train.log file was modified within the time limit
            if os.path.exists(os.path.join(model_dpath, "train.log")):
                trainlog_mod_time = os.stat(
                    os.path.join(model_dpath, "train.log")
                ).st_mtime
                if time.time() - trainlog_mod_time < TIME_LIMIT_TRAINLOG:
                    continue
            # Spare if the trainres.yaml file was modified within the time limit
            if os.path.exists(os.path.join(model_dpath, "trainres.yaml")):
                trainres_mod_time = os.stat(
                    os.path.join(model_dpath, "trainres.yaml")
                ).st_mtime
                if time.time() - trainres_mod_time < TIME_LIMIT_TRAINRES:
                    continue
            # Spare if saved_models is empty but trainres.yaml exists and the model was created within the time limit
            if os.path.exists(os.path.join(model_dpath, "saved_models")):
                if len(os.listdir(os.path.join(model_dpath, "saved_models"))) == 0:
                    if os.path.exists(os.path.join(model_dpath, "trainres.yaml")):
                        if time.time() - mod_time < TIME_LIMIT_SAVED_MODELS:
                            continue

            # Remove the subdirectory if it is empty and saved_models not modified recently
            print(f"rm -r {model_dpath}")
            if not args.dry_run:
                shutil.rmtree(model_dpath)
