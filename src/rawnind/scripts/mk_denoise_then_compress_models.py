"""
For every model in config/trained_dc_models.yaml which has attribute "data_pairing: x_x",
create a new model (copy from ../../models/rawnind_dc/<MODELNAME> to ../../models/rawnind_dc/DenoiseThenCompress<MODELNAME>)
Only copy the sub-directory "saved_models", the files "args.yaml" and "trainres.yaml". In trainres.yaml keey the "best_step" key as-is and make an empty dictionary under the key "best_val".
"""

# TODO change save_dpath, expname

import os
import shutil
from pathlib import Path
from typing import List
import yaml
import os
import shutil
from pathlib import Path
import yaml
import time


def copy_model_files(model_name: str, new_model_name) -> None:
    source_dir = f"../../models/rawnind_dc/{model_name}"
    dest_dir = f"../../models/rawnind_dc/{new_model_name}"
    # if os.path.isdir(dest_dir):
    #     return
    print(f"Creating model: {new_model_name}")
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    # Copy saved_models directory
    shutil.copytree(
        f"{source_dir}/saved_models", f"{dest_dir}/saved_models", dirs_exist_ok=True
    )
    # Copy args.yaml file
    shutil.copy2(f"{source_dir}/args.yaml", f"{dest_dir}/args.yaml")
    # Copy trainres.yaml file and modify it (unless it exists)
    if not os.path.exists(f"{dest_dir}/trainres.yaml"):
        shutil.copy2(f"{source_dir}/trainres.yaml", f"{dest_dir}/trainres.yaml")
        with open(f"{dest_dir}/trainres.yaml", "r") as file:
            trainres = yaml.safe_load(file)
        for trainres_key, trainres_val in trainres.items():
            if trainres_key == "best_step":
                continue
            trainres[trainres_key] = {}
        with open(f"{dest_dir}/trainres.yaml", "w") as file:
            yaml.dump(trainres, file)
    # Modify args.yaml file
    with open(f"{dest_dir}/args.yaml", "r") as args_file:
        args = yaml.safe_load(args_file)
    for arg_key, arg_val in args.items():
        if not isinstance(arg_val, str):
            continue
        args[arg_key] = arg_val.replace(model_name, new_model_name)
        args["arch"] = "DenoiseThenCompress"
    with open(f"{dest_dir}/args.yaml", "w") as args_file:
        yaml.dump(args, args_file)
    print(f"Created model: {dest_dir}")


def create_denoise_then_compress_models() -> None:
    with open("config/trained_dc_maninput_models.yaml", "r") as file:
        models = yaml.safe_load(file)
    added_models = {}
    for model_name, model_dict in models.items():
        if "DenoiseThenCompress" in model_name:
            continue
        new_model_name = f"DenoiseThenCompressV2{model_name}"
        # if new_model_name in models:
        #     continue
        if (
            model_dict.get("data_pairing") == "x_x"
            and model_dict.get("in_channels") == 3
        ):
            copy_model_files(model_name, new_model_name)
            new_model_dict = model_dict.copy()
            new_model_dict["denoise_then_compress"] = True
            added_models[new_model_name] = new_model_dict
    if added_models:
        # models.update(added_models)
        # backup the original models file w/ a human-readable timestamp
        # shutil.copy(
        #     "config/trained_dc_models.yaml",
        #     f"config/trained_dc_models.yaml.{time.strftime('%Y-%m-%d_%H-%M-%S')}",
        # )
        with open("config/denoise_then_compress_models.yaml", "w") as file:
            yaml.dump(added_models, file)
        with open("config/trained_dc_models.yaml", "w") as file:
            yaml.dump(models | added_models, file)


if __name__ == "__main__":
    create_denoise_then_compress_models()
