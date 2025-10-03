import os
import yaml
from typing import Literal, Optional

MODELS_ROOT_DPATH = os.path.join("..", "..", "models")


def _get_model_type(expname: str) -> Literal["rawnind_dc", "rawnind_denoise"]:
    """Get the model's root directory path for a given experiment."""

    if expname.startswith("DenoiserTraining") or expname.startswith("train_denoise"):
        return "rawnind_denoise"
    elif (
        expname.startswith("DCTraining")
        or expname.startswith("train_dc")
        or "dc" in expname
    ):
        return "rawnind_dc"
    else:
        raise ValueError(
            f"Unable to determine whether experiment is train_dc or train_denoise: {expname}"
        )


def _get_model_load_metric(
    model_dpath: str, model_type: Literal["rawnind_dc", "rawnind_denoise"]
) -> str:
    """Get the model's load metric for a given experiment."""
    with open(os.path.join(model_dpath, "trainres.yaml"), "r") as f:
        trainres = yaml.safe_load(f)
    if model_type == "rawnind_dc":
        for akey in trainres["best_step"]:
            if akey.startswith("val_combined"):
                return akey
    elif model_type == "rawnind_denoise":
        # load args.yaml to determine loss
        with open(os.path.join(model_dpath, "args.yaml"), "r") as f:
            args = yaml.safe_load(f)
        # load trainres.yaml
        for akey in trainres["best_step"]:
            if akey.startswith(f"val_{args['loss']}"):
                return akey
    raise ValueError(f"Unable to determine load metric for {model_dpath}")


def _get_next_expname_iteration_dpath(model_dpath: str) -> str:
    """Get the next model iteration (eg if the model_dpath ends is "wherever/whatever_" then return "wherever/whatever_-1", if it is wherever/whatever_-1 then return wherever/whatever_-2 and so on.)"""
    if model_dpath.endswith("_"):
        new_dpath = model_dpath + "-1"
    elif model_dpath[-2] == "-" or model_dpath[-3] == "-":
        digit = model_dpath.split("-")[-1]
        new_dpath = model_dpath[: -len(digit)] + str(int(digit) + 1)
    else:
        raise ValueError(f"Unable to determine next model iteration for {model_dpath}")
    return new_dpath


def find_latest_model_expname_iteration(
    expname: str,
    model_type: Optional[Literal["rawnind_dc", "rawnind_denoise"]] = None,
    look_no_further: bool = False,
) -> str:
    """Find the best model training iteration for a given experiment."""
    if model_type is None:
        model_type = _get_model_type(expname)
    model_dpath = os.path.join(MODELS_ROOT_DPATH, model_type, expname)
    load_metric = _get_model_load_metric(model_dpath, model_type)
    # load trainres.yaml
    with open(os.path.join(model_dpath, "trainres.yaml"), "r") as f:
        trainres = yaml.safe_load(f)
    try:
        best_iteration = trainres["best_step"][load_metric]
    except KeyError as e:
        raise ValueError(
            f"Unable to find best iteration w/{load_metric} for in {os.path.join(model_dpath, 'trainres.yaml')}"
        ) from e
    model_fpath = os.path.join(model_dpath, "saved_models", f"iter_{best_iteration}.pt")
    if look_no_further:
        return model_fpath

    next_dpath = _get_next_expname_iteration_dpath(model_dpath)
    next_expname = os.path.basename(next_dpath)
    try:
        return find_latest_model_expname_iteration(next_expname, model_type)
    except (ValueError, FileNotFoundError):
        if os.path.exists(model_fpath):
            return expname
        else:
            print(f"Best model iteration does not exist: {model_fpath}")
            raise ValueError(f"Best model iteration does not exist: {model_fpath}")


if __name__ == "__main__":
    for model_type in ("rawnind_dc", "rawnind_denoise"):
        for expname in os.listdir(os.path.join(MODELS_ROOT_DPATH, model_type)):
            if expname[-1] == "_":
                print(
                    f"{expname}:\t {find_latest_model_expname_iteration(expname, model_type)}"
                )
