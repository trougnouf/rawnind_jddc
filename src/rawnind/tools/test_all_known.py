import os
import sys
import subprocess
import tqdm
import argparse
import datetime
import yaml
import sys

sys.path.append("..")
from common.libs import utilities

TESTS: list[str] = [
    "test_manproc",
    "test_manproc_hq",
    "test_manproc_q99",
    "test_manproc_q995",
    "test_manproc_gt",
    "test_manproc_bostitch",
    "test_ext_raw_denoise",
    "test_playraw",
    "test_manproc_playraw",
    "validate_and_test",
    "test_progressive_rawnind",
    "test_progressive_manproc",
    "test_progressive_manproc_bostitch",
]
MODEL_TYPES: list[str] = ["denoise", "dc"]
MODEL_INPUTS: list[str] = ["bayer", "prgb", "proc"]

MODELS_ROOT_DIR = "/orb/benoit_phd/models/"
FAILED_TESTS_LOG = "logs/failed_tests.log"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tests", nargs="+", default=TESTS, choices=TESTS
    )  # ,  "validate"])
    parser.add_argument(
        "--model_types", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES
    )
    parser.add_argument(
        "--model_input", nargs="+", default=MODEL_INPUTS, choices=MODEL_INPUTS
    )
    parser.add_argument("--banned_models", nargs="+", default=[])
    parser.add_argument("--allowed_models", nargs="+", default=[])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    failed_tests: list[str] = []
    for model_type in tqdm.tqdm(args.model_types):
        models_root_dpath = os.path.join(MODELS_ROOT_DIR, f"rawnind_{model_type}")
        models_yaml_fpath = os.path.join("config", f"trained_{model_type}_models.yaml")
        print(f"Testing models in {models_yaml_fpath}")
        trained_models = yaml.load(open(models_yaml_fpath), Loader=yaml.FullLoader)
        try:
            trained_models = utilities.shuffle_dictionary(trained_models)
        except AttributeError as e:
            print(f"Failed to shuffle {trained_models=} ({e})")
        for model_name, model_attrs in tqdm.tqdm(trained_models.items()):
            # check if any of the banned models are contained in the model_name
            if any(banned_model in model_name for banned_model in args.banned_models):
                print(f"Skipping banned model: {model_name}")
                continue
            if args.allowed_models:
                # check if any of the allowed models are contained in the model_name
                if not any(
                    allowed_model in model_name for allowed_model in args.allowed_models
                ):
                    print(f"Skipping model not in allowed models: {model_name}")
                    continue
            model_dpath = os.path.join(models_root_dpath, model_name)

            for testname in args.tests:
                if (
                    "progressive_manproc" in testname in testname
                ) and model_type == "dc":
                    # raise NotImplementedError(
                    #     f"Cannot test progressive_manproc with dc models ({model_name=})"
                    # )
                    print(
                        f"Cannot test progressive_manproc with dc models ({model_name=})"
                    )
                    continue
                if "bm3d" in model_name and testname not in (
                    "test_manproc",
                    "test_progressive_manproc",
                ):
                    print(f"Skipping BM3D test {testname=}")
                    continue
                if (
                    "manproc_playraw" in testname
                    or "manproc_hq" in testname
                    or "manproc_q99" in testname
                    or "manproc_q995" in testname
                    or "manproc_gt" in testname
                ) and model_type == "denoise":
                    print(
                        f"Skipping denoise manproc_playraw test {testname=} (not implemented)"
                    )
                    continue
                if model_attrs["in_channels"] == 3:
                    if (
                        model_attrs.get("processed_input", False)
                        and "proc" in args.model_input
                    ):
                        model_input = model_output = "proc"
                    elif "prgb" in args.model_input and not model_attrs.get(
                        "processed_input", False
                    ):
                        model_input = model_output = "prgb"
                    else:
                        print(
                            f"Skipping model with unknown or unwanted # input channels / input type: {model_name} ({model_attrs=})"
                        )
                        continue
                elif model_attrs["in_channels"] == 4 and "bayer" in args.model_input:
                    model_input = "bayer"
                    model_output = "prgb"
                else:
                    print(
                        f"Skipping model with unknown or unwanted # input channels / input type: {model_name} ({model_attrs=})"
                    )
                    continue
                # skip non-manproc tests for proc2proc models
                if "manproc" not in testname and model_attrs.get(
                    "processed_input", False
                ):
                    continue
                # print date and time

                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                print(f"Testing model: {model_name} ({testname})")
                cmd = [
                    "python",
                    f"tools/{testname}_{model_type}_{model_input}2{model_output}.py",
                    "--config",
                    os.path.join(model_dpath, "args.yaml"),
                ]
                if args.cpu or (
                    testname == "test_manproc"
                    and (model_input != "bayer" or "preup" in model_name)
                ):
                    cmd += ["--device", "-1"]
                print(" ".join(cmd))
                res = subprocess.run(cmd, timeout=60 * 60 * 12)
                if res.returncode != 0:
                    print(f"Failed to test model: {model_name} ({res=})")
                    failed_tests += cmd
                print("Done testing model: {}".format(model_name))
    # output list of failed tests to logs/failed_tests.log
    with open(FAILED_TESTS_LOG, "w") as f:
        f.write("\n".join(failed_tests))

# python tools/test_all_known.py --cpu --tests test_manproc test_ext_raw_denoise test_playraw validate_and_test; python tools/test_all_known.py --cpu
