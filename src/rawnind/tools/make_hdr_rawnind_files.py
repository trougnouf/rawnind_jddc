"""
# This script is only used for manproc images. For training/testing/validation, crops are now made from the raw files directly.

Make OpenEXR ground-truth images.

Only works with Bayer patterns; X-Trans files need to be converted with darktable.
"""

import os
import sys
import logging
import argparse
from typing import Literal, Optional

sys.path.append("..")
from rawnind.libs import raw
from common.libs import utilities

DATA_DPATH = os.path.join("..", "..", "datasets", "RawNIND")
DEBUG = True
NUM_THREADS: int = os.cpu_count() // 4 * 3
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")
PROC_EXTENSION: Literal["exr", "tif"] = "tif"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_threads", type=int, help="Number of threads.", default=NUM_THREADS
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--single_set",
        help="Process this single image set rathen than the whole dataset.",
    )
    parser.add_argument(
        "--gt_only", help="Only process ground-truth images.", action="store_true"
    )
    parser.add_argument(
        "--test_images_only",
        help="Only process test images (as defined in config/test_reserve.yaml).",
        action="store_true",
    )
    parser.add_argument("--data_dpath", help="Path to the dataset.", default=DATA_DPATH)
    return parser.parse_args()


def proc_dataset(
    test_images_only: bool = False,
    gt_only: bool = False,
    single_set: Optional[str] = None,
    overwrite: bool = False,
    num_threads: int = NUM_THREADS,
    data_dpath: str = DATA_DPATH,
    test_reserve_fpath=os.path.join("config", "test_reserve.yaml"),
):
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.INFO,
        filemode="w",
    )
    unreadable_files = []
    bad_exposure_files = []
    # list_of_src_dest: list[tuple[str, str]] = []  # bw compat, 2022-11-09
    list_of_src_dest: list = []
    src_dpath = os.path.join(data_dpath, "src", "Bayer")
    dest_dpath = os.path.join(src_dpath, "..", "..", "proc", raw.OUTPUT_COLOR_PROFILE)
    image_sets = (single_set,) if single_set else os.listdir(src_dpath)
    if test_images_only:
        test_reserve = utilities.load_yaml(test_reserve_fpath, error_on_404=True)[
            "test_reserve"
        ]
        # check if any test set is missing
        missing_test_sets = set(test_reserve) - set(image_sets)
        if missing_test_sets:
            # logging.error(
            #     f"Missing test sets: {missing_test_sets}. "
            #     f"Please add them to {test_reserve_fpath}."
            # )
            print(
                f"Missing test sets: {missing_test_sets}. "
                f"Please add them to {test_reserve_fpath}."
            )
            sys.exit(1)
        image_sets = [set_name for set_name in image_sets if set_name in test_reserve]
    for set_name in image_sets:
        os.makedirs(os.path.join(dest_dpath, set_name, "gt"), exist_ok=True)
        src_set_dpath = os.path.join(src_dpath, set_name)
        if not gt_only:
            for fn in os.listdir(src_set_dpath):
                if fn == "gt":
                    continue
                src_fpath = os.path.join(src_set_dpath, fn)
                dest_fpath = os.path.join(
                    dest_dpath, set_name, f"{fn}.{PROC_EXTENSION}"
                )
                if not overwrite and (
                    os.path.isfile(dest_fpath)
                    and utilities.filesize(dest_fpath) > 10000000
                ):  # assuming >= 1 MB is valid
                    continue
                list_of_src_dest.append((src_fpath, dest_fpath, "lin_rec2020", 16))
        for fn in os.listdir(os.path.join(src_set_dpath, "gt")):
            src_fpath = os.path.join(src_set_dpath, "gt", fn)
            dest_fpath = os.path.join(
                dest_dpath, set_name, "gt", f"{fn}.{PROC_EXTENSION}"
            )
            if not overwrite and (
                os.path.isfile(dest_fpath) and utilities.filesize(dest_fpath) > 10000000
            ):  # assuming >= 1 MB is valid
                continue
            list_of_src_dest.append((src_fpath, dest_fpath, "lin_rec2020", 16))
    results = utilities.mt_runner(
        fun=raw.raw_fpath_to_hdr_img_file_mtrunner,
        argslist=list_of_src_dest,
        num_threads=num_threads,
        starmap=False,
        ordered=False,
    )
    logging.info("# You should clean-up bad source files by running the following:")
    os.makedirs("bad_src_files", exist_ok=True)
    for result in results:
        if result[0] is raw.ConversionOutcome.OK:
            continue
        else:
            logging.warning(f'mv "{result[1]}" bad_src_files/ # {result[0]}')

    logging.warning(f"# run bash {LOG_FPATH} to clean-up bad files.")


if __name__ == "__main__":
    args = get_args()
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"# python {' '.join(sys.argv)}")
    logging.info(f"# {args=}")
    proc_dataset(**vars(args))
