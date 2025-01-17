"""
# This script is deprecated! Crops are now made from the raw files directly.

Make OpenEXR ground-truth images.

Only works with Bayer patterns; X-Trans files need to be converted with darktable.
"""

import os
import sys
import logging
import argparse
from typing import Literal

sys.path.append("..")
from rawnind.libs import raw
from common.libs import utilities

EXTRARAW_DATA_DPATH = os.path.join("..", "..", "datasets", "extraraw")
DEBUG = True
NUM_THREADS = os.cpu_count() // 4 * 3
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")
PROC_EXTENSION: Literal["tif", "exr"] = "tif"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_threads", type=int, help="Number of threads.", default=NUM_THREADS
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset (directory) name. Default: all",
        default=None,
    )
    return parser.parse_args()


def proc_dataset(dataset: str, num_threads: int = NUM_THREADS):
    unreadable_files = []
    bad_exposure_files = []
    # list_of_src_dest: list[tuple[str, str]] = []  # bw compat, 2022-11-09
    list_of_src_dest: list = []
    if dataset:
        ds_names = [dataset]
    else:
        ds_names = sorted(os.listdir(EXTRARAW_DATA_DPATH))
    for ds_name in ds_names:
        src_dpath = os.path.join(EXTRARAW_DATA_DPATH, ds_name, "src", "Bayer")
        dest_dpath = os.path.join(
            src_dpath, "..", "..", "proc", raw.OUTPUT_COLOR_PROFILE
        )
        os.makedirs(dest_dpath, exist_ok=True)

        for fn in os.listdir(src_dpath):
            src_fpath = os.path.join(src_dpath, fn)
            dest_fpath = os.path.join(dest_dpath, fn + "." + PROC_EXTENSION)
            if (
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
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.INFO,
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    args = get_args()
    logging.info(f'# python {" ".join(sys.argv)}')
    logging.info(f"# {args=}")
    proc_dataset(dataset=args.dataset, num_threads=args.num_threads)
