import os
import sys
import tqdm
import argparse
import random

sys.path.append("..")
from rawnind.libs import rawproc  # includes DS_BASE_DPATH
from common.libs import pt_helpers
from rawnind.libs import raw
from common.libs import libimganalysis
from common.libs import utilities

BAD_SRC_FILES_DPATH = os.path.join("..", "..", "datasets", "RawNIND", "bad_src_files")


def is_valid_img_mtrunner(fpath):
    return fpath, libimganalysis.is_valid_img(fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num_threads", type=int, default=os.cpu_count() // 4 * 3)
    parser.add_argument("--directory", default=rawproc.DS_BASE_DPATH)
    args = parser.parse_args()
    list_of_files = []
    for fpath in tqdm.tqdm(list(utilities.walk(args.directory))):
        fpath = os.path.join(*fpath)
        if (
            "bad_src_files" in fpath
            or fpath.endswith(".txt")
            or fpath.endswith(".yaml")
        ):
            continue
        list_of_files.append(fpath)
    random.shuffle(list_of_files)
    results = utilities.mt_runner(is_valid_img_mtrunner, list_of_files)
    for result in results:
        fpath, is_valid = result
        if not is_valid:
            print(f"mv {fpath} {BAD_SRC_FILES_DPATH}")
