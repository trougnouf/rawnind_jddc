"""
Gather images with ISO<=MAX_ISO in orig_dpath.

eg first run:
python tools/gather_raw_gt_images.py --orig_name trougnouf --orig_dpath /orb/Pictures/ITookAPicture

eg update:
python tools/gather_raw_gt_images.py --overwrite --orig_name trougnouf --orig_dpath '/orb/Pictures/ITookAPicture/2022/'
"""

import os
import sys
import argparse
from typing import Union

sys.path.append("..")
from common.libs import libimganalysis
from common.libs import utilities

ORIG_DPATH = {"trougnouf": os.path.join(os.sep, "orb", "Pictures", "ITookAPicture")}
DEST_DPATH = os.path.join("..", "..", "datasets", "extraraw")

MAX_ISO = 100


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig_dpath", help="Input images directory.")
    parser.add_argument("--orig_name", help="Input images dataset name.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file names instead of creating new names.",
    )
    return parser.parse_args()


def check_for_duplicate(orig_fpath: str, dest_fpath: str) -> Union[bool, str]:
    """Check if file exists, returns new file name if the filename is the same but the file is different, False otherwise."""
    if os.path.exists(dest_fpath):
        this_checksum = utilities.checksum(orig_fpath)
        potential_duplicates = []
        while os.path.exists(dest_fpath):
            print(f"# File exists: {orig_fpath=}, {dest_fpath=}")
            potential_duplicates.append(utilities.checksum(dest_fpath))
            if this_checksum in potential_duplicates:
                return False
            else:
                dest_fpath = dest_fpath + "dupath." + dest_fpath.split(".")[-1]
    return dest_fpath


if __name__ == "__main__":
    args = get_args()
    if args.orig_dpath is None:
        orig_datasets = ORIG_DPATH
    else:
        assert args.orig_name is not None
        orig_datasets = {args.orig_name: args.orig_dpath}
    for ds_name, orig_dpath in orig_datasets.items():
        [
            os.makedirs(os.path.join(DEST_DPATH, ds_name, "src", adir), exist_ok=True)
            for adir in ["Bayer", "X-Trans"]
        ]
        for file in utilities.walk(root=orig_dpath):
            orig_fpath = os.path.join(*file)
            if "nind" in orig_fpath.lower():
                continue
            if libimganalysis.is_raw(orig_fpath):
                isoval = libimganalysis.get_iso(orig_fpath)
                if isoval is None:
                    utilities.popup(
                        f"gather_raw_images.py: isoval is None with {orig_fpath}"
                    )
                    continue
                    # print('Enter c to continue.')
                    # breakpoint()
                    # isoval = 9001
                if "lossy" in orig_fpath.lower():
                    continue
                if isoval <= MAX_ISO:
                    dest_dir = (
                        "X-Trans" if orig_fpath.lower().endswith("raf") else "Bayer"
                    )
                    dest_fpath = os.path.join(
                        DEST_DPATH, ds_name, "src", dest_dir, file[-1]
                    )
                    if not args.overwrite:
                        dest_fpath = check_for_duplicate(orig_fpath, dest_fpath)
                        if dest_fpath is False:
                            continue

                    utilities.cp(
                        orig_fpath, dest_fpath, verbose=True, overwrite=args.overwrite
                    )
    print(
        f"gather_raw_gt_images.py: done. Don't forget to run rmlint or similar on {DEST_DPATH} (see README.md for instructions)."
    )
