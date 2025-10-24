"""Convert x-trans raw files in the dataset to OpenEXR files using darktable-cli.

TODO skip existing files"""

import sys
import os
from typing import Literal
import tqdm

sys.path.append("..")
from rawnind.libs import raw
from rawnind.libs.rawproc import DS_BASE_DPATH

XTRANS_EXT: Literal["raf"] = "raf"
OPENEXR_EXT: Literal["exr"] = "exr"
XTRANS_DPATH: str = os.path.join(DS_BASE_DPATH, "DocScan", "x-trans")
OPENEXR_DPATH: str = os.path.join(DS_BASE_DPATH, "proc", "lin_rec2020")
if __name__ == "__main__":
    for set_name in tqdm.tqdm(os.listdir(XTRANS_DPATH)):
        os.makedirs(os.path.join(OPENEXR_DPATH, set_name), exist_ok=True)
        for fn in os.listdir(os.path.join(XTRANS_DPATH, set_name)):
            if os.path.isdir(os.path.join(XTRANS_DPATH, set_name, fn)):
                os.makedirs(os.path.join(OPENEXR_DPATH, set_name, fn), exist_ok=True)
                for fn2 in os.listdir(os.path.join(XTRANS_DPATH, set_name, fn)):
                    dest_fpath = (
                        os.path.join(OPENEXR_DPATH, set_name, fn, fn2)
                        + f".{OPENEXR_EXT}"
                    )
                    if raw.is_xtrans(
                        os.path.join(XTRANS_DPATH, set_name, fn, fn2)
                    ) and not os.path.isfile(dest_fpath):
                        raw.xtrans_fpath_to_OpenEXR(
                            os.path.join(XTRANS_DPATH, set_name, fn, fn2), dest_fpath
                        )
            else:
                dest_fpath = (
                    os.path.join(OPENEXR_DPATH, set_name, fn) + f".{OPENEXR_EXT}"
                )
                if raw.is_xtrans(
                    os.path.join(XTRANS_DPATH, set_name, fn)
                ) and not os.path.isfile(dest_fpath):
                    raw.xtrans_fpath_to_OpenEXR(
                        os.path.join(XTRANS_DPATH, set_name, fn), dest_fpath
                    )
