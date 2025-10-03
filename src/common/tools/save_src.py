import os
import shutil
import sys
from typing import Optional

sys.path.append("..")
from common.libs import utilities

SRC_EXTENSIONS: tuple = ("py", "yaml")
SRC_ROOT_DPATH: str = ".."


def save_src(
    dest_root_dpath: str,
    src_root_dpath: str = SRC_ROOT_DPATH,
    src_extensions: tuple = SRC_EXTENSIONS,
    included_dirs=None,
) -> None:
    """
    Save source files matching *.src_extensions from src_root_dpath into dest_root_dpath.

    If included_dirs: Optional[list[str]] is set, then only those directories will be included
    """
    for root, dn, fn in utilities.walk(src_root_dpath):
        if included_dirs is not None and dn not in included_dirs:
            continue
        if not any(fn.endswith("." + ext) for ext in src_extensions):
            continue
        dest_dpath: str = os.path.join(dest_root_dpath, dn)
        os.makedirs(dest_dpath, exist_ok=True)
        shutil.copyfile(os.path.join(root, dn, fn), os.path.join(dest_dpath, fn))


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: python save_src.py [DEST_ROOT_DPATH]"
    save_src(dest_root_dpath=sys.argv[1])
