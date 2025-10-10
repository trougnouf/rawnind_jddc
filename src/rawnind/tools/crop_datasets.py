"""
Make 1024x1024 raw / 2048x2048 prgb crops from raw images for faster training.
"""

import os
import logging
import sys
import argparse
from typing import Literal, Optional
import numpy as np

sys.path.append("..")
from rawnind.libs import raw
from rawnind.libs.rawproc import (
    DS_BASE_DPATH,  # os.path.join("..", "..", "datasets", "RawNIND")
    # BAYER_DS_DPATH,  # os.path.join(DS_BASE_DPATH, "src", "bayer")
    # LINREC2020_DS_DPATH,  # os.path.join(DS_BASE_DPATH, "proc", "lin_rec2020")
    MAX_SHIFT_SEARCH,
    EXTRARAW_DS_DPATH,
    DATASETS_ROOT,
)
from common.libs import utilities
from common.libs import np_imgops

LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")

# OUT_RAW_DPATH: str = os.path.join(DS_BASE_DPATH, "crops", "src", "bayer")
# OUT_PRGB_DPATH: str = os.path.join(DS_BASE_DPATH, "crops", "proc", "lin_rec2020")
# OUT_METADATA_DPATH: str = os.path.join(DS_BASE_DPATH, "metadata")
PREPROCESS_RAW_SIZE: int = 1024
TRAIN_SIZE: int = 256
OVERWRITE: bool = False
DEBUGGING = True
EXT_RAW_DENOISE_TEST_DS_DPATH = os.path.join(
    "..", "..", "datasets", "ext_raw_denoise_test"
)
EXT_RAW_DENOISE_TRAIN_DS_DPATH = os.path.join(
    "..", "..", "datasets", "ext_raw_denoise_train"
)


def create_raw_img_crops(
    in_fpath: str,
    out_raw_dpath: Optional[str],
    out_prgb_dpath: str,
    out_metadata_dpath: str,
    preprocess_raw_size: int,
    train_size: int,
    overwrite: bool = False,
) -> None:
    """This function creates many crops from a raw image.
    Overlap based on usage (train_size) s.t. every combination can be obtained

    TODO: handle x-trans
    TODO: consider how data will be loaded (is yaml list needed?)

    overlap:
    full: 0 : preprocess_size, preprocess_size-train_size+16 : preprocess_size-train_size+16+preprocess_size, ...
    half: 0 : preprocess_size, preprocess_size-train_size//2 : preprocess_size-train_size//2+preprocess_size, ...
    """
    logging.debug(
        f"start processing {in_fpath=}, {out_raw_dpath=}, {out_prgb_dpath=}, {out_metadata_dpath=}"
    )
    RAW_EXT: str = "npy"
    HDR_EXT: Literal["tif", "exr"] = "tif"
    assert preprocess_raw_size % 16 == 0 and train_size % 16 == 0
    basename: str = os.path.basename(in_fpath)
    # save_args = {}
    if in_fpath.lower().endswith(".exr") or in_fpath.lower().endswith(".tif"):
        prgb_img: np.ndarray = np_imgops.img_fpath_to_np_flt(in_fpath)
        _, h_raw, w_raw = prgb_img.shape
        h_raw = h_raw // 2
        w_raw = w_raw // 2
        pass
    elif out_raw_dpath:
        mono_image, metadata = raw.raw_fpath_to_mono_img_and_metadata(in_fpath)
        rggb_image: np.ndarray = raw.mono_to_rggb_img(mono_image, metadata)
        if not raw.is_exposure_ok(
            mono_image, metadata, ue_threshold=0.0001 if "ext_" in in_fpath else 0.001
        ):
            logging.warning(f"# bad exposure for {in_fpath} ({mono_image.mean()=})")
            return False
        prgb_img: np.ndarray = raw.demosaic(mono_image, metadata)
        prgb_img = raw.camRGB_to_profiledRGB_img(
            prgb_img, metadata, raw.OUTPUT_COLOR_PROFILE
        )
        mono_image: np.ndarray = mono_image.astype(np.float16)
        rggb_image = rggb_image.astype(np.float16)
        os.makedirs(out_raw_dpath, exist_ok=True)
        os.makedirs(out_metadata_dpath, exist_ok=True)
        # if (not overwrite) and os.path.isfile(
        #     os.path.join(out_raw_dpath, "%s_0_0.%s" % (basename, EXT))
        # :
        #     return  # return before opening the image if it looks like it's already been processed
        _, h_raw, w_raw = rggb_image.shape
    else:
        raise ValueError(
            f"No raw output path specified and image {in_fpath} is not OpenEXR or .tif"
        )
    curw_raw = curh_raw = 0

    os.makedirs(out_prgb_dpath, exist_ok=True)

    while curw_raw + train_size + MAX_SHIFT_SEARCH < w_raw - w_raw % 16:
        while curh_raw + train_size + MAX_SHIFT_SEARCH < h_raw - h_raw % 16:
            curh2_raw = min(curh_raw + preprocess_raw_size, h_raw - h_raw % 16)
            curw2_raw = min(curw_raw + preprocess_raw_size, w_raw - w_raw % 16)
            if out_raw_dpath:
                out_raw_fpath = os.path.join(
                    out_raw_dpath,
                    "%s.%u_%u.%s" % (basename, curw_raw, curh_raw, RAW_EXT),
                )
                if (
                    overwrite
                    or (not os.path.isfile(out_raw_fpath))
                    or utilities.filesize(out_raw_fpath) < 10000000
                ):
                    raw_crop = rggb_image[
                        :,
                        curh_raw:curh2_raw,
                        curw_raw:curw2_raw,
                    ]
                    np.save(out_raw_fpath, raw_crop)
            out_prgb_fpath = os.path.join(
                out_prgb_dpath, "%s.%u_%u.%s" % (basename, curw_raw, curh_raw, HDR_EXT)
            )
            if overwrite or (
                not os.path.isfile(out_prgb_fpath)
                or utilities.filesize(out_prgb_fpath) < 10000000
            ):
                prgb_crop: np.ndarray = prgb_img[
                    :,
                    curh_raw * 2 : curh2_raw * 2,
                    curw_raw * 2 : curw2_raw * 2,
                ]
                raw.hdr_nparray_to_file(
                    prgb_crop,
                    out_prgb_fpath,
                    color_profile=raw.OUTPUT_COLOR_PROFILE,
                    bit_depth=16,
                )

            curh_raw += preprocess_raw_size - train_size
        curh_raw = 0
        curw_raw += preprocess_raw_size - train_size
    if out_raw_dpath:
        utilities.dict_to_yaml(
            metadata, os.path.join(out_metadata_dpath, "%s.metadata.yaml" % basename)
        )
    logging.debug(
        f"finished processing {in_fpath=}, {out_raw_dpath=}, {out_prgb_dpath=}, {out_metadata_dpath=}"
    )


def create_raw_img_crops_mtrunner(args: list):
    """Wrapper for multiprocessing"""
    return create_raw_img_crops(*args)


def crop_paired_dataset(ds_base_dpath: str):
    # BAYER_DS_DPATH,  # os.path.join(DS_BASE_DPATH, "src", "bayer")
    # LINREC2020_DS_DPATH,  # os.path.join(DS_BASE_DPATH, "proc", "lin_rec2020")
    # OUT_RAW_DPATH_root: str = os.path.join(DS_BASE_DPATH, "crops", "src", "bayer")
    # OUT_PRGB_DPATH_root: str = os.path.join(DS_BASE_DPATH, "crops", "proc", "lin_rec2020")
    # OUT_METADATA_DPATH_root: str = os.path.join(DS_BASE_DPATH, "metadata")
    bayer_ds_dpath = os.path.join(ds_base_dpath, "src", "bayer")
    linrec2020_ds_dpath = os.path.join(ds_base_dpath, "proc", "lin_rec2020")
    out_raw_dpath_root = os.path.join(ds_base_dpath, "crops", "src", "bayer")
    out_prgb_dpath_root = os.path.join(ds_base_dpath, "crops", "proc", "lin_rec2020")
    out_metadata_dpath_root = os.path.join(ds_base_dpath, "metadata")
    in_dpaths = [bayer_ds_dpath]
    print(bayer_ds_dpath)
    if os.path.isdir(linrec2020_ds_dpath):
        in_dpaths.append(linrec2020_ds_dpath)
    for in_dpath in in_dpaths:
        if in_dpath == "darktable_exported":
            continue
        for dn in os.listdir(in_dpath):
            if not os.path.isdir(os.path.join(in_dpath, dn)):
                continue
            for fn in os.listdir(os.path.join(in_dpath, dn)):
                if fn.endswith(".xmp") or fn == "darktable_exported":
                    continue
                if os.path.isdir(os.path.join(in_dpath, dn, fn)):
                    for fn2 in os.listdir(os.path.join(in_dpath, dn, fn)):
                        if fn2.endswith(".xmp") or fn2 == "darktable_exported":
                            continue
                        in_fpath = os.path.join(in_dpath, dn, fn, fn2)
                        out_raw_dpath = (
                            None
                            if in_dpath == linrec2020_ds_dpath
                            else os.path.join(out_raw_dpath_root, dn, fn)
                        )
                        out_prgb_dpath = os.path.join(out_prgb_dpath_root, dn, fn)
                        out_metadata_dpath = os.path.join(
                            out_metadata_dpath_root, dn, fn
                        )
                        argslist.append(
                            (
                                in_fpath,
                                out_raw_dpath,
                                out_prgb_dpath,
                                out_metadata_dpath,
                                PREPROCESS_RAW_SIZE,
                                TRAIN_SIZE,
                                args.overwrite,
                            )
                        )
                else:
                    in_fpath = os.path.join(in_dpath, dn, fn)
                    out_raw_dpath = (
                        None
                        if in_dpath == linrec2020_ds_dpath
                        else os.path.join(out_raw_dpath_root, dn)
                    )
                    out_prgb_dpath = os.path.join(out_prgb_dpath_root, dn)
                    out_metadata_dpath = os.path.join(out_metadata_dpath_root, dn)
                    argslist.append(
                        (
                            in_fpath,
                            out_raw_dpath,
                            out_prgb_dpath,
                            out_metadata_dpath,
                            PREPROCESS_RAW_SIZE,
                            TRAIN_SIZE,
                            args.overwrite,
                        )
                    )


if __name__ == "__main__":
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.DEBUG if DEBUGGING else logging.INFO,
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # argument parser
    argslist = []
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="number of threads to use (default: all available)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing files"
    )
    parser.add_argument(
        "--dataset",
        help="Dataset(s) to process (extraraw, rawnind, ext_raw_denoise_test, ext_raw_denoise_train, all. default: all)",
        default="all",
    )
    parser.add_argument(
        "--subdataset",
        help="Subdataset to process (eg: raw_pixls, default: process all)",
        default=None,
    )
    args = parser.parse_args()
    opt_args = {}
    if args.num_threads is not None:
        opt_args["num_threads"] = args.num_threads
    # file getter
    if args.dataset == "all" or args.dataset == "rawnind":
        assert args.subdataset is None
        crop_paired_dataset(ds_base_dpath=DS_BASE_DPATH)
    if args.dataset == "all" or args.dataset.startswith("RawNIND"):
        assert args.subdataset is None
        crop_paired_dataset(ds_base_dpath=os.path.join(DATASETS_ROOT, args.dataset))
    if args.dataset == "all" or args.dataset == "ext_raw_denoise_test":
        assert args.subdataset is None
        if args.dataset == "all" and not os.path.isdir(EXT_RAW_DENOISE_TEST_DS_DPATH):
            print(f"Warning: {EXT_RAW_DENOISE_TEST_DS_DPATH} does not exist. Skipping.")
        else:
            crop_paired_dataset(ds_base_dpath=EXT_RAW_DENOISE_TEST_DS_DPATH)
    if args.dataset == "all" or args.dataset == "ext_raw_denoise_train":
        assert args.subdataset is None
        if args.dataset == "all" and not os.path.isdir(EXT_RAW_DENOISE_TRAIN_DS_DPATH):
            print(
                f"Warning: {EXT_RAW_DENOISE_TRAIN_DS_DPATH} does not exist. Skipping."
            )
        else:
            crop_paired_dataset(ds_base_dpath=EXT_RAW_DENOISE_TRAIN_DS_DPATH)
    if args.dataset == "all" or args.dataset == "extraraw":
        if args.subdataset is None:
            subdatasets = os.listdir(EXTRARAW_DS_DPATH)
        else:
            subdatasets = [args.subdataset]
        for dn in subdatasets:
            if not os.path.isdir(os.path.join(EXTRARAW_DS_DPATH, dn)) or dn == "crops":
                continue
            for fn in os.listdir(os.path.join(EXTRARAW_DS_DPATH, dn, "src", "bayer")):
                in_fpath = os.path.join(EXTRARAW_DS_DPATH, dn, "src", "bayer", fn)
                out_raw_dpath = os.path.join(
                    EXTRARAW_DS_DPATH, dn, "crops", "src", "bayer"
                )
                out_prgb_dpath = os.path.join(
                    EXTRARAW_DS_DPATH, dn, "crops", "proc", "lin_rec2020"
                )
                out_metadata_dpath = os.path.join(EXTRARAW_DS_DPATH, dn, "metadata")
                argslist.append(
                    (
                        in_fpath,
                        out_raw_dpath,
                        out_prgb_dpath,
                        out_metadata_dpath,
                        PREPROCESS_RAW_SIZE,
                        TRAIN_SIZE,
                        args.overwrite,
                    )
                )
    if not (
        args.dataset == "all"
        or args.dataset.lower().startswith("rawnind")
        or args.dataset == "extraraw"
    ):
        print("Invalid dataset argument. Must be one of all, rawnind, extraraw.")

    # run
    utilities.mt_runner(create_raw_img_crops_mtrunner, argslist, **opt_args)
