"""
Prepare image dataset: generate alignment and loss masks. Output a yaml list of files,alignment,mask_fpath for Bayer->RGB and RGB->RGB

Compute overexposure in Bayer (if available)

Compute alignment and loss in RGB

Problem:
cannot shift 1px in bayer
Solution:
Calculate shift in RGB image; crop a line/column as needed in Bayer->RGB, no worries for RGB->RGB

Loss mask is based on shifted image;
data loader is straightforward with RGB-RGB (pre-shift images, get loss mask)
with Bayer-to-RGB, loss_mask should be adapted ... TODO (by data loader)

metadata needed: f_bayer_fpath, f_linrec2020_fpath, gt_linrec2020_fpath, overexposure_lb, rgb_xyz_matrix
compute shift between every full-size image
compute loss mask between every full-size image
add list of crops (dict of coordinates : path)
"""

import os
import sys
import logging
import argparse

sys.path.append("..")
from rawnind.libs import rawproc
from common.libs import utilities

from rawnind.libs.rawproc import (
    DATASETS_ROOT,
    DS_DN,
    DS_BASE_DPATH,
    BAYER_DS_DPATH,
    LINREC2020_DS_DPATH,
    RAWNIND_CONTENT_FPATH,
    LOSS_THRESHOLD,
)

NUM_THREADS: int = os.cpu_count() // 4 * 3  #
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")
HDR_EXT = "tif"

"""
#align images needs: bayer_gt_fpath, profiledrgb_gt_fpath, profiledrgb_noisy_fpath
align_images needs: image_set, gt_file_endpath, f_endpath
outputs gt_rgb_fpath, f_bayer_fpath, f_rgb_fpath, best_alignment, mask_fpath
"""


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_threads", type=int, help="Number of threads.", default=NUM_THREADS
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--dataset",
        default=DS_DN,
        help="Process external dataset (ext_raw_denoise_train, ext_raw_denoise_test, RawNIND, RawNIND_Bostitch)",
    )
    return parser.parse_args()


def find_cached_result(ds_dpath, image_set, gt_file_endpath, f_endpath, cached_results):
    gt_fpath = os.path.join(ds_dpath, image_set, gt_file_endpath)
    f_fpath = os.path.join(ds_dpath, image_set, f_endpath)
    for result in cached_results:
        if result["gt_fpath"] == gt_fpath and result["f_fpath"] == f_fpath:
            return result


def fetch_crops_list(image_set, gt_fpath, f_fpath, is_bayer, ds_base_dpath):
    def get_coordinates(fn: str) -> list[int, int]:
        return [int(c) for c in fn.split(".")[-2].split("_")]

    crops = []
    gt_basename = os.path.basename(gt_fpath)
    f_basename = os.path.basename(f_fpath)
    prgb_image_set_dpath = os.path.join(
        ds_base_dpath, "crops", "proc", "lin_rec2020", image_set
    )
    if is_bayer:
        bayer_image_set_dpath = os.path.join(
            ds_base_dpath, "crops", "src", "Bayer", image_set
        )
    for f_is_gt in (True, False):
        for fn_f in os.listdir(
            os.path.join(prgb_image_set_dpath, "gt" if f_is_gt else "")
        ):
            if fn_f.startswith(f_basename):
                coordinates = get_coordinates(fn_f)
                for fn_gt in os.listdir(os.path.join(prgb_image_set_dpath, "gt")):
                    if fn_gt.startswith(gt_basename):
                        coordinates_gt = get_coordinates(fn_gt)
                        if coordinates == coordinates_gt:
                            crop = {
                                "coordinates": coordinates,
                                "f_linrec2020_fpath": os.path.join(
                                    prgb_image_set_dpath, "gt" if f_is_gt else "", fn_f
                                ),
                                "gt_linrec2020_fpath": os.path.join(
                                    prgb_image_set_dpath, "gt", fn_gt
                                ),
                            }
                            if is_bayer:
                                crop["f_bayer_fpath"] = os.path.join(
                                    bayer_image_set_dpath,
                                    "gt" if f_is_gt else "",
                                    fn_f.replace("." + HDR_EXT, ".npy"),
                                )
                                crop["gt_bayer_fpath"] = os.path.join(
                                    bayer_image_set_dpath,
                                    "gt",
                                    fn_gt.replace("." + HDR_EXT, ".npy"),
                                )
                                if not os.path.exists(
                                    crop["f_bayer_fpath"]
                                ) or not os.path.exists(crop["gt_bayer_fpath"]):
                                    logging.error(
                                        f"Missing crop: {crop['f_bayer_fpath']} and/or {crop['gt_bayer_fpath']}"
                                    )
                                    breakpoint()
                                assert os.path.exists(crop["f_bayer_fpath"])
                                assert os.path.exists(crop["gt_bayer_fpath"])
                            crops.append(crop)
    return crops


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
    if args.dataset == DS_DN:
        content_fpath = RAWNIND_CONTENT_FPATH
        bayer_ds_dpath = BAYER_DS_DPATH
        linrec_ds_dpath = LINREC2020_DS_DPATH
    else:
        content_fpath = os.path.join(
            DATASETS_ROOT, args.dataset, f"{args.dataset}_masks_and_alignments.yaml"
        )
        bayer_ds_dpath = os.path.join(DATASETS_ROOT, args.dataset, "src", "Bayer")
        linrec_ds_dpath = os.path.join(
            DATASETS_ROOT, args.dataset, "proc", "lin_rec2020"
        )

    args_in = []
    if args.overwrite or not os.path.exists(content_fpath):
        cached_results = []
    else:
        cached_results = utilities.load_yaml(content_fpath, error_on_404=True)
    for ds_dpath in (bayer_ds_dpath, linrec_ds_dpath):
        if not os.path.isdir(ds_dpath):
            continue
        for image_set in os.listdir(ds_dpath):
            if ds_dpath == linrec_ds_dpath and image_set in os.listdir(bayer_ds_dpath):
                continue  # avoid duplicate, use bayer if available
            in_image_set_dpath: str = os.path.join(ds_dpath, image_set)
            gt_files_endpaths: list[str] = [
                os.path.join("gt", fn)
                for fn in os.listdir(os.path.join(in_image_set_dpath, "gt"))
            ]
            noisy_files_endpaths: list[str] = os.listdir(in_image_set_dpath)
            noisy_files_endpaths.remove("gt")

            for gt_file_endpath in gt_files_endpaths:
                if gt_file_endpath.endswith(".xmp") or gt_file_endpath.endswith(
                    "darktable_exported"
                ):
                    continue
                for f_endpath in gt_files_endpaths + noisy_files_endpaths:
                    if f_endpath.endswith(".xmp") or f_endpath.endswith(
                        "darktable_exported"
                    ):
                        continue
                    if find_cached_result(
                        ds_dpath, image_set, gt_file_endpath, f_endpath, cached_results
                    ):
                        continue
                    args_in.append(
                        {
                            "ds_dpath": ds_dpath,
                            "image_set": image_set,
                            "gt_file_endpath": gt_file_endpath,
                            "f_endpath": f_endpath,
                            "masks_dpath": os.path.join(
                                DATASETS_ROOT, args.dataset, f"masks_{LOSS_THRESHOLD}"
                            ),
                        }
                    )
                # INPUT: gt_file_endpath, f_endpath
                # OUTPUT: gt_file_endpath, f_endpath, best_alignment, mask_fpath, mask_name

    try:
        results = utilities.mt_runner(
            rawproc.get_best_alignment_compute_gain_and_make_loss_mask,
            args_in,
            num_threads=args.num_threads,
        )

    except KeyboardInterrupt:
        logging.error(f"prep_image_dataset.py interrupted. Saving results.")

    results = results + cached_results

    for result in results:  # FIXME
        result["crops"] = fetch_crops_list(
            result["image_set"],
            result["gt_fpath"],
            result["f_fpath"],
            result["is_bayer"],
            ds_base_dpath=os.path.join(DATASETS_ROOT, args.dataset),
        )
    utilities.dict_to_yaml(
        results,
        content_fpath,
    )
