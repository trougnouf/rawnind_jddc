"""
Prepare extraraw image dataset: gather list of crops with overexposure_lb and rgb_xyz_matrix metadata. Output a yaml list.
"""

import os
import sys
import logging
import argparse
import tqdm

sys.path.append("..")
from rawnind.libs import raw
from common.libs import utilities

from rawnind.libs.rawproc import EXTRARAW_DS_DPATH

NUM_THREADS: int = os.cpu_count() // 4 * 3
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")

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
        "--dataset", type=str, help="Dataset name (default is all)", default=None
    )
    return parser.parse_args()


def find_cached_result(bayer_fpath, cached_results):
    for result in cached_results:
        if result["bayer_fpath"] == bayer_fpath:
            return result


def fetch_crops_list(fpath: str, ds_base_dpath: str) -> list[dict]:
    def get_xy_coordinates(fn: str) -> list[int, int]:
        return [int(c) for c in fn.split(".")[-2].split("_")]

    crops = []
    basename = os.path.basename(fpath)
    prgb_image_set_dpath = os.path.join(ds_base_dpath, "crops", "proc", "lin_rec2020")
    bayer_image_set_dpath = os.path.join(ds_base_dpath, "crops", "src", "Bayer")
    for fn in os.listdir(prgb_image_set_dpath):
        if fn.startswith(basename):
            coordinates = get_xy_coordinates(fn)
            crop = {
                "coordinates": coordinates,
                "gt_linrec2020_fpath": os.path.join(prgb_image_set_dpath, fn),
            }
            crop["gt_bayer_fpath"] = os.path.join(
                bayer_image_set_dpath,
                fn.replace(".exr", ".npy"),
            )
            crops.append(crop)
    return crops


def get_useful_metadata(bayer_fpath: str) -> dict:
    _, metadata = raw.raw_fpath_to_mono_img_and_metadata(bayer_fpath)
    result = {
        "bayer_fpath": bayer_fpath,
        "overexposure_lb": metadata["overexposure_lb"],
        "rgb_xyz_matrix": metadata["rgb_xyz_matrix"].tolist(),
    }
    return result


if __name__ == "__main__":
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.INFO,
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    args = get_args()
    logging.info(f"# python {' '.join(sys.argv)}")
    logging.info(f"# {args=}")
    if args.dataset is None:
        datasets = os.listdir(EXTRARAW_DS_DPATH)
    else:
        datasets = [args.dataset]
    for dataset in datasets:
        extraraw_content_fpath = os.path.join(
            EXTRARAW_DS_DPATH, dataset, "crops_metadata.yaml"
        )
        bayer_ds_dpath = os.path.join(EXTRARAW_DS_DPATH, dataset, "src", "Bayer")
        args_in = []

        if args.overwrite or not os.path.exists(extraraw_content_fpath):
            cached_results = []
        else:
            cached_results = utilities.load_yaml(
                extraraw_content_fpath, error_on_404=True
            )
        files_endpaths: list[str] = os.listdir(bayer_ds_dpath)

        for file_endpath in files_endpaths:
            bayer_fpath = os.path.join(bayer_ds_dpath, file_endpath)
            if find_cached_result(bayer_fpath, cached_results):
                continue
            args_in.append(bayer_fpath)
        # INPUT: gt_file_endpath, f_endpath
        # OUTPUT: gt_file_endpath, f_endpath, best_alignment, mask_fpath, mask_name
        logging.info(f"Getting metadata from {dataset}")
        try:
            results = utilities.mt_runner(
                get_useful_metadata,
                args_in,
                num_threads=args.num_threads,
            )

        except KeyboardInterrupt:
            logging.error("prep_image_dataset.py interrupted. Saving results.")

        results = results + cached_results
        logging.info(f"Fetching crops list for {dataset}")
        for result in tqdm.tqdm(results):  # FIXME
            try:
                result["crops"] = fetch_crops_list(
                    result["bayer_fpath"],
                    os.path.join(EXTRARAW_DS_DPATH, dataset),
                )
            except FileNotFoundError:
                logging.warning(
                    f"Crops not found for {result['bayer_fpath']}. Is {dataset} cropped? (run `python tools/crop_datasets.py --dataset extraraw` if not then run this script again with --overwrite.)"
                )
            linrec2020_fpath = (
                result["bayer_fpath"].replace("src/Bayer", "proc/lin_rec2020") + ".tif"
            )
            if os.path.exists(linrec2020_fpath):
                result["linrec2020_fpath"] = linrec2020_fpath
                # result[
                #     "crops"
                # ] = "MISSING_CROPS_LIST. Run `python tools/crop_datasets.py --dataset extraraw` then `python tools/prep_image_dataset_extraraw.py --overwrite`."
                # continue
        utilities.dict_to_yaml(
            results,
            extraraw_content_fpath,
        )
