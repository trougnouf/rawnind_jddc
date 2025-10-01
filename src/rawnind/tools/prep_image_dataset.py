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
import time
import yaml
from functools import lru_cache
import re
from typing import Dict, List, Tuple, Optional

sys.path.append("..")
from rawnind.libs import rawproc
from common.libs import utilities
from pathlib import Path
from rawnind.libs.rawproc import (
    DATASETS_ROOT,
    DS_DN,
    DS_BASE_DPATH,
    BAYER_DS_DPATH,
    LINREC2020_DS_DPATH,
    RAWNIND_CONTENT_FPATH,
    LOSS_THRESHOLD,
)
RAWNIND_CONTENT_FPATH = Path(RAWNIND_CONTENT_FPATH).resolve()

NUM_THREADS: int = os.cpu_count() // 4 * 3  #
LOG_FPATH = Path(os.path.join("logs", os.path.basename(__file__) + ".log"))
HDR_EXT = "tif"

# Performance optimization caches
@lru_cache(maxsize=256)
def cached_listdir(directory: str) -> List[str]:
    """Cached directory listing to avoid repeated filesystem calls."""
    return os.listdir(directory) if os.path.exists(directory) else []

@lru_cache(maxsize=64)
def cached_exists(filepath: str) -> bool:
    """Cached file existence check."""
    return os.path.exists(filepath)

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
    parser.add_argument(
        "--alignment_method",
        choices=["auto", "gpu", "hierarchical", "fft", "original"],
        default="auto",
        help="Alignment method to use (auto=automatically select best method)",
    )
    parser.add_argument(
        "--verbose_alignment",
        action="store_true",
        help="Enable verbose output for alignment operations",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks comparing different alignment methods",
    )
    return parser.parse_args()


def find_cached_result(ds_dpath, image_set, gt_file_endpath, f_endpath, cached_results):
    if cached_results is None:
        return None
    gt_fpath = os.path.join(ds_dpath, image_set, gt_file_endpath)
    f_fpath = os.path.join(ds_dpath, image_set, f_endpath)
    for result in cached_results:
        if result["gt_fpath"] == gt_fpath and result["f_fpath"] == f_fpath:
            return result


def extract_scene_identifier(filename: str) -> str:
    """
    Extract scene identifier from filename to match GT with corresponding noisy images.
    
    RawNIND filename patterns:
    - "Bayer_2pilesofplates_GT_ISO100_sha1=854554a34b339413462eb1538d4cb0fa95d468b5.arw" -> "2pilesofplates"
    - "Bayer_2pilesofplates_ISO1250_sha1=9121efbd50e2f2392665cb17b435b2526df8e9ae.arw" -> "2pilesofplates"
    - "Bayer_7D-1_GT_ISO100_sha1=22aef4a5b4038e241082741117827f364ce6a5ac.cr2" -> "7D-1"
    """
    # Remove directory path and extension
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # RawNIND pattern: Bayer_<SCENE>_[GT_]ISO<NUMBER>_sha1=<HASH>
    # First try with GT pattern
    rawnind_match = re.match(r'^Bayer_([^_]+(?:_[^_]+)*)_GT_ISO\d+_sha1=', basename)
    if rawnind_match:
        return rawnind_match.group(1).lower()
    
    # Then try without GT pattern
    rawnind_match = re.match(r'^Bayer_([^_]+(?:_[^_]+)*)_ISO\d+_sha1=', basename)
    if rawnind_match:
        return rawnind_match.group(1).lower()
    
    # No match found
    return ""


def files_match_same_scene(gt_file: str, noisy_file: str) -> bool:
    """
    Check if GT file and noisy file are from the same scene.
    
    Args:
        gt_file: Path to GT file (e.g., "gt/scene001_gt.dng")
        noisy_file: Path to noisy file (e.g., "iso3200/scene001_iso3200.dng")
    
    Returns:
        True if files are from the same scene, False otherwise
    """
    gt_scene = extract_scene_identifier(gt_file)
    noisy_scene = extract_scene_identifier(noisy_file)
    
    return gt_scene == noisy_scene and gt_scene != ""



def fetch_crops_list(image_set, gt_fpath, f_fpath, is_bayer, ds_base_dpath):
    """Optimized crops list fetching with caching and vectorized operations."""
    def get_coordinates(fn: str) -> Tuple[int, int]:
        return tuple(int(c) for c in fn.split(".")[-2].split("_"))

    crops = []
    gt_basename = os.path.basename(gt_fpath)
    f_basename = os.path.basename(f_fpath)
    
    prgb_image_set_dpath = os.path.join(
        ds_base_dpath, "crops", "proc", "lin_rec2020", image_set
    )
    
    # Use cached directory listings
    prgb_gt_dir = os.path.join(prgb_image_set_dpath, "gt")
    prgb_noisy_dir = prgb_image_set_dpath
    
    gt_files = cached_listdir(prgb_gt_dir)
    noisy_files = cached_listdir(prgb_noisy_dir)
    
    # Pre-filter and extract coordinates for GT files
    gt_file_coords = {}
    for fn_gt in gt_files:
        if fn_gt.startswith(gt_basename):
            coords = get_coordinates(fn_gt)
            gt_file_coords[coords] = fn_gt
    
    if is_bayer:
        bayer_image_set_dpath = os.path.join(
            ds_base_dpath, "crops", "src", "Bayer", image_set
        )
    
    # Process both GT and noisy files
    for f_is_gt in (True, False):
        file_list = gt_files if f_is_gt else noisy_files
        search_dir = prgb_gt_dir if f_is_gt else prgb_noisy_dir
        
        for fn_f in file_list:
            if fn_f.startswith(f_basename):
                coordinates = get_coordinates(fn_f)
                
                # Check if matching GT coordinates exist
                if coordinates in gt_file_coords:
                    fn_gt = gt_file_coords[coordinates]
                    
                    crop = {
                        "coordinates": list(coordinates),
                        "f_linrec2020_fpath": os.path.join(search_dir, fn_f),
                        "gt_linrec2020_fpath": os.path.join(prgb_gt_dir, fn_gt),
                    }
                    
                    if is_bayer:
                        f_bayer_path = os.path.join(
                            bayer_image_set_dpath,
                            "gt" if f_is_gt else "",
                            fn_f.replace("." + HDR_EXT, ".npy"),
                        )
                        gt_bayer_path = os.path.join(
                            bayer_image_set_dpath,
                            "gt",
                            fn_gt.replace("." + HDR_EXT, ".npy"),
                        )
                        
                        crop["f_bayer_fpath"] = f_bayer_path
                        crop["gt_bayer_fpath"] = gt_bayer_path
                        
                        # Use cached existence checks
                        if not cached_exists(f_bayer_path) or not cached_exists(gt_bayer_path):
                            logging.error(
                                f"Missing crop: {f_bayer_path} and/or {gt_bayer_path}"
                            )
                            continue  # Skip instead of breaking
                    
                    crops.append(crop)
    
    return crops


def run_alignment_benchmark(args_in: List[Dict], num_samples: int = 5) -> None:
    """Run performance benchmarks comparing different alignment methods."""
    import random
    
    # Filter out potentially problematic samples first
    valid_samples = []
    for arg in args_in:
        try:
            # Quick compatibility check
            anchor_img = rawproc.load_image(arg["gt_fpath"])
            target_img = rawproc.load_image(arg["f_fpath"])
            
            # Check if shapes are at least somewhat compatible
            if (anchor_img is not None and target_img is not None and 
                len(anchor_img.shape) == len(target_img.shape) and
                min(anchor_img.shape[-2:]) > 100 and min(target_img.shape[-2:]) > 100):
                valid_samples.append(arg)
                
            if len(valid_samples) >= num_samples * 2:  # Get enough candidates
                break
                
        except Exception:
            continue
    
    if len(valid_samples) < num_samples:
        logging.warning(f"Only found {len(valid_samples)} valid samples for benchmarking")
        num_samples = len(valid_samples)
    
    # Select a subset of valid samples for benchmarking
    sample_args = random.sample(valid_samples, min(num_samples, len(valid_samples)))
    methods = ["original", "hierarchical", "fft"]
    
    if rawproc.CUPY_IMPORTABLE:
        methods.append("gpu")
    
    logging.info(f"Running alignment benchmarks on {len(sample_args)} samples...")
    logging.info(f"Methods to test: {methods}")
    
    results = {}
    
    for method in methods:
        logging.info(f"Testing method: {method}")
        start_time = time.time()
        
        # Update method for all samples
        test_args = []
        for arg in sample_args:
            test_arg = arg.copy()
            test_arg["alignment_method"] = method
            test_arg["verbose_alignment"] = False
            test_args.append(test_arg)
        
        try:
            # Run the alignment
            method_results = utilities.mt_runner(
                rawproc.get_best_alignment_compute_gain_and_make_loss_mask,
                test_args,
                num_threads=min(os.cpu_count(), len(test_args)),  # Use all available cores for benchmarking
            )
            
            elapsed = time.time() - start_time
            results[method] = {
                "time": elapsed,
                "avg_time_per_sample": elapsed / len(sample_args),
                "success": True,
                "results": method_results
            }
            
            logging.info(f"Method '{method}': {elapsed:.2f}s total, {elapsed/len(sample_args):.2f}s per sample")
            
        except Exception as e:
            logging.error(f"Method '{method}' failed: {e}")
            results[method] = {"success": False, "error": str(e)}
    
    # Print comparison
    logging.info("\n=== BENCHMARK RESULTS ===")
    baseline_time = results.get("original", {}).get("time", 1.0)
    
    for method, result in results.items():
        if result.get("success"):
            speedup = baseline_time / result["time"] if result["time"] > 0 else float('inf')
            logging.info(f"{method:>12}: {result['time']:6.2f}s ({speedup:5.1f}x speedup)")
        else:
            logging.info(f"{method:>12}: FAILED - {result.get('error', 'Unknown error')}")


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
    
    # Performance monitoring
    start_time = time.time()
    logging.info(f"Starting dataset preparation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Using {args.num_threads} threads for processing")
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

            # Statistics for logging
            total_gt_files = 0
            matched_pairs = 0
            
            for gt_file_endpath in gt_files_endpaths:
                if gt_file_endpath.endswith(".xmp") or gt_file_endpath.endswith(
                    "darktable_exported"
                ):
                    continue
                    
                total_gt_files += 1
                
                # FIXED: Pair GT files with noisy files from the same scene directory
                # GT files are in gt/ subdirectory, noisy files are directly in scene directory
                for noisy_file in noisy_files_endpaths:
                    # Skip if it's a directory (like 'gt') or unwanted files
                    noisy_file_path = os.path.join(in_image_set_dpath, noisy_file)
                    if os.path.isdir(noisy_file_path) or noisy_file.endswith(".xmp") or noisy_file.endswith("darktable_exported"):
                        continue
                        
                    f_endpath = noisy_file  # noisy files are directly in scene directory
                    
                    # Check if this GT and noisy file are from the same scene
                    if not files_match_same_scene(gt_file_endpath, f_endpath):
                        continue
                        
                    # Check if result is already cached
                    if find_cached_result(
                        ds_dpath, image_set, gt_file_endpath, f_endpath, cached_results
                    ):
                        continue
                        
                    matched_pairs += 1
                    args_in.append(
                        {
                            "ds_dpath": ds_dpath,
                            "image_set": image_set,
                            "gt_file_endpath": gt_file_endpath,
                            "f_endpath": f_endpath,
                            "masks_dpath": os.path.join(
                                DATASETS_ROOT, args.dataset, f"masks_{LOSS_THRESHOLD}"
                            ),
                            "alignment_method": args.alignment_method,
                            "verbose_alignment": args.verbose_alignment,
                        }
                    )
            
            if total_gt_files > 0:
                logging.info(f"Image set '{image_set}': {total_gt_files} GT files, {matched_pairs} valid pairs")
                # INPUT: gt_file_endpath, f_endpath
                # OUTPUT: gt_file_endpath, f_endpath, best_alignment, mask_fpath, mask_name

    # Run benchmark if requested
    if args.benchmark and len(args_in) > 0:
        run_alignment_benchmark(args_in, num_samples=min(5, len(args_in)))
        logging.info("Benchmark completed. Proceeding with normal processing...")
    
    logging.info(f"Processing {len(args_in)} image pairs...")
    logging.info(f"Using alignment method: {args.alignment_method}")
    processing_start = time.time()
    
    try:
        results = utilities.mt_runner(
            rawproc.get_best_alignment_compute_gain_and_make_loss_mask,
            args_in,
            num_threads=args.num_threads,
        )

    except KeyboardInterrupt:
        logging.error(f"prep_image_dataset.py interrupted. Saving results.")

    processing_time = time.time() - processing_start
    logging.info(f"Alignment and mask generation completed in {processing_time:.2f} seconds")
    
    results = results + cached_results

    # Process crops with timing
    crops_start = time.time()
    logging.info(f"Processing crops for {len(results)} results...")
    
    for result in results:  # FIXME
        result["crops"] = fetch_crops_list(
            result["image_set"],
            result["gt_fpath"],
            result["f_fpath"],
            result["is_bayer"],
            ds_base_dpath=os.path.join(DATASETS_ROOT, args.dataset),
        )
    
    crops_time = time.time() - crops_start
    logging.info(f"Crops processing completed in {crops_time:.2f} seconds")
    print("trying to write to ",content_fpath.resolve())
    with content_fpath.open("w", encoding='utf-8') as f:
        yaml.dump(result, f, allow_unicode=True)

    total_time = time.time() - start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Average time per image pair: {total_time/len(args_in):.2f} seconds")
    logging.info(f"Cache hit statistics - listdir cache: {cached_listdir.cache_info()}")
    logging.info(f"Cache hit statistics - exists cache: {cached_exists.cache_info()}")
