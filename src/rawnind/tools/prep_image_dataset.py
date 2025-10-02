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
import multiprocessing

# Configure logging to show GPU diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set multiprocessing start method to spawn to avoid CUDA fork poisoning
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
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
    return os.path.exists(filepath)


def estimate_gpu_memory_for_alignment(kwargs: dict) -> Tuple[str, int]:
    """Estimate GPU memory usage for image alignment operations.
    
    Args:
        kwargs: Dictionary containing image_set, gt_file_endpath, f_endpath
        
    Returns:
        Tuple of (task_id, estimated_memory_bytes)
    """
    # Create unique task ID from the image paths
    task_id = f"{kwargs['image_set']}-{kwargs['gt_file_endpath']}-{kwargs['f_endpath']}"
    
    # Try to get actual image dimensions if possible
    try:
        gt_fpath = os.path.join(kwargs["ds_dpath"], kwargs["image_set"], kwargs["gt_file_endpath"])
        
        # Estimate image dimensions based on common sizes
        # For RAW images, typical sizes are 4K, 6K, 8K
        if "4k" in gt_fpath.lower() or "4096" in gt_fpath.lower():
            height, width = 4096, 4096
        elif "6k" in gt_fpath.lower() or "6144" in gt_fpath.lower():
            height, width = 6144, 6144
        elif "8k" in gt_fpath.lower() or "8192" in gt_fpath.lower():
            height, width = 8192, 8192
        else:
            # Default assumption for high-res images
            height, width = 4096, 4096
            
        # Use the GPU scheduler's memory estimation
        scheduler = utilities.get_gpu_scheduler()
        estimated_memory = scheduler.estimate_memory_usage(height, width, channels=3)
        
        return task_id, estimated_memory
        
    except Exception as e:
        logging.debug(f"Could not estimate memory for {task_id}: {e}")
        # Fallback: assume 4K image needs ~1.2GB
        return task_id, int(1.2e9)

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
    
    # Use first few samples for benchmarking (skip expensive validation)
    if len(args_in) == 0:
        logging.warning("No samples available for benchmarking")
        return
        
    num_samples = min(num_samples, len(args_in))
    sample_args = random.sample(args_in, num_samples)
    methods = ["fft"]
    
    if rawproc.is_accelerator_available():
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
            # Run the alignment with GPU memory management
            method_results = utilities.mt_runner(
                rawproc.get_best_alignment_compute_gain_and_make_loss_mask,
                test_args,
                num_threads=min(os.cpu_count(), len(test_args)),  # Use all available cores for benchmarking
                gpu_memory_estimator=estimate_gpu_memory_for_alignment,
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
                            "num_threads": args.num_threads,
                        }
                    )
            
            if total_gt_files > 0:
                logging.info(f"Image set '{image_set}': {total_gt_files} GT files, {matched_pairs} valid pairs")
                
                # Log detailed pairing information for debugging
                if matched_pairs > 0:
                    logging.debug(f"Detailed pairs for '{image_set}':")
                    pair_count = 0
                    for arg in args_in[-matched_pairs:]:  # Get the pairs we just added
                        pair_count += 1
                        gt_name = os.path.basename(arg['gt_file_endpath'])
                        f_name = os.path.basename(arg['f_endpath'])
                        logging.debug(f"  Pair {pair_count}: GT={gt_name} <-> Noisy={f_name}")
                        
                # INPUT: gt_file_endpath, f_endpath
                # OUTPUT: gt_file_endpath, f_endpath, best_alignment, mask_fpath, mask_name

    # Run benchmark if requested
    if args.benchmark and len(args_in) > 0:
        run_alignment_benchmark(args_in, num_samples=min(5, len(args_in)))
        logging.info("Benchmark completed. Proceeding with normal processing...")
    
    logging.info(f"Processing {len(args_in)} image pairs...")
    logging.info(f"Using alignment method: {args.alignment_method}")
    processing_start = time.time()
    
    results = []
    try:
        # GPU Hybrid Batching (Option #8): Group by GT scene for batch processing
        # This is the optimal parallelism boundary for this dataset!
        # Uses GPU-accelerated FFT from alignment_backends.py
        use_gpu_batching = rawproc.is_accelerator_available()
        
        if use_gpu_batching:
            logging.info("Using GPU hybrid batching (Option #8): grouping by GT scene")
            
            # Group args by GT file
            from collections import defaultdict
            scene_groups = defaultdict(list)
            for arg in args_in:
                gt_key = arg['gt_file_endpath']
                scene_groups[gt_key].append(arg)
            
            logging.info(f"Grouped {len(args_in)} pairs into {len(scene_groups)} GT scenes")
            logging.info(f"Scene sizes: min={min(len(v) for v in scene_groups.values())}, "
                        f"max={max(len(v) for v in scene_groups.values())}, "
                        f"avg={len(args_in)/len(scene_groups):.1f}")
            
            # Process each scene with GPU batching (scenes processed sequentially)
            # Within each scene, all noisy images are batched on GPU
            method_name = "GPU_BATCH"
            from tqdm import tqdm
            
            for gt_fpath, scene_args in tqdm(scene_groups.items(), desc=f"Method: {method_name}"):
                try:
                    # Extract common parameters from first arg
                    first_arg = scene_args[0]
                    scene_results = rawproc.process_scene_batch_gpu(
                        scene_args,
                        alignment_method=args.alignment_method,
                        verbose=args.verbose_alignment,
                        ds_dpath=first_arg['ds_dpath'],
                        masks_dpath=first_arg['masks_dpath'],
                    )
                    results.extend(scene_results)
                except Exception as e:
                    logging.error(f"Error processing scene {gt_fpath}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Fallback: traditional per-pair multiprocessing
            logging.info("Using traditional per-pair processing")
            method_name = args.alignment_method.upper() if hasattr(args, 'alignment_method') else "PROCESSING"
            
            results = utilities.mt_runner(
                rawproc.get_best_alignment_compute_gain_and_make_loss_mask,
                args_in,
                num_threads=args.num_threads,
                progress_desc=f"Method: {method_name}",
                gpu_memory_estimator=estimate_gpu_memory_for_alignment,
            )

    except KeyboardInterrupt:
        logging.error(f"prep_image_dataset.py interrupted. Saving results.")

    processing_time = time.time() - processing_start
    logging.info(f"Alignment and mask generation completed in {processing_time:.2f} seconds")
    
    if cached_results:
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
        yaml.dump(results, f, allow_unicode=True)

    total_time = time.time() - start_time
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    if len(args_in) > 0:
        logging.info(f"Average time per image pair: {total_time/len(args_in):.2f} seconds")
    else:
        logging.info("No image pairs processed")
    logging.info(f"Cache hit statistics - listdir cache: {cached_listdir.cache_info()}")
    logging.info(f"Cache hit statistics - exists cache: {cached_exists.cache_info()}")
