import os
import shutil
import subprocess
import unittest
import time
from typing import Union, Tuple, Optional
from functools import lru_cache

import colour  # colour-science, needed for the PQ OETF(-1) transfer function
import numpy as np
import scipy.ndimage
from scipy.signal import correlate
import torch

# Optional GPU acceleration - test at runtime, not import time
def setup_cuda_environment():
    """Set up CUDA environment for multiprocessing workers."""
    import os
    import sys
    
    # Try to detect virtual environment CUDA libraries
    venv_path = getattr(sys, 'prefix', None)
    if venv_path:
        cuda_lib_paths = [
            os.path.join(venv_path, 'lib', 'python*', 'site-packages', 'nvidia', '*', 'lib'),
            os.path.join(venv_path, 'lib'),
        ]
        
        # Add to LD_LIBRARY_PATH if not already there
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = []
        
        for path_pattern in cuda_lib_paths:
            import glob
            for path in glob.glob(path_pattern):
                if os.path.isdir(path) and path not in current_ld_path:
                    new_paths.append(path)
        
        if new_paths:
            if current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths) + ':' + current_ld_path
            else:
                os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)


def test_cuda_functionality():
    """Test if CUDA is actually functional, not just importable."""
    try:
        # Set up CUDA environment first (important for multiprocessing workers)
        setup_cuda_environment()
        
        import cupy as cp
        # Test device count
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            return False, None
        
        # Test basic GPU operations
        test_array = cp.array([1, 2, 3])
        result = cp.sum(test_array)
        
        # Test memory allocation
        large_array = cp.zeros((100, 100))
        _ = cp.mean(large_array)
        
        return True, cp
    except Exception as e:
        return False, None

# Don't test CUDA at import time - do it at runtime in multiprocessing workers
try:
    import cupy as cp
    CUPY_IMPORTABLE = True
except ImportError:
    cp = None
    CUPY_IMPORTABLE = False

# sys.path.append("..")
from importlib import resources
from pathlib import Path
from common.libs import np_imgops
from rawnind.libs import raw

# LOSS_THRESHOLD: float = 0.33
LOSS_THRESHOLD: float = 0.4
GT_OVEREXPOSURE_LB: float = 1.0
KEEPERS_QUANTILE: float = 0.9999
MAX_SHIFT_SEARCH: int = 128
GAMMA = 2.2
DS_DN = "RawNIND"
DATASETS_ROOT = resources.files('rawnind').joinpath('datasets')
DS_BASE_DPATH: str = DATASETS_ROOT / DS_DN
BAYER_DS_DPATH: str = DS_BASE_DPATH / "src" / "Bayer"
LINREC2020_DS_DPATH: str = DS_BASE_DPATH / "proc" / "lin_rec2020"
MASKS_DPATH = DS_BASE_DPATH / f"masks_{LOSS_THRESHOLD}"
RAWNIND_CONTENT_FPATH = DS_BASE_DPATH / "RawNIND_masks_and_alignments.yaml" # used by tools/prep_image_dataset.py and libs/rawds.py

NEIGHBORHOOD_SEARCH_WINDOW = 3
EXTRARAW_DS_DPATH = DS_BASE_DPATH / "extraraw"
EXTRARAW_CONTENT_FPATHS = (
    EXTRARAW_DS_DPATH / "trougnouf" / "crops_metadata.yaml",
    EXTRARAW_DS_DPATH / "raw-pixls" / "crops_metadata.yaml",
    # os.path.join(EXTRARAW_DS_DPATH, "SID", "crops_metadata.yaml"), # could be useful for testing
)


def np_l1(img1: np.ndarray, img2: np.ndarray, avg=True) -> Union[float, np.ndarray]:
    if avg:
        return np.abs(img1 - img2).mean()
    return np.abs(img1 - img2)


def gamma(img: np.ndarray, gamma_val: float = GAMMA, in_place=False) -> np.ndarray:
    """Apply gamma on positive values, maintain negative values as-is."""
    res = img if in_place else img.copy()
    res[res > 0] = res[res > 0] ** (1 / gamma_val)
    return res


def gamma_pt(img: torch.Tensor, gamma_val: float = GAMMA, in_place=False) -> np.ndarray:
    """Apply gamma on positive values, maintain negative values as-is."""
    res = img if in_place else img.clone()
    res[res > 0] = res[res > 0] ** (1 / gamma_val)
    return res


def scenelin_to_pq(
    img: Union[np.ndarray, torch.Tensor], compat=True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Scene linear input signal to PQ opto-electronic transfer function (OETF).
    See also:
        https://en.wikipedia.org/wiki/Perceptual_quantizer
        https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-I!!PDF-E.pdf
        https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2124-0-201901-I!!PDF-E.pdf
    https://github.com/colour-science/colour/blob/develop/colour/models/rgb/transfer_functions/itur_bt_2100.py
    : oetf_BT2100_PQ
    """
    if isinstance(img, np.ndarray):
        # in develop branch: oetf_BT2100_PQ
        return colour.models.rgb.transfer_functions.itur_bt_2100.oetf_BT2100_PQ(img)
    elif isinstance(img, torch.Tensor):
        # translation of colour.models.rgb.transfer_functions.itur_bt_2100.oetf_BT2100_PQ
        # into PyTorch
        def spow(a, p):
            a_p = torch.sign(a) * torch.abs(a) ** p
            return a_p.nan_to_num()

        def eotf_inverse_ST2084(C, L_p):
            m_1 = 2610 / 4096 * (1 / 4)
            m_2 = 2523 / 4096 * 128
            c_1 = 3424 / 4096
            c_2 = 2413 / 4096 * 32
            c_3 = 2392 / 4096 * 32
            Y_p = spow(C / L_p, m_1)

            N = spow((c_1 + c_2 * Y_p) / (c_3 * Y_p + 1), m_2)

            return N

        def eotf_BT1886(V, L_B=0, L_W=1):
            # V = to_domain_1(V)

            gamma = 2.40
            gamma_d = 1 / gamma

            n = L_W**gamma_d - L_B**gamma_d
            a = n**gamma
            b = L_B**gamma_d / n
            if compat:
                L = a * (V + b) ** gamma
            else:
                L = a * torch.clamp(V + b, min=0) ** gamma
            return L
            # return as_float(from_range_1(L))

        def oetf_BT709(L):
            E = torch.where(L < 0.018, L * 4.5, 1.099 * spow(L, 0.45) - 0.099)
            # return as_float(from_range_1(E))
            return E

        def ootf_BT2100_PQ(E):
            return 100 * eotf_BT1886(oetf_BT709(59.5208 * E))

        return eotf_inverse_ST2084(ootf_BT2100_PQ(img), 10000)
    else:
        raise NotImplementedError(f"{type(img)=}")


def pq_to_scenelin(
    img: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    PQ non-linear to scene linear signal, inverse opto-electronic transfer function (OETF^-1).
    https://github.com/colour-science/colour/blob/develop/colour/models/rgb/transfer_functions/itur_bt_2100.py
    : oetf_inverse_BT2100_PQ
    """
    return colour.models.rgb.transfer_functions.itur_bt_2100.oetf_inverse_PQ_BT2100(img)


def match_gain(
    anchor_img: Union[np.ndarray, torch.Tensor],
    other_img: Union[np.ndarray, torch.Tensor],
    return_val: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Match gain for a single or batched pair of images; other_img is adapted to anchor_img."""
    if anchor_img.ndim == 4:
        anchor_avg = anchor_img.mean((-1, -2, -3)).view(-1, 1, 1, 1)
        other_avg = other_img.mean((-1, -2, -3)).view(-1, 1, 1, 1)
    elif anchor_img.ndim == 3:  # used to prep dataset w/ RAF (EXR) source
        anchor_avg = anchor_img.mean()
        other_avg = other_img.mean()
    else:
        raise ValueError(f"{anchor_img.ndim=}")
    if return_val:
        return anchor_avg / other_avg
    return other_img * (anchor_avg / other_avg)


def shift_images(
    anchor_img: Union[np.ndarray, torch.Tensor],  # gt
    target_img: Union[np.ndarray, torch.Tensor],  # y
    shift: tuple,  # [int, int],  # python bw compat 2022-11-10
    # crop_to_bayer: bool = True,
    # maintain_shape: bool = False,  # probably not needed w/ crop_to_bayer
) -> Union[tuple, tuple]:
    #  ) -> Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]:  # python bw compat 2022-11-10
    """
    Shift images in y,x directions and crop both accordingly.

    crop_to_bayer: ensure target_img crop is %2:
        remove the first/last v/h line from both anchor and target if necessary
    maintain_shape: pad accordingly

    use-cases:
        shift two RGB images: no worries
        shift Bayer and RGB: Bayer shift is // by two. If shift%2, crop additional column/line.
    """
    anchor_img_out = anchor_img
    target_img_out = target_img
    target_is_bayer = target_img.shape[0] == 4
    if anchor_img.shape[0] == 4:
        raise NotImplementedError("shift_images: Bayer anchor_img is not implemented.")
    target_shift_divisor = target_is_bayer + 1
    if shift[0] > 0:  # y
        anchor_img_out = anchor_img_out[..., shift[0] :, :]
        target_img_out = target_img_out[
            ..., : -(shift[0] // target_shift_divisor) or None, :
        ]
        if shift[0] % 2:
            anchor_img_out = anchor_img_out[..., :-1, :]
            target_img_out = target_img_out[..., :-1, :]

    elif shift[0] < 0:
        anchor_img_out = anchor_img_out[..., : shift[0], :]
        target_img_out = target_img_out[..., -shift[0] // target_shift_divisor :, :]
        if shift[0] % 2:
            anchor_img_out = anchor_img_out[..., 1:, :]
            target_img_out = target_img_out[..., 1:, :]
    if shift[1] > 0:  # x
        anchor_img_out = anchor_img_out[..., shift[1] :]
        target_img_out = target_img_out[
            ..., : -(shift[1] // target_shift_divisor) or None
        ]
        if shift[1] % 2:
            anchor_img_out = anchor_img_out[..., :-1]
            target_img_out = target_img_out[..., :-1]
    elif shift[1] < 0:
        anchor_img_out = anchor_img_out[..., : shift[1]]
        target_img_out = target_img_out[..., -shift[1] // target_shift_divisor :]
        if shift[1] % 2:
            anchor_img_out = anchor_img_out[..., 1:]
            target_img_out = target_img_out[..., 1:]
    # try:
    assert shape_is_compatible(anchor_img_out.shape, target_img_out.shape), (
        f"{anchor_img_out.shape=}, {target_img_out.shape=}"
    )
    # except AssertionError as e:
    #    print(e)
    #    breakpoint()

    # assert (
    #     anchor_img_out.shape[1:]
    #     == np.multiply(target_img_out.shape[1:], target_shift_divisor)
    # ).all(), f"{anchor_img_out.shape=}, {target_img_out.shape=}"
    # if maintain_shape:  # unused -> deprecated
    #     assert isinstance(anchor_img_out, torch.Tensor)
    #     xpad = anchor_img.size(-1) - anchor_img_out.size(-1)
    #     ypad = anchor_img.size(-2) - anchor_img_out.size(-2)
    #     anchor_img_out = torch.nn.functional.pad(anchor_img_out, (xpad, 0, ypad, 0))
    #     target_img_out = torch.nn.functional.pad(target_img_out, (xpad, 0, ypad, 0))
    return anchor_img_out, target_img_out


#  def shape_is_compatible(shape1: tuple[int, int, int], shape2: tuple[int, int, int]):  # python bw compat 2022-11-10
def shape_is_compatible(shape1: tuple, shape2: tuple):
    """Returns True if shape1 == shape2 (after debayering if necessary)."""
    return np.all(
        np.multiply(shape1[-2:], (shape1[-3] == 4) + 1)
        == np.multiply(shape2[-2:], (shape2[-3] == 4) + 1)
    )


def shift_mask(
    mask: Union[np.ndarray, torch.Tensor],
    # shift: tuple[int, int],# python bw compat 2022-11-10
    shift: tuple,
    crop_to_bayer: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Shift single (anchor) image in x/y directions and crop accordingly.

    crop_to_bayer: cf shift_images

    TODO / FIXME: is this necessary? (or is mask already shifted when it's computed/created?)
    """
    mask_out = mask
    if shift[0] > 0:
        mask_out = mask_out[..., shift[0] :, :]
        if crop_to_bayer and shift[0] % 2:
            mask_out = mask_out[..., :-1, :]
    elif shift[0] < 0:
        mask_out = mask_out[..., : shift[0], :]
        if crop_to_bayer and shift[0] % 2:
            mask_out = mask_out[..., 1:, :]
    if shift[1] > 0:
        mask_out = mask_out[..., shift[1] :]
        if crop_to_bayer and shift[1] % 2:
            mask_out = mask_out[..., :-1]
    elif shift[1] < 0:
        mask_out = mask_out[..., : shift[1]]
        if crop_to_bayer and shift[1] % 2:
            mask_out = mask_out[..., 1:]

    return mask_out

    # mask_out = mask
    # if shift[0] > 0:  # y
    #     mask_out = mask_out[..., shift[0] :, :]
    #     if target_is_bayer and shift[0] % 2:
    #         mask_out = mask_out[..., :-1, :]
    # elif shift[0] < 0:
    #     mask_out = mask_out[..., : shift[0], :]
    #     if target_is_bayer and shift[0] % 2:
    #         mask_out = mask_out[..., 1:, :]
    # if shift[1] > 0:  # x
    #     mask_out = mask_out[..., shift[1] :]

    #     if target_is_bayer and shift[1] % 2:
    #         mask_out = mask_out[..., :-1]
    # elif shift[1] < 0:
    #     mask_out = mask_out[..., : shift[1]]
    #     if target_is_bayer and shift[1] % 2:
    #         mask_out = mask_out[..., 1:]

    # assert (
    #     anchor_img_out.shape[1:]
    #     == np.multiply(target_img_out.shape[1:], target_shift_divisor)
    # ).all(), f"{anchor_img_out.shape=}, {target_img_out.shape=}"
    # if maintain_shape:  # unused -> deprecated
    #     assert isinstance(anchor_img_out, torch.Tensor)
    #     xpad = anchor_img.size(-1) - anchor_img_out.size(-1)
    #     ypad = anchor_img.size(-2) - anchor_img_out.size(-2)
    #     anchor_img_out = torch.nn.functional.pad(anchor_img_out, (xpad, 0, ypad, 0))
    #     target_img_out = torch.nn.functional.pad(target_img_out, (xpad, 0, ypad, 0))
    return anchor_img_out, target_img_out


def make_overexposure_mask(
    anchor_img: np.ndarray, gt_overexposure_lb: float = GT_OVEREXPOSURE_LB
):
    return (anchor_img < gt_overexposure_lb).all(axis=0)


# def make_loss_mask(
#     anchor_img: np.ndarray,
#     target_img: np.ndarray,
#     loss_threshold: float = LOSS_THRESHOLD,
#     gt_overexposure_lb: float = GT_OVEREXPOSURE_LB,
#     keepers_quantile: float = KEEPERS_QUANTILE,
# ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
#     """Return a loss mask between the two (aligned) images.
#
#     loss_map is the sum of l1 loss over all 4 channels
#
#     0: ignore: if loss_map >= threshold, or anchor_img >= gt_overexposure_lb
#     1: apply loss
#
#     # TODO different keepers_quantile would make a good illustration that noise is not spatially invariant
#     """
#     loss_map = np_l1(anchor_img, match_gain(anchor_img, target_img), avg=False)
#     loss_map = loss_map.sum(axis=0)
#     loss_mask = np.ones_like(loss_map)
#     loss_mask[(anchor_img >= gt_overexposure_lb).any(axis=0)] = 0.
#     reject_threshold = min(loss_threshold, np.quantile(loss_map, keepers_quantile))
#     if reject_threshold == 0:
#         reject_threshold = 1.
#     print(f'{reject_threshold=}')
#     loss_mask[loss_map >= reject_threshold] = 0.
#     return loss_mask# if not return map else (loss_mask, loss_map)
def make_loss_mask(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    loss_threshold: float = LOSS_THRESHOLD,
    keepers_quantile: float = KEEPERS_QUANTILE,
    verbose: bool = False,
    # ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:  # # python bw compat 2022-11-10
) -> Union[np.ndarray, tuple]:  # # python bw compat 2022-11-10
    """Return a loss mask between the two (aligned) images.

    loss_map is the sum of l1 loss over all 4 channels

    0: ignore: if loss_map >= threshold, or anchor_img >= gt_overexposure_lb
    1: apply loss

    # TODO different keepers_quantile would make a good illustration that noise is not spatially invariant
    # TODO is is worth switching from gamma to scenelin_to_pq here?
    """
    # loss_map = np_l1(
    #     scenelin_to_pq(anchor_img),
    #     scenelin_to_pq(match_gain(anchor_img, target_imf)),
    #     avg=False,
    # )
    loss_map = np_l1(
        gamma(anchor_img), gamma(match_gain(anchor_img, target_img)), avg=False
    )
    loss_map = loss_map.sum(axis=0)
    loss_mask = np.ones_like(loss_map)
    reject_threshold = min(loss_threshold, np.quantile(loss_map, keepers_quantile))
    if reject_threshold == 0:
        reject_threshold = 1.0
    if verbose:
        print(f"{reject_threshold=}")
    loss_mask[loss_map >= reject_threshold] = 0.0
    loss_mask = scipy.ndimage.binary_opening(loss_mask.astype(np.uint8)).astype(
        np.float32
    )
    return loss_mask  # if not return map else (loss_mask, loss_map)


def find_best_alignment_fft(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """Fast alignment using FFT-based cross-correlation."""
    target_img = match_gain(anchor_img, target_img)
    
    # Convert to grayscale for correlation
    if len(anchor_img.shape) > 2:
        anchor_gray = anchor_img.mean(axis=0)
        target_gray = target_img.mean(axis=0)
    else:
        anchor_gray = anchor_img
        target_gray = target_img
    
    # Handle different image sizes by cropping to common region
    min_h = min(anchor_gray.shape[0], target_gray.shape[0])
    min_w = min(anchor_gray.shape[1], target_gray.shape[1])
    
    # Crop both images to same size from center
    anchor_h, anchor_w = anchor_gray.shape
    target_h, target_w = target_gray.shape
    
    anchor_y_start = (anchor_h - min_h) // 2
    anchor_x_start = (anchor_w - min_w) // 2
    target_y_start = (target_h - min_h) // 2
    target_x_start = (target_w - min_w) // 2
    
    anchor_crop = anchor_gray[anchor_y_start:anchor_y_start+min_h, anchor_x_start:anchor_x_start+min_w]
    target_crop = target_gray[target_y_start:target_y_start+min_h, target_x_start:target_x_start+min_w]
    
    # Normalize images
    anchor_crop = (anchor_crop - anchor_crop.mean()) / (anchor_crop.std() + 1e-8)
    target_crop = (target_crop - target_crop.mean()) / (target_crop.std() + 1e-8)
    
    # Cross-correlation using FFT
    correlation = correlate(anchor_crop, target_crop, mode='same')
    
    # Find peak
    y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Convert to shift coordinates
    shift_y = y_peak - anchor_crop.shape[0] // 2
    shift_x = x_peak - anchor_crop.shape[1] // 2
    
    # Clamp to search range
    shift_y = np.clip(shift_y, -max_shift_search, max_shift_search)
    shift_x = np.clip(shift_x, -max_shift_search, max_shift_search)
    
    best_shift = (int(shift_y), int(shift_x))
    
    if return_loss_too:
        # Compute actual L1 loss for the found shift
        try:
            shifted_anchor, shifted_target = shift_images(anchor_img, target_img, best_shift)
            loss = np_l1(shifted_anchor, shifted_target, avg=True)
            return best_shift, float(loss)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute loss for shift {best_shift}: {e}")
            return best_shift, float('inf')
    
    return best_shift


def find_best_alignment_hierarchical(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """Hierarchical coarse-to-fine alignment search."""
    target_img = match_gain(anchor_img, target_img)
    
    # Multi-scale pyramid
    scales = [4, 2, 1]
    best_shift = (0, 0)
    
    for scale in scales:
        if scale > 1:
            # Downsample images
            if len(anchor_img.shape) > 2:
                anchor_small = anchor_img[:, ::scale, ::scale]
                target_small = target_img[:, ::scale, ::scale]
            else:
                anchor_small = anchor_img[::scale, ::scale]
                target_small = target_img[::scale, ::scale]
            
            search_range = max(4, max_shift_search // scale)
            scale_factor = scale
        else:
            anchor_small = anchor_img
            target_small = target_img
            search_range = min(8, max_shift_search)  # Fine refinement
            scale_factor = 1
        
        # Scale previous estimate
        scaled_shift = (best_shift[0] * scale_factor, best_shift[1] * scale_factor)
        
        # Local search around scaled estimate
        best_loss = float('inf')
        current_best = scaled_shift
        
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                test_shift = (scaled_shift[0] + dy, scaled_shift[1] + dx)
                
                # Clamp to valid range
                test_shift = (
                    np.clip(test_shift[0], -max_shift_search, max_shift_search),
                    np.clip(test_shift[1], -max_shift_search, max_shift_search)
                )
                
                try:
                    shifted_anchor, shifted_target = shift_images(anchor_small, target_small, 
                                                                (test_shift[0] // scale_factor, 
                                                                 test_shift[1] // scale_factor))
                    loss = np_l1(shifted_anchor, shifted_target, avg=True)
                    
                    if loss < best_loss:
                        best_loss = loss
                        current_best = test_shift
                        
                        # Early termination for very good alignment
                        if loss < 1e-6:
                            break
                            
                except (ValueError, IndexError):
                    continue
        
        best_shift = (current_best[0] // scale_factor, current_best[1] // scale_factor)
    
    if return_loss_too:
        return best_shift, float(best_loss)
    
    return best_shift


def find_best_alignment_gpu(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """GPU-accelerated alignment search using CuPy."""
    # Test CUDA functionality at runtime (important for multiprocessing workers)
    if not CUPY_IMPORTABLE:
        if verbose:
            print("CuPy not available, falling back to FFT search")
        return find_best_alignment_fft(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    
    cuda_available, cp_runtime = test_cuda_functionality()
    if not cuda_available:
        if verbose:
            print("CUDA functionality test failed, falling back to FFT search")
        return find_best_alignment_fft(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    
    try:
        target_img = match_gain(anchor_img, target_img)
        
        # Transfer to GPU using runtime-tested CuPy
        anchor_gpu = cp_runtime.asarray(anchor_img)
        target_gpu = cp_runtime.asarray(target_img)
        
        # Use FFT-based correlation on GPU
        if len(anchor_img.shape) > 2:
            anchor_gray = cp_runtime.mean(anchor_gpu, axis=0)
            target_gray = cp_runtime.mean(target_gpu, axis=0)
        else:
            anchor_gray = anchor_gpu
            target_gray = target_gpu
        
        # Handle different image sizes by cropping to common region
        min_h = min(anchor_gray.shape[0], target_gray.shape[0])
        min_w = min(anchor_gray.shape[1], target_gray.shape[1])
        
        # Crop both images to same size from center
        anchor_h, anchor_w = anchor_gray.shape
        target_h, target_w = target_gray.shape
        
        anchor_y_start = (anchor_h - min_h) // 2
        anchor_x_start = (anchor_w - min_w) // 2
        target_y_start = (target_h - min_h) // 2
        target_x_start = (target_w - min_w) // 2
        
        anchor_crop = anchor_gray[anchor_y_start:anchor_y_start+min_h, anchor_x_start:anchor_x_start+min_w]
        target_crop = target_gray[target_y_start:target_y_start+min_h, target_x_start:target_x_start+min_w]
        
        # Normalize
        anchor_crop = (anchor_crop - cp_runtime.mean(anchor_crop)) / (cp_runtime.std(anchor_crop) + 1e-8)
        target_crop = (target_crop - cp_runtime.mean(target_crop)) / (cp_runtime.std(target_crop) + 1e-8)
        
        # Cross-correlation using CuPy's FFT
        from cupyx.scipy.signal import correlate as cp_correlate
        correlation = cp_correlate(anchor_crop, target_crop, mode='same')
        
        # Find peak
        peak_idx = cp_runtime.argmax(correlation)
        y_peak, x_peak = cp_runtime.unravel_index(peak_idx, correlation.shape)
        
        # Convert to shift coordinates
        shift_y = int(y_peak) - anchor_crop.shape[0] // 2
        shift_x = int(x_peak) - anchor_crop.shape[1] // 2
        
        # Clamp to search range
        shift_y = np.clip(shift_y, -max_shift_search, max_shift_search)
        shift_x = np.clip(shift_x, -max_shift_search, max_shift_search)
        
        best_shift = (shift_y, shift_x)
        
        if return_loss_too:
            # Compute actual L1 loss
            try:
                shifted_anchor, shifted_target = shift_images(anchor_img, target_img, best_shift)
                loss = np_l1(shifted_anchor, shifted_target, avg=True)
                return best_shift, float(loss)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not compute loss for shift {best_shift}: {e}")
                return best_shift, float('inf')
        
        return best_shift
        
    except Exception as e:
        if verbose:
            print(f"GPU alignment failed: {type(e).__name__}: {e}, falling back to FFT search")
        return find_best_alignment_fft(anchor_img, target_img, max_shift_search, return_loss_too, verbose)


def find_best_alignment(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
    method: str = "auto",
    # ) -> Union[tuple[int, int], tuple[tuple[int, int], float]]: # python bw compat 2022-11-10
) -> Union[tuple, tuple]:  # python bw compat 2022-11-10
    """Find best alignment (minimal loss) between anchor_img and target_img.
    
    Args:
        method: Alignment method to use:
            - "auto": Automatically select best method based on image size and GPU availability
            - "gpu": Use GPU acceleration if available, fallback to hierarchical
            - "hierarchical": Use hierarchical coarse-to-fine search
            - "fft": Use FFT-based cross-correlation
            - "original": Use original brute-force method
    """
    start_time = time.time() if verbose else None
    
    # Method selection
    if method == "auto":
        image_size = anchor_img.shape[-1] * anchor_img.shape[-2]
        if CUPY_IMPORTABLE and image_size > 512 * 512:
            method = "gpu"
        elif max_shift_search > 32:
            method = "hierarchical"
        elif image_size > 256 * 256:
            method = "fft"
        else:
            method = "hierarchical"
    
    # Dispatch to appropriate method
    if method == "gpu":
        result = find_best_alignment_gpu(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    elif method == "hierarchical":
        result = find_best_alignment_hierarchical(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    elif method == "fft":
        result = find_best_alignment_fft(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    elif method == "original":
        result = find_best_alignment_original(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    if verbose and start_time:
        elapsed = time.time() - start_time
        shift = result[0] if return_loss_too else result
        loss = result[1] if return_loss_too else "N/A"
        print(f"Alignment method '{method}' took {elapsed:.3f}s, shift={shift}, loss={loss}")
    
    return result


def find_best_alignment_original(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[tuple, tuple]:
    """Original brute-force alignment search (for comparison/fallback)."""
    target_img = match_gain(anchor_img, target_img)
    assert np.isclose(anchor_img.mean(), target_img.mean(), atol=1e-07), (
        f"{anchor_img.mean()=}, {target_img.mean()=}"
    )
    # current_best_shift: tuple[int, int] = (0, 0)  # python bw compat 2022-11-10
    # shifts_losses: dict[tuple[int, int], float] = {# python bw compat 2022-11-10
    current_best_shift: tuple = (0, 0)  # python bw compat 2022-11-10
    shifts_losses: dict = {  # python bw compat 2022-11-10
        current_best_shift: np_l1(anchor_img, target_img, avg=True)
    }
    if verbose:
        print(f"{shifts_losses=}")

    def explore_neighbors(
        initial_shift: tuple[int, int],
        shifts_losses: dict[tuple[int, int], float] = shifts_losses,
        anchor_img: np.ndarray = anchor_img,
        target_img: np.ndarray = target_img,
        search_window=NEIGHBORHOOD_SEARCH_WINDOW,
    ) -> None:
        """Explore initial_shift's neighbors and update shifts_losses."""
        for yshift in range(-search_window, search_window + 1, 1):
            for xshift in range(-search_window, search_window + 1, 1):
                current_shift = (initial_shift[0] + yshift, initial_shift[1] + xshift)
                if current_shift in shifts_losses:
                    continue
                shifts_losses[current_shift] = np_l1(
                    *shift_images(anchor_img, target_img, current_shift)
                )
                if verbose:
                    print(f"{current_shift=}, {shifts_losses[current_shift]}")

    while (
        min(shifts_losses.values()) > 0
        and abs(current_best_shift[0]) + abs(current_best_shift[1]) < max_shift_search
    ):
        explore_neighbors(current_best_shift)
        new_best_shift = min(shifts_losses, key=shifts_losses.get)
        if new_best_shift == current_best_shift:
            if return_loss_too:
                return new_best_shift, float(min(shifts_losses.values()))
            return new_best_shift
        current_best_shift = new_best_shift
    if return_loss_too:
        return current_best_shift, float(min(shifts_losses.values()))
    return current_best_shift


@lru_cache(maxsize=32)  # Cache recently loaded images
def img_fpath_to_np_mono_flt_and_metadata(fpath: str):
    """Load image with caching to avoid repeated I/O operations."""
    if fpath.endswith(".exr"):
        return np_imgops.img_fpath_to_np_flt(fpath), {"overexposure_lb": 1.0}
    return raw.raw_fpath_to_mono_img_and_metadata(fpath)


def get_best_alignment_compute_gain_and_make_loss_mask(kwargs: dict) -> dict:
    """
    input: image_set, gt_file_endpath, f_endpath
    output: gt_linrec2020_fpath, f_bayer_fpath, f_rgb_fpath, best_alignment, mask_fpath
    multithreading-friendly
    also returns gains
    """

    def make_mask_name(image_set: str, gt_file_endpath: str, f_endpath: str) -> str:
        return f"{kwargs['image_set']}-{kwargs['gt_file_endpath']}-{kwargs['f_endpath']}.png".replace(
            os.sep, "_"
        )

    assert set(("image_set", "gt_file_endpath", "f_endpath")).issubset(kwargs.keys())
    gt_fpath = os.path.join(
        kwargs["ds_dpath"], kwargs["image_set"], kwargs["gt_file_endpath"]
    )
    f_fpath = os.path.join(kwargs["ds_dpath"], kwargs["image_set"], kwargs["f_endpath"])
    is_bayer = not (gt_fpath.endswith(".exr") or gt_fpath.endswith(".tif"))
    gt_img, gt_metadata = img_fpath_to_np_mono_flt_and_metadata(gt_fpath)
    f_img, f_metadata = img_fpath_to_np_mono_flt_and_metadata(f_fpath)
    mask_name = make_mask_name(
        kwargs["image_set"], kwargs["gt_file_endpath"], kwargs["f_endpath"]
    )
    print(f"get_best_alignment_and_make_loss_mask: {mask_name=}")
    loss_mask = make_overexposure_mask(gt_img, gt_metadata["overexposure_lb"])
    # demosaic before finding alignment
    if is_bayer:
        raw_gain = float(match_gain(gt_img, f_img, return_val=True))
        gt_rgb = raw.demosaic(gt_img, gt_metadata)
        f_rgb = raw.demosaic(f_img, f_metadata)
        rgb_xyz_matrix = gt_metadata["rgb_xyz_matrix"].tolist()

    else:
        gt_rgb = gt_img
        f_rgb = f_img
        rgb_xyz_matrix = None
        raw_gain = None
    # Get alignment method and verbose setting from kwargs
    alignment_method = kwargs.get("alignment_method", "auto")
    verbose_alignment = kwargs.get("verbose_alignment", False)
    
    best_alignment, best_alignment_loss = find_best_alignment(
        gt_rgb, f_rgb, return_loss_too=True, method=alignment_method, verbose=verbose_alignment
    )
    rgb_gain = float(match_gain(gt_rgb, f_rgb, return_val=True))
    # gt_rgb_mean = gt_rgb.mean()
    # gain = match_gain(gt_rgb, f_rgb, return_val=True)

    print(f"{kwargs['gt_file_endpath']=}, {kwargs['f_endpath']=}, {best_alignment=}")
    gt_img_aligned, target_img_aligned = shift_images(gt_rgb, f_rgb, best_alignment)
    # align the overexposure mask generated from potentially bayer gt
    loss_mask = shift_mask(loss_mask, best_alignment)
    # add content anomalies between two images to the loss mask
    # try:
    assert gt_img_aligned.shape == target_img_aligned.shape, (
        f"{gt_img_aligned.shape=} is not equal to {target_img_aligned.shape} ({best_alignemnt=}, {loss_mask.shape=}, {kwargs=})"
    )

    loss_mask = make_loss_mask(gt_img_aligned, target_img_aligned) * loss_mask
    # except ValueError as e:
    #     print(f'get_best_alignment_and_make_loss_mask error {e=}, {kwargs=}, {loss_mask.shape=}, {gt_img.shape=}, {target_img.shape=}, {best_alignment=}, {gt_img_aligned.shape=}, {target_img_aligned.shape=}, {loss_mask.shape=}')
    #     breakpoint()
    #     raise ValueError
    print(
        f"{kwargs['image_set']=}: {loss_mask.min()=}, {loss_mask.max()=}, {loss_mask.mean()=}"
    )
    # save the mask
    masks_dpath = kwargs.get("masks_dpath", MASKS_DPATH)
    os.makedirs(masks_dpath, exist_ok=True)
    mask_fpath = os.path.join(masks_dpath, mask_name)
    np_imgops.np_to_img(loss_mask, mask_fpath, precision=8)
    return {
        "gt_fpath": gt_fpath,
        "f_fpath": f_fpath,
        "image_set": kwargs["image_set"],
        "best_alignment": list(best_alignment),
        "best_alignment_loss": best_alignment_loss,
        "mask_fpath": mask_fpath,
        "mask_mean": float(loss_mask.mean()),
        "is_bayer": is_bayer,
        "rgb_xyz_matrix": rgb_xyz_matrix,
        "overexposure_lb": gt_metadata["overexposure_lb"],
        "raw_gain": raw_gain,
        "rgb_gain": rgb_gain,
        # "gt_rgb_mean": gt_rgb_mean,
    }


def camRGB_to_lin_rec2020_images(
    camRGB_images: torch.Tensor, rgb_xyz_matrices: torch.Tensor
) -> np.ndarray:
    """Convert a batch of camRGB debayered images to the lin_rec2020 color profile."""
    # cam_to_xyzd65 = torch.linalg.inv(rgb_xyz_matrices[:, :3, :])
    # bugfix for https://github.com/pytorch/pytorch/issues/86465
    cam_to_xyzd65 = torch.linalg.inv(rgb_xyz_matrices[:, :3, :].cpu()).to(
        camRGB_images.device
    )
    xyz_to_lin_rec2020 = torch.tensor(
        [
            [1.71666343, -0.35567332, -0.25336809],
            [-0.66667384, 1.61645574, 0.0157683],
            [0.01764248, -0.04277698, 0.94224328],
        ],
        device=camRGB_images.device,
    )
    color_matrices = xyz_to_lin_rec2020 @ cam_to_xyzd65

    orig_dims = camRGB_images.shape
    # print(orig_dims)
    lin_rec2020_images = (
        color_matrices @ camRGB_images.reshape(orig_dims[0], 3, -1)
    ).reshape(orig_dims)

    return lin_rec2020_images


def demosaic(rggb_img: torch.Tensor) -> torch.Tensor:
    """
    Transform RGGB (4-ch) image or batch to camRGB colors (3-ch).
    """
    mono_img: np.ndarray = raw.rggb_to_mono_img(rggb_img)
    if len(mono_img.shape) == 3:
        return torch.from_numpy(raw.demosaic(mono_img, {"bayer_pattern": "RGGB"}))
    new_shape: list[int] = list(mono_img.shape)
    new_shape[-3] = 3
    demosaiced_image: np.ndarray = np.empty_like(mono_img, shape=new_shape)
    for i, img in enumerate(mono_img):
        demosaiced_image[i] = raw.demosaic(mono_img[i], {"bayer_pattern": "RGGB"})
    return torch.from_numpy(demosaiced_image).to(rggb_img.device)


def dt_proc_img(src_fpath: str, dest_fpath: str, xmp_fpath: str, compression=True):
    assert shutil.which("darktable-cli")
    assert dest_fpath.endswith(".tif")
    assert not os.path.isfile(dest_fpath), f"{dest_fpath} already exists"
    assert not os.path.isfile(dest_fpath), dest_fpath
    conversion_cmd: tuple = (
        "darktable-cli",
        src_fpath,
        xmp_fpath,
        dest_fpath,
        "--core",
        "--conf",
        "plugins/imageio/format/tiff/bpp=16",
    )
    # print(f"dt_proc_img: {' '.join(conversion_cmd)=}")
    subprocess.call(conversion_cmd, timeout=15 * 60)
    assert os.path.isfile(dest_fpath), f"{dest_fpath} was not written by darktable-cli"


class Test_Rawproc(unittest.TestCase):
    def test_camRGB_to_lin_rec2020_images_mt(self):
        self.longMessage = True
        rgb_xyz_matrices = torch.rand(10, 4, 3)
        images = torch.rand(10, 3, 128, 128)
        batched_conversion = camRGB_to_lin_rec2020_images(images, rgb_xyz_matrices)
        for i in range(images.shape[0]):
            single_conversion = camRGB_to_lin_rec2020_images(
                images[i].unsqueeze(0), rgb_xyz_matrices[i].unsqueeze(0)
            )
            self.assertTrue(
                torch.allclose(
                    single_conversion,
                    batched_conversion[i : i + 1],
                    atol=1e-04,
                    rtol=1e-04,
                )
            )

    def test_match_gains(self):
        self.longMessage = True
        anchor_img = torch.rand(3, 128, 128)
        target_img = torch.rand(3, 128, 128)
        target_img = match_gain(anchor_img, target_img)
        self.assertAlmostEqual(
            anchor_img.mean().item(), target_img.mean().item(), places=5
        )
        anchor_batch = torch.rand(10, 3, 128, 128)
        anchor_batch[1] *= 10
        target_batch = torch.rand(10, 3, 128, 128)
        target_batch[1] /= 10
        target_batch[5] /= 5
        target_batch[7] += 0.5
        target_batch[9] /= 90
        print(f"{anchor_batch.mean()=}, {target_batch.mean()=}")
        target_batch = match_gain(anchor_batch, target_batch)
        print(f"{anchor_batch.mean()=}, {target_batch.mean()=}")
        self.assertGreaterEqual(target_batch[1].mean(), 2.5)
        self.assertGreaterEqual(target_batch[5].mean(), 0.25)
        self.assertGreaterEqual(target_batch[1].mean(), target_batch.mean())
        for i in range(anchor_batch.shape[0]):
            self.assertAlmostEqual(
                anchor_batch[i].mean().item(), target_batch[i].mean().item(), places=5
            )


if __name__ == "__main__":
    unittest.main()
