import os
import shutil
import subprocess
import unittest
import time
import multiprocessing
from typing import Union, Tuple, Optional
from functools import lru_cache

import colour  # colour-science, needed for the PQ OETF(-1) transfer function
import numpy as np
import scipy.ndimage
from scipy.signal import correlate
import torch

def _is_multiprocessing_worker():
    """Check if we're running in a multiprocessing worker process."""
    try:
        return multiprocessing.current_process().name != 'MainProcess'
    except:
        return False

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




import torch

# GPU acceleration setup - defer CUDA initialization to avoid fork poisoning
_device_info = None

def get_device_info():
    """Get device info, called lazily to avoid fork poisoning."""
    global _device_info
    if _device_info is None:
        if torch.cuda.is_available():
            _device_info = ('cuda', torch.cuda.device_count(), torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device_info = ('mps', 1, "Apple Silicon GPU")
        else:
            _device_info = ('cpu', 0, "CPU")
    return _device_info

def get_device_type():
    return get_device_info()[0]

def get_device_count():
    return get_device_info()[1]

def get_device_name():
    return get_device_info()[2]

def is_accelerator_available():
    return get_device_type() != 'cpu'

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
    
    # Handle different image sizes by cropping to common region
    if len(anchor_img.shape) > 2:
        min_h = min(anchor_img.shape[1], target_img.shape[1])
        min_w = min(anchor_img.shape[2], target_img.shape[2])
        
        # Crop both images to same size from center
        anchor_h, anchor_w = anchor_img.shape[1], anchor_img.shape[2]
        target_h, target_w = target_img.shape[1], target_img.shape[2]
        
        anchor_y_start = (anchor_h - min_h) // 2
        anchor_x_start = (anchor_w - min_w) // 2
        target_y_start = (target_h - min_h) // 2
        target_x_start = (target_w - min_w) // 2
        
        anchor_img = anchor_img[:, anchor_y_start:anchor_y_start+min_h, anchor_x_start:anchor_x_start+min_w]
        target_img = target_img[:, target_y_start:target_y_start+min_h, target_x_start:target_x_start+min_w]
    else:
        min_h = min(anchor_img.shape[0], target_img.shape[0])
        min_w = min(anchor_img.shape[1], target_img.shape[1])
        
        anchor_h, anchor_w = anchor_img.shape
        target_h, target_w = target_img.shape
        
        anchor_y_start = (anchor_h - min_h) // 2
        anchor_x_start = (anchor_w - min_w) // 2
        target_y_start = (target_h - min_h) // 2
        target_x_start = (target_w - min_w) // 2
        
        anchor_img = anchor_img[anchor_y_start:anchor_y_start+min_h, anchor_x_start:anchor_x_start+min_w]
        target_img = target_img[target_y_start:target_y_start+min_h, target_x_start:target_x_start+min_w]
    
    # Multi-scale pyramid - start coarse, refine progressively
    scales = [4, 2, 1]
    best_shift = (0, 0)
    best_loss = float('inf')
    
    for i, scale in enumerate(scales):
        if verbose:
            print(f"Hierarchical search at scale 1/{scale}")
            
        if scale > 1:
            # Downsample images
            if len(anchor_img.shape) > 2:
                anchor_small = anchor_img[:, ::scale, ::scale]
                target_small = target_img[:, ::scale, ::scale]
            else:
                anchor_small = anchor_img[::scale, ::scale]
                target_small = target_img[::scale, ::scale]
        else:
            anchor_small = anchor_img
            target_small = target_img
        
        # Search range: start broad, narrow down
        if i == 0:  # First scale - broad search
            search_range = max(8, max_shift_search // scale)
            scaled_shift = (0, 0)  # Start from center
        else:  # Subsequent scales - refine around previous result
            search_range = max(2, 8 // scale)  # Smaller search window
            # Scale up previous result to current resolution
            scaled_shift = (best_shift[0] * scale // scales[i-1], 
                          best_shift[1] * scale // scales[i-1])
        
        # Local search around scaled estimate
        current_best = scaled_shift
        current_best_loss = float('inf')
        
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                # Test shift at current scale
                test_shift_scaled = (scaled_shift[0] + dy, scaled_shift[1] + dx)
                
                # Convert to full-resolution coordinates for clamping
                test_shift_full = (test_shift_scaled[0] * scales[0] // scale,
                                 test_shift_scaled[1] * scales[0] // scale)
                
                # Skip if outside search bounds
                if (abs(test_shift_full[0]) > max_shift_search or 
                    abs(test_shift_full[1]) > max_shift_search):
                    continue
                
                try:
                    shifted_anchor, shifted_target = shift_images(anchor_small, target_small, test_shift_scaled)
                    loss = np_l1(shifted_anchor, shifted_target, avg=True)
                    
                    if loss < current_best_loss:
                        current_best_loss = loss
                        current_best = test_shift_scaled
                        
                        # Reasonable early termination - not too aggressive
                        if loss < 0.01 and scale == 1:  # Only at finest scale
                            if verbose:
                                print(f"Early termination at loss {loss:.6f}")
                            break
                            
                except (ValueError, IndexError):
                    continue
            
            # Break outer loop too if early termination triggered
            if current_best_loss < 0.01 and scale == 1:
                break
        
        # Update best shift and loss
        best_shift = current_best
        best_loss = current_best_loss
        
        if verbose:
            print(f"Scale 1/{scale}: best_shift={best_shift}, loss={best_loss:.6f}")
    
    if return_loss_too:
        return best_shift, float(best_loss)
    
    return best_shift


def find_best_alignment_gpu(
    anchor_img: Union[np.ndarray, torch.Tensor],
    target_img: Union[np.ndarray, torch.Tensor],
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[Tuple[int, int], Tuple[Tuple[int, int], float]]:
    """GPU-accelerated alignment search using PyTorch with memory management."""
    import logging
    
    # Check if accelerator is available
    if not is_accelerator_available():
        device_type = get_device_type()
        logging.info(f"GPU alignment: No accelerator available (device: {device_type}), falling back to FFT")
        if verbose:
            print(f"No accelerator available (device: {device_type}), falling back to FFT search")
        # Convert tensors back to numpy for FFT fallback
        if isinstance(anchor_img, torch.Tensor):
            anchor_img = anchor_img.cpu().numpy()
        if isinstance(target_img, torch.Tensor):
            target_img = target_img.cpu().numpy()
        return find_best_alignment_fft(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    
    # Import GPU scheduler
    try:
        from common.libs.utilities import get_gpu_scheduler
        scheduler = get_gpu_scheduler()
    except ImportError:
        logging.warning("GPU scheduler not available, proceeding without memory management")
        scheduler = None
    
    # Estimate memory requirements
    task_id = f"gpu_align_{os.getpid()}_{id(anchor_img)}"
    required_memory = 0
    
    if scheduler:
        # Estimate memory for tensors and FFT operations
        h, w = anchor_img.shape[:2]
        # 2 input tensors + 2 FFT results + correlation result + intermediate tensors
        # Each tensor: h * w * 4 bytes (float32), FFT roughly doubles memory usage
        required_memory = h * w * 4 * 6  # Conservative estimate
        
        if not scheduler.acquire_memory(task_id, required_memory, timeout=60.0):
            logging.warning(f"GPU memory not available for alignment, falling back to FFT")
            if verbose:
                print("GPU memory not available, falling back to FFT search")
            return find_best_alignment_fft(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    
    device_type = get_device_type()
    device_name = get_device_name()
    logging.info(f"GPU alignment: Using {device_type} device ({device_name})")
    
    try:
        # Get device once
        device = torch.device(get_device_type())
        
        # Convert inputs to numpy for gain matching (required by match_gain function)
        if isinstance(anchor_img, torch.Tensor):
            anchor_np = anchor_img.cpu().numpy()
        else:
            anchor_np = anchor_img
            
        if isinstance(target_img, torch.Tensor):
            target_np = target_img.cpu().numpy()
        else:
            target_np = target_img
            
        # Apply gain matching
        target_np = match_gain(anchor_np, target_np)
        
        # Efficient tensor creation with buffer reuse
        if scheduler:
            # Try to get reusable buffers
            anchor_buffer = scheduler.get_tensor_buffer(anchor_np.shape, torch.float32, device)
            target_buffer = scheduler.get_tensor_buffer(target_np.shape, torch.float32, device)
            
            if anchor_buffer is not None and target_buffer is not None:
                # Reuse buffers - copy data directly
                anchor_tensor = anchor_buffer.copy_(torch.from_numpy(anchor_np.astype(np.float32)))
                target_tensor = target_buffer.copy_(torch.from_numpy(target_np.astype(np.float32)))
            else:
                # Create new tensors
                anchor_tensor = torch.from_numpy(anchor_np.astype(np.float32)).to(device, non_blocking=True)
                target_tensor = torch.from_numpy(target_np.astype(np.float32)).to(device, non_blocking=True)
        else:
            # No scheduler - create tensors directly with non-blocking transfer
            anchor_tensor = torch.from_numpy(anchor_np.astype(np.float32)).to(device, non_blocking=True)
            target_tensor = torch.from_numpy(target_np.astype(np.float32)).to(device, non_blocking=True)
            
        logging.debug(f"GPU alignment: Tensors moved to {device}")
        
        # Handle multi-channel images
        if len(anchor_tensor.shape) > 2:
            anchor_gray = torch.mean(anchor_tensor, dim=0)
            target_gray = torch.mean(target_tensor, dim=0)
        else:
            anchor_gray = anchor_tensor
            target_gray = target_tensor
        
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
        anchor_crop = (anchor_crop - torch.mean(anchor_crop)) / (torch.std(anchor_crop) + 1e-8)
        target_crop = (target_crop - torch.mean(target_crop)) / (torch.std(target_crop) + 1e-8)
        
        # FFT-based cross-correlation
        f_anchor = torch.fft.fft2(anchor_crop)
        f_target = torch.fft.fft2(target_crop)
        correlation = torch.fft.ifft2(f_anchor * torch.conj(f_target))
        correlation = torch.abs(correlation)
        
        # Find peak
        peak_idx = torch.argmax(correlation)
        y_peak, x_peak = torch.unravel_index(peak_idx, correlation.shape)
        
        # Convert to shift coordinates
        shift_y = int(y_peak.item()) - anchor_crop.shape[0] // 2
        shift_x = int(x_peak.item()) - anchor_crop.shape[1] // 2
        
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
        device_type = get_device_type()
        logging.warning(f"GPU alignment failed on {device_type}: {type(e).__name__}: {e}")
        if verbose:
            print(f"GPU alignment failed: {type(e).__name__}: {e}, falling back to FFT search")
        return find_best_alignment_fft(anchor_img, target_img, max_shift_search, return_loss_too, verbose)
    
    finally:
        # Return tensor buffers to pool for reuse
        if scheduler:
            try:
                if 'anchor_tensor' in locals():
                    scheduler.return_tensor_buffer(anchor_tensor)
                if 'target_tensor' in locals():
                    scheduler.return_tensor_buffer(target_tensor)
            except Exception as e:
                logging.debug(f"Failed to return tensor buffers: {e}")
        
        # Release GPU memory from scheduler
        if scheduler and required_memory > 0:
            scheduler.release_memory(task_id)
            
        # Explicit CUDA memory cleanup
        if get_device_type() == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logging.debug("GPU alignment: CUDA memory cleaned up")
            except Exception as cleanup_error:
                logging.debug(f"GPU alignment: CUDA cleanup warning: {cleanup_error}")


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
    # Method selection
    if method == "auto":
        image_size = anchor_img.shape[-1] * anchor_img.shape[-2]
        if is_accelerator_available() and image_size > 512 * 512:
            method = "gpu"
        elif max_shift_search > 32:
            method = "hierarchical"
        elif image_size > 256 * 256:
            method = "fft"
        else:
            method = "hierarchical"
    
    # Start timing after method selection
    start_time = time.time() if verbose else None
    
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
    
    # Suppress verbose output during multiprocessing to keep progress bar clean
    if verbose and start_time and not _is_multiprocessing_worker():
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
    # Only print if verbose is enabled
    verbose = kwargs.get("verbose", False)
    if verbose:
        print(f"get_best_alignment_and_make_loss_mask: {mask_name=}")
    loss_mask = make_overexposure_mask(gt_img, gt_metadata["overexposure_lb"])
    
    # Get alignment method from kwargs (used in both branches)
    alignment_method = kwargs.get("alignment_method", "auto")
    
    # NEW WORKFLOW: Align on RAW first using FFT (avoids wasteful demosaicing)
    # For RAW/Bayer images, use CFA-aware FFT alignment directly on mosaiced data
    if is_bayer:
        raw_gain = float(match_gain(gt_img, f_img, return_val=True))
        rgb_xyz_matrix = gt_metadata["rgb_xyz_matrix"].tolist()
        
        # Align on RAW using CFA-aware FFT (FAST AND ACCURATE!)
        from rawnind.libs.alignment_backends import find_best_alignment_fft_cfa
        best_alignment, best_alignment_loss = find_best_alignment_fft_cfa(
            gt_img, f_img, gt_metadata, 
            method="median", return_loss_too=True, verbose=False
        )
        
        # NOW demosaic only for loss mask computation
        gt_rgb = raw.demosaic(gt_img, gt_metadata)
        f_rgb = raw.demosaic(f_img, f_metadata)
    else:
        # For already-demosaiced images (.exr, .tif), use old method
        gt_rgb = gt_img
        f_rgb = f_img
        rgb_xyz_matrix = None
        raw_gain = None
        
        verbose_alignment = kwargs.get("verbose_alignment", False)
        
        # Disable verbose during multiprocessing to avoid spam
        is_multiprocessing = kwargs.get("num_threads", 1) > 1
        verbose_for_alignment = verbose_alignment and not is_multiprocessing
        
        # For RGB images, use old alignment method (hierarchical/fft on RGB)
        best_alignment, best_alignment_loss = find_best_alignment(
            gt_rgb, f_rgb, return_loss_too=True, method=alignment_method, verbose=verbose_for_alignment
        )
    rgb_gain = float(match_gain(gt_rgb, f_rgb, return_val=True))
    # gt_rgb_mean = gt_rgb.mean()
    # gain = match_gain(gt_rgb, f_rgb, return_val=True)

    if verbose:
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
    if verbose:
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
        "alignment_method": alignment_method,
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


def process_scene_batch_gpu(scene_args_list: list[dict], **batch_kwargs) -> list[dict]:
    """
    GPU-accelerated batch processing for ONE GT scene with multiple noisy images.
    
    This implements Option #8: batch all noisy images for a single GT scene together on GPU.
    Natural batching (avg 8 noisy/GT), avoids multiprocessing+CUDA fork poisoning.
    
    Args:
        scene_args_list: List of args dicts for one GT scene, all with same gt_file_endpath
        batch_kwargs: Additional kwargs (verbose, masks_dpath, etc.)
        
    Returns:
        List of result dicts, one per (GT, noisy) pair
    """
    if not scene_args_list:
        return []
    
    # All pairs should have same GT
    gt_fpath = scene_args_list[0]['gt_file_endpath']
    assert all(args['gt_file_endpath'] == gt_fpath for args in scene_args_list), \
        "process_scene_batch_gpu: All pairs must have same GT image"
    
    verbose = batch_kwargs.get('verbose', False)
    alignment_method = batch_kwargs.get('alignment_method', 'auto')
    
    if verbose:
        print(f"\nProcessing GT scene: {gt_fpath} with {len(scene_args_list)} noisy images")
    
    # Build full paths
    ds_dpath = batch_kwargs.get("ds_dpath", os.path.join(os.path.dirname(__file__), "..", "datasets", "RawNIND"))
    image_set = scene_args_list[0]['image_set']
    gt_full_path = os.path.join(ds_dpath, image_set, gt_fpath)
    
    # Load GT once - handle errors gracefully
    try:
        gt_img, gt_metadata = img_fpath_to_np_mono_flt_and_metadata(gt_full_path)
        is_bayer = not (gt_full_path.endswith(".exr") or gt_full_path.endswith(".tif"))
    except Exception as e:
        if verbose:
            print(f"  Error loading GT {gt_full_path}: {e}")
            print(f"  Falling back to single-pair processing for this scene")
        # Fallback to single-pair processing
        results = []
        for args in scene_args_list:
            try:
                result = get_best_alignment_compute_gain_and_make_loss_mask(
                    **args, **batch_kwargs
                )
                results.append(result)
            except Exception as e2:
                if verbose:
                    print(f"  Error processing pair: {e2}")
                continue
        return results
    
    if not is_bayer:
        # For non-Bayer images, fallback to single-pair processing
        results = []
        for args in scene_args_list:
            result = get_best_alignment_compute_gain_and_make_loss_mask(
                **args, **batch_kwargs
            )
            results.append(result)
        return results
    
    # Load all noisy images for this GT - handle errors gracefully
    target_imgs = []
    target_metadatas = []
    valid_args = []
    for args in scene_args_list:
        try:
            f_full_path = os.path.join(ds_dpath, args['image_set'], args['f_endpath'])
            f_img, f_metadata = img_fpath_to_np_mono_flt_and_metadata(f_full_path)
            target_imgs.append(f_img)
            target_metadatas.append(f_metadata)
            valid_args.append(args)
        except Exception as e:
            if verbose:
                print(f"  Error loading noisy image {args['f_endpath']}: {e}")
            continue
    
    if not valid_args:
        return []
    
    scene_args_list = valid_args  # Update to only process valid pairs
    
    # GPU batch alignment on RAW
    from rawnind.libs.alignment_backends import find_best_alignment_fft_cfa_batch
    alignments_and_losses = find_best_alignment_fft_cfa_batch(
        gt_img, target_imgs, gt_metadata,
        method="median", return_loss_too=True, verbose=verbose, use_gpu=True
    )
    
    # Process each pair individually for demosaicing and loss mask
    results = []
    for i, (args, f_img, f_metadata, (best_alignment, best_alignment_loss)) in enumerate(
        zip(scene_args_list, target_imgs, target_metadatas, alignments_and_losses)
    ):
        # Extract mask name
        mask_name = f"{args['image_set']}_{os.path.basename(args['gt_file_endpath']).split('.')[0]}_{os.path.basename(args['f_endpath']).split('.')[0]}.png"
        
        # Compute gains
        raw_gain = float(match_gain(gt_img, f_img, return_val=True))
        rgb_xyz_matrix = gt_metadata["rgb_xyz_matrix"].tolist()
        
        # Demosaic for loss mask computation
        gt_rgb = raw.demosaic(gt_img, gt_metadata)
        f_rgb = raw.demosaic(f_img, f_metadata)
        rgb_gain = float(match_gain(gt_rgb, f_rgb, return_val=True))
        
        # Align RGB images and create loss mask
        loss_mask = make_overexposure_mask(gt_img, gt_metadata["overexposure_lb"])
        gt_img_aligned, target_img_aligned = shift_images(gt_rgb, f_rgb, best_alignment)
        loss_mask = shift_mask(loss_mask, best_alignment)
        loss_mask = make_loss_mask(gt_img_aligned, target_img_aligned) * loss_mask
        
        # Save mask
        masks_dpath = batch_kwargs.get("masks_dpath", MASKS_DPATH)
        os.makedirs(masks_dpath, exist_ok=True)
        mask_fpath = os.path.join(masks_dpath, mask_name)
        np_imgops.np_to_img(loss_mask, mask_fpath, precision=8)
        
        # Build result dict
        result = {
            "gt_fpath": args['gt_file_endpath'],
            "f_fpath": args['f_endpath'],
            "image_set": args["image_set"],
            "alignment_method": "fft_gpu_batch",
            "best_alignment": list(best_alignment),
            "best_alignment_loss": best_alignment_loss,
            "mask_fpath": mask_fpath,
            "mask_mean": float(loss_mask.mean()),
            "is_bayer": is_bayer,
            "rgb_xyz_matrix": rgb_xyz_matrix,
            "overexposure_lb": gt_metadata["overexposure_lb"],
            "raw_gain": raw_gain,
            "rgb_gain": rgb_gain,
        }
        results.append(result)
        
        if verbose and i % 5 == 0:
            print(f"  Processed {i+1}/{len(scene_args_list)} images for this GT")
    
    return results


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
