# -*- coding: utf-8 -*-
"""Helper functions operating on pytorch tensors."""

from typing import Literal, Union
import torchvision
import os
from PIL import Image
import sys
import torch
import numpy as np

# import multiprocessing
# multiprocessing.set_start_method('spawn')
import cv2

# cv2.setNumThreads(0)
import math
import logging
import time

sys.path.append("..")
from common.libs import utilities
from common.libs import pt_losses
from common.libs import np_imgops
from common.libs import pt_ops

TMPDIR = "tmp"
os.makedirs(TMPDIR, exist_ok=True)


def fpath_to_tensor(
    img_fpath,
    device=torch.device("cpu"),
    batch=False,
    incl_metadata=False,
    crop_to_multiple: Union[Literal[False], int] = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
    """Load an image file into a PyTorch tensor with optional metadata.

    This helper wraps robust NumPy loading and converts into a tensor suitable for
    training/evaluation. It supports HDR/RAW-friendly paths via np_imgops and can
    optionally crop the spatial dimensions to a multiple (useful for models with
    downsampling by powers of two).

    Args:
        img_fpath: Path to image file (supports HDR formats handled by np_imgops).
        device: Target device for the returned tensor (cpu or cuda:N).
        batch: If True, add a leading batch dimension of size 1.
        incl_metadata: If True, also return a metadata dict emitted by np_imgops.
        crop_to_multiple: If not False, spatially crop H and W to be a multiple of
            this integer (e.g., 32) using pt_ops.crop_to_multiple.

    Returns:
        - tensor: Float tensor of shape (C, H, W) or (1, C, H, W) if batch=True.
        - (tensor, metadata): If incl_metadata=True, returns a tuple where metadata
          contains per-file information such as color profile, dynamic range, etc.

    Raises:
        ValueError: If image decoding fails; retried with backoff before raising.

    Notes:
        This function retries decoding up to two times with a short delay to be
        resilient to transient I/O or plugin issues. Values are not normalized
        unless np_imgops loader performs scaling. See np_imgops.img_fpath_to_np_flt
        for details about supported formats and scaling.
    """
    # totensor = torchvision.transforms.ToTensor()
    # pilimg = Image.open(imgpath).convert('RGB')
    # return totensor(pilimg)  # replaced w/ opencv to handle >8bits
    try:
        tensor = np_imgops.img_fpath_to_np_flt(img_fpath, incl_metadata=incl_metadata)
    except ValueError as e:
        try:
            logging.error(f"fpath_to_tensor error {e} with {img_fpath=}. Trying again.")
            tensor = np_imgops.img_fpath_to_np_flt(
                img_fpath, incl_metadata=incl_metadata
            )
        except ValueError as e:
            logging.error(
                f"fpath_to_tensor failed again ({e}). Trying one last time after 5 seconds."
            )
            time.sleep(5)
            tensor = np_imgops.img_fpath_to_np_flt(
                img_fpath, incl_metadata=incl_metadata
            )
    if incl_metadata:
        tensor, metadata = tensor
    tensor = torch.tensor(tensor, device=device)
    if crop_to_multiple:
        tensor = pt_ops.crop_to_multiple(tensor, crop_to_multiple)
    if batch:
        tensor = tensor.unsqueeze(0)
    if incl_metadata:
        return tensor, metadata
    else:
        return tensor


def to_smallest_type(tensor, integers=False):
    """Downcast tensor dtype to the smallest representation without information loss.

    Args:
        tensor: PyTorch tensor with integer-like values.
        integers: If True, consider integer dtypes (uint8/short). Floating-point
            downcasting is not supported and will raise NotImplementedError.

    Returns:
        Tensor with the same values but a smaller integer dtype when possible.

    Raises:
        NotImplementedError: If integers=False or when the value range cannot be
            represented by supported integer dtypes.
    """
    minval = tensor.min()
    maxval = tensor.max()
    if integers:
        if maxval <= 255 and minval >= 0:
            tensor = tensor.byte()
        elif maxval <= 32767 and minval >= 0:
            tensor = tensor.short()
        else:
            raise NotImplementedError(
                "to_smallest_type: min={}, max={}".format(minval, maxval)
            )
    else:
        raise NotImplementedError("to_smallest_type with integers=False")
    return tensor


def bits_per_value(tensor):
    """Compute the minimum integer bit-width to represent tensor values.

    This utility estimates the bit depth required to represent non-negative
    integer data exactly, ignoring compression. It is primarily intended for
    diagnostics and I/O budgeting.

    Args:
        tensor: PyTorch tensor (integer or float with non-negative values).

    Returns:
        Integer number of bits per value required to represent the range.

    Raises:
        NotImplementedError: If negative values are present (unsupported).
    """
    minval = tensor.min()
    maxval = tensor.max()
    if minval >= 0 and maxval <= 0:
        return 0
    elif minval >= 0:
        return math.floor(math.log2(maxval) + 1)
    # if minval >= 0 and maxval <= 1:
    #     return 1
    # elif minval >= 0 and maxval <= 3:
    #     return 2
    # elif minval >= 0 and maxval <= 7:
    #     return 3
    # elif minval >= 0 and maxval <= 15:
    #     return 4
    # elif minval >= 0 and maxval <= 31:
    #     return 5
    # elif minval >= 0 and maxval <= 63:
    #     return 6
    # elif minval >= 0 and maxval <= 127:
    #     return 7
    # elif minval >= 0 and maxval <= 255:
    #     return 8
    # elif minval >= 0 and maxval <= 511:
    #     return 9
    else:
        raise NotImplementedError(
            "bits_per_value w/ min={}, max={}".format(minval, maxval)
        )


def get_num_bits(tensor, integers=False, compression="lzma", try_png=True):
    """Estimate storage cost in bits for a tensor with simple compression.

    The tensor is written to disk in raw binary form to measure uncompressed
    size, then compressed using LZMA and optionally PNG (for 2D/3D images)
    to report the lowest observed bit count. If integers=True, a theoretical
    lower bound using bits_per_value is also considered.

    Args:
        tensor: PyTorch tensor (will be moved to CPU/NumPy for I/O).
        integers: If True, enforce integer casting via to_smallest_type before
            measurement and consider bits_per_value lower bound.
        compression: Which compression to apply; currently only 'lzma' is
            supported for tar.xz-like estimation.
        try_png: If True, also attempt PNG compression for image-shaped arrays.

    Returns:
        Integer bit count representing an estimate of minimal storage among the
        measured options for this tensor.
    """
    if tensor.min() == 0 and tensor.max() == 0:
        return 0
    tensor = to_smallest_type(tensor, integers=integers)
    ext = "tar.xz" if compression == "lzma" else "bin"
    tmp_fpath = os.path.join(TMPDIR, str(os.getpid()) + "." + ext)
    tensor = tensor.numpy()
    tensor.tofile(tmp_fpath)
    num_bits = (os.stat(tmp_fpath).st_size) * 8
    utilities.compress_lzma(tmp_fpath, tmp_fpath)
    num_bits = min(num_bits, (os.stat(tmp_fpath).st_size) * 8)
    png_success = try_png and utilities.compress_png(
        tensor, outfpath=tmp_fpath + ".png"
    )
    if png_success:
        num_bits = min(num_bits, (os.stat(tmp_fpath + ".png").st_size) * 8)
    if integers:
        num_bits = min(num_bits, tensor.size * bits_per_value(tensor))
    return num_bits


def get_device(device_n=None):
    """Resolve a torch.device from an index or string.

    Args:
        device_n: None (auto-detect CUDA else CPU), int index (>=0 for CUDA, -1 for CPU),
            torch.device, or string ('cpu', 'cuda', 'cuda:N').

    Returns:
        torch.device pointing to the requested or available accelerator.

    Notes:
        If CUDA is unavailable, any request for a CUDA device falls back to CPU
        with a warning printed to stdout.
    """
    if isinstance(device_n, torch.device):
        return device_n
    elif isinstance(device_n, str):
        if device_n == "cpu":
            return torch.device("cpu")
        device_n = int(device_n)
    if device_n is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("get_device: cuda not available; defaulting to cpu")
            return torch.device("cpu")
    elif torch.cuda.is_available() and device_n >= 0:
        return torch.device("cuda:%i" % device_n)
    elif device_n >= 0:
        print("get_device: cuda not available")
    return torch.device("cpu")


def sdr_pttensor_to_file(tensor: torch.Tensor, fpath: str):
    """Save a tensor as an SDR image to disk.

    Supports 8-bit JPEG and 16-bit PNG/TIFF encodings. Floating point tensors
    are clipped to [0,1] and scaled accordingly. Batches are not supported
    (except batch size 1, which is squeezed).

    Args:
        tensor: Tensor of shape (C, H, W) or (1, C, H, W), dtype float32/float16/uint8.
        fpath: Destination file path; extension determines format.

    Raises:
        AssertionError: If batch dimension > 1 is provided.
        NotImplementedError: For unsupported dtype or file extension.
    """
    if tensor.dim() == 4:
        assert tensor.size(0) == 1, (
            "sdr_pttensor_to_file: batch size > 1 is not supported"
        )
        tensor = tensor.squeeze(0)
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        if fpath[-4:].lower() in [".jpg", "jpeg"]:  # 8-bit
            return torchvision.utils.save_image(tensor.clip(0, 1), fpath)
        elif fpath[-4:].lower() in [".png", ".tif", "tiff"]:  # 16-bit?
            nptensor = (
                (tensor.clip(0, 1) * 65535)
                .round()
                .cpu()
                .numpy()
                .astype(np.uint16)
                .transpose(1, 2, 0)
            )
            # breakpoint()
            nptensor = cv2.cvtColor(nptensor, cv2.COLOR_RGB2BGR)
            outflags = None
            if fpath.endswith("tif") or fpath.endswith("tiff"):
                outflags = (cv2.IMWRITE_TIFF_COMPRESSION, 34925)  # lzma2
            cv2.imwrite(fpath, nptensor, outflags)
        else:
            raise NotImplementedError(f"Extension in {fpath}")
    elif tensor.dtype == torch.uint8:
        tensor = tensor.permute(1, 2, 0).to(torch.uint8).numpy()
        pilimg = Image.fromarray(tensor)
        pilimg.save(fpath)
    else:
        raise NotImplementedError(tensor.dtype)


def get_lossclass(lossname: str):
    """Factory for common loss functions by name.

    Args:
        lossname: One of {'msssim', 'mse'}.

    Returns:
        Instantiated PyTorch loss module.

    Raises:
        NotImplementedError: If the loss name is not recognized.

    References:
        - SSIM/MS-SSIM: Z. Wang et al., "Image Quality Assessment: From Error Visibility to
          Structural Similarity," IEEE TIP, 2004; and "Multiscale Structural Similarity for
          Image Quality Assessment," 2003.
    """
    if lossname == "msssim":
        # return piqa.MS_SSIM()
        return pt_losses.MS_SSIM_loss()
    elif lossname == "mse":
        return torch.nn.MSELoss()
    else:
        raise NotImplementedError("get_lossfun: {}".format(lossname))


def freeze_model(net):
    """Put a model in eval mode and disable gradients for all parameters.

    Args:
        net: torch.nn.Module to freeze.

    Returns:
        The same module, with requires_grad=False for all parameters.
    """
    net = net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def get_losses(img1_fpath, img2_fpath):
    """Compute basic similarity losses/metrics between two image files.

    Loads two images as tensors and returns a dict including MSE and MS-SSIM.

    Args:
        img1_fpath: Path to first image.
        img2_fpath: Path to second image.

    Returns:
        Dict with keys {'mse', 'ssim', 'msssim'} and float values.

    Raises:
        AssertionError: If the two images do not have identical shapes after load.
    """
    img1 = fpath_to_tensor(img1_fpath).unsqueeze(0)
    img2 = fpath_to_tensor(img2_fpath).unsqueeze(0)
    assert img1.shape == img2.shape, f"img1.shape={img1.shape}, img2.shape={img2.shape}"
    res = dict()
    res["mse"] = torch.nn.functional.mse_loss(img1, img2).item()
    res["ssim"] = pt_losses.SSIM_loss()(img1, img2).item()
    res["msssim"] = pt_losses.MS_SSIM_loss()(img1, img2).item()
    return res


torch_cuda_synchronize = (
    torch.cuda.synchronize if torch.cuda.is_available() else utilities.noop
)
