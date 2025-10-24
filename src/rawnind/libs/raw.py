"""Basic raw handling library; convert raw camRGB file to numpy array with LibRaw.

This file also handles HDR image export.
"""

from __future__ import annotations

import argparse
import logging
import operator
import os
import shutil
import subprocess
from enum import Enum
from typing import Literal, NamedTuple, Optional, Union

import Imath  # OpenEXR
import OpenEXR
import OpenImageIO as oiio
import imageio  # todo used in libraw_process sample, should be easy to replace
import numpy as np
import rawpy
import requests
import torch  # for typing only

from common.libs import icc

TIFF_PROVIDER = "OpenImageIO"
OPENEXR_PROVIDER = "OpenEXR"
SAMPLE_RAW_URL = ("https://nc.trougnouf.com/index.php/s/zMA8QfgPoNoByex/download/DSC01568.ARW"  # RGGB
    # "https://nc.trougnouf.com/index.php/s/9eWFyKsitoNGqQS/download/_M7D5407.CR2"  # GBRG
    # "https://nc.trougnouf.com/index.php/s/zMA8QfgPoNoByex/download/DSC01526.ARW"  # RGGB
)
SAMPLE_RAW_FPATH = os.path.join("data", SAMPLE_RAW_URL.rpartition("/")[-1])
OUTPUT_COLOR_PROFILE = "lin_rec2020"
logging.basicConfig(level=logging.INFO)


def raw_fpath_to_mono_img_and_metadata(fpath: str, force_rggb: bool = True, crop_all: bool = True,
        return_float: bool = True, ) -> tuple[np.ndarray, dict]:
    """Load a RAW camera file and extract mosaiced sensor data with metadata.

    This is the entry point for reading proprietary RAW formats (CR2, NEF, ARW, RAF, etc.)
    and converting them to a standardized representation suitable for neural network processing.
    The function handles the complexities of different camera sensors: bayer patterns (2×2 RGGB,
    GBRG, BGGR, GRBG) and x-trans patterns (6×6 non-bayer), border cropping, and conversion to
    a canonical RGGB layout when requested.

    The mosaiced image is returned as a single-channel array where each pixel contains exactly
    one color measurement (red, green, or blue depending on the color filter at that location).
    The metadata dictionary provides everything needed for subsequent processing: white balance
    coefficients, color correction matrices, black/white levels, and the exact pattern layout.

    bayer pattern conversion logic:
    Different cameras use different bayer starting points (RGGB, GBRG, BGGR, GRBG). When
    force_rggb=True, the function crops 1-2 pixels from borders to align all patterns to RGGB,
    simplifying downstream processing at the cost of slight resolution loss. x-trans sensors
    (6×6 pattern) are not converted, as their structure is fundamentally different.

    Border handling:
    RAW files often include inactive pixels at image borders (optical black regions for
    calibration). The function reads border offsets from metadata and removes these regions.
    Additionally, dimensions are cropped to multiples of the pattern size (4 for bayer, 6 for
    x-trans) to ensure clean tiling.

    Black/white point normalization:
    When return_float=True, sensor values are scaled from their native range (typically 0-4095
    for 12-bit or 0-16383 for 14-bit) to [0,1] floating point, accounting for black level
    offset and sensor saturation point. This normalization is crucial for neural network
    training, which expects inputs in a consistent range.

    Args:
        fpath: Path to RAW file (CR2, NEF, ARW, RAF, DNG, etc.)
        force_rggb: Convert all bayer patterns to RGGB by cropping borders (ignored for x-trans)
        crop_all: Crop to the region declared as valid in metadata (removes borders)
        return_float: Scale pixel values to [0,1] using black/white points; if False, returns
            raw integer sensor values

    Returns:
        Tuple of (mono_img, metadata) where:
        - mono_img: np.ndarray of shape (1, H, W), mosaiced sensor data, float32 if return_float
            else uint16
        - metadata: dict containing:
            - 'camera_whitebalance': [R, G, B, G] multipliers from camera
            - 'daylight_whitebalance': [R, G, B, G] for standard illuminant
            - 'rgb_xyz_matrix': Camera RGB to CIE XYZ conversion (4×3 for RGBG sensors)
            - 'black_level_per_channel': Sensor dark current offset per channel
            - 'white_level': Sensor saturation point
            - 'RGBG_pattern': 2×2 (bayer) or 6×6 (x-trans) array of color indices (0=R, 1=G, 2=B, 3=G)
            - 'bayer_pattern': Pattern name string ('RGGB', 'GBRG', etc.)
            - 'sizes': Dict with image dimensions (raw_height, raw_width, top_margin, etc.)

    Raises:
        ValueError: If file cannot be opened, has unsupported CFA pattern, or lacks required metadata

    Example:
        >>> mono, meta = raw_fpath_to_mono_img_and_metadata('IMG_0001.CR2')
        >>> mono.shape
        (1, 3472, 5208)
        >>> meta['bayer_pattern']
        'RGGB'
        >>> meta['camera_whitebalance']
        array([2.3984, 1.0, 1.5234, 1.0])

    Note:
        x-trans files (Fujifilm RAF) are handled but not converted to RGGB. Their 6×6 pattern
        remains intact. For x-trans processing, use the specialized xtrans_to_OpenEXR workflow
        instead of attempting to demosaic directly.
    """

    def mono_any_to_mono_rggb(mono_img: np.ndarray, metadata: dict, whole_image_raw_pattern) -> np.ndarray:
        """
        Converts a monochrome image captured with an arbitrary Bayer pattern into a format compatible with the RGGB
        pattern expected by downstream processing. The function performs the following steps:

        * Detects the original Bayer pattern from the provided ``metadata``.
        * If the pattern is already RGGB, the image is returned unchanged.
        * For non‑RGGB patterns, the image is cropped to remove the necessary border columns/rows, updates the raw
        data pattern, and adjusts the size fields in ``metadata`` so that the remaining data matches the RGGB layout.
        * After processing, the ``metadata`` dictionary is updated with the new ``RGBG_pattern`` and a consistent
        ``bayer_pattern`` name of “RGGB”.
        * The function returns the cropped (or unchanged) monochrome image along with the updated 2×2 pattern that
        represents the channel order.

        The conversion assumes the following relationships between Bayer patterns and the required border removal:

        * **GBRG** – remove one column from each side of the image and the first column of the pattern.
        * **BGGR** – remove one column and one row from each side and the first row and column of the pattern.
        * **GRBG** – remove one row from each side and the first row of the pattern.

        If an unsupported pattern or mismatched borders are encountered, ``NotImplementedError`` is raised.

        Args:
            mono_img: 2‑D NumPy array containing the raw sensor data. metadata: Dictionary containing information
            about the Bayer pattern
                and image dimensions.  Keys include ``bayer_pattern``, ``RGBG_pattern``,
                and ``sizes`` with sub‑keys ``raw_height``, ``raw_width``,
                ``height``, ``width``, ``iheight``, ``iwidth``.
            whole_image_raw_pattern: 2‑D array describing the original 2×2
                Bayer pattern of the whole image.

        Returns:
            Tuple containing:
                - The processed monochrome image as a NumPy array.
                - The updated 2×2 raw pattern array that now corresponds to the
                  RGGB layout.

        """
        if metadata["bayer_pattern"] == "RGGB":
            pass
        else:
            if not (metadata["sizes"]["top_margin"] == metadata["sizes"]["left_margin"] == 0):
                raise NotImplementedError(f"{metadata["sizes"]=}, {metadata["bayer_pattern"]=} with borders")
            if metadata["bayer_pattern"] == "GBRG":
                assert (metadata["RGBG_pattern"] == [[3, 2], [0, 1]]).all(), (f"{metadata['RGBG_pattern']=}")
                mono_img = mono_img[:, 1:-1]
                # metadata["cropped_y"] = True
                whole_image_raw_pattern = whole_image_raw_pattern[1:-1]
                metadata["sizes"]["raw_height"] -= 2
                metadata["sizes"]["height"] -= 2
                metadata["sizes"]["iheight"] -= 2
            elif metadata["bayer_pattern"] == "BGGR":
                assert (metadata["RGBG_pattern"] == [[2, 3], [1, 0]]).all(), (f"{metadata['RGBG_pattern']=}")
                mono_img = mono_img[:, 1:-1, 1:-1]
                # metadata["cropped_x"] = metadata["cropped_y"] = True
                whole_image_raw_pattern = whole_image_raw_pattern[1:-1, 1:-1]
                metadata["sizes"]["raw_height"] -= 2
                metadata["sizes"]["height"] -= 2
                metadata["sizes"]["iheight"] -= 2
                metadata["sizes"]["raw_width"] -= 2
                metadata["sizes"]["width"] -= 2
                metadata["sizes"]["iwidth"] -= 2
            elif metadata["bayer_pattern"] == "GRBG":
                assert (metadata["RGBG_pattern"] == [[1, 0], [2, 3]]).all(), (f"{metadata['RGBG_pattern']=}")
                mono_img = mono_img[:, :, 1:-1]
                # metadata["cropped_x"] = True
                whole_image_raw_pattern = whole_image_raw_pattern[:, 1:-1]
                metadata["sizes"]["raw_width"] -= 2
                metadata["sizes"]["width"] -= 2
                metadata["sizes"]["iwidth"] -= 2
            else:
                raise NotImplementedError(f"{metadata["bayer_pattern"]=}")
        # try:  # DBG
        assert metadata["sizes"]["raw_height"] >= metadata["sizes"]["height"], (f"Wrong height: {metadata['sizes']=}")
        assert metadata["sizes"]["raw_width"] >= metadata["sizes"]["width"], (f"Wrong width: {metadata['sizes']=}")
        # except AssertionError as e:
        #    print(e)
        #    breakpoint()
        metadata["RGBG_pattern"] = whole_image_raw_pattern[:2, :2]
        set_bayer_pattern_name(metadata)
        assert metadata["bayer_pattern"] == "RGGB", f"{metadata["bayer_pattern"]=}"
        return mono_img, whole_image_raw_pattern

    def rm_empty_borders(mono_img: np.ndarray, metadata: dict, whole_image_raw_pattern: np.ndarray, crop_all: bool, ) -> \
    tuple[np.ndarray, np.ndarray]:
        """
        Remove empty borders declared in the metadata.

        args:
            mono_img: 1 x h x w
            metadata: contains top_margin, left_margin
            whole_image_raw_pattern: h x w
        """

        if metadata["sizes"]["top_margin"] or metadata["sizes"]["left_margin"]:
            mono_img = mono_img[:, metadata["sizes"]["top_margin"]:, metadata["sizes"]["left_margin"]:]
            metadata["sizes"]["raw_height"] -= metadata["sizes"]["top_margin"]
            metadata["sizes"]["raw_width"] -= metadata["sizes"]["left_margin"]
            whole_image_raw_pattern = whole_image_raw_pattern[
                metadata["sizes"]["top_margin"]:, metadata["sizes"]["left_margin"]:]
            metadata["sizes"]["top_margin"] = metadata["sizes"]["left_margin"] = 0
        if crop_all:
            _, h, w = mono_img.shape
            min_h = min(h, metadata["sizes"]["height"], metadata["sizes"]["iheight"], metadata["sizes"]["raw_height"], )
            min_w = min(w, metadata["sizes"]["width"], metadata["sizes"]["iwidth"], metadata["sizes"]["raw_width"], )
            h = metadata["sizes"]["height"] = metadata["sizes"]["iheight"] = metadata["sizes"]["raw_height"] = min_h
            w = metadata["sizes"]["width"] = metadata["sizes"]["iwidth"] = metadata["sizes"]["raw_width"] = min_w
            mono_img = mono_img[:, :h, :w]

        assert mono_img.shape[1:] == (metadata["sizes"]["raw_height"],
                                      metadata["sizes"]["raw_width"],), f"{mono_img.shape[1:]=}, {metadata[" sizes"]=}"
        # Keep original pattern size - don't force to 2x2 for X-Trans
        # metadata["RGBG_pattern"] = whole_image_raw_pattern[:2, :2]  # Old bayer-only code
        # set_bayer_pattern_name(metadata)  # Pattern already set earlier
        return mono_img, whole_image_raw_pattern

    def set_bayer_pattern_name(metadata: dict):
        """Set bayer_pattern string from RGBG (raw_pattern) indices."""
        metadata["bayer_pattern"] = "".join(["RGBG"[i] for i in metadata["RGBG_pattern"].flatten()])

    def ensure_correct_shape(mono_img: np.ndarray, metadata: dict, whole_image_raw_pattern: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        """Ensure dimension % pattern_size == 0 (4 for bayer, 6 for x-trans)."""
        _, h, w = mono_img.shape
        assert (metadata["sizes"]["raw_height"], metadata["sizes"]["raw_width"]) == (h, w,)
        # Determine pattern size from CFA pattern
        pattern_shape = metadata["RGBG_pattern"].shape
        if pattern_shape == (2, 2):
            pattern_size = 2  # bayer uses 2x2, but we ensure % 4 for processing
            divisor = 4
        elif pattern_shape == (6, 6):
            pattern_size = 6  # X-Trans uses 6x6
            divisor = 6
        else:
            raise ValueError(f"Unsupported pattern shape: {pattern_shape}")

        # crop raw width/height if bigger than width/height
        if h % divisor > 0:
            mono_img = mono_img[:, : -(h % divisor)]
            metadata["sizes"]["raw_height"] -= h % divisor
            metadata["sizes"]["height"] -= h % divisor
            metadata["sizes"]["iheight"] -= h % divisor
        if w % divisor > 0:
            mono_img = mono_img[:, :, : -(w % divisor)]
            metadata["sizes"]["raw_width"] -= w % divisor
            metadata["sizes"]["width"] -= w % divisor
            metadata["sizes"]["iwidth"] -= w % divisor

        assert not (mono_img.shape[1] % divisor or mono_img.shape[2] % divisor), (mono_img.shape)
        return mono_img, whole_image_raw_pattern

    # step 1: get raw data and metadata
    try:
        rawpy_img = rawpy.imread(fpath)
    except rawpy._rawpy.LibRawFileUnsupportedError as e:
        raise ValueError(f"raw.raw_fpath_to_mono_img_and_metadata error opening {fpath}: {e}")
    except rawpy._rawpy.LibRawIOError as e:
        raise ValueError(f"raw.raw_fpath_to_mono_img_and_metadata error opening {fpath}: {e}")
    metadata = dict()
    metadata["camera_whitebalance"] = rawpy_img.camera_whitebalance
    metadata["black_level_per_channel"] = rawpy_img.black_level_per_channel
    metadata["white_level"] = rawpy_img.white_level  # imgdata.rawdata.color.maximum
    metadata["camera_white_level_per_channel"] = (
        rawpy_img.camera_white_level_per_channel)  # imgdata.rawdata.color.linear_max
    #  daylight_whitebalance = float rawdata.color.pre_mul[4]; in libraw
    #  White balance coefficients for daylight (daylight balance). Either read
    #  from file, or calculated on the basis of file data, or taken from
    metadata["daylight_whitebalance"] = rawpy_img.daylight_whitebalance
    #  float cam_xyz[4][3];
    #  Camera RGB - XYZ conversion matrix. This matrix is constant (different
    #  for different models). Last row is zero for RGB cameras and non-zero for
    #  different color models (CMYG and so on).
    metadata["rgb_xyz_matrix"] = rawpy_img.rgb_xyz_matrix
    assert metadata["rgb_xyz_matrix"].any(), (f"rgb_xyz_matrix of {fpath} is empty ({metadata=})")
    metadata["sizes"] = rawpy_img.sizes._asdict()
    assert rawpy_img.color_desc.decode() == "RGBG", (
        f"{fpath} does not seem to have bayer pattern ({rawpy_img.color_desc.decode()})")
    metadata["RGBG_pattern"] = rawpy_img.raw_pattern
    assert metadata["RGBG_pattern"] is not None, (f"{fpath} has no bayer pattern information")
    # processed metadata:
    # Replace raw_pattern from libraw (which always returns RGBG)
    set_bayer_pattern_name(metadata)

    whole_image_raw_pattern = rawpy_img.raw_colors

    # Handle both bayer (2x2) and X-Trans (6x6) patterns
    pattern_shape = metadata["RGBG_pattern"].shape
    if pattern_shape == (2, 2):
        # bayer pattern
        assert_correct_metadata = (metadata["RGBG_pattern"] == whole_image_raw_pattern[:2, :2])
        assert (
            assert_correct_metadata if isinstance(assert_correct_metadata, bool) else assert_correct_metadata.all()), (
            f"bayer pattern decoding did not match ({fpath=}, {metadata['RGBG_pattern']=}, {whole_image_raw_pattern[:2, :2]=})")
    elif pattern_shape == (6, 6):
        # X-Trans pattern
        assert_correct_metadata = (metadata["RGBG_pattern"] == whole_image_raw_pattern[:6, :6])
        assert (
            assert_correct_metadata if isinstance(assert_correct_metadata, bool) else assert_correct_metadata.all()), (
            f"x-trans pattern decoding did not match ({fpath=}, {metadata['RGBG_pattern']=}, {whole_image_raw_pattern[:6, :6]=})")
    else:
        raise ValueError(f"Unsupported CFA pattern shape: {pattern_shape}")
    for a_wb in ("daylight", "camera"):
        metadata[f"{a_wb}_whitebalance_norm"] = np.array(metadata[f"{a_wb}_whitebalance"], dtype=np.float32)
        if metadata[f"{a_wb}_whitebalance_norm"][3] == 0:
            metadata[f"{a_wb}_whitebalance_norm"][3] = metadata[f"{a_wb}_whitebalance_norm"][1]
        metadata[f"{a_wb}_whitebalance_norm"] /= metadata[f"{a_wb}_whitebalance_norm"][1]
    mono_img = rawpy_img.raw_image
    mono_img = np.expand_dims(mono_img, axis=0)
    mono_img, whole_image_raw_pattern = rm_empty_borders(mono_img, metadata, whole_image_raw_pattern, crop_all=crop_all)
    # Only convert to RGGB for bayer patterns, skip for X-Trans
    pattern_shape = metadata["RGBG_pattern"].shape
    if force_rggb and pattern_shape == (2, 2):
        mono_img, whole_image_raw_pattern = mono_any_to_mono_rggb(mono_img, metadata, whole_image_raw_pattern)
    mono_img, whole_image_raw_pattern = ensure_correct_shape(mono_img, metadata, whole_image_raw_pattern)
    if return_float:
        mono_img = scale_img_to_bw_points(mono_img, metadata)
    return mono_img, metadata


def libraw_process(raw_fpath: str, out_fpath) -> None:
    """Sanity check uses libraw/dcraw. Actually does not return a desired output."""
    rawpy_img = rawpy.imread(raw_fpath)

    params = rawpy.Params(gamma=(1, 1), demosaic_algorithm=rawpy.DemosaicAlgorithm.VNG, use_camera_wb=False,
        use_auto_wb=False, output_color=rawpy.ColorSpace.sRGB,  # XYZ here looks closer ...
        no_auto_bright=True, # no_auto_scale=True, # dark, I guess using 14 bits out of 16 ?
    )
    img = rawpy_img.postprocess(params)
    try:
        imageio.imsave(out_fpath, img)
    except TypeError as e:
        print(f"libraw_process failed to write image {out_fpath=}: {e=}")


def mono_to_rggb_img(mono_img: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Convert from mono (1xy) image or batch to 4-layers RGGB.

    BGGR can be supported in the future by cropping the first/last row/column.
    """
    if metadata["bayer_pattern"] != "RGGB":
        raise NotImplementedError(f"{metadata['bayer_pattern']=} is not RGGB")
        log

    assert mono_img.shape[-3] == 1, f"not a mono image ({mono_img.shape})"
    mono_img = mono_img[..., 0, :, :]
    # step 2: convert to 4-channels RGGB
    rggb_img = np.array([# RGGB
        mono_img[..., 0::2, 0::2],  # R
        mono_img[..., 0::2, 1::2],  # G
        mono_img[..., 1::2, 0::2],  # G
        mono_img[..., 1::2, 1::2],  # B
    ], dtype=np.float32, )
    return rggb_img


def rggb_to_mono_img(rggb_img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    assert rggb_img.shape[-3] == 4, rggb_img.shape
    mono_img = np.empty_like(rggb_img,
        shape=(*rggb_img.shape[:-3], 1, rggb_img.shape[-2] * 2, rggb_img.shape[-1] * 2), )
    mono_img[..., 0::2, 0::2] = rggb_img[..., 0:1, :, :]
    mono_img[..., 0::2, 1::2] = rggb_img[..., 1:2, :, :]
    mono_img[..., 1::2, 0::2] = rggb_img[..., 2:3, :, :]
    mono_img[..., 1::2, 1::2] = rggb_img[..., 3:4, :, :]
    return mono_img


def scale_img_to_bw_points(img: np.ndarray, metadata: dict, compat: bool = True) -> np.ndarray:
    """
    Scale image to black/white points described in metadata.

    compat: bool: use the conservative (higher) "white_level" value as in darktable.
    """
    scaled_img = img.astype(np.float32)
    metadata["overexposure_lb"] = 1.0

    for ch in range(img.shape[-3]):
        scaled_img[ch] -= metadata["black_level_per_channel"][ch]
        # step 4: normalize s.t. white level is 1: divide each value by (white level - black level)
        # darktable uses global max only, aka "white_level", which is higher.
        if compat:
            vrange = metadata["white_level"] - metadata["black_level_per_channel"][ch]
            if metadata["camera_white_level_per_channel"] is not None:
                metadata["overexposure_lb"] = min(metadata["overexposure_lb"], (metadata[
                                                                                    "camera_white_level_per_channel"][
                                                                                    ch] -
                                                                                metadata["black_level_per_channel"][
                                                                                    ch]) / vrange, )
        else:
            vrange = (metadata["camera_white_level_per_channel"][ch] - metadata["black_level_per_channel"][ch])
        scaled_img[ch] /= vrange
    return scaled_img


def apply_whitebalance(img: np.ndarray, metadata: dict, wb_type: str = "daylight", in_place: bool = True,
        reverse: bool = False, ) -> None:
    """Apply or remove white balance correction to neutralize color casts from illumination.

    White balance compensates for the color temperature of the scene illumination. Indoor
    tungsten lighting appears orange to cameras calibrated for daylight; fluorescent lighting
    appears greenish. Without white balance correction, colors are distorted—what should be
    neutral gray appears tinted.

    Camera sensors measure scene radiance, which is the product of surface reflectance and
    illumination spectrum. Under tungsten lighting, a white surface reflects more red light
    (because tungsten emits more red). White balance multiplies each color channel to
    compensate: boost blue to counter the excess red, producing neutral colors.

    This function applies per-channel multipliers stored in the RAW file metadata. The
    'daylight' multipliers normalize to standard D65 daylight illuminant. The 'camera'
    multipliers use whatever white balance the camera computed when the image was captured
    (often auto white balance based on scene analysis).

    The multipliers are normalized such that green = 1.0, with red and blue scaled relative
    to green. Typical values might be [2.4, 1.0, 1.5], indicating the blue channel needs 1.5×
    amplification and red needs 2.4× to neutralize a warm (low color temperature) illuminant.

    bayer vs. RGB handling:
    For mosaiced bayer data (shape (1, H, W)), multipliers are applied in a checkerboard
    pattern matching the RGGB layout. For demosaiced RGB (shape (3, H, W)) or 4-channel
    bayer (4, H/2, W/2), multipliers apply to entire channels.

    Reversing white balance:
    Setting reverse=True divides instead of multiplies, useful for "de-white-balancing" to
    recover original sensor measurements from already-corrected images.

    Args:
        img: Image to white balance, shape (1, H, W) for mosaiced bayer, (3, H, W) for RGB,
            or (4, H/2, W/2) for 4-channel bayer
        metadata: Dict containing '{wb_type}_whitebalance_norm' with [R, G, B, G] multipliers
        wb_type: 'daylight' for D65 normalization or 'camera' for as-shot white balance
        in_place: If True, modify img directly; if False, return modified copy
        reverse: If True, divide by multipliers (undo white balance) instead of multiply

    Returns:
        None if in_place=True (img is modified), otherwise returns the corrected image

    Example:
        >>> mono, meta = raw_fpath_to_mono_img_and_metadata('IMG_0001.CR2')
        >>> meta['daylight_whitebalance_norm']
        array([2.3984, 1.0, 1.5234, 1.0])
        >>> apply_whitebalance(mono, meta, wb_type='daylight')
        >>> # Red pixels now multiplied by 2.4, blue by 1.5, greens unchanged
        >>> # Image colors now appear as they would under daylight

    Note:
        White balance amplifies channels, which can push values above 1.0 even if the
        original image was normalized to [0,1]. This is expected and correct—highlights
        that were near saturation in one channel may exceed 1.0 in others after balancing.

        The function assumes RGGB pattern when operating on mosaiced bayer data. Other
        patterns will raise NotImplementedError.
    """
    op = operator.truediv if reverse else operator.mul
    # step 5: apply camera reference white balance
    assert f"{wb_type}_whitebalance_norm" in metadata, f"{wb_type=}, {metadata=}"
    if metadata["bayer_pattern"] != "RGGB" and img.shape[-3] != 3:
        raise NotImplementedError(f"{metadata["bayer_pattern"]=} is not RGGB")
    if not in_place:
        img = img.copy()
    if img.shape[-3] == 1:  # mono
        # RGGB
        img[0, 0::2, 0::2] = op(img[0, 0::2, 0::2], metadata[f"{wb_type}_whitebalance_norm"][0])  # R
        img[0, 0::2, 1::2] = op(img[0, 0::2, 1::2], metadata[f"{wb_type}_whitebalance_norm"][1])  # G
        img[0, 1::2, 1::2] = op(img[0, 1::2, 1::2], metadata[f"{wb_type}_whitebalance_norm"][2])  # B
        img[0, 1::2, 0::2] = op(img[0, 1::2, 0::2], metadata[f"{wb_type}_whitebalance_norm"][3])  # G

    else:
        for ch_img, ch_wb in enumerate(range(3) if img.shape[-3] == 3 else (0, 1, 3, 2)):
            img[ch_img] = op(img[ch_img], metadata[f"{wb_type}_whitebalance_norm"][ch_wb])
    # assert img.max() <= 1.0, img.max()

    if not in_place:
        return img


def raw_fpath_to_rggb_img_and_metadata(fpath: str, return_float: bool = True):
    """Load RAW file and return 4-channel bayer RGGB image plus metadata.

    Convenience wrapper around raw_fpath_to_mono_img_and_metadata that reshapes the
    single-channel mosaiced data into a 4-channel representation where each channel
    contains one color component from the bayer pattern. This format is more convenient
    for neural networks that process bayer data directly, as it allows standard 2D
    convolutions to operate independently on each color channel without mixing pixels
    from different color filters.

    The RGGB layout separates:
    - Channel 0: Red pixels (top-left in each 2×2 block)
    - Channel 1: Green pixels (top-right)
    - Channel 2: Green pixels (bottom-left)
    - Channel 3: Blue pixels (bottom-right)

    Spatial resolution in each channel is half the original in each dimension
    (width/2 × height/2), since each 2×2 bayer block yields one pixel per channel.

    Args:
        fpath: Path to RAW camera file
        return_float: Scale to [0,1] using black/white points if True

    Returns:
        Tuple of (rggb_img, metadata) where:
        - rggb_img: np.ndarray of shape (4, H/2, W/2), bayer channels separated
        - metadata: Same as raw_fpath_to_mono_img_and_metadata

    Example:
        >>> rggb, meta = raw_fpath_to_rggb_img_and_metadata('IMG_0001.CR2')
        >>> rggb.shape  # 4 channels, each half resolution
        (4, 1736, 2604)
    """
    mono_img, metadata = raw_fpath_to_mono_img_and_metadata(fpath, return_float=return_float)
    return mono_to_rggb_img(mono_img, metadata), metadata


def demosaic(mono_img: np.ndarray, metadata: dict, algorithm: str = 'linear') -> np.ndarray:
    """Reconstruct full-color RGB image from mosaiced bayer sensor data.

    Demosaicing (also called debayering) is the process of interpolating missing color channels
    at each pixel location. A bayer sensor measures only one color per pixel—red, green, or
    blue depending on the color filter at that location. To produce a full-color image, the
    other two channels must be inferred from neighboring pixels.

    This is an inherently ill-posed problem. Near edges, naïve interpolation creates color
    fringing artifacts (false colors where sharp transitions occur). In textured regions,
    repetitive patterns can cause moiré. The demosaicing algorithm must balance between
    preserving sharp edges and avoiding artifacts in smooth regions.

    This function uses OpenImageIO's demosaicing implementation (default), which handles
    float32 data natively without uint16 conversion. OIIO provides better HDR support
    and integrates seamlessly with the rest of the pipeline (EXR I/O uses OIIO).

    Color space note:
    The output is in "camera RGB"—the native color space of the sensor. The spectral
    responses of the camera's red, green, and blue filters don't match human color perception
    or standard color spaces. Further color correction via camRGB_to_profiledRGB_img() is
    required to convert to perceptually meaningful colors (linear Rec.2020, sRGB, etc.).

    Args:
        mono_img: Mosaiced image of shape (1, H, W), single-channel bayer pattern data
        metadata: Dict containing 'bayer_pattern' key with value 'RGGB' (other patterns
            should be converted to RGGB before calling this function)
        method: DEPRECATED (kept for backward compatibility). Use backend/algorithm instead.
        backend: Demosaic backend ('oiio' or 'cv2'). Default 'oiio' recommended.
        algorithm: Demosaic algorithm. For OIIO: 'linear' (bilinear). 
                  For cv2: 'bilinear' or 'edge_aware'. Default 'linear'.

    Returns:
        RGB image as np.ndarray of shape (3, H, W), in camera RGB color space, same data
        type as input (float32). Intensity range matches input (may exceed [0,1] for HDR).

    Raises:
        AssertionError: If input is not shape (1, H, W), if bayer_pattern != 'RGGB'
        ValueError: If invalid backend specified

    Example:
        >>> mono, meta = raw_fpath_to_mono_img_and_metadata('IMG_0001.CR2')
        >>> mono.shape
        (1, 3472, 5208)
        >>> camRGB = demosaic(mono, meta)  # Uses OIIO by default
        >>> camRGB.shape
        (3, 3472, 5208)
        >>> # Now apply color correction:
        >>> pRGB = camRGB_to_profiledRGB_img(camRGB, meta, 'lin_rec2020')

    Note:
        This function assumes the input has already been converted to RGGB pattern via
        raw_fpath_to_mono_img_and_metadata(..., force_rggb=True). Other bayer patterns
        (GBRG, BGGR, GRBG) will fail the assertion.

        For x-trans sensors, demosaicing requires specialized algorithms not provided here.
        Use the xtrans_to_OpenEXR workflow instead.
    """
    # Validate inputs
    assert mono_img.shape[0] == 1, f"{mono_img.shape=}"

    # Only validate bayer_pattern for bayer sensors
    if metadata.get("is_bayer", True):
        bayer_pattern = metadata.get("bayer_pattern", "RGGB")
        assert bayer_pattern == "RGGB", f"{bayer_pattern=} (only RGGB supported)"

    # OIIO demosaic (supports both bayer and X-Trans)
    # OIIO demosaic: native float32 support, no conversions needed

    # Create ImageBuf from numpy array (1, H, W) -> (H, W, 1)
    mono_2d = mono_img[0, :, :].astype(np.float32)

    # OIIO ImageBuf needs (H, W, C) shape
    src_buf = oiio.ImageBuf(oiio.ImageSpec(mono_2d.shape[1], mono_2d.shape[0], 1, oiio.FLOAT))
    src_buf.set_pixels(oiio.ROI(), mono_2d.reshape(mono_2d.shape[0], mono_2d.shape[1], 1))

    # Infer pattern type from metadata
    is_bayer = metadata.get("is_bayer", True)  # Default to Bayer if not specified

    if is_bayer:
        pattern = "bayer"
        layout = metadata.get("bayer_pattern", "RGGB")
        # Use MHC (Malvar-He-Cutler) - edge-aware algorithm similar to Li's
        algorithm = "MHC" if algorithm == "linear" else algorithm
    else:
        # X-Trans: use OIIO's X-Trans support
        pattern = "xtrans"
        layout = ""  # X-Trans doesn't use layout parameter  # Pass algorithm through - OIIO may support additional algorithms for X-Trans  # beyond those documented (e.g., markesteijn variant)

    # No white balance adjustment (neutral 1.0, 1.0, 1.0)
    # White balance should be applied during color correction, not demosaic
    white_balance = (1.0, 1.0, 1.0)

    dst_buf = oiio.ImageBufAlgo.demosaic(src_buf, pattern=pattern, layout=layout, algorithm=algorithm,
        white_balance=white_balance)

    if not dst_buf or dst_buf.has_error:
        raise RuntimeError(f"OIIO demosaic failed: {dst_buf.geterror()}")

    # Convert back to numpy (H, W, 3) -> (3, H, W)
    rgb_img = dst_buf.get_pixels(oiio.FLOAT)
    rgb_img = rgb_img.transpose(2, 0, 1).astype(np.float32)

    return rgb_img


def rgb_to_cielab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to CIELab color space.
    
    CIELab is perceptually uniform - differences in Lab values correspond to 
    perceptual differences. This is critical for homogeneity detection.
    
    Args:
        rgb: RGB values, shape (..., 3), values in [0, 1]
    
    Returns:
        lab: CIELab values, shape (..., 3)
            L: lightness [0, 100]
            a: green-red axis [-128, 127]
            b: blue-yellow axis [-128, 127]
    """
    # Simplified RGB→XYZ→Lab conversion
    # Assumes RGB is already linear (no gamma)

    # RGB to XYZ (using sRGB/Rec.709 matrix)
    # Simplified - proper implementation would use color profiles
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722  # Y is luminance
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    # XYZ to Lab
    # Reference white D65
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    def f(t):
        """CIELab f function"""
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)

    lab = np.stack([L, a, b_lab], axis=-1)
    return lab.astype(np.float32)


def markesteijn_demosaic(mono_img: np.ndarray, metadata: dict) -> np.ndarray:
    """(UNUSED): works, but verrrry slow --- x-trans demosaicing using Markesteijn 3-pass algorithm.
    
    The x-trans color filter array uses a 6×6 non-repeating pattern designed to reduce
    moiré artifacts without an optical low-pass filter. Standard bayer demosaicing
    algorithms produce severe artifacts on x-trans data due to the irregular pattern.
    
    The Markesteijn algorithm is a 3-pass edge-directed method specifically designed
    for x-trans sensors:
    
    1. Green interpolation: Multi-directional gradients with homogeneity detection
    2. Red/Blue interpolation: Guided by interpolated green channel
    3. Refinement: Color difference propagation with edge preservation
    
    This implementation will be ported from darktable's OpenCL kernel:
    https://github.com/darktable-org/darktable/blob/master/data/kernels/demosaic_markesteijn.cl
    
    Args:
        mono_img: x-trans mosaiced image of shape (H, W), single-channel raw data
        metadata: Dict containing 'RGBG_pattern' key with 6×6 array where:
            0 = Red, 1 = Green, 2 = Blue (sensor color at each position)
    
    Returns:
        RGB image as np.ndarray of shape (3, H, W), in camera RGB color space,
        float32 data type. Intensity range matches input.
    
    Raises:
        NotImplementedError: Currently stubbed - implementation pending
        AssertionError: If RGBG_pattern is not 6×6
    
    Example:
        >>> mono, meta = raw_fpath_to_mono_img_and_metadata('DSCF0001.RAF')
        >>> pattern = meta['RGBG_pattern']
        >>> pattern.shape
        (6, 6)
        >>> camRGB = markesteijn_demosaic(mono, meta)
        >>> camRGB.shape
        (3, 3472, 5208)
    
    Note:
        This is a computationally intensive algorithm. For batch processing,
        consider tiling large images or using GPU acceleration.

    Implementation based on LibRaw's xtrans_interpolate
        https://github.com/LibRaw/LibRaw/blob/master/src/demosaic/xtrans_demosaic.cpp
    """
    # Validate inputs
    if mono_img.ndim == 3:
        assert mono_img.shape[0] == 1, f"Expected shape (1,H,W), got {mono_img.shape}"
        mono_img = mono_img[0]  # Convert to 2D

    assert mono_img.ndim == 2, f"Expected 2D array, got shape {mono_img.shape}"
    pattern = metadata.get("RGBG_pattern")
    assert pattern is not None, "metadata must contain 'RGBG_pattern' key"
    assert pattern.shape == (6, 6), f"x-trans pattern must be 6×6, got {pattern.shape}"
    height, width = mono_img.shape
    xtrans = pattern  # 6x6 color filter array

    # Define fcol helper: maps (row, col) to color channel (0=R, 1=G, 2=B)
    def fcol(row: int, col: int) -> int:
        return int(xtrans[(row + 6) % 6, (col + 6) % 6])

    # Validate X-Trans pattern
    cstat = [0, 0, 0, 0]
    for row in range(6):
        for col in range(6):
            cstat[fcol(row, col)] += 1

    if not (6 <= cstat[0] <= 10 and 16 <= cstat[1] <= 24 and 6 <= cstat[2] <= 10 and cstat[3] == 0):
        raise ValueError(f"Invalid x-trans pattern color distribution: {cstat}")

    # Constants for hexagonal neighborhood sampling
    orth = np.array([1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1], dtype=np.int16)
    patt = np.array(
        [[0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0], [0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1]],
        dtype=np.int16).reshape(2, 8, 2)

    # Initialize allhex table for hexagonal neighborhoods
    # CRITICAL FIX: Store as (dy, dx) pairs directly to avoid Python divmod bugs with negatives
    # allhex_2d[row%3][col%3][0=image,1=tile][hex_idx] = (dy, dx)
    allhex_2d = np.zeros((3, 3, 2, 8, 2), dtype=np.int16)  # Extra dim for (dy, dx)

    sgrow, sgcol = 0, 0  # Solitary green pixel location

    # Build hex table mapping
    for row in range(3):
        for col in range(3):
            ng = 0
            for d in range(0, 10, 2):
                g = 1 if fcol(row, col) == 1 else 0
                if fcol(row + orth[d], col + orth[d + 2]) == 1:
                    ng = 0
                else:
                    ng += 1

                if ng == 4:
                    sgrow, sgcol = row, col

                if ng == g + 1:
                    for c in range(8):
                        v = orth[d] * patt[g, c, 0] + orth[d + 1] * patt[g, c, 1]  # dy
                        h = orth[d + 2] * patt[g, c, 0] + orth[d + 3] * patt[g, c, 1]  # dx
                        # Store as (dy, dx) pair instead of encoded offset
                        allhex_2d[row, col, 0, c ^ (g * 2 & d)] = [v, h]
                        allhex_2d[row, col, 1, c ^ (g * 2 & d)] = [v, h]  # tile version

    # Initialize RGB output arrays
    # LibRaw uses 4 directional versions: rgb[0-3] for different interpolation directions
    # After interpolation, homogeneity map picks best direction per pixel
    num_directions = 4
    rgb = np.zeros((num_directions, height, width, 3), dtype=np.float32)

    # Copy raw values to appropriate channels (all directions start with same raw data)
    for d in range(num_directions):
        for row in range(height):
            for col in range(width):
                rgb[d, row, col, fcol(row, col)] = mono_img[row, col]

    # Set green min/max bounds for non-green pixels  
    green_min = np.full((height, width), np.inf, dtype=np.float32)
    green_max = np.full((height, width), -np.inf, dtype=np.float32)

    for row in range(2, height - 2):
        for col in range(2, width - 2):
            if fcol(row, col) == 1:
                continue

            hex_2d = allhex_2d[row % 3, col % 3, 0]
            for c in range(6):
                dy, dx = hex_2d[c]
                nr, nc = row + dy, col + dx
                if 0 <= nr < height and 0 <= nc < width and fcol(nr, nc) == 1:
                    val = mono_img[nr, nc]
                    green_min[row, col] = min(green_min[row, col], val)
                    green_max[row, col] = max(green_max[row, col], val)

    # Interpolate green channel using LibRaw's 4-directional formula
    # This is the KEY quality improvement - directional interpolation preserves edges
    for row in range(3, height - 3):
        for col in range(3, width - 3):
            f = fcol(row, col)
            if f == 1:  # Already green, copy to all directions
                for d in range(num_directions):
                    rgb[d, row, col, 1] = mono_img[row, col]
                continue

            hex_2d = allhex_2d[row % 3, col % 3, 0]

            # Helper to safely get pixel value at offset (can be 1x, 2x, 3x hex distance)
            def get_pix_at_offset(offset_multiplier, hex_offset_idx):
                """Get pixel at offset_multiplier * hex_2d[hex_offset_idx]
                
                Now using direct (dy, dx) storage instead of encoded offsets.
                """
                base_dy, base_dx = hex_2d[hex_offset_idx]
                dy = int(base_dy) * offset_multiplier
                dx = int(base_dx) * offset_multiplier
                nr, nc = row + dy, col + dx
                if 0 <= nr < height and 0 <= nc < width:
                    return mono_img[nr, nc]
                return 0.0

            # LibRaw's 4-directional green interpolation formulas
            # These are the KEY - each formula emphasizes a different direction

            # Direction 0: horizontal/vertical using hex[0] and hex[1]
            # Weighted combo: strong weight to 1x neighbors, negative weight to 2x (edge sharpening)
            color_0 = (174 * (get_pix_at_offset(1, 1) + get_pix_at_offset(1, 0)) - 46 * (
                        get_pix_at_offset(2, 1) + get_pix_at_offset(2, 0))) / 256.0

            # Direction 1: another h/v variant using hex[2] and hex[3]
            # Includes current pixel value (f channel) for correction
            color_1 = (223 * get_pix_at_offset(1, 3) + 33 * get_pix_at_offset(1, 2) + 92 * (
                        mono_img[row, col] - get_pix_at_offset(-1, 2))) / 256.0

            # Directions 2 and 3: diagonal interpolations using hex[4] and hex[5]
            # Uses ±2x and ±3x distances for longer-range edge detection
            color_2 = (164 * get_pix_at_offset(1, 4) + 92 * get_pix_at_offset(-2, 4) + 33 * (
                        2 * mono_img[row, col] - get_pix_at_offset(3, 4) - get_pix_at_offset(-3, 4))) / 256.0

            color_3 = (164 * get_pix_at_offset(1, 5) + 92 * get_pix_at_offset(-2, 5) + 33 * (
                        2 * mono_img[row, col] - get_pix_at_offset(3, 5) - get_pix_at_offset(-3, 5))) / 256.0

            # Clamp each direction to min/max green bounds
            colors = [color_0, color_1, color_2, color_3]
            for d in range(num_directions):
                if not np.isinf(green_min[row, col]):
                    colors[d] = np.clip(colors[d], green_min[row, col], green_max[row, col])

                # Store in corresponding RGB direction array
                rgb[d, row, col, 1] = colors[d]

    # Interpolate red and blue for solitary green pixels
    # Simplified version: just average nearest neighbors of same color
    for row in range(max(0, sgrow), min(height - 3, height), 3):
        for col in range(max(0, sgcol), min(width - 3, width), 3):
            if fcol(row, col) != 1:
                continue

            # Find neighboring R and B values
            for channel in [0, 2]:  # R and B
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r, c = row + dr, col + dc
                        if 0 <= r < height and 0 <= c < width and fcol(r, c) == channel:
                            neighbors.append(mono_img[r, c])

                if neighbors:
                    avg_val = np.mean(neighbors)
                    for d in range(num_directions):
                        rgb[d, row, col, channel] = avg_val

    # Fill remaining R/B values by bilinear interpolation
    for channel in [0, 2]:  # R and B
        for row in range(3, height - 3):
            for col in range(3, width - 3):
                # Check if any direction needs filling
                if rgb[0, row, col, channel] == 0:
                    # Bilinear from nearest same-color pixels
                    neighbors = []
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < height and 0 <= nc < width and fcol(nr, nc) == channel:
                                neighbors.append(mono_img[nr, nc])

                    if neighbors:
                        avg_val = np.mean(neighbors)
                        for d in range(num_directions):
                            rgb[d, row, col, channel] = avg_val
                    else:
                        # Fallback: use green channel scaled
                        for d in range(num_directions):
                            rgb[d, row, col, channel] = rgb[d, row, col, 1] * 0.5

    # ==================== HOMOGENEITY-BASED DIRECTION SELECTION ====================
    # This is the KEY to Markesteijn's quality: pick best direction per pixel

    # Step 1: Convert each directional RGB to CIELab for perceptual comparison
    print(f"Computing homogeneity maps for {num_directions} directions...")
    lab = np.zeros((num_directions, height, width, 3), dtype=np.float32)

    for d in range(num_directions):
        # Convert RGB to Lab (shape: H,W,3 -> H,W,3)
        lab[d] = rgb_to_cielab(rgb[d])

    # Step 2: Calculate directional derivatives in Lab space
    # dir_offsets: which direction to sample for derivative (horizontal, vertical, diagonals)
    dir_offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]  # right, down, diag-right, diag-left

    drv = np.zeros((num_directions, height, width), dtype=np.float32)

    for d in range(num_directions):
        dy, dx = dir_offsets[d]

        for row in range(3, height - 3):
            for col in range(3, width - 3):
                # Get Lab values at center and offset positions
                lab_center = lab[d, row, col]

                # Forward and backward neighbors
                if 0 <= row + dy < height and 0 <= col + dx < width:
                    lab_fwd = lab[d, row + dy, col + dx]
                else:
                    lab_fwd = lab_center

                if 0 <= row - dy < height and 0 <= col - dx < width:
                    lab_bwd = lab[d, row - dy, col - dx]
                else:
                    lab_bwd = lab_center

                # Second derivative in Lab space (measures edge strength)
                # LibRaw formula (line ~337-340):
                # g = 2*L[0] - L[fwd] - L[bwd]
                # drv = g^2 + (2*a[0] - a[fwd] - a[bwd] + g*500/232)^2 + (2*b[0] - b[fwd] - b[bwd] - g*500/580)^2

                L_diff = 2 * lab_center[0] - lab_fwd[0] - lab_bwd[0]
                a_diff = 2 * lab_center[1] - lab_fwd[1] - lab_bwd[1] + L_diff * 500.0 / 232.0
                b_diff = 2 * lab_center[2] - lab_fwd[2] - lab_bwd[2] - L_diff * 500.0 / 580.0

                drv[d, row, col] = L_diff ** 2 + a_diff ** 2 + b_diff ** 2

    # Step 3: Build homogeneity maps
    # For each direction, count how many neighbors have similar (low) derivatives
    homo = np.zeros((num_directions, height, width), dtype=np.int32)

    for row in range(4, height - 4):
        for col in range(4, width - 4):
            # Find minimum derivative across all directions at this pixel
            min_drv = np.min(drv[:, row, col])
            threshold = min_drv * 8.0  # LibRaw uses 8x threshold

            # For each direction, count neighbors below threshold
            for d in range(num_directions):
                count = 0
                for v in range(-1, 2):
                    for h in range(-1, 2):
                        if drv[d, row + v, col + h] <= threshold:
                            count += 1
                homo[d, row, col] = count

    # Step 4: Select best direction(s) and average them
    # LibRaw averages all directions with homogeneity above (max - max/8)
    result = np.zeros((height, width, 3), dtype=np.float32)

    # Use smaller border for homogeneity selection (was 8, now 5)
    border = 5
    for row in range(border, height - border):
        for col in range(border, width - border):
            # Sum homogeneity over 5x5 neighborhood for each direction
            hm = np.zeros(num_directions, dtype=np.int32)
            for d in range(num_directions):
                for v in range(-2, 3):
                    for h in range(-2, 3):
                        hm[d] += homo[d, row + v, col + h]

            # Find max homogeneity
            max_hm = np.max(hm)
            threshold = max_hm - max_hm // 8

            # Average RGB from all directions above threshold
            avg_rgb = np.zeros(3, dtype=np.float32)
            count = 0
            for d in range(num_directions):
                if hm[d] >= threshold:
                    avg_rgb += rgb[d, row, col]
                    count += 1

            if count > 0:
                result[row, col] = avg_rgb / count
            else:
                # Fallback: use direction 0
                result[row, col] = rgb[0, row, col]

    # Fill borders (not processed above) with simple average
    for row in range(height):
        for col in range(width):
            if row < border or row >= height - border or col < border or col >= width - border:
                result[row, col] = np.mean(rgb[:, row, col], axis=0)

    print("Homogeneity selection complete")

    # Convert to (3, H, W) format
    result = result.transpose(2, 0, 1).astype(np.float32)

    return result


def get_XYZ_to_profiledRGB_matrix(profile: str) -> np.ndarray:
    """Returns a static XYZ->profile matrix."""
    if profile == "lin_rec2020":
        return np.array([[1.71666343, -0.35567332, -0.25336809], [-0.66667384, 1.61645574, 0.0157683],
            [0.01764248, -0.04277698, 0.94224328], ], dtype=np.float32, )
    elif "sRGB" in profile:
        # conversion_matrix = colour.models.dataset.srgb.XYZ_TO_sRGB_MATRIX
        return np.array([[3.24100326, -1.53739899, -0.49861587], [-0.96922426, 1.87592999, 0.04155422],
            [0.05563942, -0.2040112, 1.05714897], ], dtype=np.float32, )
    else:
        raise NotImplementedError(f"get_std_profile_matrix: {profile} not *_sRGB or lin_rec2020.")


def get_camRGB_to_profiledRGB_img_matrix(metadata: dict, output_color_profile: str) -> np.ndarray:
    """
    Get conversion matrix from camRGB to a given color profile.

    compat=True yields an output which might be closer to darktable's,
    otherwise the output is slightly darker.
    """
    # if compat:
    #     cam_to_xyzd65 = np.linalg.inv(
    #         metadata["rgb_xyz_matrix"][:3] / metadata["daylight_whitebalance"][1]
    #     )
    # else:
    cam_to_xyzd65 = np.linalg.inv(metadata["rgb_xyz_matrix"][:3])
    if output_color_profile.lower() == "xyz":
        return cam_to_xyzd65
    xyz_to_profiledRGB = get_XYZ_to_profiledRGB_matrix(output_color_profile)
    color_matrix = xyz_to_profiledRGB @ cam_to_xyzd65
    return color_matrix


def camRGB_to_profiledRGB_img(camRGB_img: np.ndarray, metadata: dict, output_color_profile: str) -> np.ndarray:
    """Convert camera RGB to a standardized perceptual color space.

    Camera RGB (the output of demosaicing) exists in a device-specific color space determined
    by the spectral sensitivities of that particular camera sensor's color filters. A pixel
    that measures as (1.0, 0.5, 0.3) in camera RGB might represent different physical
    wavelengths on a Canon vs. a Nikon sensor. This makes camera RGB unsuitable for training
    generalizable neural networks—models would learn camera-specific color relationships
    rather than perceptual ones.

    This function applies a 3×3 color correction matrix to transform camera RGB into a
    standardized color space (typically linear Rec.2020 for this codebase). The matrix
    accounts for:
    1. The difference between the camera's actual spectral sensitivities and the target
       color space primaries
    2. Chromatic adaptation to a standard illuminant (typically D65 daylight)
    3. White balance normalization

    The transformation is linear (matrix multiplication), preserving radiometric relationships.
    A pixel with twice the radiant energy will have twice the value after transformation.
    This linearity is crucial for neural network training with perceptual loss functions.

    Output color spaces:
    - 'lin_rec2020': Linear Rec.2020, wide gamut, standard for HDR/professional work
    - 'lin_srgb': Linear sRGB, narrower gamut but more widely supported
    - 'gamma22': Rec.2020 with gamma 2.2 encoding (for visualization)
    - Other gamma-encoded spaces apply a power curve after the linear transformation

    Mathematical form:
        profiledRGB = M @ camRGB
    where M is a 3×3 matrix derived from the camera's XYZ matrix and the target color space.

    Args:
        camRGB_img: Demosaiced RGB image in camera color space, shape (3, H, W)
        metadata: Dict containing 'rgb_xyz_matrix' (camera RGB → CIE XYZ transform)
        output_color_profile: Target color space string ('lin_rec2020', 'lin_srgb',
            'gamma22', etc.)

    Returns:
        RGB image in the target color space, same shape as input (3, H, W). If output
        profile is gamma-encoded, the transfer function is applied in-place.

    Example:
        >>> mono, meta = raw_fpath_to_mono_img_and_metadata('IMG_0001.CR2')
        >>> camRGB = demosaic(mono, meta)
        >>> pRGB = camRGB_to_profiledRGB_img(camRGB, meta, 'lin_rec2020')
        >>> # pRGB is now in linear Rec.2020, suitable for neural network training
        >>> display_RGB = camRGB_to_profiledRGB_img(camRGB, meta, 'gamma22')
        >>> # display_RGB has gamma encoding applied for human viewing

    Note:
        The operation modifies the array in-place when gamma encoding is applied, but
        returns a new array for the linear transformation. This inconsistency is a
        historical artifact that should be ignored—treat the return value as the output.

        Values outside [0,1] are preserved (highlights exceeding white point, or
        reconstruction artifacts from demosaicing). Clipping to [0,1] should be done
        downstream if needed for a specific application.
    """
    color_matrix = get_camRGB_to_profiledRGB_img_matrix(metadata, output_color_profile)
    orig_dims = camRGB_img.shape
    # color_matrix /= metadata['daylight_whitebalance'][1]

    profiledRGB_img = (color_matrix @ camRGB_img.reshape(3, -1)).reshape(orig_dims)
    if output_color_profile.startswith("gamma"):
        apply_gamma(profiledRGB_img, output_color_profile)
    return profiledRGB_img


def apply_gamma(profiledRGB_img: np.ndarray, color_profile: str) -> None:
    """Apply gamma correction (in-place)."""
    if color_profile == "gamma_sRGB":
        #  See https://en.wikipedia.org/wiki/SRGB
        img_mask = profiledRGB_img > 0.0031308
        profiledRGB_img[img_mask] = (1.055 * np.power(profiledRGB_img[img_mask], 1.0 / 2.4) - 0.055)
        profiledRGB_img[~img_mask] *= 12.92
    else:
        raise NotImplementedError(f"apply_gamma with {color_profile=}")


def get_sample_raw_file(url: str = SAMPLE_RAW_URL) -> str:
    """Get a testing image online."""
    fn = url.split("/")[-1]
    fpath = os.path.join("data", fn)
    if not os.path.exists(fpath):
        os.makedirs("data", exist_ok=True)
        r = requests.get(url, allow_redirects=True, verify=False)
        open(fpath, "wb").write(r.content)
    return fpath


def is_exposure_ok(mono_float_img: np.ndarray, metadata: dict, oe_threshold=0.99, ue_threshold=0.001,
        qty_threshold=0.75, ) -> bool:
    """Check that the image exposure is useable in all channels."""
    rggb_img = mono_to_rggb_img(mono_float_img, metadata)
    overexposed = (rggb_img >= oe_threshold * metadata["overexposure_lb"]).any(0)
    if ue_threshold > 0:
        underexposed = (rggb_img <= ue_threshold).all(0)
        return (overexposed + underexposed).sum() / overexposed.size <= qty_threshold
    return overexposed.sum() / overexposed.size <= qty_threshold


ConversionOutcome = Enum("ConversionOutcome", "OK BAD_EXPOSURE UNREADABLE_ERROR UNKNOWN_ERROR")
ConversionResult = NamedTuple("ConversionResult",
    [("outcome", ConversionOutcome), ("src_fpath", str), ("dest_fpath", str)], )


def is_xtrans(fpath) -> bool:
    """Check if a file is an X-Trans sensor raw format (Fujifilm).
    
    Delegates to common.libs.libimganalysis.is_xtrans for single source of truth.
    
    Args:
        fpath: File path to check
        
    Returns:
        True if the file extension matches an X-Trans format, False otherwise
    """
    from common.libs import libimganalysis
    return libimganalysis.is_xtrans(fpath)


# ============================================================================
# CFA-AWARE FFT PHASE CORRELATION FOR ALIGNMENT
# ============================================================================


def extract_bayer_channels(img: np.ndarray, pattern: np.ndarray) -> dict[str, np.ndarray]:
    """Extract 4 bayer channels via strided slicing.

    Args:
        img: Mosaiced RAW image [1, H, W]
        pattern: 2x2 bayer pattern array (values 0=R, 1=G, 2=B, 3=G)

    Returns:
        Dict with keys 'R', 'G1', 'G2', 'B' containing downsampled channel arrays
    """
    values = img[0]

    # RawPy pattern encoding: 0=R, 1=G, 2=B, 3=G
    color_map = {0: "R", 1: "G", 2: "B", 3: "G"}
    positions = {}

    for row in range(2):
        for col in range(2):
            color_code = pattern[row, col]
            color = color_map[color_code]

            if color == "G":
                key = "G1" if "G1" not in positions else "G2"
            else:
                key = color

            positions[key] = (row, col)

    # Extract channels via strided slicing
    channels = {}
    for key, (row, col) in positions.items():
        channels[key] = values[row::2, col::2].copy()

    return channels


def extract_xtrans_channels(img: np.ndarray, pattern: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Extract x-trans channels via pixel shuffle without averaging.

    Uses pixel shuffle to reorganize the 6x6 pattern into 36 position channels,
    returning all position grids for each color separately for pixel-by-pixel comparison.

    Args:
        img: Mosaiced RAW image [1, H, W]
        pattern: 6x6 x-trans pattern array

    Returns:
        Dict with keys 'R', 'G', 'B' containing lists of position grids (no averaging)
    """
    values = img[0]
    h, w = values.shape
    pattern_h, pattern_w = pattern.shape

    sampled_h = h // pattern_h
    sampled_w = w // pattern_w

    color_map = {0: "R", 1: "G", 2: "B"}

    # Pixel shuffle: reorganize into all 36 position grids
    # Trim to multiple of pattern size
    h_trim = sampled_h * pattern_h
    w_trim = sampled_w * pattern_w
    values = values[:h_trim, :w_trim]

    # Reshape: [H, W] -> [H/6, 6, W/6, 6] -> [H/6, W/6, 6, 6] -> [H/6, W/6, 36]
    blocks = values.reshape(sampled_h, pattern_h, sampled_w, pattern_w)
    blocks = blocks.transpose(0, 2, 1, 3)  # [H/6, W/6, 6, 6]
    position_grids = blocks.reshape(sampled_h, sampled_w, pattern_h * pattern_w)

    # Group by color - return ALL position grids without averaging
    channels = {}
    for color_code in [0, 1, 2]:
        color_name = color_map[color_code]

        # Find all flat indices for this color in the pattern
        color_positions = []
        for pos_y in range(pattern_h):
            for pos_x in range(pattern_w):
                if pattern[pos_y, pos_x] == color_code:
                    flat_idx = pos_y * pattern_w + pos_x
                    color_positions.append(flat_idx)

        # Extract all grids for this color - return as list
        if color_positions:
            color_grids = [position_grids[:, :, pos] for pos in color_positions]
            channels[color_name] = color_grids

    return channels


def _fft_phase_correlate_single(anchor_ch: np.ndarray, target_ch: np.ndarray) -> tuple[int, int]:
    """Single-channel FFT phase correlation (hot path).

    Performance-critical function with minimal branching.

    Args:
        anchor_ch: Reference channel [H, W]
        target_ch: Target channel [H, W]

    Returns:
        (dy, dx): Detected shift in pixels
    """
    from scipy import signal

    # Normalize to zero mean
    anchor_norm = anchor_ch - anchor_ch.mean()
    target_norm = target_ch - target_ch.mean()

    # FFT cross-correlation
    correlation = signal.fftconvolve(anchor_norm, target_norm[::-1, ::-1], mode="same")

    # Find peak
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    center = np.array(correlation.shape) // 2

    # Convert to shift (displacement convention)
    shift = np.array(peak_idx) - center

    return int(shift[0]), int(shift[1])


def fft_phase_correlate_cfa(anchor: np.ndarray, target: np.ndarray, anchor_metadata: dict,
        method: Literal["median", "mean"] = "median", verbose: bool = False, ) -> tuple[
    tuple[int, int], list[tuple[int, int]]]:
    """CFA-aware FFT phase correlation for RAW image alignment.

    Performs phase correlation per CFA color channel and aggregates results to
    estimate integer-pixel shifts between mosaiced RAW frames. The CFA-aware
    separation improves robustness to color-specific noise and aliasing.

    Background:
    Phase correlation is a frequency-domain registration technique based on the
    Fourier shift theorem. The displacement between two images manifests as a
    phase ramp in the frequency domain; locating the peak in the inverse FFT of
    the normalized cross power spectrum yields the shift. Applying it per CFA
    channel better respects RAW data statistics.

    Args:
        anchor: Reference RAW image [1, H, W].
        target: Target RAW image to align [1, H, W].
        anchor_metadata: Metadata dict containing 'RGBG_pattern' (2x2 bayer or 6x6 X-Trans).
        method: 'median' or 'mean' for combining channel shifts.
        verbose: If True, print per-channel shift detections.

    Returns:
        - shift: (dy, dx) tuple — final combined shift estimate snapped to CFA blocks.
        - channel_shifts: List of per-channel (dy, dx) detections at sensor resolution.

    Notes:
        - Output is snapped to CFA block boundaries (2 pixels for bayer; 3 for X-Trans).
        - This implementation estimates integer-pixel shifts; subpixel refinement is
          possible but not implemented here.

    References:
        - Kuglin, C. D., & Hines, D. C. (1975). The phase correlation image alignment method.
        - Bayer, B. E. (1976). Color imaging array. US Patent 3,971,065.
        - Suggested reading for X-Trans demosaicing/registration nuances: RawTherapee and
          LibRaw documentation, and OIIO demosaic literature.
    """
    pattern = anchor_metadata["RGBG_pattern"]
    pattern_shape = pattern.shape

    # Auto-detect CFA type and extract channels
    if pattern_shape == (2, 2):
        # bayer CFA
        channels_anchor = extract_bayer_channels(anchor, pattern)
        channels_target = extract_bayer_channels(target, pattern)
        scale_factor = 2
    elif pattern_shape == (6, 6):
        # X-Trans CFA
        channels_anchor = extract_xtrans_channels(anchor, pattern)
        channels_target = extract_xtrans_channels(target, pattern)
        scale_factor = 6
    else:
        raise ValueError(f"Unsupported CFA pattern shape: {pattern_shape}")

    # Correlate each channel pair
    channel_shifts = []
    for color in channels_anchor.keys():
        ch_anchor = channels_anchor[color]
        ch_target = channels_target[color]

        # Handle different return types: bayer=single array, X-Trans=list of arrays
        if isinstance(ch_anchor, list):
            # X-Trans: compare each position grid pair, collect all shift estimates
            for anchor_grid, target_grid in zip(ch_anchor, ch_target):
                dy_down, dx_down = _fft_phase_correlate_single(anchor_grid, target_grid)
                dy = dy_down * scale_factor
                dx = dx_down * scale_factor
                channel_shifts.append((dy, dx))
                if verbose:
                    print(f"  {color:3s} pos: ({dy:3d}, {dx:3d})")
        else:
            # bayer: single array per color
            dy_down, dx_down = _fft_phase_correlate_single(ch_anchor, ch_target)
            dy = dy_down * scale_factor
            dx = dx_down * scale_factor
            channel_shifts.append((dy, dx))
            if verbose:
                print(f"  {color:3s}: ({dy:3d}, {dx:3d})")

    # Combine channel results
    if method == "median":
        dy_final = int(np.median([s[0] for s in channel_shifts]))
        dx_final = int(np.median([s[1] for s in channel_shifts]))
    elif method == "mean":
        dy_final = int(np.mean([s[0] for s in channel_shifts]))
        dx_final = int(np.mean([s[1] for s in channel_shifts]))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Snap to CFA block boundaries
    # bayer: 2x2 blocks, X-Trans: 3x3 processing blocks (per vkdt)
    if pattern_shape == (6, 6):
        # X-Trans: snap to 3x3 blocks
        dy_final = (dy_final // 3) * 3
        dx_final = (dx_final // 3) * 3
    elif pattern_shape == (2, 2):
        # bayer: snap to 2x2 blocks
        dy_final = (dy_final // 2) * 2
        dx_final = (dx_final // 2) * 2

    return (dy_final, dx_final), channel_shifts


def xtrans_fpath_to_OpenEXR(src_fpath: str, dest_fpath: str, output_color_profile: str = OUTPUT_COLOR_PROFILE):
    assert output_color_profile == OUTPUT_COLOR_PROFILE
    assert is_xtrans(src_fpath)
    assert shutil.which("darktable-cli")
    conversion_cmd: tuple = ("darktable-cli", src_fpath,
                             os.path.join(os.path.curdir, "DocScan/rawnind/config", "dt4_xtrans_to_linrec2020.xmp"),
                             dest_fpath, "--core", "--conf", "plugins/imageio/format/exr/bpp=16",)
    subprocess.call(conversion_cmd)


# todo: this function (and any others following the same pattern i'm about to describe) need to be refactored for
# better seperation of concerns - there should not be one function that mutates according to which backend we're
# using, but rather we need a strategy pattern or (some other pattern even more suited to the case at hand) if we want
# to keep the option of using different backends for file I/O. alternative is to cut support for using OpenCV and
# possibly for openEXR - just need to track down whether there is any downside to using OpenImageIO over OpenEXR (
# there was NOT for using it instead of OpenCV when dealing with .tiffs)
def hdr_nparray_to_file(img: Union[np.ndarray, torch.Tensor], fpath: str,
        color_profile: Literal["lin_rec2020", "lin_sRGB", "gamma_sRGB"], bit_depth: Optional[int] = None,
        src_fpath: Optional[str] = None, ) -> None:
    """Write a high dynamic range image to disk (OpenEXR or TIFF).

    Expects channel-first arrays (C, H, W) in linear color. For EXR, floating
    point encodings are used (16-bit HALF or 32-bit FLOAT). For TIFF, OpenCV or
    OpenImageIO is used depending on availability.

    Args:
        img: Image as np.ndarray or torch.Tensor (C, H, W), float16/float32.
        fpath: Destination path; file extension determines container ('.exr' or '.tif/.tiff').
        color_profile: One of {'lin_rec2020', 'lin_sRGB', 'gamma_sRGB'}. For EXR, encodes
            chromaticities metadata when possible. For TIFF, may be ignored depending on backend.
        bit_depth: Desired float bit depth for EXR (16 or 32). If None, inferred from dtype.
        src_fpath: Optional path to source RAW; reserved for future EXIF/XMP metadata copy.

    Raises:
        NotImplementedError: For unsupported dtype/bit_depth combinations or unsupported profile.
        RuntimeError: If the image writer fails to open or write the file.

    Notes:
        - OpenImageIO backend provides robust EXR/TIFF I/O with explicit color space metadata.
        - When writing EXR via OpenEXR, chromaticities for Rec.2020 and linear sRGB are embedded.
        - Images should be in linear light; apply gamma only if exporting 'gamma_sRGB' TIFFs.
    """
    if isinstance(img, torch.Tensor):
        img: np.ndarray = img.numpy()
    if fpath.endswith("exr"):
        if bit_depth is None:
            bit_depth = 16 if img.dtype == np.float16 else 32
        elif bit_depth == 16 and img.dtype != np.float16:
            img = img.astype(np.float16)
        elif bit_depth == 32 and img.dtype != np.float32:
            img = img.astype(np.float32)
        else:
            raise NotImplementedError(f"hdr_nparray_to_file: {bit_depth=} with OpenEXR and {img.dtype=}")
        if OPENEXR_PROVIDER == "OpenImageIO":
            output = oiio.ImageOutput.create(fpath)
            if not output:
                raise RuntimeError(f"Could not create output for {fpath}")
            # Set the format and metadata

            spec = oiio.ImageSpec(img.shape[2], img.shape[1], img.shape[0],
                oiio.HALF if bit_depth == 16 else oiio.FLOAT, )
            if color_profile == "lin_rec2020":
                # Set chromaticities for Rec. 2020
                spec.attribute("oiio:ColorSpace", "Rec2020")
                spec.attribute("chromaticities", oiio.TypeDesc("float[8]"),
                    [0.708, 0.292, 0.17, 0.797, 0.131, 0.046, 0.3127, 0.3290], )
            elif color_profile == "lin_sRGB":
                # Set chromaticities for linear sRGB
                spec.attribute("oiio:ColorSpace", "lin_srgb")
                spec.attribute("chromaticities", oiio.TypeDesc("float[8]"),
                    [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290], )
            else:
                print(f"warning: no color profile for {fpath}")

            # Set compression to ZIPS
            spec.attribute("compression", "zips")
            # Bit depth
            if not bit_depth:
                if img.dtype == np.float16:
                    bit_depth = 16
                elif img.dtype == np.float32:
                    bit_depth = 32
                else:
                    raise NotImplementedError(f"hdr_nparray_to_file: {img.dtype=} with OpenEXR")

            # spec.set_format(oiio.HALF if bit_depth == 16 else oiio.FLOAT)
            # Open the output file and write the image data
            if output.open(fpath, spec):
                success = output.write_image(np.ascontiguousarray(img.transpose(1, 2, 0)))  # .tolist())
                output.close()
                if not success:
                    breakpoint()
                    raise RuntimeError(f"Error writing {fpath}: {output.geterror()} ({img.shape=})")
            else:
                raise RuntimeError(f"Error opening output image: {fpath}")
        elif OPENEXR_PROVIDER == "OpenEXR":
            # Init OpenEXR header
            header = OpenEXR.Header(img.shape[-1], img.shape[-2])
            header["Compression"] = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)
            # Chromaticities
            assert color_profile is None or color_profile.startswith("lin"), (f"{color_profile=}")
            if color_profile == "lin_rec2020":
                header["chromaticities"] = Imath.Chromaticities(Imath.chromaticity(0.708, 0.292),
                    Imath.chromaticity(0.17, 0.797), Imath.chromaticity(0.131, 0.046),
                    Imath.chromaticity(0.3127, 0.3290), )
            elif color_profile == "lin_sRGB":
                header["chromaticities"] = Imath.Chromaticities(Imath.chromaticity(0.64, 0.33),
                    Imath.chromaticity(0.30, 0.60), Imath.chromaticity(0.15, 0.06),
                    Imath.chromaticity(0.3127, 0.3290), )
            elif color_profile is None:
                pass
            else:
                raise NotImplementedError(f"hdr_nparray_to_file: OpenEXR with {color_profile=}")
            if bit_depth == 16:
                header["channels"] = {"R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    "G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    "B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)), }
                np_data_type = np.float16
            elif bit_depth == 32:
                # converting to np.float32 even though it's already the dtype, otherwise
                # *** TypeError: Unsupported buffer structure for channel 'B'
                # with negative values.
                header["channels"] = {"R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)), }
                np_data_type = np.float32
            else:
                raise NotImplementedError(f"hdr_nparray_to_file: OpenEXR with {bit_depth=}")
            # Save
            # TODO include EXIF metadata
            exr = OpenEXR.OutputFile(fpath, header)

            exr.writePixels({"R": img[0].astype(np_data_type), "G": img[1].astype(np_data_type),
                "B": img[2].astype(np_data_type), })
            exr.close()
        else:
            # todo: this should be verified/fail much earlier
            raise NotImplementedError(f"hdr_nparray_to_file: {OPENEXR_PROVIDER=}")
    else:  # not .exr so must be .tiff. is that actually a safe assumption?
        output = oiio.ImageOutput.create(fpath)
        if not output:
            raise RuntimeError(f"Could not create output for {fpath}")
        # Set the format and metadata

        spec = oiio.ImageSpec(img.shape[2], img.shape[1], img.shape[0], oiio.HALF if bit_depth == 16 else oiio.FLOAT, )
        if bit_depth == 16:
            spec.attribute("tiff:half", 1)
        if color_profile == "lin_rec2020":
            # Set chromaticities for Rec. 2020
            spec.attribute("chromaticities", oiio.TypeDesc("float[8]"),
                [0.708, 0.292, 0.17, 0.797, 0.131, 0.046, 0.3127, 0.3290], )
            spec.attribute("oiio:ColorSpace", "Rec2020")
            spec.attribute("ICCProfile", oiio.TypeDesc("uint8[904]"),
                           icc.rec2020)  # with open(  #     os.path.join("..", "common", "cfg", "icc", "rec2020.icc"),  #     "rb",  # ) as f:  #     spec.attribute("ICCProfile", f.read())  # load ICC profile from ../common/cfg/icc/ITU-R_BT2020.icc  # spec.attribute(  #     "ICCProfile",  #     np.fromfile(  #         os.path.join("..", "common", "cfg", "icc", "ITU-R_BT2020.icc"),  #         dtype="uint8",  #     ),  # )
        elif color_profile == "lin_sRGB":
            # Set chromaticities for linear sRGB
            spec.attribute("oiio:ColorSpace", "lin_srgb")
        else:
            print(f"warning: no color profile for {fpath}")
        assert img.dtype == np.float16 or img.dtype == np.float32, img.dtype
        if output.open(fpath, spec):
            success = output.write_image(np.ascontiguousarray(img.transpose(1, 2, 0)))  # .tolist())
            output.close()
            if not success:
                breakpoint()
                raise RuntimeError(f"Error writing {fpath}: {output.geterror()} ({img.shape=})")
        else:
            raise RuntimeError(f"Error opening output image: {fpath}")
    # copy metadata using exiftool if it exists and src_fpath is provided
    if src_fpath and shutil.which("exiftool"):
        subprocess.call(["exiftool", "-overwrite_original", "-TagsFromFile", src_fpath, fpath])


def raw_fpath_to_hdr_img_file(src_fpath: str, dest_fpath: str,
        output_color_profile: Literal["lin_rec2020", "lin_sRGB"] = OUTPUT_COLOR_PROFILE,
        bit_depth: Optional[int] = None, check_exposure: bool = True, crop_all: bool = True, ) -> tuple[
    ConversionOutcome, str, str]:
    """Convert a RAW file into an HDR image with color management.

    Pipeline:
    1) Read RAW file to mosaiced mono image and metadata.
    2) Optionally verify exposure quality (reject under/over-exposed frames).
    3) Demosaic to camera RGB.
    4) Convert camera RGB to the requested linear output color space.
    5) Write HDR image (EXR/TIFF) with appropriate metadata.

    Args:
        src_fpath: Input RAW file path.
        dest_fpath: Output HDR file path ('.exr' or '.tif/.tiff').
        output_color_profile: Target color space, typically 'lin_rec2020' or 'lin_sRGB'.
        bit_depth: For EXR, floating point bit depth (16 or 32). If None, inferred from dtype.
        check_exposure: If True, reject files whose exposure quality is poor.
        crop_all: If True, crop to valid sensor area per metadata.

    Returns:
        ConversionResult tuple: (outcome, src_fpath, dest_fpath).

    Raises:
        Various exceptions from underlying readers/writers are caught and mapped to
        ConversionOutcome codes; unexpected errors return UNKNOWN_ERROR.

    References:
        - Malvar, H. S., He, L.-W., & Cutler, R. (2004). High-Quality Linear Interpolation
          for Demosaicing of Bayer-Patterned Color Images. IEEE ICASSP 2004.
        - Bruce Lindbloom, Color Science: XYZ⇄RGB conversions (for color matrix context).
    """

    def log(msg):
        """Multiprocessing-safe logging."""
        try:
            logging.info(msg)
        except NameError:
            print(msg)

    try:
        img, metadata = raw_fpath_to_mono_img_and_metadata(src_fpath, crop_all=crop_all, return_float=True)
        if check_exposure and not is_exposure_ok(img, metadata):
            log(f"# bad exposure for {src_fpath} ({img.mean()=})")
            return ConversionResult(ConversionOutcome.BAD_EXPOSURE, src_fpath, dest_fpath)
        img = demosaic(img, metadata)
        img = camRGB_to_profiledRGB_img(img, metadata, output_color_profile=output_color_profile)
    except Exception as e:
        if (isinstance(e, AssertionError) or isinstance(e, rawpy._rawpy.LibRawFileUnsupportedError) or isinstance(e,
                                                                                                                  rawpy._rawpy.LibRawIOError)):
            log(f"# Unable to read {src_fpath=}, {e=}")
            return ConversionResult(ConversionOutcome.UNREADABLE_ERROR, src_fpath, dest_fpath)
        else:
            log(f"# Unknown error {e} with {src_fpath}")
            return ConversionResult(ConversionOutcome.UNKNOWN_ERROR, src_fpath, dest_fpath)
    hdr_nparray_to_file(img, dest_fpath, OUTPUT_COLOR_PROFILE, bit_depth, src_fpath)
    log(f"# Wrote {dest_fpath}")
    return ConversionResult(ConversionOutcome.OK, src_fpath, dest_fpath)


def raw_fpath_to_hdr_img_file_mtrunner(argslist):
    return raw_fpath_to_hdr_img_file(*argslist)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--raw_fpath", help="Input image file path.")
    parser.add_argument("-o", "--out_base_path", help="Output image base file path.")
    parser.add_argument("--no_wb", action="store_true", help="No white balance is applied")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not args.raw_fpath:
        args.raw_fpath = get_sample_raw_file(url=SAMPLE_RAW_URL)
    if not args.out_base_path:
        args.out_base_path = os.path.join("tests_output", "raw.py.main")
    # prepare image as neural network input
    mono_img, metadata = raw_fpath_to_mono_img_and_metadata(args.raw_fpath)
    print(f"raw.py: opened {args.raw_fpath} with {metadata=}\n")
    if args.no_wb:
        nn_input_rggb_img = mono_to_rggb_img(mono_img, metadata)
        # prepare image as neural network ground-truth
        camRGB_img = demosaic(mono_img, metadata)  # NN GT
        camRGB_img_nowb = camRGB_img

    else:
        mono_img_wb = apply_whitebalance(mono_img, metadata, wb_type="daylight", in_place=False)
        nn_input_rggb_img = mono_to_rggb_img(mono_img_wb, metadata)
        # prepare image as neural network ground-truth
        camRGB_img = demosaic(mono_img_wb, metadata)  # NN GT
        camRGB_img_nowb = apply_whitebalance(camRGB_img, metadata, wb_type="daylight", in_place=False, reverse=True)
    # output to file for visualization
    lin_sRGB_img = camRGB_to_profiledRGB_img(camRGB_img_nowb, metadata, "lin_sRGB")
    lin_rec2020_img = camRGB_to_profiledRGB_img(camRGB_img_nowb, metadata, "lin_rec2020")
    gamma_sRGB_img = camRGB_to_profiledRGB_img(camRGB_img_nowb, metadata, "gamma_sRGB")
    os.makedirs("tests_output", exist_ok=True)
    hdr_nparray_to_file(lin_rec2020_img, args.out_base_path + ".lin_rec2020.exr", color_profile="lin_rec2020", )
    hdr_nparray_to_file(lin_sRGB_img, args.out_base_path + ".lin_sRGB.exr", color_profile="lin_sRGB")

    hdr_nparray_to_file(gamma_sRGB_img, args.out_base_path + ".gamma_sRGB.tif", color_profile="gamma_sRGB", )
    hdr_nparray_to_file(img=camRGB_img, fpath=args.out_base_path + ".camRGB.exr", color_profile=None)
    libraw_process(args.raw_fpath, args.out_base_path + ".libraw.tif")
    print(f"raw.py: output images saved to {args.out_base_path}.*")
