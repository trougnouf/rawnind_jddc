"""Basic raw handling library; convert raw camRGB file to numpy array with LibRaw.

This file also handles HDR image export.
"""

from __future__ import annotations
import os
import Imath  # OpenEXR
import requests
import argparse
import operator
import subprocess
import shutil
from collections import OrderedDict
from enum import Enum
from typing import Literal, NamedTuple, Optional, Union
import numpy as np
import rawpy

from common.libs import icc


try:
    import OpenImageIO as oiio
    OPENEXR_PROVIDER = "OpenImageIO"
except ImportError:
    try:
        import OpenEXR
        OPENEXR_PROVIDER = "OpenEXR"
    except ImportError:
        raise ImportError("OpenImageIO or OpenEXR must be installed")

# import multiprocessing
# multiprocessing.set_start_method('spawn')
import cv2
#cv2.setNumThreads(0)

try:
    import OpenImageIO as oiio
    TIFF_PROVIDER = "OpenImageIO"
except ImportError:
    print(
        "raw.py warning: OpenImageIO not found, using OpenCV for TIFFs. Install OpenImageIO for better TIFF support."
    )
    TIFF_PROVIDER = "OpenCV"
import matplotlib.pyplot as plt
import torch  # for typing only


# import colour
try:
    import imageio  # used in libraw_process sample, should be easy to replace
except ImportError:
    print("raw.py warning: imageio not found, libraw_process will not work")

SAMPLE_RAW_URL = (
    "https://nc.trougnouf.com/index.php/s/zMA8QfgPoNoByex/download/DSC01568.ARW"  # RGGB
    # "https://nc.trougnouf.com/index.php/s/9eWFyKsitoNGqQS/download/_M7D5407.CR2"  # GBRG
    # "https://nc.trougnouf.com/index.php/s/zMA8QfgPoNoByex/download/DSC01526.ARW"  # RGGB
)
SAMPLE_RAW_FPATH = os.path.join("data", SAMPLE_RAW_URL.rpartition("/")[-1])
OUTPUT_COLOR_PROFILE = "lin_rec2020"


def raw_fpath_to_mono_img_and_metadata(
    fpath: str,
    force_rggb: bool = True,
    crop_all: bool = True,
    return_float: bool = True,
) -> tuple[np.ndarray, dict]:
    """Convert from raw fpath to (tuple) [1,h,w] numpy array and metadata dictionary."""

    def mono_any_to_mono_rggb(
        mono_img: np.ndarray, metadata: dict, whole_image_raw_pattern
    ) -> np.ndarray:
        """
        Convert (crop) any Bayer pattern to RGGB.

        Assumes cropped margins (with rm_empty_borders).

        args:
            force_rggb: convert (crop) any Bayer pattern to RGGB
            crop_all: crop raw_dimensions to dimensions declared in metadata
        """
        if metadata["bayer_pattern"] == "RGGB":
            pass
        else:
            if not (
                metadata["sizes"]["top_margin"] == metadata["sizes"]["left_margin"] == 0
            ):
                raise NotImplementedError(
                    f'{metadata["sizes"]=}, {metadata["bayer_pattern"]=} with borders'
                )
            if metadata["bayer_pattern"] == "GBRG":
                assert (
                    metadata["RGBG_pattern"] == [[3, 2], [0, 1]]
                ).all(), f"{metadata['RGBG_pattern']=}"
                mono_img = mono_img[:, 1:-1]
                # metadata["cropped_y"] = True
                whole_image_raw_pattern = whole_image_raw_pattern[1:-1]
                metadata["sizes"]["raw_height"] -= 2
                metadata["sizes"]["height"] -= 2
                metadata["sizes"]["iheight"] -= 2
            elif metadata["bayer_pattern"] == "BGGR":
                assert (
                    metadata["RGBG_pattern"] == [[2, 3], [1, 0]]
                ).all(), f"{metadata['RGBG_pattern']=}"
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
                assert (
                    metadata["RGBG_pattern"] == [[1, 0], [2, 3]]
                ).all(), f"{metadata['RGBG_pattern']=}"
                mono_img = mono_img[:, :, 1:-1]
                # metadata["cropped_x"] = True
                whole_image_raw_pattern = whole_image_raw_pattern[:, 1:-1]
                metadata["sizes"]["raw_width"] -= 2
                metadata["sizes"]["width"] -= 2
                metadata["sizes"]["iwidth"] -= 2
            else:
                raise NotImplementedError(f'{metadata["bayer_pattern"]=}')
        # try:  # DBG
        assert (
            metadata["sizes"]["raw_height"] >= metadata["sizes"]["height"]
        ), f"Wrong height: {metadata['sizes']=}"
        assert (
            metadata["sizes"]["raw_width"] >= metadata["sizes"]["width"]
        ), f"Wrong width: {metadata['sizes']=}"
        # except AssertionError as e:
        #    print(e)
        #    breakpoint()
        metadata["RGBG_pattern"] = whole_image_raw_pattern[:2, :2]
        set_bayer_pattern_name(metadata)
        assert metadata["bayer_pattern"] == "RGGB", f'{metadata["bayer_pattern"]=}'
        return mono_img, whole_image_raw_pattern

    def rm_empty_borders(
        mono_img: np.ndarray,
        metadata: dict,
        whole_image_raw_pattern: np.ndarray,
        crop_all: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove empty borders declared in the metadata.

        args:
            mono_img: 1 x h x w
            metadata: contains top_margin, left_margin
            whole_image_raw_pattern: h x w
        """

        if metadata["sizes"]["top_margin"] or metadata["sizes"]["left_margin"]:
            mono_img = mono_img[
                :, metadata["sizes"]["top_margin"] :, metadata["sizes"]["left_margin"] :
            ]
            metadata["sizes"]["raw_height"] -= metadata["sizes"]["top_margin"]
            metadata["sizes"]["raw_width"] -= metadata["sizes"]["left_margin"]
            whole_image_raw_pattern = whole_image_raw_pattern[
                metadata["sizes"]["top_margin"] :, metadata["sizes"]["left_margin"] :
            ]
            metadata["sizes"]["top_margin"] = metadata["sizes"]["left_margin"] = 0
        if crop_all:
            _, h, w = mono_img.shape
            min_h = min(
                h,
                metadata["sizes"]["height"],
                metadata["sizes"]["iheight"],
                metadata["sizes"]["raw_height"],
            )
            min_w = min(
                w,
                metadata["sizes"]["width"],
                metadata["sizes"]["iwidth"],
                metadata["sizes"]["raw_width"],
            )
            h = metadata["sizes"]["height"] = metadata["sizes"]["iheight"] = metadata[
                "sizes"
            ]["raw_height"] = min_h
            w = metadata["sizes"]["width"] = metadata["sizes"]["iwidth"] = metadata[
                "sizes"
            ]["raw_width"] = min_w
            mono_img = mono_img[:, :h, :w]

        assert mono_img.shape[1:] == (
            metadata["sizes"]["raw_height"],
            metadata["sizes"]["raw_width"],
        ), f'{mono_img.shape[1:]=}, {metadata["sizes"]=}'
        metadata["RGBG_pattern"] = whole_image_raw_pattern[:2, :2]
        set_bayer_pattern_name(metadata)
        return mono_img, whole_image_raw_pattern

    def set_bayer_pattern_name(metadata: dict):
        """Set bayer_pattern string from RGBG (raw_pattern) indices."""
        metadata["bayer_pattern"] = "".join(
            ["RGBG"[i] for i in metadata["RGBG_pattern"].flatten()]
        )

    def ensure_correct_shape(
        mono_img: np.ndarray, metadata: dict, whole_image_raw_pattern: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure dimension % 4 == 0."""
        _, h, w = mono_img.shape
        assert (metadata["sizes"]["raw_height"], metadata["sizes"]["raw_width"]) == (
            h,
            w,
        )
        # crop raw width/height if bigger than width/height
        if h % 4 > 0:
            mono_img = mono_img[:, : -(h % 4)]
            metadata["sizes"]["raw_height"] -= h % 4
            metadata["sizes"]["height"] -= h % 4
            metadata["sizes"]["iheight"] -= h % 4
        if w % 4 > 0:
            mono_img = mono_img[:, :, : -(w % 4)]
            metadata["sizes"]["raw_width"] -= w % 4
            metadata["sizes"]["width"] -= w % 4
            metadata["sizes"]["iwidth"] -= w % 4

        assert not (mono_img.shape[1] % 4 or mono_img.shape[2] % 4), mono_img.shape
        return mono_img, whole_image_raw_pattern

    # step 1: get raw data and metadata
    try:
        rawpy_img = rawpy.imread(fpath)
    except rawpy._rawpy.LibRawFileUnsupportedError as e:
        raise ValueError(
            f"raw.raw_fpath_to_mono_img_and_metadata error opening {fpath}: {e}"
        )
    except rawpy._rawpy.LibRawIOError as e:
        raise ValueError(
            f"raw.raw_fpath_to_mono_img_and_metadata error opening {fpath}: {e}"
        )
    metadata = dict()
    metadata["camera_whitebalance"] = rawpy_img.camera_whitebalance
    metadata["black_level_per_channel"] = rawpy_img.black_level_per_channel
    metadata["white_level"] = rawpy_img.white_level  # imgdata.rawdata.color.maximum
    metadata[
        "camera_white_level_per_channel"
    ] = rawpy_img.camera_white_level_per_channel  # imgdata.rawdata.color.linear_max
    #  daylight_whitebalance = float rawdata.color.pre_mul[4]; in libraw
    #  White balance coefficients for daylight (daylight balance). Either read
    #  from file, or calculated on the basis of file data, or taken from
    metadata["daylight_whitebalance"] = rawpy_img.daylight_whitebalance
    #  float cam_xyz[4][3];
    #  Camera RGB - XYZ conversion matrix. This matrix is constant (different
    #  for different models). Last row is zero for RGB cameras and non-zero for
    #  different color models (CMYG and so on).
    metadata["rgb_xyz_matrix"] = rawpy_img.rgb_xyz_matrix
    assert metadata[
        "rgb_xyz_matrix"
    ].any(), f"rgb_xyz_matrix of {fpath} is empty ({metadata=})"
    metadata["sizes"] = rawpy_img.sizes._asdict()
    assert (
        rawpy_img.color_desc.decode() == "RGBG"
    ), f"{fpath} does not seem to have bayer pattern ({rawpy_img.color_desc.decode()})"
    metadata["RGBG_pattern"] = rawpy_img.raw_pattern
    assert (
        metadata["RGBG_pattern"] is not None
    ), f"{fpath} has no bayer pattern information"
    # processed metadata:
    # Replace raw_pattern from libraw (which always returns RGBG)
    set_bayer_pattern_name(metadata)

    whole_image_raw_pattern = rawpy_img.raw_colors
    assert_correct_metadata = (
        metadata["RGBG_pattern"] == whole_image_raw_pattern[:2, :2]
    )
    assert (
        assert_correct_metadata
        if isinstance(assert_correct_metadata, bool)
        else assert_correct_metadata.all()
    ), f"Bayer pattern decoding did not match ({fpath=}, {metadata['RGBG_pattern']=}, {whole_image_raw_pattern[:2, :2]=})"
    for a_wb in ("daylight", "camera"):
        metadata[f"{a_wb}_whitebalance_norm"] = np.array(
            metadata[f"{a_wb}_whitebalance"], dtype=np.float32
        )
        if metadata[f"{a_wb}_whitebalance_norm"][3] == 0:
            metadata[f"{a_wb}_whitebalance_norm"][3] = metadata[
                f"{a_wb}_whitebalance_norm"
            ][1]
        metadata[f"{a_wb}_whitebalance_norm"] /= metadata[f"{a_wb}_whitebalance_norm"][
            1
        ]
    mono_img = rawpy_img.raw_image
    mono_img = np.expand_dims(mono_img, axis=0)
    mono_img, whole_image_raw_pattern = rm_empty_borders(
        mono_img, metadata, whole_image_raw_pattern, crop_all=crop_all
    )
    if force_rggb:
        mono_img, whole_image_raw_pattern = mono_any_to_mono_rggb(
            mono_img, metadata, whole_image_raw_pattern
        )
    mono_img, whole_image_raw_pattern = ensure_correct_shape(
        mono_img, metadata, whole_image_raw_pattern
    )
    if return_float:
        mono_img = scale_img_to_bw_points(mono_img, metadata)
    return mono_img, metadata


def libraw_process(raw_fpath: str, out_fpath) -> None:
    """Sanity check uses libraw/dcraw. Actually does not return a desired output."""
    rawpy_img = rawpy.imread(raw_fpath)

    params = rawpy.Params(
        gamma=(1, 1),
        demosaic_algorithm=rawpy.DemosaicAlgorithm.VNG,
        use_camera_wb=False,
        use_auto_wb=False,
        output_color=rawpy.ColorSpace.sRGB,  # XYZ here looks closer ...
        no_auto_bright=True,
        # no_auto_scale=True, # dark, I guess using 14 bits out of 16 ?
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
        raise NotImplementedError(f'{metadata["bayer_pattern"]=} is not RGGB')
    assert mono_img.shape[-3] == 1, f"not a mono image ({mono_img.shape})"
    mono_img = mono_img[..., 0, :, :]
    # step 2: convert to 4-channels RGGB
    rggb_img = np.array(
        [
            # RGGB
            mono_img[..., 0::2, 0::2],  # R
            mono_img[..., 0::2, 1::2],  # G
            mono_img[..., 1::2, 0::2],  # G
            mono_img[..., 1::2, 1::2],  # B
        ],
        dtype=np.float32,
    )
    return rggb_img


def rggb_to_mono_img(rggb_img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    assert rggb_img.shape[-3] == 4, rggb_img.shape
    mono_img = np.empty_like(
        rggb_img,
        shape=(*rggb_img.shape[:-3], 1, rggb_img.shape[-2] * 2, rggb_img.shape[-1] * 2),
    )
    mono_img[..., 0::2, 0::2] = rggb_img[..., 0:1, :, :]
    mono_img[..., 0::2, 1::2] = rggb_img[..., 1:2, :, :]
    mono_img[..., 1::2, 0::2] = rggb_img[..., 2:3, :, :]
    mono_img[..., 1::2, 1::2] = rggb_img[..., 3:4, :, :]
    return mono_img


def scale_img_to_bw_points(
    img: np.ndarray, metadata: dict, compat: bool = True
) -> np.ndarray:
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
                metadata["overexposure_lb"] = min(
                    metadata["overexposure_lb"],
                    (
                        metadata["camera_white_level_per_channel"][ch]
                        - metadata["black_level_per_channel"][ch]
                    )
                    / vrange,
                )
        else:
            vrange = (
                metadata["camera_white_level_per_channel"][ch]
                - metadata["black_level_per_channel"][ch]
            )
        scaled_img[ch] /= vrange
    return scaled_img


def apply_whitebalance(
    img: np.ndarray,
    metadata: dict,
    wb_type: str = "daylight",
    in_place: bool = True,
    reverse: bool = False,
) -> None:
    """
    Apply white balance.

    wb_type values: daylight or camera.
    reverse: undo white balance.
    """
    op = operator.truediv if reverse else operator.mul
    # step 5: apply camera reference white balance
    assert f"{wb_type}_whitebalance_norm" in metadata, f"{wb_type=}, {metadata=}"
    if metadata["bayer_pattern"] != "RGGB" and img.shape[-3] != 3:
        raise NotImplementedError(f'{metadata["bayer_pattern"]=} is not RGGB')
    if not in_place:
        img = img.copy()
    if img.shape[-3] == 1:  # mono
        # RGGB
        img[0, 0::2, 0::2] = op(
            img[0, 0::2, 0::2], metadata[f"{wb_type}_whitebalance_norm"][0]
        )  # R
        img[0, 0::2, 1::2] = op(
            img[0, 0::2, 1::2], metadata[f"{wb_type}_whitebalance_norm"][1]
        )  # G
        img[0, 1::2, 1::2] = op(
            img[0, 1::2, 1::2], metadata[f"{wb_type}_whitebalance_norm"][2]
        )  # B
        img[0, 1::2, 0::2] = op(
            img[0, 1::2, 0::2], metadata[f"{wb_type}_whitebalance_norm"][3]
        )  # G

    else:
        for ch_img, ch_wb in enumerate(
            range(3) if img.shape[-3] == 3 else (0, 1, 3, 2)
        ):
            img[ch_img] = op(
                img[ch_img], metadata[f"{wb_type}_whitebalance_norm"][ch_wb]
            )
    # assert img.max() <= 1.0, img.max()

    if not in_place:
        return img


def raw_fpath_to_rggb_img_and_metadata(fpath: str, return_float: bool = True):
    mono_img, metadata = raw_fpath_to_mono_img_and_metadata(
        fpath, return_float=return_float
    )
    return mono_to_rggb_img(mono_img, metadata), metadata


def demosaic(
    mono_img: np.ndarray, metadata: dict, method=cv2.COLOR_BayerRGGB2RGB_EA
) -> np.ndarray:
    """
    Transform mono image to camRGB colors.

    Debayering methods include COLOR_BayerRGGB2RGB, COLOR_BayerRGGB2RGB_EA.
    """
    assert method in (
        cv2.COLOR_BayerRGGB2RGB,
        cv2.COLOR_BayerRGGB2RGB_EA,
    ), f"Wrong debayering method: {method}"
    assert mono_img.shape[0] == 1, f"{mono_img.shape=}"
    assert metadata["bayer_pattern"] == "RGGB", f'{metadata["bayer_pattern"]=}'
    mono_img: np.ndarray = mono_img.copy()
    dbg_img = mono_img.copy()
    # convert to uint16 and scale to ensure we don't lose negative / large values
    black_offset: float = 0.0 if mono_img.min() >= 0.0 else -mono_img.min()
    mono_img += black_offset
    assert (
        mono_img >= 0
    ).all(), f"{black_offset=}, {dbg_img.min()=}, {mono_img.min()=}"
    max_value: float = 1.0 if mono_img.max() <= 1.0 else mono_img.max()
    mono_img /= max_value
    try:
        assert mono_img.min() >= 0 and mono_img.max() <= 1.0, (
            f"demosaic: image is out of bound; destructive operation. {mono_img.min()=}, "
            f"{mono_img.max()=}"
        )
    except AssertionError as e:
        print(f"{e}; {dbg_img.min()=}, {dbg_img.max()=}")
        breakpoint()
    mono_img *= 65535
    mono_img = mono_img.astype(np.uint16).reshape(mono_img.shape[1:] + (1,))
    rgb_img = cv2.demosaicing(mono_img, method)
    rgb_img = rgb_img.transpose(2, 0, 1)  #  opencv h, w, ch to numpy ch, h, w
    rgb_img = rgb_img.astype(np.float32) / 65535.0 * max_value - black_offset
    return rgb_img


def get_XYZ_to_profiledRGB_matrix(profile: str) -> np.ndarray:
    """Returns a static XYZ->profile matrix."""
    if profile == "lin_rec2020":
        return np.array(
            [
                [1.71666343, -0.35567332, -0.25336809],
                [-0.66667384, 1.61645574, 0.0157683],
                [0.01764248, -0.04277698, 0.94224328],
            ],
            dtype=np.float32,
        )
    elif "sRGB" in profile:
        # conversion_matrix = colour.models.dataset.srgb.XYZ_TO_sRGB_MATRIX
        return np.array(
            [
                [3.24100326, -1.53739899, -0.49861587],
                [-0.96922426, 1.87592999, 0.04155422],
                [0.05563942, -0.2040112, 1.05714897],
            ],
            dtype=np.float32,
        )
    else:
        raise NotImplementedError(
            f"get_std_profile_matrix: {profile} not *_sRGB or lin_rec2020."
        )


def get_camRGB_to_profiledRGB_img_matrix(
    metadata: dict, output_color_profile: str
) -> np.ndarray:
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


def camRGB_to_profiledRGB_img(
    camRGB_img: np.ndarray, metadata: dict, output_color_profile: str
) -> np.ndarray:
    """Convert camRGB debayered image to a given RGB color profile (in-place)."""
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
        profiledRGB_img[img_mask] = (
            1.055 * np.power(profiledRGB_img[img_mask], 1.0 / 2.4) - 0.055
        )
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


def is_exposure_ok(
    mono_float_img: np.ndarray,
    metadata: dict,
    oe_threshold=0.99,
    ue_threshold=0.001,
    qty_threshold=0.75,
) -> bool:
    """Check that the image exposure is useable in all channels."""
    rggb_img = mono_to_rggb_img(mono_float_img, metadata)
    overexposed = (rggb_img >= oe_threshold * metadata["overexposure_lb"]).any(0)
    if ue_threshold > 0:
        underexposed = (rggb_img <= ue_threshold).all(0)
        return (overexposed + underexposed).sum() / overexposed.size <= qty_threshold
    return overexposed.sum() / overexposed.size <= qty_threshold


ConversionOutcome = Enum(
    "ConversionOutcome", "OK BAD_EXPOSURE UNREADABLE_ERROR UNKNOWN_ERROR"
)
ConversionResult = NamedTuple(
    "ConversionResult",
    [("outcome", ConversionOutcome), ("src_fpath", str), ("dest_fpath", str)],
)


def is_xtrans(fpath) -> bool:
    return fpath.lower().endswith(".raf")


def xtrans_fpath_to_OpenEXR(
    src_fpath: str, dest_fpath: str, output_color_profile: str = OUTPUT_COLOR_PROFILE
):
    assert output_color_profile == OUTPUT_COLOR_PROFILE
    assert is_xtrans(src_fpath)
    assert shutil.which("darktable-cli")
    conversion_cmd: tuple = (
        "darktable-cli",
        src_fpath,
        os.path.join("config", "dt4_xtrans_to_linrec2020.xmp"),
        dest_fpath,
        "--core",
        "--conf",
        "plugins/imageio/format/exr/bpp=16",
    )
    subprocess.call(conversion_cmd)


def hdr_nparray_to_file(
    img: Union[np.ndarray, torch.Tensor],
    fpath: str,
    color_profile: Literal["lin_rec2020", "lin_sRGB", "gamma_sRGB"],
    bit_depth: Optional[int] = None,
    src_fpath: Optional[str] = None,
) -> None:
    """Save (c,h,w) numpy array to HDR image file. (OpenEXR or TIFF)

    src_fpath can be used to copy metadata over using exiftool.
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
            raise NotImplementedError(
                f"hdr_nparray_to_file: {bit_depth=} with OpenEXR and {img.dtype=}"
            )
        if OPENEXR_PROVIDER == "OpenImageIO":
            output = oiio.ImageOutput.create(fpath)
            if not output:
                raise RuntimeError(f"Could not create output for {fpath}")
            # Set the format and metadata

            spec = oiio.ImageSpec(
                img.shape[2],
                img.shape[1],
                img.shape[0],
                oiio.HALF if bit_depth == 16 else oiio.FLOAT,
            )
            if color_profile == "lin_rec2020":
                # Set chromaticities for Rec. 2020
                spec.attribute("oiio:ColorSpace", "Rec2020")
                spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), [0.708, 0.292, 0.17, 0.797, 0.131, 0.046, 0.3127, 0.3290])
            elif color_profile == "lin_sRGB":
                # Set chromaticities for linear sRGB
                spec.attribute("oiio:ColorSpace", "lin_srgb")
                spec.attribute('chromaticities', oiio.TypeDesc("float[8]"), [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290])
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
                    raise NotImplementedError(
                        f"hdr_nparray_to_file: {img.dtype=} with OpenEXR"
                    )

            # spec.set_format(oiio.HALF if bit_depth == 16 else oiio.FLOAT)
            # Open the output file and write the image data
            if output.open(fpath, spec):
                success = output.write_image(
                    np.ascontiguousarray(img.transpose(1, 2, 0))
                )  # .tolist())
                output.close()
                if not success:
                    breakpoint()
                    raise RuntimeError(
                        f"Error writing {fpath}: {output.geterror()} ({img.shape=})"
                    )
            else:
                raise RuntimeError(f"Error opening output image: {fpath}")
        elif OPENEXR_PROVIDER == "OpenEXR":
            # Init OpenEXR header
            header = OpenEXR.Header(img.shape[-1], img.shape[-2])
            header["Compression"] = Imath.Compression(
                Imath.Compression.ZIPS_COMPRESSION
            )
            # Chromaticities
            assert color_profile is None or color_profile.startswith(
                "lin"
            ), f"{color_profile=}"
            if color_profile == "lin_rec2020":
                header["chromaticities"] = Imath.Chromaticities(
                    Imath.chromaticity(0.708, 0.292),
                    Imath.chromaticity(0.17, 0.797),
                    Imath.chromaticity(0.131, 0.046),
                    Imath.chromaticity(0.3127, 0.3290),
                )
            elif color_profile == "lin_sRGB":
                header["chromaticities"] = Imath.Chromaticities(
                    Imath.chromaticity(0.64, 0.33),
                    Imath.chromaticity(0.30, 0.60),
                    Imath.chromaticity(0.15, 0.06),
                    Imath.chromaticity(0.3127, 0.3290),
                )
            elif color_profile is None:
                pass
            else:
                raise NotImplementedError(
                    f"hdr_nparray_to_file: OpenEXR with {color_profile=}"
                )
            if bit_depth == 16:
                header["channels"] = {
                    "R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    "G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    "B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                }
                np_data_type = np.float16
            elif bit_depth == 32:
                # converting to np.float32 even though it's already the dtype, otherwise
                # *** TypeError: Unsupported buffer structure for channel 'B'
                # with negative values.
                header["channels"] = {
                    "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                }
                np_data_type = np.float32
            else:
                raise NotImplementedError(
                    f"hdr_nparray_to_file: OpenEXR with {bit_depth=}"
                )
            # Save
            # TODO include EXIF metadata
            exr = OpenEXR.OutputFile(fpath, header)

            exr.writePixels(
                {
                    "R": img[0].astype(np_data_type),
                    "G": img[1].astype(np_data_type),
                    "B": img[2].astype(np_data_type),
                }
            )
            exr.close()
        else:
            raise NotImplementedError(f"hdr_nparray_to_file: {OPENEXR_PROVIDER=}")
    else:
        if TIFF_PROVIDER == "OpenCV":
            if img.dtype == np.float32 and (img.min() <= 0 or img.max() >= 1):
                raise NotImplementedError(
                    f"hdr_nparray_to_file warning: DATA LOSS: image range out of bounds "
                    f"({img.min()=}, {img.max()=}). Consider using OpenImageIO or saving {fpath=} to "
                    "OpenEXR in order to maintain data integrity."
                )
            if color_profile != "gamma_sRGB":
                print(
                    f"hdr_nparray_to_file warning: {color_profile=} not saved to "
                    f"{fpath=}. Viewer will wrongly assume sRGB."
                )
            hwc_img = img.transpose(1, 2, 0)
            hwc_img = cv2.cvtColor(hwc_img, cv2.COLOR_RGB2BGR)
            hwc_img = (hwc_img * 65535).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(fpath, hwc_img)
        elif TIFF_PROVIDER == "OpenImageIO":
            output = oiio.ImageOutput.create(fpath)
            if not output:
                raise RuntimeError(f"Could not create output for {fpath}")
            # Set the format and metadata

            spec = oiio.ImageSpec(
                img.shape[2],
                img.shape[1],
                img.shape[0],
                oiio.HALF if bit_depth == 16 else oiio.FLOAT,
            )
            if bit_depth == 16:
                spec.attribute("tiff:half", 1)
            if color_profile == "lin_rec2020":
                # Set chromaticities for Rec. 2020
                spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), [0.708, 0.292, 0.17, 0.797, 0.131, 0.046, 0.3127, 0.3290])
                spec.attribute("oiio:ColorSpace", "Rec2020")
                spec.attribute("ICCProfile", oiio.TypeDesc("uint8[904]"), icc.rec2020)
                # with open(
                #     os.path.join("..", "common", "cfg", "icc", "rec2020.icc"),
                #     "rb",
                # ) as f:
                #     spec.attribute("ICCProfile", f.read())
                # load ICC profile from ../common/cfg/icc/ITU-R_BT2020.icc
                # spec.attribute(
                #     "ICCProfile",
                #     np.fromfile(
                #         os.path.join("..", "common", "cfg", "icc", "ITU-R_BT2020.icc"),
                #         dtype="uint8",
                #     ),
                # )
            elif color_profile == "lin_sRGB":
                # Set chromaticities for linear sRGB
                spec.attribute("oiio:ColorSpace", "lin_srgb")
            else:
                print(f"warning: no color profile for {fpath}")
            assert img.dtype == np.float16 or img.dtype == np.float32, img.dtype
            if output.open(fpath, spec):
                success = output.write_image(
                    np.ascontiguousarray(img.transpose(1, 2, 0))
                )  # .tolist())
                output.close()
                if not success:
                    breakpoint()
                    raise RuntimeError(
                        f"Error writing {fpath}: {output.geterror()} ({img.shape=})"
                    )
            else:
                raise RuntimeError(f"Error opening output image: {fpath}")

        else:
            raise NotImplementedError(f"hdr_nparray_to_file: {TIFF_PROVIDER=}")
    # copy metadata using exiftool if it exists and src_fpath is provided
    if src_fpath and shutil.which("exiftool"):
        subprocess.call(
            ["exiftool", "-overwrite_original", "-TagsFromFile", src_fpath, fpath]
        )


def raw_fpath_to_hdr_img_file(
    src_fpath: str,
    dest_fpath: str,
    output_color_profile: Literal['lin_rec2020', 'lin_sRGB'] = OUTPUT_COLOR_PROFILE,
    bit_depth: Optional[int] = None,
    check_exposure: bool = True,
    crop_all: bool = True,
) -> tuple[ConversionOutcome, str, str]:
    """
    Converts a raw file to OpenEXR or TIFF HDR.

    if check_exposure: will not perform conversion if image is under/over-exposed

    Returns (ConversionOutcome, src_fpath, dest_fpath)
    """

    def log(msg):
        """Multiprocessing-safe logging."""
        try:
            logging.info(msg)
        except NameError:
            print(msg)

    try:
        img, metadata = raw_fpath_to_mono_img_and_metadata(
            src_fpath, crop_all=crop_all, return_float=True
        )
        if check_exposure and not is_exposure_ok(img, metadata):
            log(f"# bad exposure for {src_fpath} ({img.mean()=})")
            return ConversionOutcome.BAD_EXPOSURE, src_fpath, dest_fpath
        img = demosaic(img, metadata)
        img = camRGB_to_profiledRGB_img(
            img, metadata, output_color_profile=output_color_profile
        )
    except Exception as e:
        if (
            isinstance(e, AssertionError)
            or isinstance(e, rawpy._rawpy.LibRawFileUnsupportedError)
            or isinstance(e, rawpy._rawpy.LibRawIOError)
        ):
            log(f"# Unable to read {src_fpath=}, {e=}")
            return ConversionResult(
                ConversionOutcome.UNREADABLE_ERROR, src_fpath, dest_fpath
            )
        else:
            log(f"# Unknown error {e} with {src_fpath}")
            return ConversionResult(
                ConversionOutcome.UNKNOWN_ERROR, src_fpath, dest_fpath
            )
    hdr_nparray_to_file(img, dest_fpath, OUTPUT_COLOR_PROFILE, bit_depth, src_fpath)
    log(f"# Wrote {dest_fpath}")
    return ConversionResult(ConversionOutcome.OK, src_fpath, dest_fpath)


def raw_fpath_to_hdr_img_file_mtrunner(argslist):
    return raw_fpath_to_hdr_img_file(*argslist)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--raw_fpath", help="Input image file path.")
    parser.add_argument("-o", "--out_base_path", help="Output image base file path.")
    parser.add_argument(
        "--no_wb", action="store_true", help="No white balance is applied"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not args.raw_fpath:
        args.raw_fpath = get_sample_raw_file(url=SAMPLE_RAW_URL)
    if not args.out_base_path:
        args.out_base_path = os.path.join("tests_output", f"raw.py.main")
    # prepare image as neural network input
    mono_img, metadata = raw_fpath_to_mono_img_and_metadata(args.raw_fpath)
    print(f"raw.py: opened {args.raw_fpath} with {metadata=}\n")
    if args.no_wb:
        nn_input_rggb_img = mono_to_rggb_img(mono_img, metadata)
        # prepare image as neural network ground-truth
        camRGB_img = demosaic(mono_img, metadata)  # NN GT
        camRGB_img_nowb = camRGB_img

    else:
        mono_img_wb = apply_whitebalance(
            mono_img, metadata, wb_type="daylight", in_place=False
        )
        nn_input_rggb_img = mono_to_rggb_img(mono_img_wb, metadata)
        # prepare image as neural network ground-truth
        camRGB_img = demosaic(mono_img_wb, metadata)  # NN GT
        camRGB_img_nowb = apply_whitebalance(
            camRGB_img, metadata, wb_type="daylight", in_place=False, reverse=True
        )
    # output to file for visualization
    lin_sRGB_img = camRGB_to_profiledRGB_img(camRGB_img_nowb, metadata, "lin_sRGB")
    lin_rec2020_img = camRGB_to_profiledRGB_img(
        camRGB_img_nowb, metadata, "lin_rec2020"
    )
    gamma_sRGB_img = camRGB_to_profiledRGB_img(camRGB_img_nowb, metadata, "gamma_sRGB")
    os.makedirs("tests_output", exist_ok=True)
    hdr_nparray_to_file(
        lin_rec2020_img,
        args.out_base_path + ".lin_rec2020.exr",
        color_profile="lin_rec2020",
    )
    hdr_nparray_to_file(
        lin_sRGB_img, args.out_base_path + ".lin_sRGB.exr", color_profile="lin_sRGB"
    )

    hdr_nparray_to_file(
        gamma_sRGB_img,
        args.out_base_path + ".gamma_sRGB.tif",
        color_profile="gamma_sRGB",
    )
    hdr_nparray_to_file(
        img=camRGB_img, fpath=args.out_base_path + ".camRGB.exr", color_profile=None
    )
    libraw_process(args.raw_fpath, args.out_base_path + ".libraw.tif")
    print(f"raw.py: output images saved to {args.out_base_path}.*")
