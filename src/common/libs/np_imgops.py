# cv2.setNumThreads(0)
import os
import random
import sys
import unittest
from enum import Enum, auto
from typing import Tuple, Union

# import multiprocessing
# multiprocessing.set_start_method('spawn')
try:
    import cv2
except ImportError:
    cv2 = None  # Optional dependency
import numpy as np

try:
    import OpenImageIO as oiio

    TIFF_PROVIDER = "OpenImageIO"
except ImportError:
    TIFF_PROVIDER = "OpenCV"
    print(
        "np_imgops.py warning: missing OpenImageIO library; falling back to OpenCV which cannot open 16-bit float tiff images"
    )


sys.path.append("..")
from rawnind.libs import raw
from common.libs import libimganalysis

TMP_DPATH = "tmp"


class CropMethod(Enum):
    RAND = auto()
    CENTER = auto()


def _oiio_img_fpath_to_np(fpath: str):
    inp = oiio.ImageInput.open(fpath)
    spec = inp.spec()
    pixels = inp.read_image(0, 0, 0, spec.nchannels, spec.format)
    # move channels to first dimension
    pixels = np.moveaxis(pixels, -1, 0)
    inp.close()
    return pixels


def _opencv_img_fpath_to_np(fpath: str):
    try:
        return cv2.cvtColor(
            cv2.imread(fpath, flags=cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH),
            cv2.COLOR_BGR2RGB,
        ).transpose(2, 0, 1)
    except cv2.error as e:
        raise ValueError(
            f"img_fpath_to_np_flt: error {e} with {fpath} (hint: consider installing OpenImageIO instead of OpenCV backend)"
        )


def img_fpath_to_np_flt(
    fpath: str,
    incl_metadata=False,  # , bit_depth: Optional[int] = None
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """returns a numpy float32 array from RGB image path (8-16 bits per component)
    shape: c, h, w
    FROM common.libimgops"""
    if not os.path.isfile(fpath):
        raise ValueError(f"File not found {fpath}")
    if fpath.endswith(".npy"):
        assert not incl_metadata
        return np.load(fpath)
    if libimganalysis.is_raw(fpath):
        rggb_img, metadata = raw.raw_fpath_to_rggb_img_and_metadata(fpath)
        if incl_metadata:
            return rggb_img, metadata
        print("img_fpath_to_np_flt warning: ignoring raw image metadata")
        return rggb_img
    if (
        fpath.lower().endswith(".tif") or fpath.lower().endswith(".tiff")
    ) and TIFF_PROVIDER == "OpenImageIO":
        rgb_img = _oiio_img_fpath_to_np(fpath)
    else:
        rgb_img = _opencv_img_fpath_to_np(fpath)

    if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float16:
        res = rgb_img
        # if bit_depth is None:
        #     return rgb_img
        # elif bit_depth == 16:
        #     return rgb_img.astype(np.float16)
        # elif bit_depth == 32:
        #     return rgb_img.astype(np.float32)
    elif rgb_img.dtype == np.ubyte:
        res = rgb_img.astype(np.single) / 255
    elif rgb_img.dtype == np.ushort:
        res = rgb_img.astype(np.single) / 65535
    else:
        raise TypeError(
            f"img_fpath_to_np_flt: Error: fpath={fpath} has unknown format ({rgb_img.dtype})"
        )
    if incl_metadata:
        return res, {}
    else:
        return res


def np_pad_img_pair(img1, img2, cs):
    xpad0 = max(0, (cs - img1.shape[2]) // 2)
    xpad1 = max(0, cs - img1.shape[2] - xpad0)
    ypad0 = max(0, (cs - img1.shape[1]) // 2)
    ypad1 = max(0, cs - img1.shape[1] - ypad0)
    padding = ((0, 0), (ypad0, ypad1), (xpad0, xpad1))
    return np.pad(img1, padding), np.pad(img2, padding)


def np_crop_img_pair(img1, img2, cs: int, crop_method=CropMethod.RAND):
    """
    crop an image pair into cs
    also compatible with pytorch tensors
    """
    if crop_method is CropMethod.RAND:
        x0 = random.randint(0, img1.shape[2] - cs)
        y0 = random.randint(0, img1.shape[1] - cs)
    elif crop_method is CropMethod.CENTER:
        x0 = (img1.shape[2] - cs) // 2
        y0 = (img1.shape[1] - cs) // 2
    return img1[:, y0 : y0 + cs, x0 : x0 + cs], img2[:, y0 : y0 + cs, x0 : x0 + cs]


def np_to_img(img: np.ndarray, fpath: str, precision: int = 16):
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    hwc_img = img.transpose(1, 2, 0)
    hwc_img = cv2.cvtColor(hwc_img, cv2.COLOR_RGB2BGR)
    if precision == 16:
        hwc_img = (hwc_img * 65535).clip(0, 65535).astype(np.uint16)
    elif precision == 8:
        hwc_img = (hwc_img * 255).clip(0, 255).astype(np.uint8)
    else:
        raise NotImplementedError(precision)
    cv2.imwrite(fpath, hwc_img)


class TestImgOps(unittest.TestCase):
    def setUp(self):
        import tifffile

        self.imgeven1 = np.random.rand(3, 8, 8)
        self.imgeven2 = np.random.rand(3, 8, 8)
        self.imgodd1 = np.random.rand(3, 5, 5)
        self.imgodd2 = np.random.rand(3, 5, 5)
        # create a RGB image file
        self.random_image = np.random.rand(3, 512, 768).astype(np.float32)
        self.random_image_fpath = os.path.join(TMP_DPATH, "rand.tiff")
        os.makedirs(TMP_DPATH, exist_ok=True)
        tifffile.imwrite(self.random_image_fpath, self.random_image.transpose(1, 2, 0))

    def tearDown(self):
        os.remove(self.random_image_fpath)

    def test_pad(self):
        imgeven1_padded, imgeven2_padded = np_pad_img_pair(
            self.imgeven1, self.imgeven2, 16
        )
        imgodd1_padded, imgodd2_padded = np_pad_img_pair(self.imgodd1, self.imgodd2, 16)
        self.assertTupleEqual(imgeven1_padded.shape, (3, 16, 16), imgeven1_padded.shape)
        self.assertTupleEqual(imgodd2_padded.shape, (3, 16, 16), imgodd2_padded.shape)
        self.assertEqual(imgeven1_padded[0, 4, 4], self.imgeven1[0, 0, 0])

    def test_crop(self):
        # random crop: check size
        imgeven1_randcropped, imgeven2_randcropped = np_crop_img_pair(
            self.imgeven1, self.imgeven2, 4, CropMethod.RAND
        )
        self.assertTupleEqual(
            imgeven1_randcropped.shape, (3, 4, 4), imgeven1_randcropped.shape
        )

        # center crop: check size and value
        imgeven1_centercropped, imgeven2_centercropped = np_crop_img_pair(
            self.imgeven1, self.imgeven2, 4, CropMethod.CENTER
        )
        self.assertTupleEqual(
            imgeven1_centercropped.shape, (3, 4, 4), imgeven1_centercropped.shape
        )
        # orig:    0 1 2 3 4 5 6 7
        # cropped: x x 2 3 4 5 x x
        self.assertEqual(
            imgeven1_centercropped[0, 0, 0],
            self.imgeven1[0, 2, 2],
            f"imgeven1_centercropped[0]={imgeven1_centercropped[0]}, self.imgeven1[0]={self.imgeven1[0]}",
        )

        # crop w/ same size: check identity
        imgeven1_randcropped, imgeven2_randcropped = np_crop_img_pair(
            self.imgeven1, self.imgeven2, 8, CropMethod.CENTER
        )
        self.assertTrue(
            (imgeven1_randcropped == self.imgeven1).all(), "Crop to same size is broken"
        )

    def test_read_img_opencv_equals_oiio(self):
        # load the image with opencv and with oiio
        cvimg = _opencv_img_fpath_to_np(self.random_image_fpath)
        oiioimg = _oiio_img_fpath_to_np(self.random_image_fpath)
        default_img = img_fpath_to_np_flt(self.random_image_fpath, incl_metadata=False)
        # check that the images match each other and self.random_image
        self.assertTrue(
            (cvimg == oiioimg).all(), "OpenCV and OpenImageIO images do not match"
        )
        self.assertTrue(
            (cvimg == self.random_image).all(), "OpenCV and random image do not match"
        )
        self.assertTrue(
            (cvimg == default_img).all(), "OpenCV and default image do not match"
        )


if __name__ == "__main__":
    unittest.main()
