# -*- coding: utf-8 -*-
"""
Image analysis on file paths
"""
import subprocess
from typing import Union
from PIL import Image
import unittest
import os
import shutil
import numpy as np

if shutil.which("exiftool") is None:
    print("libimganalysis warning: exiftool binary is missing. get_iso is impacted.")
try:
    import piexif
except ModuleNotFoundError:
    print(
        "filter_dataset_by_iso.py: warning: piexif library not found, using exiftool instead"
    )
from typing import Optional
import sys

#try:
#    import tensorflow as tf
#except ModuleNotFoundError as e:
#    print("libimganalysis.py: tensorflow is missing, some functions are disabled")

import piqa

import sys

sys.path.append("..")
#try:
#    from common.libs import tf_helpers
#except ModuleNotFoundError as e:
#    print("libimganalysis: warning: {}".format(e))
from common.libs import pt_helpers
from rawnind.libs import rawproc

VALID_IMG_EXT = ["png", "jpg", "jpeg", "bmp", "gif", "tiff", "ppm", "j2k", "webp"]


def get_iso(fpath):
    def piexif_get_iso(fpath):
        """
        supports jpeg and maybe tiff. should be slightly faster than calling exiftool.
        """
        try:
            exifdata = piexif.load(fpath)["Exif"]
        except Exception as e:
            print(f"piexif_get_iso: {e} on {fpath}; reverting to exiftool_get_iso")
            return
        if 34855 in exifdata:
            isoval = exifdata[34855]
            if not isinstance(isoval, int):
                print(
                    f"piexif_get_iso: invalid non-int format for {fpath} ({isoval}), skipping."
                )
                isoval = None
            return isoval

    def exiftool_get_iso(fpath):
        cmd = "exiftool", "-S", "-ISO", fpath
        try:
            res = subprocess.run(cmd, text=True, capture_output=True, timeout=30).stdout
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"exiftool_get_iso error: exiftool binary not present ({e}"
            )
        if res == "":
            return None
        else:
            try:
                return int(res.split(": ")[-1])
            except ValueError as e:
                print(f"exiftool_get_iso: got {e} on {fpath}, skipping.")

    exiftool_installed = True if shutil.which("exiftool") else False
    ext = fpath[-4:].lower()
    isoval = False
    if (
        ext.endswith("jpg")
        or ext.endswith("jpeg")
        or ext.endswith("tif")
        or ext.endswith("tiff")
    ) and "piexif" in sys.modules:
        isoval = piexif_get_iso(fpath)
    if exiftool_installed and not isoval:
        isoval = exiftool_get_iso(fpath)
    if isoval is False:
        isoval = None
        print(
            "get_iso: no suitable tools found. Install piexif library and/or exiftool executable."
        )
    assert isoval is None or isinstance(isoval, int), f"fpath={fpath}, isoval={isoval}"
    return isoval


def is_raw(fpath: str) -> bool:
    if fpath.split(".")[-1].lower() in (
        "3fr",
        "ari",
        "arw",
        "bay",
        "braw",
        "crw",
        "cr2",
        "cr3",
        "cap",
        "data",
        "dcs",
        "dcr",
        "dng",
        "drf",
        "eip",
        "erf",
        "fff",
        "gpr",
        "iiq",
        "k25",
        "kdc",
        "mdc",
        "mef",
        "mos",
        "mrw",
        "nef",
        "nrw",
        "obm",
        "orf",
        "pef",
        "ptx",
        "pxn",
        "r3d",
        "raf",
        "raw",
        "rwl",
        "rw2",
        "rwz",
        "sr2",
        "srf",
        "srw",
        # "tif",
        "x3f",
        "cri",
        "jxs",
        "tco",
    ):
        return True
    return False


def piqa_msssim(img1path: str, img2path: str):
    img1 = pt_helpers.fpath_to_tensor(img1path, batch=True)
    img2 = pt_helpers.fpath_to_tensor(img2path, batch=True)
    return piqa.MS_SSIM()(img1, img2).item()


def ipx_psnr(img1path: str, img2path: str):
    """
    intoPIX psnr_compare implementation
    multithread-safe
    """
    args = ["psnr_compare", img1path, img2path]
    res = subprocess.run(args, capture_output=True)
    try:
        return min(
            float(res.stdout.decode().split("Psnr Global:\t")[1].split(" dB")[0]), 100
        )
    except IndexError as e:
        raise ValueError(
            "ipx_psnr: unable to get psnr for %s, %s: %s" % (img1path, img2path, e)
        )


if "tensorflow" in sys.modules:

    def tf_psnr(img1: Union[str, tf.Tensor], img2: Union[str, tf.Tensor]):
        """
        Tensorflow PSNR implementation
        img1 and img2 can be paths or tf tensors.
        not multithread-safe"""
        img1 = tf_helpers.tf_open_img(img1)
        img2 = tf_helpers.tf_open_img(img2)
        return max(100, float(tf.image.psnr(img1, img2, 255)))

    def tf_msssim(img1: Union[str, tf.Tensor], img2: Union[str, tf.Tensor]):
        """
        Tensorflow MSSSIM implementation
        img1 and img2 can be paths or tf tensors.
        not multithread-safe"""
        # with tf.device("/cpu:0"):
        img1 = tf_helpers.tf_open_img(img1)
        img2 = tf_helpers.tf_open_img(img2)
        return float(tf.image.ssim_multiscale(img1, img2, 255))

    def tf_psnr_cpu(img1: Union[str, tf.Tensor], img2: Union[str, tf.Tensor]):
        """not multithread-safe"""
        with tf.device("/cpu:0"):
            return tf_psnr(img1, img2)

    def tf_msssim_cpu(img1: Union[str, tf.Tensor], img2: Union[str, tf.Tensor]):
        """not multithread-safe"""
        with tf.device("/cpu:0"):
            return tf_msssim(img1, img2)

    def tf_compare(img1, img2, bitstream: Optional[str] = None, ignore_border: int = 4):
        """img1 and img2 can be fpath strings or tf tensors. bitstream is an optional fpath"""
        res = dict()
        if isinstance(img1, str):
            img1 = tf_helpers.tff_open_img(img1)
        if isinstance(img2, str):
            img2 = tf_helpers.tff_open_img(img2)
        npixels = img1.shape[0] * img1.shape[1]
        assert img1.shape == img2.shape
        if ignore_border > 0:
            img1 = img1[ignore_border:-ignore_border, ignore_border:-ignore_border, :]
            img2 = img2[ignore_border:-ignore_border, ignore_border:-ignore_border, :]
        if bitstream is not None:
            res["bytes"] = os.path.getsize(bitstream)
            res["bpp"] = (res["bytes"] * 8) / npixels
        res["psnr"] = float(tf.image.psnr(img1, img2, 1))
        res["msssim"] = float(tf.image.ssim_multiscale(img1, img2, 1))
        res["ssim"] = float(tf.image.ssim(img1, img2, 1))
        res["mse"] = float(
            tf.reduce_mean(tf.keras.losses.MeanSquaredError()(img1, img2))
        )
        return res

    msssim = tf_msssim
    psnr = tf_psnr
else:
    msssim = piqa_msssim


def pil_get_resolution(imgpath):
    return Image.open(imgpath).size


def is_valid_img(img_fpath, open_img=False, save_img=False, clean=False):
    """
    Check if an image is valid.
    open_img = True: use PIL's verify function
    save_img = True: use PIL to resize the image and save as png (slower but more effective)
    open_img = save_img = False: just check the extension (default)
    clean = True: remove deffective images, otherwise just return False
    """
    if is_raw(img_fpath) or img_fpath.split(".")[-1].lower() == "exr":
        try:
            img, metadata = rawproc.img_fpath_to_np_mono_flt_and_metadata(img_fpath)
            return True
        except AssertionError:
            if img_fpath.lower().endswith(".raf"):  # cannot check RAF atm
                return True
            return False
        except ValueError:
            return False
    if img_fpath.lower().endswith(".npy"):
        try:
            np.load(img_fpath)
            return True
        except ValueError:
            return False
    Image.MAX_IMAGE_PIXELS = 15000**2
    ext_is_valid = img_fpath.split(".")[-1].lower() in VALID_IMG_EXT
    if not open_img and not save_img:
        return ext_is_valid
    try:
        img = Image.open(img_fpath)
        if save_img:
            img = img.resize((128, 128))
            img.save("tmp.png")
        else:
            img.verify()
        return True
    except OSError as e:
        print(e)
        if clean:
            os.remove(img_fpath)
            print("rm {}".format(img_fpath))
        return False
    except Image.DecompressionBombError as e:
        print(e)
        if clean:
            os.remove(img_fpath)
            print("rm {}".format(img_fpath))
        return False


class Test_libimganalysis(unittest.TestCase):
    """
    [common]$ python -m unittest discover libs/ -p *.py
    """

    DSROOTPATH = os.path.join("..", "..", "datasets")  # running from common/
    TESTIMG = os.path.join(
        "FeaturedPictures",
        "Isaac Lake during golden hour, moments before a storm (DSCF2631).jpg",
    )

    def setUp(self):
        from common.libs.libimgops import bpg_encdec

        self.inimgpath = os.path.join(self.DSROOTPATH, self.TESTIMG)
        self.tmpimgpath = "test_libimganalysis.png"
        bpg_encdec(self.inimgpath, self.tmpimgpath, qp=29)

    def tearDown(self):
        os.remove(self.tmpimgpath)

    def test_tf_msssim(self):
        with tf.device("/cpu:0"):
            score = tf_msssim(self.inimgpath, self.tmpimgpath)
            print("msssim score: {}".format(score))
        self.assertGreater(score, 0.5)

    def test_tf_psnr(self):
        with tf.device("/cpu:0"):
            score = tf_psnr(self.inimgpath, self.tmpimgpath)
        self.assertGreater(score, 10)


if __name__ == "__main__":
    unittest.main()
