"""Denoise an image using the bm3d binary from https://github.com/gfacciol/bm3d"""

import sys
import os
import shutil
import random
import string
import subprocess
import torch
import cv2
import numpy as np

sys.path.append("..")
from typing import Union
from rawnind.models import raw_denoiser
from rawnind.libs import raw
from common.libs import pt_helpers
from common.libs import np_imgops

TMPDIR = f"tmp_{os.uname().nodename}"


class BM3D_Denoiser(raw_denoiser.Denoiser):
    def __init__(self, in_channels: int, funit: Union[int, str], *args, **kwargs):
        super().__init__(in_channels=in_channels)
        assert in_channels == 3, f"{in_channels=} should be 3 for BM3D"
        self.sigma = (
            funit  # we use the funit parameter because it's common to other models
        )
        self.dummy_parameter = torch.nn.Parameter(torch.randn(3))
        # check that the bm3d binary exists
        assert shutil.which("bm3d"), "bm3d binary not found in PATH"
        os.makedirs(TMPDIR, exist_ok=True)

    def forward(self, noisy_image):
        # store values which are out of 0,1 range. They'll probably contain some noise but it seems to be the only way to conserve their color values since bm3d implementation only works w/ 8-bit png
        # out_of_range_values = noisy_image - torch.clamp(noisy_image, 0, 1)
        # check that we are not working on a batch, ie 3 dimensions and 3 channels
        noisy_image = noisy_image.squeeze(0).numpy()
        print(f"{noisy_image.shape=}, {noisy_image.mean()=}")
        assert noisy_image.shape[0] == 3, f"{noisy_image.shape=} should be (3, H, W)"
        assert len(noisy_image.shape) == 3, f"{noisy_image.shape=} should be (3, H, W)"
        # denoise the image. use png because bm3d doesn't seem to support tiff
        tmp_str = "".join(random.choices(string.ascii_letters + string.digits, k=23))
        tmp_input_img_fpath = os.path.join(TMPDIR, f"{tmp_str}_input.png")
        tmp_denoised_img_fpath = os.path.join(TMPDIR, f"{tmp_str}_denoised.png")

        # # scale image to 0,1
        # img_min = noisy_image.min()
        # img_max = noisy_image.max()
        # noisy_image = (noisy_image - img_min) / (img_max - img_min)
        # save to disk and denoise
        np_imgops.np_to_img(noisy_image, tmp_input_img_fpath, precision=8)
        cmd = ("bm3d", tmp_input_img_fpath, str(self.sigma), tmp_denoised_img_fpath)
        cmd_res = subprocess.run(cmd)
        assert cmd_res.returncode == 0 and os.path.isfile(tmp_denoised_img_fpath)
        denoised_image = pt_helpers.fpath_to_tensor(
            tmp_denoised_img_fpath, device=noisy_image.device, batch=True
        )
        # denoised_image = denoised_image + out_of_range_values
        # restore image to original scale
        # denoised_image = denoised_image * (img_max - img_min) + img_min
        # cleanup and return
        # os.remove(tmp_input_img_fpath)
        # os.remove(tmp_denoised_img_fpath)

        # cv2 method, did not work (on RGB images?)
        # orig_dtype = noisy_image.dtype
        # # convert image to opencv dimension order
        # noisy_image = np.moveaxis(noisy_image, 0, -1)
        # # convert noisy_image to uint8
        # noisy_image = (noisy_image * 255).astype(np.uint8)

        # denoised_image = cv2.xphoto.bm3dDenoising(src=noisy_image, h=float(self.sigma))
        # denoised_image = (
        #     torch.from_numpy(denoised_image).to(dtype=orig_dtype) / 255
        # ).unsqueeze(0)
        return denoised_image


architectures = {"bm3d": BM3D_Denoiser}

if __name__ == "__main__":
    assert len(sys.argv) == 4, (
        f"Usage: python {sys.argv[0]} <noisy_image_fpath> <sigma> <denoised_fpath>"
    )
    noisy_image = pt_helpers.fpath_to_tensor(sys.argv[1])
    sigma = sys.argv[2]
    denoiser = architectures["bm3d"](in_channels=3, funit=sigma)
    denoised_image = denoiser(noisy_image)
    raw.hdr_nparray_to_file(
        denoised_image.squeeze(0).numpy(), sys.argv[3], color_profile="lin_rec2020"
    )
