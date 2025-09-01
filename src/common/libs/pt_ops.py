# -*- coding: utf-8 -*-

import unittest
import math
import torch
from torch import nn

# def img_to_batch(imgtensor, patch_size: int):
#     _, ch, height, width = imgtensor.shape
#     assert height%patch_size == 0 and width%patch_size == 0, 'img_to_batch: dims must be dividable by patch_size. {}%{}!=0'.format(imgtensor.shape, patch_size)
#     bs = math.ceil(height/patch_size) * math.ceil(width/patch_size)
#     btensor = torch.zeros([bs,ch,patch_size, patch_size], device=imgtensor.device, dtype=imgtensor.dtype)
#     xstart = ystart = 0
#     for i in range(bs):
#         btensor[i] = imgtensor[:, :, ystart:ystart+patch_size, xstart:xstart+patch_size]
#         xstart += patch_size
#         if xstart+patch_size > width:
#             xstart = 0
#             ystart += patch_size
#     return btensor


def pt_crop_batch(batch, cs: int):
    """
    center crop an image batch to cs
    also compatible with numpy tensors
    """
    x0 = (batch.shape[3] - cs) // 2
    y0 = (batch.shape[2] - cs) // 2
    return batch[:, :, y0 : y0 + cs, x0 : x0 + cs]


def img_to_batch(img, patch_size: int, nchans_per_prior: int = None):
    _, ch, height, width = img.shape
    if nchans_per_prior is None:
        nchans_per_prior = ch
    assert img.ndim == 4
    if not (height % patch_size == 0 and width % patch_size == 0):
        print(
            "img_to_batch: warning: "
            "img_to_batch: dims must be dividable by patch_size. {}%{}!=0".format(
                img.shape, patch_size
            )
        )
        _, _, imgy, imgx = img.shape
        img = img[
            :,
            :,
            : (imgy // patch_size) * patch_size,
            : (imgx // patch_size) * patch_size,
        ]

    return (
        img.unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
        .transpose(1, 0)
        .reshape(ch, -1, patch_size, patch_size)
        .transpose(1, 0)
        .reshape(-1, nchans_per_prior, patch_size, patch_size)
    )
    # incorrect
    # return img.unfold(2,patch_size,patch_size).unfold(
    #     3,patch_size,patch_size).contiguous().view(
    #         ch,-1,patch_size,patch_size).permute((1,0,2,3))


def batch_to_img(btensor, height: int, width: int, ch=3):
    """
    This one isn't differentiable (FIXME if needed)
    """
    imgtensor = torch.zeros(
        [1, ch, height, width], device=btensor.device, dtype=btensor.dtype
    )
    patch_size = btensor.shape[-1]
    xstart = ystart = 0
    for i in range(btensor.size(0)):
        imgtensor[0, :, ystart : ystart + patch_size, xstart : xstart + patch_size] = (
            btensor[i]
        )
        xstart += patch_size
        if xstart + patch_size > width:
            xstart = 0
            ystart += patch_size
    return imgtensor


def pixel_unshuffle(input, downscale_factor):
    """
    https://github.com/SsnL/pytorch/blob/c0a9167d2397f9064336bbb7ac73e0ed9ed44d78/torch/nn/functional.py
    input should have ch= chout*(1/scale_factor)**2
    (for pixel_shuffle, input should have ch= chout*scale_factor**2)
    """
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor
    input_view = input.reshape(
        batch_size, channels, out_height, downscale_factor, out_width, downscale_factor
    )
    channels *= downscale_factor**2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4)
    return unshuffle_out.reshape(batch_size, channels, out_height, out_width)


def oneloss(x, y):
    return torch.ones(1).to(x.device)


def fragile_checksum(atensor):
    return (
        torch.mean(atensor.float()).item(),
        atensor.float().std().item(),
        torch.sum(atensor).item(),
    )


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


def crop_to_multiple(tensor: torch.Tensor, multiple: int = 64) -> torch.Tensor:
    return tensor[
        ...,
        : tensor.size(-2) - tensor.size(-2) % multiple,
        : tensor.size(-1) - tensor.size(-1) % multiple,
    ]


class Test_PTOPS(unittest.TestCase):
    def test_pixel_shuffle_size(self):
        # pytorch img tensor is NxCxHxW
        dim = 256
        bs = 4
        ch = 3
        img = torch.rand(bs, ch, dim, dim)
        scale_factor = 4
        aconv = nn.Conv2d(ch, ch * scale_factor**2, 3, padding=3 // 2)
        preshuffle_img = aconv(img)
        self.assertListEqual(
            [bs, ch * scale_factor**2, dim, dim], list(preshuffle_img.shape)
        )
        upscaled_img = nn.PixelShuffle(4)
        # # fails because ch is too small
        # aconv = nn.Conv2d(ch, int(ch*(1/scale_factor)**2), 3, padding=3//2)
        # preshuffle_img = aconv(img)
        # downscaled_img = pixel_unshuffle(preshuffle_img, scale_factor)

    def test_img_to_batch_to_img(self):
        imgtensor = torch.rand(1, 3, 768, 512)
        # pix23 = imgtensor[0,:,23,23]
        btensor = img_to_batch(imgtensor, 64)
        self.assertListEqual(list(btensor.shape), [96, 3, 64, 64])
        imgtensorback = batch_to_img(btensor, 768, 512)
        self.assertEqual((imgtensor != imgtensorback).sum(), 0)


if __name__ == "__main__":
    unittest.main()
