# -*- coding: utf-8 -*-

import torch

# import piqa  # disabled due to https://github.com/francois-rozet/piqa/issues/25
import pytorch_msssim
import sys

sys.path.append("..")
# from common.extlibs import DISTS_pt

# class MS_SSIM_loss(piqa.MS_SSIM):
#     def __init__(self, **kwargs):
#         r""""""
#         super().__init__(**kwargs)
#     def forward(self, input, target):
#         return 1-super().forward(input, target)


class MS_SSIM_loss(pytorch_msssim.MS_SSIM):
    def __init__(self, data_range=1.0, **kwargs):
        r""""""
        super().__init__(data_range=data_range, **kwargs)

    def forward(self, input, target):
        return 1 - super().forward(input, target)


class MS_SSIM_metric(pytorch_msssim.MS_SSIM):
    def __init__(self, data_range=1.0, **kwargs):
        r""""""
        super().__init__(data_range=data_range, **kwargs)


# class SSIM_loss(piqa.SSIM):
#     def __init__(self, **kwargs):
#         r""""""
#         super().__init__(**kwargs)
#     def forward(self, input, target):
#         return 1-super().forward(input, target)


# class DISTS_loss(DISTS_pt.DISTS):
#     def __init__(self, **kwargs):
#         super().__init__()
#
#     def forward(self, x, y):
#         return super().forward(x, y, require_grad=True, batch_average=True)


losses = {
    "mse": torch.nn.MSELoss,
    "msssim_loss": MS_SSIM_loss,
}  # , "dists": DISTS_loss}
# metrics = losses | {"msssim": MS_SSIM_metric}  # python 3.8 / 3.10 compat
metrics = {
    "msssim": MS_SSIM_metric,
    "mse": torch.nn.MSELoss,
    "msssim_loss": MS_SSIM_loss,
    # "dists": DISTS_loss,
}  # python 3.8 / 3.10 compat


if __name__ == "__main__":

    def findvaliddim(start):
        try:
            piqa.MS_SSIM()(
                torch.rand(1, 3, start, start), torch.rand(1, 3, start, start)
            )
            print(start)
            return start
        except RuntimeError:
            print(start)
            findvalid(start + 1)

    findvaliddim(1)  # result is 162
