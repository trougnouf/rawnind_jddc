# -*- coding: utf-8 -*-
"""PyTorch loss and metric wrappers used in training/evaluation.

This module provides friendly wrappers around popular perceptual and pixelwise
criteria used in low-level vision tasks:

- MS-SSIM loss/metric: Multiscale Structural Similarity, a perceptual index that
  correlates better with human judgments than MSE for image fidelity tasks.
- MSE loss: Mean squared error, the de-facto L2 pixel loss.

References:
- Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality
  Assessment: From Error Visibility to Structural Similarity," IEEE Trans. Image
  Processing, 13(4):600–612, 2004.
- Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale Structural Similarity for
  Image Quality Assessment," 37th Asilomar Conf. on Signals, Systems and Computers, 2003.
- pytorch-msssim: https://github.com/VainF/pytorch-msssim
"""

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
    """MS-SSIM as a loss (1 - MS-SSIM) for optimization.

    This wraps pytorch_msssim.MS_SSIM and flips it into a loss by returning
    1 - MS-SSIM. Inputs are expected to be in [0, data_range].

    Args:
        data_range: Dynamic range of the input images. For normalized floats,
            use 1.0; for 8-bit integers, use 255, etc.
        **kwargs: Passed through to pytorch_msssim.MS_SSIM.
    """
    def __init__(self, data_range=1.0, **kwargs):
        super().__init__(data_range=data_range, **kwargs)

    def forward(self, input, target):
        """Compute 1 - MS-SSIM between input and target.

        Args:
            input: Tensor of shape (N, C, H, W) in [0, data_range].
            target: Tensor of shape (N, C, H, W) in [0, data_range].

        Returns:
            Scalar tensor representing the loss.
        """
        return 1 - super().forward(input, target)


class MS_SSIM_metric(pytorch_msssim.MS_SSIM):
    """MS-SSIM as a metric (higher is better).

    Thin wrapper around pytorch_msssim.MS_SSIM retaining the original meaning
    of the score (1.0 = identical images, 0 = poor similarity).

    Args:
        data_range: Dynamic range of the input images.
        **kwargs: Passed through to pytorch_msssim.MS_SSIM.
    """
    def __init__(self, data_range=1.0, **kwargs):
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
    raise NotImplementedError
    # def findvaliddim(start):
    #     try:
    #         piqa.MS_SSIM()(
    #             torch.rand(1, 3, start, start), torch.rand(1, 3, start, start)
    #         )
    #         print(start)
    #         return start
    #     except RuntimeError:
    #         print(start)
    #         findvalid(start + 1)
    #
    # findvaliddim(1)  # result is 162
