import os
import sys
import statistics
import time
import logging
from typing import Optional
from collections.abc import Iterable
import torch

sys.path.append("..")
from rawnind.libs import abstract_trainer
from rawnind.libs import rawproc
from rawnind.libs import raw

APPROX_EXPOSURE_DIFF_PENALTY = 1 / 10000


class DenoiserTrainingBayerToProfiledRGB(
    abstract_trainer.DenoiserTraining,
    abstract_trainer.BayerImageToImageNNTraining,
    abstract_trainer.BayerDenoiser,
):
    CLS_CONFIG_FPATHS = abstract_trainer.DenoiserTraining.CLS_CONFIG_FPATHS + [
        os.path.join("config", "train_denoise_bayer2prgb.yaml")
    ]

    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def autocomplete_args(self, args) -> None:
        if not args.in_channels:
            args.in_channels = 4
        super().autocomplete_args(args)


if __name__ == "__main__":
    # try:
    #     os.nice(1)
    # except OSError:
    #     pass
    denoiserTraining = DenoiserTrainingBayerToProfiledRGB()
    denoiserTraining.training_loop()
