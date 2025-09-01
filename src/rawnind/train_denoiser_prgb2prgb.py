import os
import statistics
import logging
import sys
from collections.abc import Iterable
import multiprocessing

sys.path.append("..")
from rawnind.libs import abstract_trainer
from rawnind.libs import raw
from rawnind.libs import rawproc
from common.libs import pt_helpers


class DenoiserTrainingProfiledRGBToProfiledRGB(
    abstract_trainer.DenoiserTraining, abstract_trainer.PRGBImageToImageNNTraining
):
    CLS_CONFIG_FPATHS = abstract_trainer.DenoiserTraining.CLS_CONFIG_FPATHS + [
        os.path.join("config", "train_denoise_prgb2prgb.yaml")
    ]

    def __init__(self, launch=False, **kwargs):
        super().__init__(launch=launch, **kwargs)

    def autocomplete_args(self, args):
        if not args.in_channels:
            args.in_channels = 3
        super().autocomplete_args(args)


if __name__ == "__main__":
    # check if the args contain proc2proc anywhere
    if any("proc2proc" in arg or "opencv" in arg for arg in sys.argv):
        try:
            print("setting multiprocessing.set_start_method('spawn')")
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            print("multiprocessing.set_start_method('spawn') failed")
            pass
    # try:
    #     os.nice(1)
    # except OSError:
    #     pass
    denoiserTraining = DenoiserTrainingProfiledRGBToProfiledRGB()
    denoiserTraining.training_loop()
