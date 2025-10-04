import os
import sys

sys.path.append("..")
from rawnind.libs import abstract_trainer


class DCTrainingBayerToProfiledRGB(
    abstract_trainer.DenoiseCompressTraining,
    abstract_trainer.BayerImageToImageNNTraining,
):
    CLS_CONFIG_FPATHS = abstract_trainer.DenoiseCompressTraining.CLS_CONFIG_FPATHS + [
        os.path.join("config", "train_dc_bayer2prgb.yaml")
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
    # logging.getLogger().setLevel(logging.DEBUG)
    denoiserTraining = DCTrainingBayerToProfiledRGB()
    denoiserTraining.training_loop()
