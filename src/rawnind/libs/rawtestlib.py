import sys

sys.path.append("..")
from rawnind import train_dc_bayer2prgb
from rawnind import train_dc_prgb2prgb
from rawnind import train_denoiser_bayer2prgb
from rawnind import train_denoiser_prgb2prgb


class DCTestCustomDataloaderBayerToProfiledRGB(
    train_dc_bayer2prgb.DCTrainingBayerToProfiledRGB
):
    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        return None


class DenoiseTestCustomDataloaderBayerToProfiledRGB(
    train_denoiser_bayer2prgb.DenoiserTrainingBayerToProfiledRGB
):
    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        return None


class DCTestCustomDataloaderProfiledRGBToProfiledRGB(
    train_dc_prgb2prgb.DCTrainingProfiledRGBToProfiledRGB
):
    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        return None


class DenoiseTestCustomDataloaderProfiledRGBToProfiledRGB(
    train_denoiser_prgb2prgb.DenoiserTrainingProfiledRGBToProfiledRGB
):
    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        return None
