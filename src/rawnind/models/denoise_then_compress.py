"""This model serves for testing comparison; an image is denoised w/ a given denoising model then compressed with a non-denoising compression model."""

import sys
from typing import Literal, Optional
import torch

sys.path.append("..")
from rawnind.models import raw_denoiser, manynets_compression

from rawnind.libs import rawproc

# DBG
# from rawnind.libs import raw
# import os
# ENDDBG


class DenoiseThenCompress(torch.nn.Module):
    DENOISING_ARCH = raw_denoiser.UtNet2
    BAYER_DENOISING_MODEL_FPATH = "../../models/rawnind_denoise/DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4/saved_models/iter_4350000.pt"
    PRGB_DENOISING_MODEL_FPATH = "../../models/rawnind_denoise/DenoiserTrainingProfiledRGBToProfiledRGB_3ch_2024-10-09-prgb_ms-ssim_mgout_notrans_valeither_-1/saved_models/iter_3900000.pt"

    def __init__(
        self,
        in_channels: Literal[3, 4],
        encoder_cls: Optional[torch.nn.Module],
        decoder_cls: Optional[torch.nn.Module],
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
        num_distributions: int = 64,
        preupsample: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.compressor = manynets_compression.ManyPriors_RawImageCompressor(
            hidden_out_channels=hidden_out_channels,
            bitstream_out_channels=bitstream_out_channels,
            in_channels=3,
            device=device,
            # min_feat=min_feat,
            # max_feat=max_feat,
            # precision=precision,
            # entropy_coding=entropy_coding,
            encoder_cls=encoder_cls,
            decoder_cls=decoder_cls,
            preupsample=preupsample,
            num_distributions=num_distributions,
        )
        self.denoiser = self.DENOISING_ARCH(in_channels=in_channels, funit=32)

        if in_channels == 3:
            denoiser_model_fpath = self.PRGB_DENOISING_MODEL_FPATH
        elif in_channels == 4:
            denoiser_model_fpath = self.BAYER_DENOISING_MODEL_FPATH
        else:
            raise ValueError(f"Unknown in_channels: {in_channels}")

        self.denoiser = self.denoiser.to(device)
        self.denoiser.load_state_dict(
            torch.load(denoiser_model_fpath, map_location=device)
        )
        self.denoiser.eval()

    def forward(self, x: torch.Tensor):
        # DBG: output input image
        # raw.hdr_nparray_to_file(
        #     x[0].detach().cpu().numpy(),
        #     os.path.join(
        #         "dbg",
        #         "x.tif",
        #     ),
        #     color_profile="lin_rec2020",
        # )
        # ENDDBG
        # xmin = x.min()
        # xmax = x.max()
        x_denoised = self.denoiser(x)
        # # scale to the original values
        # x_denoised = xmin + (x_denoised - x_denoised.min()) * (xmax - xmin) / (x_denoised.max() - x_denoised.min())
        # # check that xmin and x.min() are the same, and xmax and x.max() are the same
        # assert torch.allclose(xmin, x_denoised.min())
        # assert torch.allclose(xmax, x_denoised.max())
        x_denoised = rawproc.match_gain(x, x_denoised)
        # DBG
        # raw.hdr_nparray_to_file(
        #     x_denoised[0].detach().cpu().numpy(),
        #     os.path.join(
        #         "dbg",
        #         "denoised_x.tif",
        #     ),
        #     color_profile="lin_rec2020",
        # )

        # ENDDBG
        x_compressed = self.compressor(x_denoised)
        # breakpoint()
        return x_compressed
        # DBG
        compressed = self.compressor(x)
        compressed["reconstructed_image"] = x  # dbg
        return compressed

    def parameters(self, *args, **kwargs):  # needed for optimizer
        return self.compressor.parameters(*args, **kwargs)

    def load_state_dict(self, state_dict: dict):
        self.compressor.load_state_dict(state_dict)

    def get_parameters(self, *args, **kwargs):
        return self.compressor.get_parameters(*args, **kwargs)

    # def
