"""
Model described in End-to-end optimized image compression with competition of prior distributions.

Based on End-to-end Optimized Image Compression by Johannes Ballé et al.

The autoencoder is left out of this library.
"""

import math
import logging
import unittest
import sys
import torch
import numpy as np
from typing import Literal, Optional, Union

sys.path.append("..")
from rawnind.models import bitEstimator
from common.libs import pt_helpers
from common.libs import pt_ops
from rawnind.models import compression_autoencoders


# logger = logging.getLogger("ImageCompression")
# try:
#     import torchac
# except ModuleNotFoundError:
#     logging.info(
#         "manynets_compressor: torchac not available; entropy coding is disabled"
#     )
# except RuntimeError:
#     logging.info(
#         "manynets_compressor: torchac build failure; entropy coding is disabled"
#     )
# FIXME needs to switch to a maintained entropy coder (https://github.com/fab-jul/torchac)
try:
    import png
except ModuleNotFoundError:
    logging.info(
        "manynets_compressor: png is not available (currently used in encode/decode)"
    )

NUMSCALES: int = 64  # 384 is the max I can fit with 48GB of RAM


class ManyPriors_RawImageCompressor(
    #  TODO use abstract_model.AbstractImageCompressor instead and complete it as needed
    compression_autoencoders.AbstractRawImageCompressor
):
    def __init__(
        self,
        in_channels: Literal[3, 4],
        encoder_cls: Optional[torch.nn.Module],
        decoder_cls: Optional[torch.nn.Module],
        device: Union[torch.device, Literal["cpu"]],
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
        num_distributions: int = 64,
        preupsample: bool = False,
        # min_feat: int = -127,
        # max_feat: int = 128,
        # precision: int = 16,
        # entropy_coding: bool = False,
        **kwargs,
    ):
        """
        max cost to encode the prior: (bits_per_prior) / (patch_size)**2; typically 6 * (16**2) = 0.0234375
        for SpaceChans encoding:
        (nch * (bits_per_prior)) / (patch_size**2 * nchan_per_prior) bpp for priors
        eg: (256*8)/(128**2 * 4) = 0.0313 bpp
        eg: (256*8)/(32**2 * 64) = 0.0313


        """
        super().__init__(
            hidden_out_channels=hidden_out_channels,
            bitstream_out_channels=bitstream_out_channels,
            in_channels=in_channels,
            device=device,
            # min_feat=min_feat,
            # max_feat=max_feat,
            # precision=precision,
            # entropy_coding=entropy_coding,
            encoder_cls=encoder_cls,
            decoder_cls=decoder_cls,
            preupsample=preupsample,
        )
        if not (hasattr(self, "Encoder") and hasattr(self, "Decoder")):
            raise ValueError(f"{self}: encoder_cls and decoder_cls must be specified")
            # self.Encoder = compression_autoencoders.BalleEncoder(
            #     hidden_out_channels=hidden_out_channels,
            #     bitstream_out_channels=bitstream_out_channels,
            #     in_channels=in_channels,
            #     device=device,
            # )
            # self.Decoder = compression_autoencoders.BalleDecoder(
            #     hidden_out_channels=hidden_out_channels,
            #     bitstream_out_channels=bitstream_out_channels,
            #     # demosaic=in_channels == 4,
            #     device=device,
            # )
        self.num_distributions = num_distributions
        self.bitEstimators = bitEstimator.MultiHeadBitEstimator(
            bitstream_out_channels,
            nb_head=self.num_distributions,
            shape=("g", "bs", "ch", "h", "w"),
            **kwargs,
        )
        self.dists_last_use = np.zeros(self.num_distributions, dtype=int)

        # ntargets = int((-self.min_feat + self.max_feat) / self.q_intv + 1)
        # self.entropy_table = torch.zeros(
        #     self.num_distributions, out_channel_M, ntargets, dtype=torch.short
        # )
        self._bpp_px_mult = (
            1 if in_channels == 3 else 4
        )  # multiply # of pixels by this value to get accurate bpp

    def get_parameters(
        self, lr=None, bitEstimator_lr_multiplier: Optional[float] = None
    ):
        assert lr is not None
        assert bitEstimator_lr_multiplier is not None
        param_list = [
            {"params": self.Encoder.parameters(), "name": "encoder"},
            {"params": self.Decoder.parameters(), "name": "decoder"},
            {
                "params": self.bitEstimators.parameters(),
                "lr": lr * bitEstimator_lr_multiplier,
                "name": "bit_estimator",
            },
        ]
        return param_list

    def forward(self, input_image, **kwargs):
        """
        Send input_image through autoencoder.

        Return (reconstructed_image, bpp_main, bpp_sidestring, used_distributions,
        num_dists_to_force_trains)
        """
        num_dists_to_force_train = 0

        im_shape = input_image.shape
        feature = self.Encoder(input_image)

        if self.training:
            # quant_noise_feature = torch.zeros_like(feature)
            # quant_noise_feature = torch.zeros(
            #     input_image.size(0),
            #     self.bitstream_out_channels,
            #     input_image.size(2) // 16,
            #     input_image.size(3) // 16,
            #     device=self.device,
            # )
            quant_noise_feature = torch.nn.init.uniform_(
                torch.zeros_like(feature), -0.5, 0.5
            )
            compressed_feature_entropy = feature + quant_noise_feature
            compressed_feature_ae = pt_ops.RoundNoGradient().apply(feature)

        else:
            # ae and entropy: round
            compressed_feature_entropy = torch.round(feature)
            compressed_feature_ae = compressed_feature_entropy
        recon_image = self.Decoder(compressed_feature_ae)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0.0, 1.0)

        total_bits_feature = 0

        probs = self.bitEstimators(
            compressed_feature_entropy + 0.5
        ) - self.bitEstimators(compressed_feature_entropy - 0.5)
        total_bits = torch.sum(torch.clamp(-torch.log2(probs + 1e-10), 0, 50), dim=(2))
        minbits, dist_select = total_bits.min(0)
        feature_batched_shape = compressed_feature_entropy.shape
        max_num_dists_to_force_train = int(
            compressed_feature_entropy.size(0)
            * compressed_feature_entropy.size(2)
            * compressed_feature_entropy.size(3)
            * 0.1
        )

        # print(dist_select)
        used_dists = dist_select.unique()
        used_dists_cpu = used_dists.cpu()
        self.dists_last_use[used_dists.cpu()] = 0
        unused_dists = np.setdiff1d(
            np.arange(self.num_distributions), used_dists_cpu, assume_unique=True
        )
        self.dists_last_use[unused_dists] += 1
        if (
            self.training and self.num_distributions > 1
        ):  # and len(used_dists) < min((self.num_distributions // 4*3), self.out_channel_M//4*3):
            dists_i_to_train = np.argwhere(self.dists_last_use > 50).flatten()
            num_dists_to_force_train = min(
                dists_i_to_train.size, max_num_dists_to_force_train
            )

            victims = (
                minbits.flatten()
                .sort(descending=True)
                .indices[:num_dists_to_force_train]
            )
            dist_select.flatten()[victims] = torch.tensor(
                np.random.choice(unused_dists, victims.size(), replace=False),
                device=dist_select.device,
            )

        if self.training or not (
            hasattr(self, "entropy_coding") and self.entropy_coding
        ):
            total_bits_feature = torch.gather(
                total_bits, 0, dist_select.unsqueeze(0)
            ).sum()
        else:
            total_bits_feature_theo = torch.gather(
                total_bits, 0, dist_select.unsqueeze(0)
            ).sum()
            logging.info("theobits")
            logging.info(total_bits_feature_theo)
            bitstream, nbytes = self.entropy_encode(
                compressed_feature_entropy, dist_select
            )
            total_bits_feature = nbytes * 8
            logging.info("actualbits")
            logging.info(total_bits_feature)
            # breakpoint()
        # I could just have used the reconstructed image to compute bpp instead of the input shape multiplied by _bpp_px_mult. oh well
        if self.training:
            bpp_sidestring = torch.tensor(
                (feature_batched_shape[0] * math.log2(self.num_distributions))
                / (im_shape[0] * im_shape[2] * im_shape[3] * self._bpp_px_mult)
            )
        else:
            bpp_sidestring = torch.tensor(
                pt_helpers.get_num_bits(dist_select.cpu(), integers=True),
                dtype=torch.float32,
            ) / (im_shape[0] * im_shape[2] * im_shape[3] * self._bpp_px_mult)
        bpp_feature = total_bits_feature / (
            im_shape[0] * im_shape[2] * im_shape[3] * self._bpp_px_mult
        )
        bpp = bpp_feature + bpp_sidestring
        return {
            "reconstructed_image": clipped_recon_image,
            "bpp_feature": bpp_feature,
            "bpp_sidestring": bpp_sidestring,
            "bpp": bpp,
            "used_dists": used_dists_cpu.tolist(),
            "num_forced_dists": num_dists_to_force_train,
        }


class ManyPriorsEncoder(torch.nn.Module):
    def __init__(self, input_channels: int):
        pass


class ManyPriorsDecoder(torch.nn.Module):
    def __init__(self):
        pass


class TestRawManyPriors(unittest.TestCase):
    def test_raw_many_priors(self):
        """
        Instantiate the 3 and 4 ch. models and check that the losses decrease.
        """
        for in_channels in (4, 3):
            # init
            input_image = torch.rand(1, in_channels, 512, 512)
            if in_channels == 4:
                input_image[:, 1] = input_image[:, 3]
                input_image_upscaled = torch.nn.functional.interpolate(
                    input_image, scale_factor=2, mode="bilinear"
                )[:, :3, :, :]
            ae_model = ManyPriors_RawImageCompressor(
                in_channels, device=torch.device("cpu")
            )
            optimizer = torch.optim.Adam(
                ae_model.get_parameters(lr=1e-4, bitEstimator_lr_multiplier=10), lr=1e-4
            )  # 0.0001)
            # get initial loss
            output = ae_model(input_image)
            init_loss_bpp = output["bpp"]
            init_loss_visual = torch.nn.functional.mse_loss(
                output["reconstructed_image"],
                input_image if in_channels == 3 else input_image_upscaled,
            )
            # train
            for i in range(100):
                optimizer.zero_grad()
                output = ae_model(input_image)
                loss_bpp = output["bpp"]
                loss_visual = torch.nn.functional.mse_loss(
                    output["reconstructed_image"],
                    input_image if in_channels == 3 else input_image_upscaled,
                )
                loss = loss_bpp + loss_visual
                loss.backward()
                optimizer.step()
                print(
                    f"{i=}: loss: {loss.item():.4f} ({loss_bpp.item()=:.4f} + {loss_visual.item()=:.4f})"
                )
            self.assertLess(loss_bpp.item(), init_loss_bpp.item())
            self.assertLess(loss_visual.item(), init_loss_visual.item())


if __name__ == "__main__":
    pass
