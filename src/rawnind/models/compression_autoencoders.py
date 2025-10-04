"""Base model for hyperprior/manyprior based network."""

# -*- coding: utf-8 -*-

import math
import sys
from typing import Optional, Type, Literal
from typing_extensions import Self
import torch
from torch import nn


sys.path.append("..")
from common.extlibs import gdn

# logger = logging.getLogger("ImageCompression")


class AbstractRawImageCompressor(nn.Module):
    def __init__(
        self,
        device: torch.device,
        in_channels: int,
        hidden_out_channels: Optional[int] = None,
        bitstream_out_channels: Optional[int] = None,
        encoder_cls: Optional[Type[nn.Module]] = None,
        decoder_cls: Optional[Type[nn.Module]] = None,
        preupsample=False,
    ):
        super().__init__()
        self.device: torch.device = device
        self.in_channels: int = in_channels
        if encoder_cls and decoder_cls:
            self.Encoder = encoder_cls(
                hidden_out_channels=hidden_out_channels,
                bitstream_out_channels=bitstream_out_channels,
                in_channels=in_channels,
                device=device,
                preupsample=preupsample,
            )
            self.Decoder = decoder_cls(
                hidden_out_channels=hidden_out_channels,
                bitstream_out_channels=bitstream_out_channels,
                device=device,
            )

    def forward(self, input_image: torch.Tensor) -> dict:
        """
        Takes an input image batch (b,c,h,w), returns a dictionary containing
        "reconstructed_image": (b,c,h,w) tensor, encoded-decoded image,
        "visual_loss": float tensor, visual loss used to train the model,
        "bpp": float tensor, total bpp of the bitstream
        "bpp_feature": (optional) float tensor, bpp of the main features
        "bpp_sidestring": (optional) float tensor, bpp of the side information
        """
        pass

    def cpu(self) -> Self:
        self.device = torch.device("cpu")
        return self.to(self.device)

    def todev(self, device: torch.device) -> Self:
        self.device = device
        return self.to(self.device)


class BalleEncoder(nn.Module):
    """
    Image encoder for RGB (3ch) or Bayer (4ch) images.
    """

    def __init__(
        self,
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
        in_channels: Literal[3, 4] = 3,
        preupsample: bool = False,
    ):
        super().__init__()
        assert (in_channels == 3 and not preupsample) or in_channels == 4
        self.gdn1 = gdn.GDN(hidden_out_channels, device=device)
        self.gdn2 = gdn.GDN(hidden_out_channels, device=device)
        self.gdn3 = gdn.GDN(hidden_out_channels, device=device)
        if preupsample:
            self.preprocess = torch.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
        else:
            self.preprocess = torch.nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, hidden_out_channels, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(
            self.conv1.weight.data,
            (math.sqrt(2 * (in_channels + hidden_out_channels) / (6))),
        )
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

        self.conv2 = nn.Conv2d(
            hidden_out_channels, hidden_out_channels, 5, stride=2, padding=2
        )
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)

        self.conv3 = nn.Conv2d(
            hidden_out_channels, hidden_out_channels, 5, stride=2, padding=2
        )
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

        self.conv4 = nn.Conv2d(
            hidden_out_channels, bitstream_out_channels, 5, stride=2, padding=2
        )
        torch.nn.init.xavier_normal_(
            self.conv4.weight.data,
            (
                math.sqrt(
                    2
                    * (bitstream_out_channels + hidden_out_channels)
                    / (hidden_out_channels + hidden_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)


# class BayerPreUpEncoder(BalleEncoder):
#     def __init__(
#         self,
#         device: torch.device,
#         hidden_out_channels: int = 192,
#         bitstream_out_channels: int = 320,
#         in_channels: Literal[3, 4] = 4,
#     ):
#         assert in_channels == 4
#         super().__init__(
#             device, hidden_out_channels, bitstream_out_channels, in_channels
#         )
#         # bicubic upsampler
#         self.upsampler = nn.Upsample(
#             scale_factor=2, mode="bicubic", align_corners=False
#         )

#     def forward(self, x):
#         x = self.upsampler(x)
#         return super().forward(x)


class BalleDecoder(nn.Module):
    """
    Image decoder for RGB (3ch) images.
    """

    def __init__(
        self,
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
    ):
        super().__init__()
        self.igdn1 = gdn.GDN(ch=hidden_out_channels, inverse=True, device=device)
        self.igdn2 = gdn.GDN(ch=hidden_out_channels, inverse=True, device=device)
        self.igdn3 = gdn.GDN(ch=hidden_out_channels, inverse=True, device=device)

        self.deconv1 = nn.ConvTranspose2d(
            bitstream_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(
            self.deconv1.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (bitstream_out_channels + hidden_out_channels)
                    / (bitstream_out_channels + bitstream_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        # self.igdn1 = GDN.GDN(hidden_out_channels, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(
            hidden_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        # self.igdn2 = GDN.GDN(hidden_out_channels, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(
            hidden_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        # self.igdn3 = GDN.GDN(hidden_out_channels, inverse=True)

        self.output_module = nn.ConvTranspose2d(
            hidden_out_channels,
            3,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(
            self.output_module.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (hidden_out_channels + 3)
                    / (hidden_out_channels + hidden_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(self.output_module.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        return self.output_module(x)


class BayerPSDecoder(BalleDecoder):
    def __init__(
        self,
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
    ):
        super().__init__(device, hidden_out_channels, bitstream_out_channels)
        deconv4 = nn.ConvTranspose2d(
            hidden_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(deconv4.weight.data, (math.sqrt(2 * 1)))
        torch.nn.init.constant_(deconv4.bias.data, 0.01)
        finalconv = torch.nn.Conv2d(hidden_out_channels, 4 * 3, 1)
        torch.nn.init.xavier_normal_(
            finalconv.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (hidden_out_channels + 4 * 3)
                    / (hidden_out_channels + hidden_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(finalconv.bias.data, 0.01)
        self.output_module = nn.Sequential(deconv4, finalconv, nn.PixelShuffle(2))


class BayerTCDecoder(BalleDecoder):
    def __init__(
        self,
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
    ):
        super().__init__(device, hidden_out_channels, bitstream_out_channels)
        deconv4 = nn.ConvTranspose2d(
            hidden_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(deconv4.weight.data, (math.sqrt(2 * 1)))
        torch.nn.init.constant_(deconv4.bias.data, 0.01)
        final_act = nn.LeakyReLU()
        finalconv = nn.ConvTranspose2d(
            hidden_out_channels,
            3,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(
            finalconv.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (hidden_out_channels + 3)
                    / (hidden_out_channels + hidden_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(finalconv.bias.data, 0.01)

        self.output_module = nn.Sequential(deconv4, final_act, finalconv)


# class RawImageEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         raise NotImplementedError
#         # self.compressor = compressor

#     # def forward(self, input_image):
#     # return self.compressor.encode(input_image, entropy_coding=False)


# class RawImageDecoder(nn.Module):
#     def __init__(self, compressor):
#         super().__init__()
#         raise NotImplementedError
#         # self.compressor = compressor

#     # def forward(self, input_image):
#     # return self.compressor.decode(input_image)
