# -*- coding: utf-8 -*-
from torch import nn
import torch
import sys

sys.path.append("..")
from rawnind.libs import rawproc

"""
# U-Net with transposed convolutions (consistent shape) and concatenations rather than additions.
"""


class Denoiser(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        assert in_channels == 3 or in_channels == 4, f"{in_channels=} should be 3 or 4"


class Passthrough(Denoiser):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        self.dummy_parameter = torch.nn.Parameter(torch.randn(3))
        if kwargs:
            print(f"Passthrough: ignoring unexpected kwargs: {kwargs}")

    def forward(self, batch: torch.Tensor):
        if self.in_channels == 3:
            return batch
        debayered_batch: torch.Tensor = rawproc.demosaic(batch)
        debayered_batch.requires_grad_()
        return debayered_batch


def get_activation_class_params(activation: str) -> tuple:
    if activation == "PReLU":
        return nn.PReLU, {}
    elif activation == "ELU":
        return nn.ELU, {"inplace": True}
    elif activation == "Hardswish":
        return nn.Hardswish, {"inplace": True}
    elif activation == "LeakyReLU":
        return nn.LeakyReLU, {"inplace": True, "negative_slope": 0.2}
        # negative_slope from # per https://github.com/lavi135246/pytorch-Learning-to-See-in-the-Dark/blob/master/model.py
    else:
        exit(f"get_activation_class: unknown activation function: {activation}")


class UtNet2(Denoiser):
    def __init__(
        self,
        in_channels: int,
        funit: int = 32,
        activation: str = "LeakyReLU",
        preupsample: bool = False,
    ):
        super().__init__(in_channels=in_channels)
        assert (in_channels == 3 and not preupsample) or in_channels == 4
        activation_fun, activation_params = get_activation_class_params(activation)
        # self.pad = nn.ReflectionPad2d(2)
        if preupsample:
            self.preprocess = torch.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
        else:
            self.preprocess = torch.nn.Identity()
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels, funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(funit, funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.maxpool = nn.MaxPool2d(2)
        self.convs2 = nn.Sequential(
            nn.Conv2d(funit, 2 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(2 * funit, 2 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.convs3 = nn.Sequential(
            nn.Conv2d(2 * funit, 4 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(4 * funit, 4 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.convs4 = nn.Sequential(
            nn.Conv2d(4 * funit, 8 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(8 * funit, 8 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.bottom = nn.Sequential(
            nn.Conv2d(8 * funit, 16 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(16 * funit, 16 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.up1 = nn.ConvTranspose2d(16 * funit, 8 * funit, 2, stride=2)
        self.tconvs1 = nn.Sequential(
            nn.Conv2d(16 * funit, 8 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(8 * funit, 8 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.up2 = nn.ConvTranspose2d(8 * funit, 4 * funit, 2, stride=2)
        self.tconvs2 = nn.Sequential(
            nn.Conv2d(8 * funit, 4 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(4 * funit, 4 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.up3 = nn.ConvTranspose2d(4 * funit, 2 * funit, 2, stride=2)
        self.tconvs3 = nn.Sequential(
            nn.Conv2d(4 * funit, 2 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(2 * funit, 2 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.up4 = nn.ConvTranspose2d(2 * funit, funit, 2, stride=2)
        self.tconvs4 = nn.Sequential(
            nn.Conv2d(2 * funit, funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(funit, funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        if in_channels == 3 or preupsample:
            self.output_module = nn.Sequential(nn.Conv2d(funit, 3, 1))
        elif in_channels == 4:
            self.output_module = nn.Sequential(
                nn.Conv2d(funit, 4 * 3, 1), nn.PixelShuffle(2)
            )
        else:
            raise NotImplementedError(f"{in_channels=}")
        # self.unpad = nn.ZeroPad2d(-2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # TODO try xavier_normal_ ?

    def forward(self, l):
        l1 = self.preprocess(l)
        # l = self.pad(l)
        l1 = self.convs1(l1)
        l2 = self.convs2(self.maxpool(l1))
        l3 = self.convs3(self.maxpool(l2))
        l4 = self.convs4(self.maxpool(l3))
        l = torch.cat([self.up1(self.bottom(self.maxpool(l4))), l4], dim=1)
        l = torch.cat([self.up2(self.tconvs1(l)), l3], dim=1)
        l = torch.cat([self.up3(self.tconvs2(l)), l2], dim=1)
        l = torch.cat([self.up4(self.tconvs3(l)), l1], dim=1)
        l = self.tconvs4(l)
        # l = self.unpad(l)
        return self.output_module(l)


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        activation="LeakyReLU",
    ):
        super().__init__()
        activation_fun, activation_params = get_activation_class_params(activation)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            activation_fun(**activation_params),
        )

    def forward(self, x):
        return self.conv(x) + x


class UtNet3(UtNet2):
    def __init__(self, in_channels: int = 4, funit: int = 32, activation="LeakyReLU"):
        super().__init__(in_channels=in_channels, funit=funit, activation=activation)
        assert in_channels == 4
        self.output_module = nn.Sequential(
            torch.nn.Conv2d(funit, funit * 8, 1),
            ResBlock(
                funit * 8,
            ),
            ResBlock(funit * 8),
            torch.nn.Conv2d(funit * 8, funit, 1),
            self.output_module,
        )


architectures = {"UtNet2": UtNet2, "Passthrough": Passthrough}


if __name__ == "__main__":
    utnet3 = UtNet3(in_channels=4)
    rawtensor = torch.rand(1, 4, 16, 16)
    output = utnet3(rawtensor)
    print(f"{rawtensor.shape=}, {output.shape=}")

    rawnet = UtNet2(in_channels=4)
    rgbnet = UtNet2(in_channels=3)
    rgbtensor = torch.rand(1, 3, 16, 16)

    print(f"{rawtensor.shape=}, {rawnet(rawtensor).shape=}")
    print(f"{rgbtensor.shape=}, {rgbnet(rgbtensor).shape=}")
