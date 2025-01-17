"""Entropy model (copied ManyPriors classes from compression/models/bitEstimator.py)."""

import torch
from torch.nn import functional as F


class MultiHeadBitEstimator(torch.nn.Module):
    """
    Estimate cumulative distribution function.
    """

    def __init__(
        self,
        channel: int,
        nb_head: int,
        shape=("g", "bs", "ch", "h", "w"),
        bitparm_init_mode="normal",
        bitparm_init_range=0.01,
        **kwargs
    ):
        super(MultiHeadBitEstimator, self).__init__()
        self.f1 = MultiHeadBitparm(
            channel,
            nb_head=nb_head,
            shape=shape,
            bitparm_init_mode=bitparm_init_mode,
            bitparm_init_range=bitparm_init_range,
        )
        self.f2 = MultiHeadBitparm(
            channel,
            nb_head=nb_head,
            shape=shape,
            bitparm_init_mode=bitparm_init_mode,
            bitparm_init_range=bitparm_init_range,
        )
        self.f3 = MultiHeadBitparm(
            channel,
            nb_head=nb_head,
            shape=shape,
            bitparm_init_mode=bitparm_init_mode,
            bitparm_init_range=bitparm_init_range,
        )
        self.f4 = MultiHeadBitparm(
            channel,
            final=True,
            nb_head=nb_head,
            shape=shape,
            bitparm_init_mode=bitparm_init_mode,
            bitparm_init_range=bitparm_init_range,
        )

    #        if bs_first:
    #            self.prep_input_fun = lambda x: x.unsqueeze(0)
    #        else:
    #            self.prep_input_fun = lambda x: x

    def forward(self, x):
        # x = self.prep_input_fun(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


class MultiHeadBitparm(torch.nn.Module):
    """
    MultiHeadBitEstimator component.
    """

    def __init__(
        self,
        channel,
        nb_head,
        final=False,
        shape=("g", "bs", "ch", "h", "w"),
        bitparm_init_mode="normal",
        bitparm_init_range=0.01,
    ):
        super(MultiHeadBitparm, self).__init__()
        self.final = final
        if shape == (
            "g",
            "bs",
            "ch",
            "h",
            "w",
        ):  # used in Balle2017ManyPriors_ImageCompressor
            params_shape = (nb_head, 1, channel, 1, 1)
        elif shape == ("bs", "ch", "g", "h", "w"):
            params_shape = (1, channel, nb_head, 1, 1)
        if bitparm_init_mode == "normal":
            init_fun = torch.nn.init.normal_
            init_params = 0, bitparm_init_range
        elif bitparm_init_mode == "xavier_uniform":
            init_fun = torch.nn.init.xavier_uniform_
            init_params = [bitparm_init_range]
        else:
            raise NotImplementedError(bitparm_init_mode)
        self.h = torch.nn.Parameter(
            init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params)
        )
        self.b = torch.nn.Parameter(
            init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params)
        )
        if not final:
            self.a = torch.nn.Parameter(
                init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params)
            )
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)
