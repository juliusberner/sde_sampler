from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn


class Model(nn.Module):
    """
    Base class for different models.
    """

    def __init__(self, dim: int):
        super().__init__()

        # Dims
        self.dim_in = dim + 1
        self.dim_out = dim

    @staticmethod
    def init_linear(
        layer: nn.Linear,
        bias_init: Callable | None = None,
        weight_init: Callable | None = None,
    ):
        if bias_init:
            bias_init(layer.bias)
        if weight_init:
            weight_init(layer.weight)

    def flatten(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0 or t.shape[0] == 1:
            t = t.expand(x.shape[0], 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        assert x.shape[-1] == self.dim_out
        assert t.shape == (x.shape[0], 1)
        return torch.cat([t, x], dim=1)


class TimeEmbed(Model):
    def __init__(
        self,
        dim: int,
        activation: Callable,
        num_layers: int = 2,
        channels: int = 64,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(dim=dim)
        self.channels = channels
        self.activation = activation
        self.register_buffer(
            "timestep_coeff",
            torch.linspace(start=0.1, end=100, steps=self.channels).unsqueeze(0),
            persistent=False,
        )
        self.timestep_phase = nn.Parameter(torch.randn(1, self.channels))
        self.hidden_layer = nn.ModuleList([nn.Linear(2 * self.channels, self.channels)])
        self.hidden_layer += [
            nn.Linear(self.channels, self.channels) for _ in range(num_layers - 2)
        ]
        self.out_layer = nn.Linear(self.channels, self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )

    def forward(self, t: torch.Tensor, *args) -> torch.Tensor:
        assert t.ndim in [0, 1, 2]
        if t.ndim == 2:
            assert t.shape[1] == 1
        t = t.view(-1, 1).float()
        sin_embed_t = torch.sin((self.timestep_coeff * t) + self.timestep_phase)
        cos_embed_t = torch.cos((self.timestep_coeff * t) + self.timestep_phase)
        assert cos_embed_t.shape == (t.shape[0], self.channels)
        embed_t = torch.cat([sin_embed_t, cos_embed_t], dim=1)
        for layer in self.hidden_layer:
            embed_t = self.activation(layer(embed_t))
        return self.out_layer(embed_t)


class FourierMLP(Model):
    def __init__(
        self,
        dim: int,
        activation: Callable,
        num_layers: int = 4,
        channels: int = 64,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(dim=dim)
        self.channels = channels
        self.activation = activation
        self.input_embed = nn.Linear(self.dim_out, self.channels)
        self.timestep_embed = TimeEmbed(
            dim=self.channels,
            activation=self.activation,
            num_layers=2,
            channels=self.channels,
        )
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(self.channels, self.channels) for _ in range(num_layers - 2)]
        )
        self.out_layer = nn.Linear(self.channels, self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1).expand(x.shape[0], 1).float()
        embed_t = self.timestep_embed(t)
        embed_x = self.input_embed(x)
        assert embed_t.shape == embed_x.shape
        embed = embed_x + embed_t
        for layer in self.hidden_layer:
            embed = layer(self.activation(embed))
        return self.out_layer(self.activation(embed))


class FeedForward(Model):
    def __init__(
        self,
        dim: int,
        arch: Sequence[int],
        activation: Callable,
        normalization_factory=None,
        normalization_kwargs=None,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(dim=dim)

        # Affine linear layer
        bias = normalization_factory is None
        self.hidden_layer = nn.ModuleList([nn.Linear(self.dim_in, arch[0], bias=bias)])
        self.hidden_layer += [
            nn.Linear(arch[i], arch[i + 1], bias=bias) for i in range(len(arch) - 1)
        ]
        self.out_layer = nn.Linear(arch[-1], self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )

        # Activation function
        self.activation = activation

        # Normalization layer
        if normalization_factory:
            normalization_kwargs = normalization_kwargs or {}
            self.norm_layer = nn.ModuleList(
                [
                    normalization_factory(num_features, **normalization_kwargs)
                    for num_features in arch
                ]
            )
        else:
            self.norm_layer = None

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tensor = self.flatten(t, x)

        for i, linear in enumerate(self.hidden_layer):
            tensor = self.activation(linear(tensor))
            if self.norm_layer is not None:
                tensor = self.norm_layer[i](tensor)

        return self.out_layer(tensor)


class DenseNet(Model):
    def __init__(
        self,
        dim: int,
        arch: list[int],
        activation: Callable,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(dim=dim)
        self.nn_dims = [self.dim_in] + arch
        self.hidden_layer = nn.ModuleList(
            [
                nn.Linear(sum(self.nn_dims[: i + 1]), self.nn_dims[i + 1])
                for i in range(len(self.nn_dims) - 1)
            ]
        )
        self.out_layer = nn.Linear(sum(self.nn_dims), self.dim_out)
        Model.init_linear(
            self.out_layer, bias_init=last_bias_init, weight_init=last_weight_init
        )
        self.activation = activation

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tensor = self.flatten(t, x)
        for layer in self.hidden_layer:
            tensor = torch.cat([tensor, self.activation(layer(tensor))], dim=1)
        return self.out_layer(tensor)


class LevelNet(Model):
    """
    Network module for a single level
    """

    def __init__(
        self,
        dim,
        dim_embed,
        level,
        activation,
        normalization_factory=None,
        normalization_kwargs=None,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(dim=dim)

        self.level = level
        bias = normalization_factory is None
        self.dense_layers = nn.ModuleList(
            [nn.Linear(self.dim_in, dim_embed, bias=bias)]
        )
        self.dense_layers += [
            nn.Linear(dim_embed, dim_embed, bias=bias) for _ in range(2**level - 1)
        ]
        self.dense_layers.append(nn.Linear(dim_embed, self.dim_out))
        Model.init_linear(
            self.dense_layers[-1],
            bias_init=last_bias_init,
            weight_init=last_weight_init,
        )  # \U$1: ignore
        if normalization_factory is None:
            self.norm_layers = None
        else:
            normalization_kwargs = normalization_kwargs or {}
            self.norm_layers = nn.ModuleList(
                [
                    normalization_factory(dim_embed, **normalization_kwargs)
                    for _ in range(2**level)
                ]
            )
        self.act = activation

    def forward(self, t, x, res_tensors=None):
        tensor = self.flatten(t, x)
        out_tensors = []
        tensor = self.dense_layers[0](tensor)
        for i, dense in enumerate(self.dense_layers[1:]):  # \U$1: ignore
            if self.norm_layers is not None:
                tensor = self.norm_layers[i](tensor)
            tensor = self.act(tensor)
            tensor = dense(tensor)
            if res_tensors:
                tensor = tensor + res_tensors[i]
            if i % 2 or self.level == 0:
                out_tensors.append(tensor)
        return out_tensors


class MultilevelNet(Model):
    """
    Multilevel net
    """

    def __init__(
        self,
        dim,
        activation,
        factor=5,
        levels=4,
        normalization_factory=None,
        normalization_kwargs=None,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(dim=dim)
        self.nets = nn.ModuleList(
            [
                LevelNet(
                    dim=dim,
                    dim_embed=factor * self.dim_in,
                    level=level,
                    activation=activation,
                    normalization_factory=normalization_factory,
                    normalization_kwargs=normalization_kwargs,
                    last_bias_init=last_bias_init,
                    last_weight_init=last_weight_init,
                )
                for level in range(levels)
            ]
        )

    def forward(self, t, x):
        res_tensors = None
        for net in self.nets[::-1]:  # \U$1: ignore
            res_tensors = net(t, x, res_tensors)
        assert res_tensors is not None
        return res_tensors[-1]
