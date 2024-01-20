from __future__ import annotations

import logging
import math
from numbers import Number

import torch
from torch import distributions
from torch.nn.init import trunc_normal_

from .base import Distribution


def gmm_params(name: str = "heart", dim: int = 2):
    if name == "heart":
        loc = 1.5 * torch.tensor(
            [
                [-0.5, -0.25],
                [0.0, -1],
                [0.5, -0.25],
                [-1.0, 0.5],
                [-0.5, 1.0],
                [0.0, 0.5],
                [0.5, 1.0],
                [1.0, 0.5],
            ]
        )
        factor = 1 / len(loc)

    elif name == "dist":
        loc = torch.tensor(
            [
                [0.0, 0.0],
                [2, 0.0],
                [0.0, 3.0],
                [-4, 0.0],
                [0.0, -5],
            ]
        )
        factor = math.sqrt(0.2)

    elif name in ["fab", "multi"]:
        n_mixes, loc_scaling = (40, 40) if name == "fab" else (80, 80)
        generator = torch.Generator()
        generator.manual_seed(42)
        loc = (torch.rand((n_mixes, 2), generator=generator) - 0.5) * 2 * loc_scaling
        factor = torch.nn.functional.softplus(torch.tensor(1.0, device=loc.device))
    elif name == "grid":
        x_coords = torch.linspace(-5, 5, 3)
        loc = torch.cartesian_prod(x_coords, x_coords)
        factor = math.sqrt(0.3)
    elif name == "circle":
        freq = 2 * torch.pi * torch.arange(1, 9) / 8
        loc = torch.stack([4.0 * freq.cos(), 4.0 * freq.sin()], dim=1)
        factor = math.sqrt(0.3)
    else:
        raise ValueError("Unknown mode for the Gaussian mixture.")

    if dim > 2:
        loc = torch.cat([loc, torch.zeros(8, dim - 2)], dim=1)
    scale = factor * torch.ones_like(loc)
    mixture_weights = torch.ones(loc.shape[0], device=loc.device)
    return loc, scale, mixture_weights


class GMM(Distribution):
    def __init__(
        self,
        dim: int = 2,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        mixture_weights: torch.Tensor | None = None,
        n_reference_samples: int = int(1e7),
        name: str | None = None,
        log_norm_const: float = 0.0,
        domain_scale: float = 5,
        domain_tol: float | None = 1e-5,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )
        if name is not None:
            if any(t is not None for t in [loc, scale, mixture_weights]):
                logging.warning(
                    "Ignoring loc, scale, and mixture weights since name is specified."
                )
            loc, scale, mixture_weights = gmm_params(name, dim=dim)

        # Check shapes
        n_mixtures = loc.shape[0]
        if not loc.shape == scale.shape == (n_mixtures, self.dim):
            raise ValueError("Shape missmatch between loc and scale.")
        if mixture_weights is None and n_mixtures > 1:
            raise ValueError("Require mixture weights.")
        if not (mixture_weights is None or mixture_weights.shape == (n_mixtures,)):
            raise ValueError("Shape missmatch for the mixture weights.")

        # Initialize
        self.register_buffer("loc", loc, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("mixture_weights", mixture_weights, persistent=False)
        self._initialize_distr()

        # Check domain
        if self.domain is None:
            deviation = domain_scale * self.scale.max(dim=0).values
            deviation = torch.stack([-deviation, deviation], dim=-1)
            pos = torch.stack(
                [self.loc.min(dim=0).values, self.loc.max(dim=0).values], dim=-1
            )
            self.set_domain(pos + deviation)
        if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
            raise ValueError("domain does not satisfy tolerance at the boundary.")

    @property
    def stddevs(self) -> torch.Tensor:
        return self.distr.variance.sqrt()

    def _initialize_distr(
        self,
    ) -> distributions.MixtureSameFamily | distributions.Independent:
        if self.mixture_weights is None:
            self.distr = distributions.Independent(
                distributions.Normal(self.loc.squeeze(0), self.scale.squeeze(0)), 1
            )
        else:
            modes = distributions.Independent(
                distributions.Normal(self.loc, self.scale), 1
            )
            mix = distributions.Categorical(self.mixture_weights)
            self.distr = distributions.MixtureSameFamily(mix, modes)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.distr.log_prob(x).unsqueeze(-1) + self.log_norm_const
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        if self.mixture_weights is None:
            return distributions.Normal(self.loc[0, dim], self.scale[0, dim])
        modes = distributions.Normal(self.loc[:, dim], self.scale[:, dim])
        mix = distributions.Categorical(self.mixture_weights)
        return distributions.MixtureSameFamily(mix, modes)

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        return self.marginal_distr(dim=dim).log_prob(x).exp()

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        return self.distr.sample(torch.Size(shape))


class Gauss(GMM):
    def __init__(
        self,
        dim: int = 1,
        loc: torch.Tensor | Number = 0.0,
        scale: torch.Tensor | Number = 1.0,
        **kwargs,
    ):

        # Setup parameters
        params = {"loc": loc, "scale": scale}
        params = {k: Gauss._prepare_input(p, dim) for k, p in params.items()}
        super().__init__(dim=dim, **params, **kwargs)
        self.stddevs = self.scale.squeeze(0)

    @staticmethod
    def _prepare_input(param: torch.Tensor | Number, dim: int = 1):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float)
        param = torch.atleast_2d(param)
        if param.numel() == 1:
            param = param.repeat(1, dim)
        return param

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.loc - x) / self.scale**2


class IsotropicGauss(Gauss):
    # Typially used as prior (supports truncation and faster methods)
    def __init__(
        self,
        dim: int = 1,
        loc: float = 0.0,
        scale: float = 1.0,
        truncate_quartile: float | None = None,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            loc=loc,
            scale=scale,
            **kwargs,
        )

        assert torch.allclose(self.loc, self.loc[0, 0])
        assert torch.allclose(self.scale, self.scale[0, 0])

        # Calculate truncation values
        if truncate_quartile is not None:
            quartiles = torch.tensor(
                [truncate_quartile / 2, 1 - truncate_quartile / 2],
                device=self.domain.device,
            )
            truncate_quartile = self.marginal_distr().icdf(quartiles).tolist()
        self.truncate_quartile = truncate_quartile

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        var = self.scale[0, 0] ** 2
        norm_const = -0.5 * self.dim * (2.0 * math.pi * var).log()
        norm_const += self.log_norm_const
        sq_sum = torch.sum((x - self.loc[0, 0]) ** 2, dim=-1, keepdim=True)
        return norm_const - 0.5 * sq_sum / var

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.loc[0, 0] - x) / self.scale[0, 0] ** 2

    def marginal(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.marginal_distr().log_prob(x).exp()

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        if self.truncate_quartile is None:
            return self.loc[0, 0] + self.scale[0, 0] * torch.randn(
                *shape, self.dim, device=self.domain.device
            )
        tensor = torch.empty(*shape, self.dim, device=self.domain.device)
        return trunc_normal_(
            tensor,
            mean=self.loc[0, 0],
            std=self.scale[0, 0],
            a=self.truncate_quartile[0],
            b=self.truncate_quartile[1],
        )
