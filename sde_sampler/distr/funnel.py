from __future__ import annotations

import math

import torch

from .base import Distribution
from .gauss import IsotropicGauss


class Funnel(Distribution):
    def __init__(
        self,
        dim: int = 10,
        variance: float | None = None,
        n_reference_samples: int = int(1e7),
        log_norm_const: float = 0.0,
        domain_first_scale: float = 5.0,
        domain_other_scale: float = 5.0,
        domain_tol: float | None = 1e-5,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )
        self.variance = variance
        if self.variance is None:
            self.variance = self.dim - 1
        self.distr_first = IsotropicGauss(
            dim=1,
            scale=math.sqrt(self.variance),
            domain_scale=domain_first_scale,
            domain_tol=domain_tol,
        )
        self._initialize_distr()

        # Check domains
        if self.domain is None:
            domain_other = (
                self.distr_first.domain.sgn()
                * (self.distr_first.domain.abs() / domain_other_scale).exp()
            )
            self.set_domain(
                torch.cat(
                    [self.distr_first.domain, domain_other.repeat(self.dim - 1, 1)]
                )
            )
        if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
            raise ValueError("Domain does not satisfy tolerance at the boundary.")

    @staticmethod
    def log_prob_other(x_other, x_first):
        norm_const = -x_other.shape[-1] * (x_first + math.log(2.0 * math.pi)) / 2.0
        x_sq_sum = (x_other**2).sum(dim=-1, keepdim=True)
        return norm_const - 0.5 * x_sq_sum * (-x_first).exp()

    def _initialize_distr(self):
        self.distr_first._initialize_distr()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_first = x[:, 0].unsqueeze(-1)
        log_prob_first = self.distr_first.unnorm_log_prob(x_first)
        log_prob_other = Funnel.log_prob_other(x[:, 1:], x_first)
        assert log_prob_other.shape == log_prob_first.shape == (x.shape[0], 1)
        log_prob = log_prob_first + log_prob_other
        return log_prob + self.log_norm_const

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[-1] == self.dim
        x_first = x[:, 0].unsqueeze(-1)
        x_other = x[:, 1:]
        inv_var_other = (-x_first).exp()
        score_first = self.distr_first.score(x_first) - 0.5 * x_other.shape[-1]
        score_first += 0.5 * (x_other**2).sum(dim=-1, keepdim=True) * inv_var_other
        assert score_first.shape == (x.shape[0], 1)
        score_other = -x_other * inv_var_other
        return torch.cat([score_first, score_other], dim=-1)

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[-1] == 1
        if dim == 0:
            return self.distr_first.marginal(x)
        samples_first = self.distr_first.sample((self.n_reference_samples, 1))
        log_prob = self.log_prob_other(x, samples_first)
        return log_prob.exp().mean(axis=0)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples_first = self.distr_first.sample(shape)
        stdd_other = (0.5 * samples_first).exp()
        samples_other = torch.randn(*shape, self.dim - 1, device=samples_first.device)
        return torch.cat((samples_first, samples_other * stdd_other), dim=-1)
