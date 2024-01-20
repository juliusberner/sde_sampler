"""
Adapted from https://github.com/qsh-zh/pis/
"""
from __future__ import annotations

import math

import torch
import torchquad

from .base import Distribution


class Rings(Distribution):
    def __init__(
        self,
        dim: int = 2,
        lower_rad: float = 1.0,
        upper_rad: float = 5.0,
        num_rad: int = 3,
        scale: float = 100.0,
        grid_points: int = 2001**2,
        scale_domain: float = 10.0,
        domain_tol: float | None = 1e-5,
        eps: float = 1e-8,
        **kwargs,
    ):
        if dim != 2:
            raise ValueError("The rings should be two-dimensional.")
        super().__init__(dim=dim, grid_points=grid_points, **kwargs)
        self.register_buffer(
            "r_centers", torch.linspace(lower_rad, upper_rad, num_rad), persistent=False
        )
        self.scale = scale
        self.eps = eps

        # Set domain
        self.domain_tol = domain_tol
        if self.domain is None:
            self.set_domain(
                self.r_centers.max() + scale_domain / math.sqrt(self.scale / 2)
            )

    def compute_stats(self):
        super().compute_stats()
        if (
            self.domain_tol is not None
            and (self.pdf(self.domain.T) > self.domain_tol).any()
        ):
            raise ValueError("Domain does not satisfy tolerance at the boundary.")

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        radius = torch.norm(x, dim=-1, keepdim=True)
        return (
            -self.scale
            * (radius - self.r_centers).square().min(dim=-1, keepdim=True).values
        )

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        radius = torch.norm(x, dim=-1, keepdim=True)
        indices = (radius - self.r_centers).square().min(dim=-1).indices
        centers = self.r_centers[indices].unsqueeze(-1)
        assert centers.shape == (x.shape[0], 1)
        return -2.0 * self.scale * (1 - centers / (radius + self.eps)) * x

    def _integrand(
        self, y: torch.Tensor, x: torch.Tensor, dim: int = 0
    ) -> torch.Tensor:
        x = x.tile(y.shape[0], 1, 1)
        y = y.unsqueeze(1).tile(1, x.shape[1], 1)
        assert y.shape == x.shape
        if dim == 0:
            tensor = torch.cat([x, y], dim=-1)
        else:
            tensor = torch.cat([y, x], dim=-1)
        return self.pdf(tensor).squeeze(-1)

    def marginal(self, x: torch.Tensor, dim: int = 0) -> torch.Tenor:
        integrator = torchquad.Boole()
        with torch.device(self.domain.device):
            domain = self.domain[dim].unsqueeze(0)
            integral = integrator.integrate(
                lambda y: self._integrand(y, x, dim=dim),
                dim=1,
                N=2001,
                integration_domain=domain,
            )
        return integral.unsqueeze(-1)
