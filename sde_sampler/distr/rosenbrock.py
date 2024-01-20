from __future__ import annotations

import torch

from .base import Distribution


class Rosenbrock(Distribution):
    def __init__(self, dim: int = 5, flatness: float = 1.0, scale: float = 1.0):
        super().__init__(dim=dim)
        self.register_buffer("flatness", torch.tensor(flatness), persistent=False)
        self.register_buffer("scale", torch.tensor(scale), persistent=False)

    @staticmethod
    def objective(x: torch.Tensor) -> torch.Tensor:
        return (100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2).sum(
            dim=-1, keepdim=True
        )

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -Rosenbrock.objective(x) / self.flatness + torch.log(self.scale)
