from __future__ import annotations

import torch

from .gauss import Gauss


class Delta(Gauss):
    def __init__(
        self,
        dim: int = 1,
        loc: torch.Tensor | float = 0.0,
        approx_scale: float = 1e-3,
        domain_scale: float = 10,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            loc=loc,
            scale=approx_scale,
            domain_scale=domain_scale,
            **kwargs,
        )

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        return self.loc.repeat(*shape, 1)
