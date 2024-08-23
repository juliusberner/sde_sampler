from __future__ import annotations

import torch
import math

from .base import Distribution


class Phi4Distr(Distribution):
    r"""Action (unnormalized logprob) for the real-valued :math:`\phi^4` scalar field Theory in 1+1 dimensions.
    For reference see eq. :math:`S = \dots`, in https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.126.032001.
    It has two free parameters the bare coupling :math:`\lambda` and the hopping parameter :math:`\kappa`.
    The action reads

    .. math::

        S(\phi) = \sum_{x \in \Lambda} - 2 \kappa \sum_{\hat{\mu}=1}^2 \phi(x) \phi(x + \hat{\mu}) +
        (1 - 2 \lambda) \phi(x)^2 + \lambda \, \phi(x)^4 \,.

    The first sum runs over all sites :math:`x` of a lattice with volume :math:`\Lambda` and the second sum runs
    over nearest neighbours in two dimensions indicated by :math:`\hat{\mu}`.

    """
    def __init__(
        self,
        dim: int = 16,
        kappa: float = 0.3,
        lambd: float = 0.022,
        lat_shape: list = None,
        **kwargs,
    ):
        """Constructs all the necessary attributes for the `Phi4Action` action.

        Parameters
        ----------
        dim : int
            Dimensionality of the distribution.
        kappa : float
            The hopping parameter.
        lambd : float
            The bare coupling.
        lat_shape: :py:obj:torc.Tensor
            Desired shape of the lattice.

        """
        super().__init__(dim=2, **kwargs)
        if lat_shape is None:
            lat_shape = [4, 4]

        self.kappa = kappa
        self.lambd = lambd
        prod = math.prod(lat_shape)

        if len(lat_shape) != 2:
            raise ValueError(f"The lattice configuration has an invalid shape {len(self.lat_shape)} instead of 2.\n "
                             "Only 2D systems are supported for the `Phi4Action` action.")

        if prod != dim:
            raise ValueError(f"The number of dimension {dim} does not match the desired lattice"
                             f" shape {prod}. Please check and try again!")

        self.lat_shape = lat_shape

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Takes a batch of lattice configurations and evaluates the `Phi4Action` action in batches.

        Parameters
        ----------
        x : :py:obj:`torch.Tensor`
            Batch of the lattice configurations.

        Returns
        -------
        :py:obj:`torch.Tensor`
           Action evaluated at the given lattice configurations.

        Raises
        ------
        ValueError
            When the shape of the lattice configuration is not of length 3.

        """
        x = x.unsqueeze(dim=-1).reshape(len(x), self.lat_shape[0], self.lat_shape[1])
        kinetic = (-2 * self.kappa) * x * (torch.roll(x, 1, -1) + torch.roll(x, 1, -2))
        mass = (1 - 2 * self.lambd) * x ** 2
        inter = self.lambd * x ** 4
        action = (kinetic + mass + inter).reshape(len(x), -1)
        return action.sum(-1, keepdim=True)

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Computes the derivative of the phi-4 action, i.e., the score term for the Boltzmann distribution.

        Parameters
        ----------
        x : :py:obj:`torch.Tensor`
            Batch of the lattice configurations.

        Returns
        -------
        :py:obj:`torch.Tensor`
           Score function (grad of unnormalized log prob) evaluated at the given lattice configurations.

        """
        # Reshape not needed here as there's no interaction term, i.e., sum over nearest neighbours.
        # x = x.unsqueeze(dim=-1).reshape(len(x), self.lat_shape[0], self.lat_shape[1])
        free = 2 * (1 - 2 * self.lambd - 2 * self.kappa) * x
        inter = 4 * self.lambd * x ** 3
        der_action = (free + inter)
        return der_action
