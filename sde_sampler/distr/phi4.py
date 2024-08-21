from __future__ import annotations

import torch

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
        dim: int = 2,
        kappa: float = 0.3,
        lambd: float = 0.022,
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

        """
        # if not dim == 2:
        #   raise ValueError(r"`dim` needs to be `2` for $\phi^4$-theory.")
        super().__init__(dim=2, **kwargs)
        self.kappa = kappa
        self.lambd = lambd

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

        # if len(x.shape) != 3:
        #    raise ValueError(f"The lattice configuration has an invalid shape {x.shape} instead of 3.\n "
        #                     "Only 2D systems are supported for the `Phi4Action` action.")

        kinetic = (-2 * self.kappa) * x * (torch.roll(x, 1, -1) + torch.roll(x, 1, -2))
        mass = (1 - 2 * self.lambd) * x ** 2
        inter = self.lambd * x ** 4

        return (kinetic + mass + inter).sum(-1).sum(-1)

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
        kinetic = (-2 * self.kappa) * (torch.roll(x, 1, -1) + torch.roll(x, 1, -2))
        mass = 2 * (1 - 2 * self.lambd) * x
        inter = 4 * self.lambd * x ** 3
        return (kinetic + mass + inter).sum(-1).sum(-1)

    def marginal(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.pdf(x)
