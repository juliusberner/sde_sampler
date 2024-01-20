from __future__ import annotations

import math

import plotly.graph_objects as go
import torch

from sde_sampler.eval.plots import plot_marginal

from .base import Distribution, rejection_sampling
from .gauss import GMM, IsotropicGauss


class DoubleWell(Distribution):
    def __init__(
        self,
        dim: int = 1,
        separation: float = 2.0,
        shift: float = 0.0,
        grid_points: int = 2001,
        rejection_sampling_scaling: float = 3.0,
        domain_delta: float = 2.5,
        **kwargs,
    ):
        if not dim == 1:
            raise ValueError("`dim` needs to be `1`. Consider using `MultiWell`.")
        super().__init__(dim=1, grid_points=grid_points, **kwargs)
        self.rejection_sampling_scaling = rejection_sampling_scaling
        self.register_buffer("separation", torch.tensor(separation), persistent=False)
        self.register_buffer("shift", torch.tensor(shift), persistent=False)

        # Set domain
        if self.domain is None:
            domain = self.shift + (
                self.separation.sqrt() + domain_delta
            ) * torch.tensor([[-1.0, 1.0]])
            self.set_domain(domain)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.shift
        return -((x**2 - self.separation) ** 2)

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x - self.shift
        return -4.0 * (x**2 - self.separation) * x

    def marginal(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.pdf(x)

    def get_proposal_distr(self):
        device = self.domain.device
        loc = self.shift + self.separation.sqrt() * torch.tensor(
            [[-1.0], [1.0]], device=device
        )
        scale = 1 / self.separation.sqrt() * torch.ones(2, 1, device=device)
        proposal = GMM(
            dim=1,
            loc=loc,
            scale=scale,
            mixture_weights=torch.ones(2, device=device),
            domain_tol=None,
        )
        proposal.to(device)
        return proposal

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        proposal = self.get_proposal_distr()
        return rejection_sampling(
            shape=shape,
            target=self,
            proposal=proposal,
            scaling=self.rejection_sampling_scaling,
        )

    def plots(self, samples, nbins=100) -> torch.Tensor:
        samples = self.sample((samples.shape[0],))
        fig = plot_marginal(
            x=samples,
            marginal=lambda x, **kwargs: self.pdf(x),
            dim=0,
            nbins=nbins,
            domain=self.domain,
        )

        x = torch.linspace(*self.domain[0], steps=nbins, device=self.domain.device)
        y = (
            self.get_proposal_distr().pdf(x.unsqueeze(-1))
            * self.rejection_sampling_scaling
        )
        fig.add_trace(
            go.Scatter(
                x=x.cpu(),
                y=y.squeeze(-1).cpu(),
                mode="lines",
                name="proposal",
            )
        )
        return {"plots/rejection_sampling": fig}


class MultiWell(Distribution):
    def __init__(
        self,
        dim: int = 2,
        n_double_wells: int = 1,
        separation: float = 2.0,
        shift: float = 0.0,
        domain_dw_delta: float = 2.5,
        domain_gauss_scale: float = 5.0,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        # Define parameters
        self.separation = separation
        if n_double_wells > dim or n_double_wells == 0:
            raise ValueError(f"Please specify between 1 and {dim} double wells.")
        self.n_double_wells = n_double_wells
        self.n_gauss = self.dim - self.n_double_wells

        # Initialize distributions
        self.double_well = DoubleWell(
            separation=separation, shift=shift, domain_delta=domain_dw_delta
        )
        domain = self.double_well.domain.repeat(self.n_double_wells, 1)
        self.gauss = None
        if self.n_gauss > 0:
            self.gauss = IsotropicGauss(
                dim=self.n_gauss,
                loc=shift,
                log_norm_const=0.5 * math.log(2.0 * math.pi) * self.n_gauss,
                domain_scale=domain_gauss_scale,
            )
            domain = torch.cat([domain, self.gauss.domain])

        # Set domain
        self.set_domain(domain)

    def _initialize_distr(self):
        if self.gauss is not None:
            return self.gauss._initialize_distr()

    def compute_stats(self):
        # Double well
        self.double_well.compute_stats()
        self.log_norm_const = self.double_well.log_norm_const * self.n_double_wells
        self.expectations = {
            name: exp * self.n_double_wells
            for name, exp in self.double_well.expectations.items()
        }
        self.stddevs = torch.cat([self.double_well.stddevs] * self.n_double_wells)

        # Gauss
        if self.gauss is not None:
            self.gauss.compute_stats()
            self.log_norm_const += self.gauss.log_norm_const
            for name in self.expectations:
                # This assumes that the expectations are reducing the dim via a sum
                self.expectations[name] += self.gauss.expectations[name]
            self.stddevs = torch.cat([self.stddevs, self.gauss.stddevs])

        assert (self.pdf(self.domain.T) < 1e-5).all()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.double_well.unnorm_log_prob(x[:, : self.n_double_wells]).sum(
            dim=-1, keepdim=True
        )
        if self.gauss is not None:
            log_prob += self.gauss.unnorm_log_prob(x[:, self.n_double_wells :])
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        score = self.double_well.score(x[:, : self.n_double_wells])
        if self.gauss is not None:
            score_gauss = self.gauss.score(x[:, self.n_double_wells :])
            score = torch.cat([score, score_gauss], dim=-1)
        return score

    def marginal(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if dim < self.n_double_wells:
            return self.double_well.marginal(x)
        return self.gauss.marginal(x)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples = self.double_well.sample(shape + (self.n_double_wells,)).squeeze(-1)
        if self.gauss is not None:
            samples_gauss = self.gauss.sample(shape)
            samples = torch.cat([samples, samples_gauss], dim=-1)
        return samples
