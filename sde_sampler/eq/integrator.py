from __future__ import annotations

import logging

import torch
import torchsde

from sde_sampler.eq.sdes import TorchSDE
from sde_sampler.utils.common import get_timesteps


class Integrator:
    def integrate(
        self,
        sde: TorchSDE,
        ts: torch.Tensor,
        x_init: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        bm: torchsde.BaseBrownian | None = None,
    ):
        raise NotImplementedError


class TorchSDEIntegrator(Integrator):
    def __init__(
        self,
        sdeint_kwargs: dict | None = None,
        brownian_interval_kwargs: dict | None = None,
        adjoint: bool = False,
        stratonovich: bool = False,
    ):
        self.sdeint_kwargs = sdeint_kwargs or {}
        self.brownian_interval_kwargs = brownian_interval_kwargs or {}
        self.adjoint = adjoint
        self.stratonovich = stratonovich

    def integrate(
        self,
        sde: TorchSDE,
        ts: torch.Tensor,
        x_init: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        bm: torchsde.BaseBrownian | None = None,
    ) -> torch.Tensor:
        if bm is None:
            bm = torchsde.BrownianInterval(
                t0=ts[0],
                t1=ts[-1],
                size=x_init.shape,
                device=x_init.device,
                **self.brownian_interval_kwargs,
            )
        if timesteps is not None:
            logging.warning("Timesteps cannot be specified for torchsde integrator.")
        sde_type = sde.sde_type
        if self.stratonovich:
            sde.sde_type = "stratonovich"
        else:
            sde.sde_type = "ito"
        sdeint = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint
        output = sdeint(sde=sde, y0=x_init, ts=ts, bm=bm, **self.sdeint_kwargs)
        sde.sde_type = sde_type
        return output


def interpolate(
    ts: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    xs: torch.Tensor,
    xt: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    ind = torch.searchsorted(ts, t + eps, side="right")
    t_eval = ts[:ind]
    assert (s <= t_eval).all() and (t_eval <= t + eps).all()
    return torch.lerp(xs, xt, (t_eval.view(-1, 1, 1) - s) / (t - s))


class EulerIntegrator(Integrator):
    def __init__(
        self,
        dt: float | None = 0.01,
        steps: int | None = None,
        rescale_t: str | None = None,
        eps: float = 1e-8,
    ):
        self.dt = dt
        self.steps = steps
        self.rescale_t = rescale_t
        self.eps = eps

    def integrate(
        self,
        sde: TorchSDE,
        ts: torch.Tensor,
        x_init: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        bm: torchsde.BaseBrownian | None = None,
    ) -> torch.Tensor:
        if timesteps is None:
            timesteps = get_timesteps(
                ts[0],
                ts[-1],
                dt=self.dt,
                steps=self.steps,
                rescale_t=self.rescale_t,
                device=ts.device,
            )
        ts_count = 0
        xs_out = []
        xs = x_init
        for s, t in zip(timesteps[:-1], timesteps[1:]):
            if bm is None:
                noise = torch.randn(*xs.shape, device=xs.device) * torch.sqrt(t - s)
            else:
                noise = bm(s, t)
            xt = xs + sde.drift(s, xs) * (t - s) + sde.diff(s, xs) * noise

            if ts[ts_count] <= t + self.eps:
                xs_out.append(interpolate(ts[ts_count:], s, t, xs, xt, eps=self.eps))
                ts_count += xs_out[-1].shape[0]
            xs = xt

        xs_out = torch.cat(xs_out)
        assert ts_count == xs_out.shape[0]
        return xs_out
