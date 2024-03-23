from __future__ import annotations

from typing import Callable

import torch
from torch.nn import Module

from sde_sampler.distr.gauss import Gauss
from sde_sampler.utils.common import clip_and_log


class TorchSDE(Module):
    noise_type: str = "diagonal"
    sde_type: str = "ito"

    def __init__(
        self,
        terminal_t: float = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            "terminal_t", torch.tensor(terminal_t, dtype=torch.float), persistent=False
        )

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.drift(t, x).expand_as(x)

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.diff(t, x).expand_as(x)


class LangevinSDE(TorchSDE):
    def __init__(
        self,
        target_score: Callable,
        diff_coeff: float = 1.0,
        clip_score: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_score = target_score
        self.register_buffer(
            "diff_coeff", torch.tensor(diff_coeff, dtype=torch.float), persistent=False
        )
        self.clip_score = clip_score

    # Drift
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        drift = self.target_score(x) * self.diff_coeff**2 / 2.0
        return clip_and_log(
            drift,
            max_norm=self.clip_score,
            name="score",
            t=t,
        )

    # Diffusion coefficient
    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.diff_coeff


class OU(TorchSDE):
    def __init__(
        self,
        generative=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Distinguish between coeff. functions for the inference and generative SDEs
        self.generative = generative
        self.sign = 1.0 if self.generative else -1.0

    def drift_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diff_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def drift_div(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.drift_coeff_t(t) * x.shape[-1]

    def drift_div_int(
        self, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return self.int_drift_coeff_t(s, t) * x.shape[-1]

    # Drift
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.drift_coeff_t(t) * x

    # Diffusion coefficient
    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.diff_coeff_t(t)

    def int_drift_coeff_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def int_diff_coeff_sq_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def marginal_params(
        self,
        t: torch.Tensor,
        x_init: torch.Tensor,
        var_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def marginal_distr(
        self,
        t: torch.Tensor,
        x_init: torch.Tensor,
        var_init: torch.Tensor | None = None,
    ) -> Gauss:
        loc, var = self.marginal_params(t, x_init, var_init=var_init)
        return Gauss(dim=x_init.shape[-1], loc=loc, scale=var.sqrt(), domain_tol=None)


class ConstOU(OU):
    def __init__(self, drift_coeff: float = 2.0, diff_coeff: float = 2.0, **kwargs):
        if drift_coeff < 0 or diff_coeff <= 0:
            raise ValueError("Choose non-negative drift_coeff and positive diff_coeff.")
        super().__init__(**kwargs)
        self.register_buffer(
            "drift_coeff",
            torch.tensor(drift_coeff, dtype=torch.float),
            persistent=False,
        )
        self.drift_coeff: torch.Tensor
        self.register_buffer(
            "diff_coeff", torch.tensor(diff_coeff, dtype=torch.float), persistent=False
        )
        self.diff_coeff: torch.Tensor

    def drift_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sign * self.drift_coeff

    def diff_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.diff_coeff

    def int_drift_coeff_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dt = t - s
        assert (dt >= 0).all()
        return self.sign * self.drift_coeff * dt

    def int_diff_coeff_sq_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dt = t - s
        assert (dt >= 0).all()
        return self.diff_coeff**2 * dt

    def marginal_params(
        self,
        t: torch.Tensor,
        x_init: torch.Tensor,
        var_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drift_coeff = self.sign * self.drift_coeff
        loc = torch.exp(drift_coeff * t)
        var = (
            -self.diff_coeff**2
            / (2 * drift_coeff)
            * (1 - torch.exp(2 * drift_coeff * t))
        )
        if var_init is not None:
            var = var + loc**2 * var_init
        return loc * x_init, var


class ScaledBM(ConstOU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, drift_coeff=0.0, **kwargs)

    def marginal_params(
        self,
        t: torch.Tensor,
        x_init: torch.Tensor,
        var_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var = self.diff_coeff**2 * t
        if var_init is not None:
            var = var + var_init
        return x_init, var


class VP(OU):
    def __init__(
        self,
        diff_coeff_sq_min: float = 0.1,
        diff_coeff_sq_max: float = 20.0,
        scale_diff_coeff: float = 1.0,
        **kwargs,
    ):
        """Construct a Variance Preserving SDE.

        Based on https://github.com/yang-song/score_sde
        """
        super().__init__(**kwargs)
        self.register_buffer(
            "scale_diff_coeff",
            torch.tensor(scale_diff_coeff, dtype=torch.float),
            persistent=False,
        )
        self.register_buffer(
            "diff_coeff_sq_min",
            torch.tensor(diff_coeff_sq_min, dtype=torch.float),
            persistent=False,
        )
        self.diff_coeff_sq_min: torch.Tensor
        self.register_buffer(
            "diff_coeff_sq_max",
            torch.tensor(diff_coeff_sq_max, dtype=torch.float),
            persistent=False,
        )
        self.diff_coeff_sq_max: torch.Tensor

    def _diff_coeff_sq_t(self, t: torch.Tensor) -> torch.Tensor:
        if self.generative:
            return torch.lerp(
                self.diff_coeff_sq_max, self.diff_coeff_sq_min, t / self.terminal_t
            )
        return torch.lerp(
            self.diff_coeff_sq_min, self.diff_coeff_sq_max, t / self.terminal_t
        )

    def drift_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sign * 0.5 * self._diff_coeff_sq_t(t)

    def diff_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.scale_diff_coeff * torch.sqrt(self._diff_coeff_sq_t(t))

    def int_drift_coeff_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dt = t - s
        assert (dt >= 0).all()
        return (
            self.sign
            * 0.25
            * (self._diff_coeff_sq_t(t) + self._diff_coeff_sq_t(s))
            * dt
        )

    def int_diff_coeff_sq_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dt = t - s
        assert (dt >= 0).all()
        return (
            0.5
            * self.scale_diff_coeff**2
            * (self._diff_coeff_sq_t(t) + self._diff_coeff_sq_t(s))
            * dt
        )

    def marginal_params(
        self,
        t: torch.Tensor,
        x_init: torch.Tensor,
        var_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.zeros(1, device=t.device)
        int_drift_coeff = self.int_drift_coeff_t(s, t)
        loc = torch.exp(int_drift_coeff)
        var = (1 - torch.exp(2 * int_drift_coeff)) * self.scale_diff_coeff**2
        if var_init is not None:
            var = var + loc**2 * var_init
        return loc * x_init, var


class ControlledSDE(TorchSDE):
    def __init__(
        self,
        sde: OU,
        ctrl: Callable | None,
        **kwargs,
    ):
        super().__init__(terminal_t=sde.terminal_t.item(), **kwargs)
        self.sde = sde
        self.sde_type = self.sde.sde_type
        self.noise_type = self.sde.noise_type
        self.ctrl = ctrl

    def drift(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.f_and_g(t, x)[0]

    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.sde.diff(t, x)

    # Minimal speedup by saving one diff evaluation for torchsde
    def f_and_g(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sde_diff = self.sde.diff(t, x)
        sde_drift = self.sde.drift(t, x)
        if self.ctrl is not None:
            if not self.sde.generative:
                t = self.terminal_t - t
            sde_drift += sde_diff * self.ctrl(t, x)
        return sde_drift, sde_diff.expand_as(x)
