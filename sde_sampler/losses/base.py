from __future__ import annotations

from typing import Callable

import torch
from torchdiffeq import odeint, odeint_adjoint

from sde_sampler.utils.common import Results


class BaseSDELoss:
    methods = ["kl", "kl_ito", "lv", "lv_traj"]

    def __init__(
        self,
        method: str = "kl",
        traj_per_sample: int = 1,
        filter_samples: Callable | None = None,
        max_log_rnd: float | None = None,
    ):
        if method not in self.methods:
            raise ValueError("Unknown loss method.")
        self.method = method
        if traj_per_sample == 1 and self.method == "lv_traj":
            raise ValueError("Cannot compute variance over a single trajectory.")
        self.traj_per_sample = traj_per_sample

        # Filter
        self.filter_samples = filter_samples
        self.max_log_rnd = max_log_rnd

        # Metrics
        self.n_filtered = 0

    def filter(
        self, log_rnd: torch.Tensor, samples: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = True
        if samples is not None and self.filter_samples is not None:
            mask = self.filter_samples(samples)
        if self.max_log_rnd is None:
            return mask & log_rnd.isfinite()
        return mask & (log_rnd < self.max_log_rnd)

    def compute_loss(
        self, log_rnd: torch.Tensor, samples: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        # Compute loss
        mask = self.filter(log_rnd, samples=samples)
        assert mask.shape == log_rnd.shape
        if self.method == "lv_traj":
            # compute variance over the trajectories
            log_rnd = log_rnd.reshape(self.traj_per_sample, -1, 1)
            mask = mask.reshape(self.traj_per_sample, -1, 1)
            mask = mask.all(dim=0)
            self.n_filtered += self.traj_per_sample * (mask.numel() - mask.sum()).item()
            loss = log_rnd[:, mask].var(dim=0).mean()
        else:
            self.n_filtered += (mask.numel() - mask.sum()).item()
            if self.method == "lv":
                loss = log_rnd[mask].var()
            else:
                loss = log_rnd[mask].mean()

        return loss, {"train/n_filtered_cumulative": self.n_filtered}

    @staticmethod
    def compute_results(
        log_rnd: torch.Tensor,
        compute_weights: bool = False,
        ts: torch.Tensor | None = None,
        samples: torch.Tensor | None = None,
        xs: torch.Tensor | None = None,
    ):
        metrics = {}
        neg_log_rnd = -log_rnd
        log_norm_const_preds = {"log_norm_const_lb": neg_log_rnd.mean().item()}
        if compute_weights:
            # Compute importance weights
            log_weights_max = neg_log_rnd.max()
            weights = (neg_log_rnd - log_weights_max).exp()
            log_norm_const_preds["log_norm_const_is"] = (
                weights.mean().log() + log_weights_max
            ).item()
            metrics["eval/lv_loss"] = log_rnd.var().item()
        else:
            weights = None
        return Results(
            samples=samples,
            weights=weights,
            log_norm_const_preds=log_norm_const_preds,
            ts=ts,
            xs=xs,
            metrics=metrics,
        )

    def __call__(
        self, ts: torch.Tensor, x: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def eval(self, ts: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> Results:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        self.n_filtered = state_dict["n_filtered"]

    def state_dict(self) -> dict:
        return {"n_filtered": self.n_filtered}


class ODELoss(BaseSDELoss):
    methods = ["kl", "lv"]

    def __init__(
        self,
        *args,
        divdrift_and_drift: Callable | None = None,
        odeint_method: str = "rk4",
        odeint_options: dict | None = None,
        use_adjoint: bool = True,
        adjoint_params: tuple | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.divdrift_and_drift = divdrift_and_drift

        # odeint settings
        self.odeint_method = odeint_method
        self.odeint_options = odeint_options or {}
        self.use_adjoint = use_adjoint
        self.adjoint_params = adjoint_params

    def simulate(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        initial_log_prob: Callable | None,
        train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Initial cost
        if train and self.method == "kl":
            log_rnd = 0.0
        else:
            log_rnd = initial_log_prob(x)
            assert log_rnd.shape == (x.shape[0], 1)
        y0 = torch.cat([x, torch.zeros(x.shape[0], 1, device=x.device)], dim=-1)

        def ode_func(t, y):
            x, _ = torch.split(y, split_size_or_sections=[y.shape[-1] - 1, 1], dim=-1)
            divdrift, drift = self.divdrift_and_drift(t, x, create_graph=train)
            return torch.cat([drift, divdrift], dim=-1)

        # Simulate
        if train and self.use_adjoint:
            ys = odeint_adjoint(
                ode_func,
                y0=y0,
                t=ts,
                method=self.odeint_method,
                options=self.odeint_options,
                adjoint_params=self.adjoint_params,
            )
        else:
            ys = odeint(
                ode_func,
                y0=y0,
                t=ts,
                method=self.odeint_method,
                options=self.odeint_options,
            )
        xs, log_probs = torch.split(ys, split_size_or_sections=[x.shape[-1], 1], dim=-1)
        log_rnd -= log_probs[-1]

        # Terminal cost
        log_rnd -= terminal_unnorm_log_prob(xs[-1])
        assert log_rnd.shape == (x.shape[0], 1)

        return xs[-1], log_rnd, xs

    def __call__(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        initial_log_prob: Callable | None,
    ) -> tuple[torch.Tensor, dict]:
        # Repeat initial values
        if self.traj_per_sample != 1:
            x = x.repeat(self.traj_per_sample, 1, 1).reshape(-1, x.shape[-1])

        # Simulate
        samples, log_rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            train=True,
        )
        return self.compute_loss(log_rnd, samples=samples)

    def eval(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        initial_log_prob: Callable = None,
        compute_weights: bool = True,
    ) -> Results:
        samples, log_rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            train=False,
        )
        return BaseSDELoss.compute_results(
            log_rnd,
            compute_weights=compute_weights,
            ts=ts,
            samples=samples.contiguous(),
            xs=xs,
        )
