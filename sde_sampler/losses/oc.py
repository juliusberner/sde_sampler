from __future__ import annotations

import logging
from typing import Callable

import torch

from sde_sampler.eq.sdes import OU
from sde_sampler.utils.autograd import compute_divx
from sde_sampler.utils.common import Results


class BaseOCLoss:
    def __init__(
        self,
        generative_ctrl: Callable,
        sde: OU | None = None,
        method: str = "kl",
        traj_per_sample: int = 1,
        filter_samples: Callable | None = None,
        max_rnd: float | None = None,
        sde_ctrl_dropout: float | None = None,
        sde_ctrl_noise: float | None = None,
        **kwargs,
    ):
        self.generative_ctrl = generative_ctrl
        self.sde = sde
        if method not in ["kl", "kl_ito", "lv", "lv_traj"]:
            raise ValueError("Unknown loss method.")
        self.method = method
        if traj_per_sample == 1 and self.method == "lv_traj":
            raise ValueError("Cannot compute variance over a single trajectory.")
        self.traj_per_sample = traj_per_sample

        # Filter
        self.filter_samples = filter_samples
        self.max_rnd = max_rnd

        # SDE controls
        if self.sde is not None:
            self.sde_ctrl_noise = sde_ctrl_noise
            self.sde_ctrl_dropout = sde_ctrl_dropout
            if self.method in ["kl", "kl_ito"]:
                for attr in ["sde_ctrl_noise", "sde_ctrl_dropout"]:
                    if getattr(self, attr) is not None:
                        logging.warning(
                            "%s should only be used for the log-variance loss."
                        )

        # Metrics
        self.n_filtered = 0

    def filter(
        self, rnd: torch.Tensor, samples: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = True
        if samples is not None and self.filter_samples is not None:
            mask = self.filter_samples(samples)
        if self.max_rnd is None:
            return mask & rnd.isfinite()
        return mask & (rnd < self.max_rnd)

    def generative_and_sde_ctrl(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generative_ctrl = self.generative_ctrl(t, x)
        sde_ctrl = generative_ctrl.detach()
        if self.sde_ctrl_noise is not None:
            sde_ctrl += self.sde_ctrl_noise * torch.randn_like(sde_ctrl)
        if self.sde_ctrl_dropout is not None:
            mask = torch.rand_like(sde_ctrl) > self.sde_ctrl_dropout
            sde_ctrl[mask] = -(self.sde.drift(t, x) / self.sde.diff(t, x))[mask]
        return generative_ctrl, sde_ctrl

    def compute_loss(
        self, rnd: torch.Tensor, samples: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        # Compute loss
        mask = self.filter(rnd, samples=samples)
        assert mask.shape == rnd.shape
        if self.method == "lv_traj":
            # compute variance over the trajectories
            rnd = rnd.reshape(self.traj_per_sample, -1, 1)
            mask = mask.reshape(self.traj_per_sample, -1, 1)
            mask = mask.all(dim=0)
            self.n_filtered += self.traj_per_sample * (mask.numel() - mask.sum()).item()
            loss = rnd[:, mask].var(dim=0).mean()
        else:
            self.n_filtered += (mask.numel() - mask.sum()).item()
            if self.method == "lv":
                loss = rnd[mask].var()
            else:
                loss = rnd[mask].mean()

        return loss, {"train/n_filtered_cumulative": self.n_filtered}

    @staticmethod
    def compute_results(
        rnd: torch.Tensor,
        compute_weights: bool = False,
        ts: torch.Tensor | None = None,
        samples: torch.Tensor | None = None,
        xs: torch.Tensor | None = None,
    ):
        metrics = {}
        neg_rnd = -rnd
        if compute_weights:
            # Compute importance weights
            log_weights_max = neg_rnd.max()
            weights = (neg_rnd - log_weights_max).exp()
            log_norm_const_preds = {
                "log_norm_const_lb_ito": neg_rnd.mean().item(),
                "log_norm_const_is": (weights.mean().log() + log_weights_max).item(),
            }
            metrics["eval/lv_loss"] = rnd.var().item()
        else:
            weights = None
            log_norm_const_preds = {"log_norm_const_lb": neg_rnd.mean().item()}
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


class TimeReversalLoss(BaseOCLoss):
    def __init__(self, *args, inference_ctrl: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_ctrl = inference_ctrl

    def simulate(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        initial_log_prob: Callable | None,
        train: bool = True,
        compute_ito_int: bool = False,
        change_sde_ctrl: bool = False,
        return_traj: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Initial cost
        if train and self.method in ["kl", "kl_ito"]:
            rnd = 0.0
        else:
            rnd = initial_log_prob(x)
            assert rnd.shape == (x.shape[0], 1)

        xs = [x] if return_traj else None
        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl, sde_ctrl = self.generative_and_sde_ctrl(s, x)
            else:
                sde_ctrl = generative_ctrl = self.generative_ctrl(s, x)
            sde_diff = self.sde.diff(s, x)
            dt = t - s

            # Loss increments
            if self.inference_ctrl is None:
                gen_plus_inf_ctrl = gen_minus_inf_ctrl = generative_ctrl

            else:
                div_ctrl, inference_ctrl = compute_divx(self.inference_ctrl, s, x)
                # This assumes the diffusion coeff. to be independent of x
                rnd += sde_diff * div_ctrl * dt
                gen_plus_inf_ctrl = generative_ctrl + inference_ctrl
                gen_minus_inf_ctrl = generative_ctrl - inference_ctrl

            if change_sde_ctrl:
                cost = gen_plus_inf_ctrl * (sde_ctrl - 0.5 * gen_minus_inf_ctrl)
                rnd += cost.sum(dim=-1, keepdim=True) * dt
            else:
                rnd += 0.5 * (gen_plus_inf_ctrl**2).sum(dim=-1, keepdim=True) * dt

            if not train:
                rnd -= self.sde.drift_div_int(s, t, x)

            # Euler-Maruyama
            db = torch.randn_like(x) * dt.sqrt()
            x = x + (self.sde.drift(s, x) + sde_diff * sde_ctrl) * dt + sde_diff * db

            # Compute ito integral
            if compute_ito_int:
                rnd += (gen_plus_inf_ctrl * db).sum(dim=-1, keepdim=True)

            if return_traj:
                xs.append(x)

        # Terminal cost
        rnd -= terminal_unnorm_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        if return_traj:
            xs = torch.stack(xs)
        return x, rnd, xs

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
        compute_ito_int = self.method != "kl"
        change_sde_ctrl = self.method in ["lv", "lv_traj"]
        samples, rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            compute_ito_int=compute_ito_int,
            change_sde_ctrl=change_sde_ctrl,
            train=True,
            return_traj=False,
        )
        return self.compute_loss(rnd, samples=samples)

    def eval(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        initial_log_prob: Callable | None = None,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        samples, rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            compute_ito_int=compute_weights,
            train=False,
            return_traj=return_traj,
        )
        return BaseOCLoss.compute_results(
            rnd, compute_weights=compute_weights, ts=ts, samples=samples, xs=xs
        )


class ReferenceSDELoss(BaseOCLoss):
    def __init__(self, *args, reference_ctrl: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_ctrl = reference_ctrl

    def simulate(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        reference_log_prob: Callable,
        compute_ito_int: bool = False,
        change_sde_ctrl: bool = False,
        return_traj: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Initial cost
        rnd = 0.0

        xs = [x] if return_traj else None
        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl, sde_ctrl = self.generative_and_sde_ctrl(s, x)
            else:
                sde_ctrl = generative_ctrl = self.generative_ctrl(s, x)
            sde_diff = self.sde.diff(s, x)
            dt = t - s

            # Loss increments
            if self.reference_ctrl is None:
                gen_minus_ref_ctrl = generative_ctrl
                gen_plus_ref_ctrl = generative_ctrl
            else:
                reference_ctrl = self.reference_ctrl(s, x)
                gen_minus_ref_ctrl = generative_ctrl - reference_ctrl
                gen_plus_ref_ctrl = reference_ctrl + generative_ctrl

            if change_sde_ctrl:
                running_cost = gen_minus_ref_ctrl * (sde_ctrl - 0.5 * gen_plus_ref_ctrl)
                rnd += running_cost.sum(dim=-1, keepdim=True) * dt
            else:
                rnd += 0.5 * (gen_minus_ref_ctrl**2).sum(dim=-1, keepdim=True) * dt

            # Euler-Maruyama
            db = torch.randn_like(x) * dt.sqrt()
            x = x + (self.sde.drift(s, x) + sde_diff * sde_ctrl) * dt + sde_diff * db

            # Compute ito integral
            if compute_ito_int:
                rnd += (gen_minus_ref_ctrl * db).sum(dim=-1, keepdim=True)

            if return_traj:
                xs.append(x)

        # Terminal cost
        rnd += reference_log_prob(x) - terminal_unnorm_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        if return_traj:
            xs = torch.stack(xs)

        return x, rnd, xs

    def __call__(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        reference_log_prob: Callable,
    ) -> tuple[torch.Tensor, dict]:
        # Repeat initial values
        if self.traj_per_sample != 1:
            x = x.repeat(self.traj_per_sample, 1, 1).reshape(-1, x.shape[-1])

        # Simulate
        compute_ito_int = self.method != "kl"
        change_sde_ctrl = self.method in ["lv", "lv_traj"]
        samples, rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            reference_log_prob=reference_log_prob,
            compute_ito_int=compute_ito_int,
            change_sde_ctrl=change_sde_ctrl,
            return_traj=False,
        )

        return self.compute_loss(rnd, samples=samples)

    def eval(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        reference_log_prob: Callable | None = None,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        samples, rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            reference_log_prob=reference_log_prob,
            compute_ito_int=compute_weights,
            change_sde_ctrl=False,
            return_traj=return_traj,
        )
        return BaseOCLoss.compute_results(
            rnd, compute_weights=compute_weights, ts=ts, samples=samples, xs=xs
        )


class ExponentialIntegratorSDELoss(BaseOCLoss):
    def __init__(self, *args, **kwargs):
        # Only loss.method = KL implemented
        super().__init__(*args, **kwargs)
        self.alpha = kwargs.get("alpha", 1)
        self.sigma = kwargs.get("sigma", 1)

    def simulate(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        reference_log_prob: Callable | None = None,  # unused
        compute_ito_int: bool = False,
        change_sde_ctrl: bool = False,
        return_traj: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Initial cost

        if change_sde_ctrl:
            raise NotImplementedError

        rnd = 0.0

        xs = [x] if return_traj else None

        # Simulate
        # s,t = old time, new time
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            generative_ctrl = self.generative_ctrl(s, x)
            dt = t - s

            # Exponential Scheme as implemented by Vargas et.al
            beta_k = torch.clip(self.alpha * dt.sqrt(), 0, 1)
            alpha_k = torch.sqrt(1.0 - beta_k**2)

            rnd += (
                0.5
                * (beta_k**2)
                * (self.sigma**2)
                * (generative_ctrl**2).sum(dim=-1, keepdim=True)
            )

            noise = torch.randn_like(x)

            x = (
                x * alpha_k
                + (beta_k**2) * (self.sigma**2) * generative_ctrl
                + self.sigma * beta_k * noise
            )

            # Compute ito integral for importance sampling
            if compute_ito_int:
                rnd += (self.sigma * generative_ctrl * noise * beta_k).sum(
                    dim=-1, keepdim=True
                )

            if return_traj:
                xs.append(x)

        # compute reference log prob value based on based in prior
        reference_log_prob_value = reference_log_prob(x)
        rnd += reference_log_prob_value - terminal_unnorm_log_prob(x)

        assert rnd.shape == (x.shape[0], 1)  # one loss number for each sample

        if return_traj:
            xs = torch.stack(xs)

        return x, rnd, xs

    def __call__(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        reference_log_prob: Callable,
    ) -> tuple[torch.Tensor, dict]:
        # Repeat initial values
        if self.traj_per_sample != 1:
            x = x.repeat(self.traj_per_sample, 1, 1).reshape(-1, x.shape[-1])

        # Simulate
        compute_ito_int = self.method != "kl"
        change_sde_ctrl = self.method in ["lv", "lv_traj"]
        samples, rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            reference_log_prob=reference_log_prob,
            compute_ito_int=compute_ito_int,
            change_sde_ctrl=change_sde_ctrl,
            return_traj=False,
        )

        return self.compute_loss(rnd, samples=samples)

    def eval(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        terminal_unnorm_log_prob: Callable,
        reference_log_prob: Callable | None = None,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        samples, rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            reference_log_prob=reference_log_prob,
            compute_ito_int=compute_weights,
            change_sde_ctrl=False,
            return_traj=return_traj,
        )
        return BaseOCLoss.compute_results(
            rnd, compute_weights=compute_weights, ts=ts, samples=samples, xs=xs
        )
