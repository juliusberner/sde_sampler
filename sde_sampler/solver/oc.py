from __future__ import annotations

import time
from typing import Callable

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module

from sde_sampler.distr.base import Distribution, sample_uniform
from sde_sampler.distr.delta import Delta
from sde_sampler.distr.gauss import Gauss
from sde_sampler.eq.integrator import EulerIntegrator
from sde_sampler.eq.sdes import OU, ControlledSDE
from sde_sampler.eval.plots import get_plots
from sde_sampler.losses.oc import BaseOCLoss
from sde_sampler.solver.base import Trainable
from sde_sampler.utils.common import Results, clip_and_log


class TrainableDiff(Trainable):
    save_attrs = Trainable.save_attrs + ["generative_ctrl"]

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

        # Train
        self.train_batch_size: int = self.cfg.train_batch_size
        self.train_timesteps: Callable = instantiate(self.cfg.train_timesteps)
        self.clip_target: float | None = self.cfg.get("clip_target")

        # Eval
        self.eval_timesteps: Callable = instantiate(self.cfg.eval_timesteps)
        self.eval_batch_size: int = self.cfg.eval_batch_size
        self.eval_integrator = EulerIntegrator()

    def setup_models(self):
        self.prior: Distribution = instantiate(self.cfg.prior)
        self.sde: OU = instantiate(self.cfg.sde)
        self.generative_ctrl: Module = instantiate(
            self.cfg.generative_ctrl,
            sde=self.sde,
            prior_score=self.prior.score,
            target_score=self.target.score,
        )

    def clipped_target_unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        output = clip_and_log(
            self.target.unnorm_log_prob(x),
            max_norm=self.clip_target,
            name="target",
        )
        return output

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        raise NotImplementedError

    def compute_loss(self) -> tuple[torch.Tensor, dict]:
        x = self.prior.sample((self.train_batch_size,))
        ts = self.train_timesteps(device=x.device)
        return self._compute_loss(ts, x)

    def compute_results(self) -> Results:
        # Sample trajectories
        x = self.prior.sample((self.eval_batch_size,))
        ts = self.eval_timesteps(device=x.device)

        results = self._compute_results(
            ts,
            x,
            compute_weights=True,
        )
        assert results.xs.shape == (len(ts), *results.samples.shape)

        # Sample w/o ito integral
        start_time = time.time()
        add_results = self._compute_results(
            ts,
            x,
            compute_weights=False,
            return_traj=False,
        )

        # Update results
        results.metrics["eval/sample_time"] = time.time() - start_time
        results.metrics.update(add_results.metrics)
        results.log_norm_const_preds.update(add_results.log_norm_const_preds)

        # Sample trajectories of inference proc
        if (
            self.plot_results
            and hasattr(self, "inference_sde")
            and hasattr(self.target, "sample")
        ):
            x_target = self.target.sample((self.eval_batch_size,))
            xs = self.eval_integrator.integrate(
                sde=self.inference_sde, ts=ts, x_init=x_target, timesteps=ts
            )
            plots = get_plots(
                distr=self.prior,
                samples=xs[-1],
                ts=ts,
                xs=xs,
                marginal_dims=self.eval_marginal_dims,
                domain=self.target.domain,
            )
            results.plots.update({f"{k}_inference": v for k, v in plots.items()})

        return results


class Bridge(TrainableDiff):
    save_attrs = TrainableDiff.save_attrs + ["inference_ctrl", "loss"]

    def setup_models(self):
        super().setup_models()
        self.inference_ctrl = self.cfg.get("inference_ctrl")
        self.inference_sde: OU = instantiate(
            self.cfg.sde,
            generative=False,
        )
        if self.inference_ctrl is not None:
            self.inference_ctrl: Module = instantiate(
                self.cfg.inference_ctrl,
                sde=self.sde,
                prior_score=self.prior.score,
                target_score=self.target.score,
            )
            self.inference_sde = ControlledSDE(
                sde=self.inference_sde, ctrl=self.inference_ctrl
            )
        elif not isinstance(self.prior, Gauss):
            raise ValueError("Can only be used with Gaussian prior.")

        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            inference_ctrl=self.inference_ctrl,
            filter_samples=getattr(self.target, "filter", None),
        )

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            initial_log_prob=self.prior.log_prob,
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            initial_log_prob=self.prior.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class PIS(TrainableDiff):
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def setup_models(self):
        super().setup_models()
        if not isinstance(self.prior, Delta):
            raise ValueError("Can only be used with dirac delta prior.")
        self.reference_distr = self.sde.marginal_distr(
            t=self.sde.terminal_t, x_init=self.prior.loc
        )
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            filter_samples=getattr(self.target, "filter", None),
        )

        # Inference SDE
        inference_sde: OU = instantiate(
            self.cfg.sde,
            generative=False,
        )
        self.inference_sde = ControlledSDE(sde=inference_sde, ctrl=self.inference_ctrl)

    def inference_ctrl(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        reference_distr = self.sde.marginal_distr(t=t, x_init=self.prior.loc)
        return self.sde.diff(t, x) * reference_distr.score(x).clip(max=1e5)

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )

class DDSExponential(TrainableDiff):
    # This implements the basic DDS algorithm
    # with the exponential integrator
    # https://arxiv.org/abs/2302.13834
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def setup_models(self):
        super().setup_models()
        if not isinstance(self.prior, Gauss):
            raise ValueError("Can only be used with Gaussian prior.")

        # prior = reference_distr for terminal loss
        self.reference_distr = self.prior 
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            filter_samples=getattr(self.target, "filter", None),
        )

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )

class DDS(TrainableDiff):
    # This implementation induces the same objectives in the DDS paper (https://arxiv.org/abs/2302.13834).
    # However, we do not use the exponential integrator and the same parametrization.
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def setup_models(self):
        super().setup_models()
        if not isinstance(self.prior, Gauss):
            raise ValueError("Can only be used with Gaussian prior.")
        self.inference_sde = instantiate(self.cfg.sde, generative=False)
        self.reference_distr = self.sde.marginal_distr(
            self.sde.terminal_t, x_init=self.prior.loc, var_init=self.prior.scale**2
        )
        if not torch.allclose(
            self.reference_distr.loc, self.prior.loc
        ) and torch.allclose(self.reference_distr.scale, self.prior.scale):
            raise ValueError(
                "Make sure that the Gaussian is the invariant distribution of the SDE."
            )
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            reference_ctrl=self.reference_ctrl,
            filter_samples=getattr(self.target, "filter", None),
        )

    def reference_ctrl(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.sde.diff(t, x) * self.prior.score(x)

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class SubtrajBridge(Bridge):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        if not hasattr(self.generative_ctrl, "unnorm_log_prob"):
            raise ValueError("Needs an unnormalized log density.")
        if not self.loss.method in ["lv", "lv_traj"]:
            raise ValueError("Can only be used with log-variance loss.")
        if self.target.domain is None:
            raise ValueError("Need a domain for sampling.")
        self.subtraj_prob = self.cfg.get("subtraj_prob", 0.5)
        self.fix_terminal = self.cfg.get("fix_terminal", False)
        self.subtraj_steps = self.cfg.get("subtraj_steps")
        if self.fix_terminal and self.subtraj_steps is not None:
            raise ValueError("Cannot fix subtrajectory steps with fixed terminal time.")
        self.lerp_domain = self.cfg.get("lerp_domain", True)

    def get_log_prob(self, t: torch.Tensor, detach=False) -> Callable:
        if torch.isclose(t, self.sde.terminal_t):
            return self.clipped_target_unnorm_log_prob
        if torch.isclose(t, torch.zeros_like(t)):
            return self.prior.log_prob

        def log_prob(x: torch.Tensor) -> torch.Tensor:
            with torch.set_grad_enabled(detach):
                output = self.generative_ctrl.unnorm_log_prob(t=t, x=x)
                if self.inference_ctrl is not None:
                    output += self.inference_ctrl.unnorm_log_prob(t=t, x=x)
                return output

        return log_prob

    def compute_loss(
        self,
    ) -> tuple[torch.Tensor, dict]:
        if torch.rand(1) > self.subtraj_prob:
            return super().compute_loss()

        # Timesteps
        ts = self.train_timesteps(device=self.target.domain.device)
        idx_init = torch.randint(0, len(ts) - 1, tuple())

        if self.fix_terminal:
            idx_end = len(ts) - 1
        elif self.subtraj_steps is not None:
            idx_end = torch.minimum(
                idx_init + self.subtraj_steps, torch.tensor(len(ts)) - 1
            )
        else:
            idx_end = torch.randint(idx_init + 1, len(ts), tuple())

        # Get initial points
        domain = self.target.domain
        if self.lerp_domain:
            domain = torch.lerp(
                self.prior.domain, domain, ts[idx_init] / self.sde.terminal_t
            )

        x = sample_uniform(domain=domain, batchsize=self.train_batch_size)

        # Simulate loss
        subts = ts[idx_init : idx_end + 1]
        initial_log_prob = self.get_log_prob(t=ts[idx_init], detach=True)
        target_unnorm_log_prob = self.get_log_prob(t=ts[idx_end], detach=False)
        loss, metrics = self.loss(
            ts, x, target_unnorm_log_prob, initial_log_prob=initial_log_prob
        )
        loss *= len(subts) / len(ts)
        return loss, metrics
