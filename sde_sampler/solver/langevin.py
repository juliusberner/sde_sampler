from __future__ import annotations

import time

from hydra.utils import instantiate
from omegaconf import DictConfig

from sde_sampler.eq.integrator import Integrator
from sde_sampler.eq.sdes import LangevinSDE
from sde_sampler.eval.metrics import EXPECTATION_FNS
from sde_sampler.solver.base import Solver
from sde_sampler.utils.common import Results


class LangevinSolver(Solver):
    save_attrs: list[str] = []

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self.integrator: Integrator = instantiate(self.cfg.integrator)
        self.sde: LangevinSDE = instantiate(
            self.cfg.sde, target_score=self.target.score
        )
        self.sde.to(self.device)

        self.prior = instantiate(self.cfg.prior)
        self.prior.to(self.device)

        self.eval_timesteps = instantiate(self.cfg.eval_timesteps)
        self.burn_steps = self.cfg.get("eval_expectation_burn", 0)
        if self.burn_steps >= len(self.eval_timesteps()):
            raise ValueError("Specify more eval_steps than burn_steps.")

    def run(self):
        start_time = time.time()

        # Sample prior
        x = self.prior.sample((self.cfg.eval_batch_size,))

        # Simulate SDE
        ts = self.eval_timesteps(device=self.device)
        xs = self.integrator.integrate(self.sde, ts=ts, x_init=x)

        # Other metrics
        metrics = {
            "eval/sample_time": time.time() - start_time,
        }

        # Expectation predictions over time
        exp_samples = xs[self.burn_steps :].reshape(-1, self.target.dim)
        expectation_preds = {
            name: fn(exp_samples).mean() for name, fn in EXPECTATION_FNS.items()
        }

        return Results(
            samples=xs[-1],
            weights=None,
            log_norm_const_preds=None,
            ts=ts,
            xs=xs,
            metrics=metrics,
            expectation_preds=expectation_preds,
        )
