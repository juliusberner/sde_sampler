from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import typing as tp
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, MutableMapping, MutableSequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from torch_ema import ExponentialMovingAverage

from sde_sampler.distr.base import Distribution
from sde_sampler.eval.metrics import get_metrics
from sde_sampler.eval.plots import get_plots, save_fig
from sde_sampler.utils import wandb as wandb_utils
from sde_sampler.utils.common import CKPT_DIR, Results


class Solver(torch.nn.Module):
    # Attributes to be checkpointed and loaded
    save_attrs: list[str] = []

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Configuration and setup
        self.cfg = deepcopy(cfg)
        OmegaConf.resolve(self.cfg)
        if cfg.get("setup") is not None:
            for fn_cfg in cfg.setup:
                call(fn_cfg)
        if cfg.get("num_threads"):
            torch.set_num_threads(cfg.num_threads)

        # Set output directory
        if self.cfg.get("out_dir") is None:
            self.out_dir = Path.cwd()
        else:
            self.out_dir = Path(cfg.out_dir)

        # Seed pytorch and numpy (e.g., torchsde uses the numpy seed)
        if "seed" in self.cfg:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # Device
        self.device = self.cfg.get("device")
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: torch.device | str

        # Problem
        self.target: Distribution = instantiate(self.cfg.target)
        self.target.to(self.device)

        # Sample losses
        self.eval_sample_losses: dict[str, tp.Callable] | None = None
        if cfg.get("eval_sample_losses") is not None:
            self.eval_sample_losses = {
                name: instantiate(loss_cfg, out_dir=self.out_dir)
                for name, loss_cfg in cfg.get("eval_sample_losses").items()
            }

        # Logging and checkpoints
        self.plot_results: bool = self.cfg.get("plot_results", True)
        self.store_last_ckpt: bool = self.cfg.get("store_last_ckpt", False)
        self.restore_ckpt_from_wandb: bool | None = self.cfg.get(
            "restore_ckpt_from_wandb"
        )
        self.upload_ckpt_to_wandb: str | bool | None = self.cfg.get(
            "upload_ckpt_to_wandb"
        )
        if (
            isinstance(self.upload_ckpt_to_wandb, str)
            and self.upload_ckpt_to_wandb != "last"
        ):
            raise ValueError("Unknown upload mode.")
        self.eval_marginal_dims: list = self.cfg.get("eval_marginal_dims", [])

        # Paths
        self.ckpt_file: str | None = self.cfg.get("ckpt_file")
        self.ckpt_dir = self.out_dir / CKPT_DIR
        logging.info("Checkpoint directory: %s", self.ckpt_dir)
        # See https://jsonlines.org/ for the JSON Lines text file format
        self.metrics_file = self.out_dir / "metrics.jsonl"

        # Weights & Biases
        self.wandb = wandb.run
        if self.wandb is None:
            wandb.init(mode="disabled")
        else:
            self.wandb.summary["device"] = str(self.device)

        self.initialized = False
        self.initial_time = time.time()

    def setup(self):
        logging.info("Setting up solver.")
        self.target.compute_stats()
        self.load_checkpoint(self.ckpt_file)
        self.initialized = True

    def get_metrics_and_plots(
        self, results: Results, decimals: int = 6, nbins: int = 100
    ) -> tuple[dict, dict]:
        metrics = results.metrics
        plots = results.plots

        metrics["eval/overall_time"] = time.time() - self.initial_time
        if results.samples is not None:
            nonfinite = ~results.samples.isfinite().all(dim=-1).sum()
            if nonfinite > 0:
                logging.warning("Found %d non-finite samples.", nonfinite)

            if self.plot_results:
                plots.update(
                    get_plots(
                        distr=self.target,
                        samples=results.samples,
                        weights=results.weights,
                        ts=results.ts,
                        xs=results.xs,
                        marginal_dims=self.eval_marginal_dims,
                        decimals=decimals,
                        nbins=nbins,
                    )
                )
                if hasattr(self.target, "plots"):
                    plots.update(self.target.plots(results.samples))

            eval_metrics = get_metrics(
                distr=self.target,
                samples=results.samples,
                weights=results.weights,
                log_norm_const_preds=results.log_norm_const_preds,
                expectation_preds=results.expectation_preds,
                marginal_dims=self.eval_marginal_dims,
                sample_losses=self.eval_sample_losses,
            )
            metrics.update(eval_metrics)
            if hasattr(self.target, "metrics"):
                metrics.update(self.target.metrics(results.samples))

        return metrics, plots

    def log(
        self,
        results: Results,
        step=None,
    ) -> dict:
        metrics, plots = self.get_metrics_and_plots(results)

        # Save figures to disk
        for k, fig in plots.items():
            name = f"{k}.png" if step is None else f"{k}_step_{step}.png"
            save_fig(fig, self.out_dir / name)

        # Save metrics to disk
        with self.metrics_file.open(mode="a") as f:
            f.write(json.dumps(metrics) + "\n")

        plots = {k: wandb_utils.format_fig(fig) for k, fig in plots.items()}
        wandb.log({**metrics, **plots}, step=step)
        logging.info("Metrics:\n%s", yaml.dump(metrics))
        return metrics

    def run(self) -> Results:
        return NotImplementedError

    def forward(self) -> Results:
        # Setup
        if not self.initialized:
            self.setup()

        logging.info("Running solver 🏃")

        # Checkpoint and log
        results = self.run()
        if self.store_last_ckpt:
            self.store_checkpoint(suffix="_final")

        logging.info("Logging final results.")
        self.log(results)

        if self.upload_ckpt_to_wandb == "last":
            wandb_utils.delete_old_wandb_ckpts()
        return results

    def state_dict(self) -> dict:
        state_dict = {}
        for key in self.save_attrs:
            attr = getattr(self, key)
            if getattr(attr, "load_state_dict", None):
                state_dict[key] = attr.state_dict()
            else:
                state_dict[key] = attr
        return state_dict

    def load_state_dict(self, state_dict: dict):
        for key in self.save_attrs:
            if key in state_dict:
                attr = getattr(self, key)
                if getattr(attr, "load_state_dict", None):
                    attr.load_state_dict(state_dict[key])
                else:
                    setattr(self, key, state_dict[key])

    def latest_checkpoint(self) -> Path | None:
        if self.restore_ckpt_from_wandb:
            wandb_utils.restore_ckpt(self.out_dir)

        ckpts = list(self.ckpt_dir.glob("ckpt*.pt"))
        if ckpts:
            return max(ckpts, key=os.path.getmtime)

    def store_checkpoint(self, suffix="") -> Path:
        name = f"ckpt{suffix}.pt"
        path = self.ckpt_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Save checkpoint at %s", path)
        torch.save(self.state_dict(), path)

        # Upload
        if self.upload_ckpt_to_wandb:
            wandb_utils.upload_ckpt(path, name=name)
        return path

    def load_checkpoint(self, ckpt_file: str | Path | None = None):
        if ckpt_file is None:
            ckpt_file = self.latest_checkpoint()
        if ckpt_file is not None:
            logging.info("Loading checkpoint %s", ckpt_file)
            ckpt = torch.load(ckpt_file, map_location=self.device)
            self.load_state_dict(ckpt)


class Trainable(Solver):
    save_attrs: list[str] = [
        "n_steps",
        "time",
        "optim",
        "scheduler",
        "ema",
    ]

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        # Eval and EMA devices
        self.eval_device = self.cfg.get("eval_device")
        if self.eval_device is None:
            self.eval_device = self.device
        self.eval_device: torch.device | str
        self.ema_device = self.cfg.get("ema_device")
        if self.ema_device is None:
            self.ema_device = self.device
        self.ema_device: torch.device | str

        # Model
        self.setup_models()
        self.to(self.device)

        # EMA
        if self.cfg.get("ema"):
            self.ema = instantiate(self.cfg.ema, parameters=self.trainable_parameters())
            self.ema.to(self.ema_device)
        else:
            self.ema = None
        self.ema: EMA | None

        # Optimization
        self.train_steps = self.cfg.train_steps
        self.grad_clip: tp.Callable | None = instantiate(self.cfg.get("grad_clip"))
        self.max_grad: float | None = self.cfg.get("max_grad")
        self.max_loss: float | None = self.cfg.get("max_loss")
        self.scale_loss: float | None = self.cfg.get("scale_loss")

        if self.cfg.get("param_groups"):
            parameters = [
                {"params": getattr(self, name).parameters(), **options}
                for name, options in self.cfg.param_groups.items()
            ]
        else:
            parameters = self.trainable_parameters()
        self.optim: torch.optim.Optimizer = instantiate(self.cfg.optim, parameters)
        assert sum(p.numel() for p in self.trainable_parameters()) == sum(
            sum(p.numel() for p in group["params"]) for group in self.optim.param_groups
        )

        # Scheduler
        schedulers = []
        if self.cfg.get("lr_scheduler"):
            schedulers.append(instantiate(self.cfg.lr_scheduler, optimizer=self.optim))
        for scheduler_cfg in self.cfg.get("scheduler", []):
            schedulers.append(instantiate(scheduler_cfg, obj=self))
        self.scheduler = CombinedScheduler(schedulers)

        # Evaluation, logging, and checkpoints
        self.time = 0.0
        self.eval_stddev_steps: int | None = self.cfg.get("eval_stddev_steps")
        self.eval_init = self.cfg.eval_init
        self.eval_interval: int = self.cfg.get("eval_interval") or self.train_steps
        self.log_interval: int = self.cfg.get("log_interval") or self.train_steps
        self.ckpt_interval: int = self.cfg.get("ckpt_interval") or self.train_steps
        self.n_steps: int = 0
        self.n_steps_skip: int = 0

        # Weights & Biases
        if self.cfg.get("model_watcher") is not None:
            call(self.cfg.model_watcher, models=self)

        if self.wandb is not None:
            self.wandb.summary["ema_device"] = str(self.ema_device)
            self.wandb.summary["eval_device"] = str(self.eval_device)
            self.wandb.summary["params/all"] = sum(p.numel() for p in self.parameters())
            self.wandb.summary["params/trainable"] = sum(
                p.numel() for p in self.trainable_parameters()
            )

    def setup_models(self):
        raise NotImplementedError

    def compute_results(self) -> Results:
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, use_ema=True, log=True) -> Results:
        logging.info("Evaluate at step %d (%.0f min).", self.n_steps, self.time // 60)
        training = self.training
        self.eval()
        self.to(self.eval_device)

        if self.ema and use_ema:
            self.ema.to(self.eval_device)
            with self.ema.average_parameters():
                results = self.compute_results()
            self.ema.to(self.ema_device)
        else:
            results = self.compute_results()

        if self.eval_stddev_steps is not None:
            results.metrics.update(self.loss_and_grad_var())

        if log:
            self.log(results, step=self.n_steps)

        self.train(training)
        self.to(self.device)
        return results

    def compute_loss(self) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def loss_and_grad_var(self) -> dict[str, float]:
        losses, grads = [], []
        for _ in range(self.eval_stddev_steps):
            self.optim.zero_grad()
            loss, _ = self.compute_loss()
            loss.backward()
            losses.append(loss.detach())
            grad = torch.cat(
                [
                    p.grad.flatten()
                    for p in self.trainable_parameters()
                    if p.grad is not None
                ]
            )
            grads.append(grad)
        loss_std = torch.stack(losses).var()
        grads_std = torch.stack(grads).var(dim=0)
        return {
            "eval/grad_stddev_mean": grads_std.mean().item(),
            "eval/grad_stddev_max": grads_std.max().item(),
            "eval/loss_stddev": loss_std.item(),
        }

    def grad_norm(self, norm_type: float = 2.0):
        norms = [
            torch.linalg.vector_norm(p.grad, norm_type)
            for p in self.trainable_parameters()
            if p.grad is not None
        ]
        return torch.linalg.vector_norm(torch.stack(norms), norm_type)

    def trainable_parameters(self) -> tp.Iterator[torch.Tensor]:
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def step(self) -> dict[str, float]:
        start_t = time.time()
        self.optim.zero_grad()

        # Loss
        loss, metrics = self.compute_loss()
        if self.scale_loss is not None:
            loss = self.scale_loss * loss
        loss.backward()

        # Check max. loss and grad
        loss_ok = (
            loss.isfinite() if self.max_loss is None else loss.abs() <= self.max_loss
        )
        if self.max_grad is None:
            grad_ok = all(
                p.grad.isfinite().all()
                for p in self.trainable_parameters()
                if p.grad is not None
            )
        else:
            max_grad = self.grad_norm(norm_type=torch.inf)
            grad_ok = max_grad <= self.max_grad
            metrics["train/max_grad"] = max_grad.item()

        # Step optimizer, scheduler, and ema
        if loss_ok and grad_ok:

            # Clip grads
            if self.grad_clip is not None:
                metrics["train/grad_clip_norm"] = self.grad_clip(
                    self.trainable_parameters()
                ).item()

            self.optim.step()
            self.scheduler.step()
            if self.ema:
                self.ema.update()
                metrics["train/ema_decay"] = self.ema.get_current_decay()
        else:
            self.n_steps_skip += 1
        time_step = time.time() - start_t

        metrics.update(
            {
                "train/time_per_step": time_step,
                "train/loss": loss.item(),
                "train/skipped_steps": self.n_steps_skip,
                "train/no_grad": sum(
                    p.grad is None for p in self.trainable_parameters()
                ),
            }
        )

        self.n_steps += 1
        return metrics

    def run(self) -> Results:
        if self.n_steps == 0 and self.eval_init:
            self.evaluate()

        logging.info("Start training at step %d.", self.n_steps)
        self.train()
        for _ in range(self.n_steps, self.train_steps):
            # Train
            t_start = time.time()
            metrics = self.step()
            self.time += time.time() - t_start
            metrics.update(
                {
                    "train/time": self.time,
                    "train/step": self.n_steps,
                }
            )
            metrics.update({f"params/{k}": v for k, v in self.scheduler.get().items()})

            # Last step is treated differently
            last_step = self.n_steps == self.train_steps

            # Logging
            if self.n_steps % self.log_interval == 0 or last_step:
                wandb.log(metrics, step=self.n_steps)
                logging.info(f"Train metrics:\n%s", yaml.dump(metrics))

                # Save metrics to disk
                with self.metrics_file.open(mode="a") as f:
                    f.write(json.dumps(metrics) + "\n")

            if not last_step:
                # Evaluation
                if self.n_steps % self.eval_interval == 0:
                    results = self.evaluate()

                # Checkpointing
                if self.n_steps % self.ckpt_interval == 0:
                    self.store_checkpoint(suffix=f"{self.n_steps:06}")

        logging.info("Finished training at step %d.", self.n_steps)
        results = self.evaluate()
        return results

    def load_checkpoint(self, ckpt_file: str | Path | None):
        super().load_checkpoint(ckpt_file=ckpt_file)
        if self.ema is not None:
            self.ema.to(self.ema_device)


class CombinedScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def get(self):
        output = {}
        for scheduler in self.schedulers:
            if isinstance(scheduler, MultiStepParams):
                output.update(scheduler.get())
            # This assumes the optimizer to adhere to the pytorch api
            elif hasattr(scheduler, "optimizer"):
                for i, group in enumerate(scheduler.optimizer.param_groups):
                    output[f"lr_{i}"] = group["lr"]
        return output

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict:
        return {
            idx: scheduler.state_dict() for idx, scheduler in enumerate(self.schedulers)
        }

    def load_state_dict(self, state_dict: dict):
        for idx, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(state_dict[idx])


class MultiStepParams:
    sep: str = "."

    def __init__(
        self,
        obj: tp.Any,
        milestones: list[int],
        gammas: dict[str, float | int],
        last_step: int = 0,
    ):
        self.obj = obj
        self.milestones = Counter(milestones)
        self.gammas = gammas

        # Initialize step and base value
        self.base_values = {k: v for k, v in self.get().items() if v is not None}

        # Discard non-existant keys
        missing = set(self.gammas).difference(self.base_values)
        if len(missing) > 0:
            logging.warning("The keys %s are missing and cannot be scheduled.", missing)
            self.gammas = {k: self.gammas[k] for k in self.base_values}

        # Update
        self.last_step = last_step
        self.update()

    def dotted_get(self, key: str, default=None) -> float | int:
        obj = self.obj
        for attr in key.split(MultiStepParams.sep):
            if isinstance(obj, MutableSequence):
                # Sequence
                attr = int(attr)
                if attr < len(obj):
                    obj = attr[attr]
                else:
                    obj = default
            elif isinstance(obj, MutableMapping):
                # Mapping
                obj = obj.get(attr, default)
            else:
                # Object
                obj = getattr(obj, attr, default)

            if obj is default:
                return default
        return obj

    def get(self) -> dict[str, float | int]:
        return {key: self.dotted_get(key) for key in self.gammas}

    def set(self, values: dict[str, float | int]):
        for key in self.gammas:
            obj = self.obj
            attr = key
            if MultiStepParams.sep in key:
                subkeys, attr = key.rsplit(MultiStepParams.sep, 1)
                obj = self.dotted_get(subkeys)
            if isinstance(obj, MutableSequence):
                attr = int(attr)
            elif not isinstance(obj, MutableMapping):
                obj = obj.__dict__
            obj[attr] = values[key]  # type: ignore

    def step(self):
        self.last_step += 1
        if self.last_step in self.milestones:
            values = {
                k: v * self.gammas[k] ** self.milestones[self.last_step]
                for k, v in self.get().items()
            }
            self.set(values)

    def update(self):
        milestones = list(sorted(self.milestones.elements()))
        values = {
            k: v * self.gammas[k] ** bisect_right(milestones, self.last_step)
            for k, v in self.base_values.items()
        }
        self.set(values)

    def state_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != "obj"}

    def load_state_dict(self, state_dict: dict):
        self.__dict__.update(state_dict)
        self.update()


class EMA(ExponentialMovingAverage):
    def __init__(
        self,
        *args,
        update_after_step: int = 100,
        update_every: int = 10,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, use_num_updates=True, **kwargs)
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

    def get_current_decay(self) -> float:
        # From https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
        epoch = max(self.num_updates - self.update_after_step - 1, 0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch <= 0:
            return 0.0

        return min(max(value, self.min_value), self.decay)

    def update(self, parameters: Iterable[torch.nn.Parameter] | None = None) -> None:
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        self.num_updates += 1

        if (self.num_updates % self.update_every) != 0:
            return

        parameters = self._get_parameters(parameters)
        device = self.shadow_params[0].device

        if self.num_updates <= self.update_after_step:
            self.shadow_params = [p.clone().detach().to(device) for p in parameters]
            return

        decay = self.get_current_decay()
        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                param = param.to(device)
                tmp = s_param - param
                # Tmp will be a new tensor so we can do in-place
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    @contextlib.contextmanager
    def average_parameters(
        self, parameters: Iterable[torch.nn.Parameter] | None = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        Equivalent to:

            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)
            # Free memory of collected params
            self.collected_params = None
