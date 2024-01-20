from __future__ import annotations

import math
from collections import namedtuple

import torch

import wandb

Results = namedtuple(
    "Results",
    "samples weights log_norm_const_preds expectation_preds ts xs metrics plots",
    defaults=[{}, {}, None, None, None, None, {}, {}],
)

CKPT_DIR = "ckpt"


def get_timesteps(
    start: torch.Tensor | float,
    end: torch.Tensor | float,
    dt: torch.Tensor | float | None = None,
    steps: int | None = None,
    rescale_t: str | None = None,
    device: str | torch.device = None,
) -> torch.Tensor:
    if (steps is None) is (dt is None):
        raise ValueError("Exactly one of `dt` and `steps` should be defined.")
    if steps is None:
        steps = int(math.ceil((end - start) / dt))
    if rescale_t is None:
        return torch.linspace(start, end, steps=steps + 1, device=device)
    elif rescale_t == "quad":
        return torch.sqrt(
            torch.linspace(start, end.square(), steps=steps + 1, device=device)
        ).clip(max=end)
    raise ValueError("Unkown timestep rescaling method.")


def clip_and_log(
    tensor: torch.Tensor,
    max_norm: float | None = None,
    name: str | None = None,
    t: torch.Tensor | None = None,
    log_dt: float = 0.2,
) -> torch.Tensor:
    # Log
    if __debug__ and name is not None and wandb.run is not None:
        log = (
            t is None
            or torch.isclose(
                t % log_dt,
                torch.tensor([0.0, log_dt], device=t.device),
                atol=1e-4,
            ).any()
        )
        if log:
            name = name if t is None else f"{name}_{t:.3f}"
            wandb.log(
                {"clip/" + name: tensor.abs().max().item()},
                commit=False,
            )

    # Clip
    if max_norm is not None:
        tensor = tensor.clip(min=-1.0 * max_norm, max=max_norm)
    return tensor
