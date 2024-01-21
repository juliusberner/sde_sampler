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
    elif rescale_t == "cosine":
        """
        Copied verbatim from 
        https://github.com/franciscovargas/denoising_diffusion_samplers/blob/main/dds/discretisation_schemes.py#L50
        """
        s = 0.008 # Choice from original paper
        pre_phase = torch.linspace(start, end, steps + 1, device=device) / end
        phase = ((pre_phase + s) / (1 + s)) * torch.pi * 0.5

        dts = torch.cos(phase)**4

        dts /= dts.sum()
        dts *= end  # We normalise s.t. \sum_k \beta_k = T (where beta_k = b_m*cos^4)

        dts_out = torch.concat((torch.tensor([start], device = device), torch.cumsum(dts,-1)))

        # New Adaptation for this repo: 
        # We remove excess 1s at end 
        ptr = len(dts_out) 
        eps = 1e-6
        while abs(dts_out[ptr - 1] - end) < eps * end:
            ptr -= 1

        return dts_out[:ptr]
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
