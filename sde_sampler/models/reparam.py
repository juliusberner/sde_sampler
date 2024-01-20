from __future__ import annotations

from typing import Callable

import torch
from torch.nn import Module

from sde_sampler.eq.sdes import OU
from sde_sampler.utils.autograd import compute_gradx
from sde_sampler.utils.common import clip_and_log


class ClippedCtrl(Module):
    def __init__(
        self,
        base_model: Module,
        clip_model: float | None = None,
        name: str = "ctrl",
        **kwargs,
    ):
        super().__init__()
        self.base_model = base_model
        self.clip_model = clip_model
        self.name = name

    def clipped_base_model(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        output = clip_and_log(
            self.base_model(t, x),
            max_norm=self.clip_model,
            name=self.name + "_model",
            t=t,
        )
        return output

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.clipped_base_model(t, x)


class ScoreCtrl(ClippedCtrl):
    def __init__(
        self,
        *args,
        target_score: Callable,
        score_model: Module | None = None,
        detach_score: bool = True,
        scale_score: float = 1.0,
        clip_score: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.score_model = score_model
        self.target_score = target_score
        self.detach_score = detach_score
        self.scale_score = scale_score
        self.clip_score = clip_score

    def clipped_target_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = clip_and_log(
            self.target_score(x, create_graph=self.detach_score),
            max_norm=self.clip_score,
            name=self.name + "_score",
            t=t,
        )
        assert output.shape == x.shape
        return output

    def clipped_score_model(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        output = clip_and_log(
            self.score_model(t, x),
            max_norm=self.clip_model,
            name=self.name + "_score_model",
            t=t,
        )
        assert output.shape in [(1, 1), (1, x.shape[-1]), x.shape, (x.shape[0], 1)]
        return output

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctrl = self.clipped_base_model(t, x)
        score = self.scale_score * self.clipped_target_score(t, x)
        if self.score_model is not None:
            score *= self.clipped_score_model(t, x)
        return ctrl + score


class CancelDriftCtrl(ScoreCtrl):
    def __init__(self, *args, sde: OU, langevin_init: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if sde.noise_type not in ["diagonal", "scalar"]:
            raise ValueError(f"Invalid sde noise type {sde.noise_type}.")
        self.sde = sde
        self.langevin_init = langevin_init

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctrl = self.clipped_base_model(t, x)

        # Score
        sde_diff = self.sde.diff(t, x)
        if self.langevin_init:
            # This yields langevin dynamis if ctrl == 0
            # And score_model is None or score_model == 1
            scale = sde_diff**2 / 2
        else:
            scale = self.scale_score
        score = scale * self.clipped_target_score(t, x)

        if self.score_model is not None:
            score *= self.clipped_score_model(t, x)

        return ctrl + (score - self.sde.drift(t, x)) / sde_diff


class LerpCtrl(ScoreCtrl):
    def __init__(
        self,
        *args,
        sde: OU,
        prior_score: Callable,
        hard_constrain: bool = False,
        scale_lerp: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sde.noise_type not in ["diagonal", "scalar"]:
            raise ValueError(f"Invalid sde noise type {sde.noise_type}.")
        self.sde = sde
        self.prior_score = prior_score
        self.hard_constrain = hard_constrain
        self.scale_lerp = scale_lerp

    def clipped_interpolated_score(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = self.target_score(x, create_graph=self.detach_score)
        output = torch.lerp(self.prior_score(x), output, t / self.sde.terminal_t)
        output = clip_and_log(
            output,
            max_norm=self.clip_score,
            name=self.name + "_score",
            t=t,
        )
        assert output.shape == x.shape
        return output

    def constrain(self, output, t):
        return 4 * output * (self.sde.terminal_t - t) * t / self.terminal_t**2

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctrl = self.clipped_base_model(t, x)
        if self.hard_constrain:
            ctrl = self.constrain(ctrl, t)

        # Interpolated score
        score = self.scale_score * self.clipped_interpolated_score(t, x)
        if self.score_model is not None:
            score_model = self.clipped_score_model(t, x)
            if self.hard_constrain:
                score_model = self.constrain(score_model, t)
            score *= score_model

        return ctrl + self.sde.diff(t, x) * score


class LerpPriorCtrl(LerpCtrl):
    def clipped_interpolated_score(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = (1.0 - t / self.sde.terminal_t) * self.prior_score(x)
        output = clip_and_log(
            output,
            max_norm=self.clip_score,
            name=self.name + "_score",
            t=t,
        )
        assert output.shape == x.shape
        return output

    def constrain(self, output, t):
        return 2 * output * t / self.terminal_t


class LerpTargetCtrl(LerpCtrl):
    def clipped_interpolated_score(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = t / self.sde.terminal_t * self.target_score(x)
        output = clip_and_log(
            output,
            max_norm=self.clip_score,
            name=self.name + "_score",
            t=t,
        )
        assert output.shape == x.shape
        return output

    def constrain(self, output, t):
        return 2 * output * (1.0 - t / self.terminal_t)


class PotentialCtrl(ClippedCtrl):
    def __init__(
        self,
        *args,
        sde: OU,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sde = sde

    def unnorm_log_prob(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.clipped_base_model(t, x)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (
            self.sde.diff(t, x)
            * compute_gradx(self.clipped_base_model, t=t, x=x, retain_graph=True)[0]
        )
