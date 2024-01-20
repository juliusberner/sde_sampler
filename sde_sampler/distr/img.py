"""
Adapted from https://github.com/noegroup/stochastic_normalizing_flows
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from .base import DATA_DIR, Distribution


class Img(Distribution):
    def __init__(
        self,
        dim: int = 2,
        mean: Sequence[int] = (350, 350),
        scale: Sequence[int] = (100, 100),
        outside_penalty=1.0,
        path: str | Path = DATA_DIR / "labrador.jpg",
        embed: Sequence[int] | None = None,
        crop: Sequence[int] = (10, 710, 240, 940),
        white_cutoff: float = 225.0,
        gauss_sigma: float = 3.0,
        background: float = 0.01,
        domain: float | torch.Tensor | None = 3.5,
        n_reference_samples: int = int(1e7),
        **kwargs,
    ):
        super().__init__(
            dim=dim, domain=domain, n_reference_samples=n_reference_samples, **kwargs
        )

        self.path = path
        self.outside_penalty = outside_penalty

        # Init
        img = mpimg.imread(self.path)

        # Make one channel
        img = img.mean(axis=2)

        # Make background white
        img = img.astype(np.float32)
        img[img > white_cutoff] = 255

        # Normalize
        img /= img.max()

        if crop is not None:
            # Crop
            img = img[crop[0] : crop[1], crop[2] : crop[3]]

        if embed is not None:
            tmp = np.ones((embed[0], embed[1]), dtype=np.float32)
            shift_x = (embed[0] - img.shape[0]) // 2
            shift_y = (embed[1] - img.shape[1]) // 2
            tmp[
                shift_x : img.shape[0] + shift_x, shift_y : img.shape[1] + shift_y
            ] = img
            img = tmp

        # Convolve with Gaussian
        img2 = gaussian_filter(img, sigma=gauss_sigma)

        # Add background
        background1 = gaussian_filter(img, sigma=10)
        background2 = gaussian_filter(img, sigma=20)
        background3 = gaussian_filter(img, sigma=50)
        density = (1.0 - img2) + background * (background1 + background2 + background3)

        density = density[::-1]
        energy = -np.log(density)
        energy -= energy.min()

        # Setup sampler
        Ix, Iy = np.meshgrid(np.arange(density.shape[1]), np.arange(density.shape[0]))
        density_normed = density.astype(np.float64)
        density_normed /= density_normed.sum()

        # Set attributes
        self.register_buffer(
            "idx",
            torch.from_numpy(np.vstack([Ix.flatten(), Iy.flatten()]).T),
            persistent=False,
        )
        self.register_buffer(
            "density_flat", torch.from_numpy(density_normed.flatten()), persistent=False
        )
        self.register_buffer("pixel_energy", torch.from_numpy(energy), persistent=False)
        self.register_buffer(
            "maxindex_x",
            torch.tensor([self.pixel_energy.shape[1] - 1]),
            persistent=False,
        )
        self.register_buffer(
            "maxindex_y",
            torch.tensor([self.pixel_energy.shape[0] - 1]),
            persistent=False,
        )
        self.register_buffer("mean", torch.tensor([mean]), persistent=False)
        self.register_buffer("scale", torch.tensor([scale]), persistent=False)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        Xs = x * self.scale + self.mean
        I = Xs.to(dtype=torch.long)
        Ix = I[:, 0]
        Iy = I[:, 1]
        dx = Xs[:, 0] - Ix.to(dtype=torch.float32)
        dy = Xs[:, 1] - Iy.to(dtype=torch.float32)

        zero = torch.tensor([0], device=x.device)
        # Select closest pixel inside image
        Ix_inside = torch.max(torch.min(Ix, self.maxindex_x - 1), zero + 1)
        Iy_inside = torch.max(torch.min(Iy, self.maxindex_y - 1), zero + 1)
        E0 = self.pixel_energy[Iy_inside, Ix_inside]
        dEdx = 0.5 * (
            self.pixel_energy[Iy_inside, Ix_inside + 1]
            - self.pixel_energy[Iy_inside, Ix_inside - 1]
        )
        dEdy = 0.5 * (
            self.pixel_energy[Iy_inside + 1, Ix_inside]
            - self.pixel_energy[Iy_inside - 1, Ix_inside]
        )

        image_energy = E0 + dx * dEdx + dy * dEdy
        image_energy = image_energy.unsqueeze(-1)

        # Penalty factor from being outside image
        dx_left = torch.max(-Ix, zero)
        dx_right = torch.max(Ix - self.maxindex_x, zero)
        dx = torch.max(dx_left, dx_right)
        dy_down = torch.max(-Iy, zero)
        dy_up = torch.max(Iy - self.maxindex_y, zero)
        dy = torch.max(dy_down, dy_up)
        penalty = self.outside_penalty * (dx**2 + dy**2).to(dtype=torch.float32)
        penalty = penalty.unsqueeze(-1)

        return -image_energy - penalty

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        # Draw random index
        assert self.density_flat.numel() == self.idx.shape[0]
        i = self.density_flat.multinomial(
            num_samples=math.prod(shape), replacement=True
        )
        ixy = self.idx[i, :].reshape(*shape, 2)

        # Add random noise
        xy = ixy + torch.rand(*shape, 2, device=ixy.device) - 0.5

        # Normalize shape
        xy = (xy - self.mean) / self.scale
        return xy

    def plot_energies(self, axs: plt.Axes | None = None) -> plt.Figure:
        if axs is None:
            _, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 6))
        axs[0].imshow(mpimg.imread(self.path))
        energy = self.pixel_energy.cpu().numpy()
        axs[1].imshow(np.exp(-energy[::-1]), vmin=0, vmax=1, cmap="gray_r")
        axs[2].contourf(
            energy, 20, vmin=energy.min(), vmax=energy.max(), cmap="Spectral"
        )
        for ax in axs:
            ax.axis("off")
        return axs[0].get_figure()

    def plot_samples(
        self,
        samples: torch.Tensor,
        nbins: int = 100,
        vmax: float = 250.0,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        samples = samples.cpu().numpy()

        # Domain
        domain = self.domain
        if domain is not None:
            domain = domain.cpu()

        if ax is None:
            _, ax = plt.subplots()
        ax.hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=nbins,
            vmax=vmax,
            range=domain,
            cmap="gray_r",
        )
        if domain is not None:
            ax.set_xlim(*domain[0])
            ax.set_ylim(*domain[1])
        ax.axis("off")
        return ax.get_figure()

    def plots(
        self, samples: torch.Tensor, nbins: int = 100, vmax: float = 250.0
    ) -> dict[float, plt.Figure]:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
        fig.subplots_adjust(wspace=0.05)
        results = {
            "prediction": samples,
            "groundtruth": self.sample((samples.shape[0],)),
        }
        for ax, (name, x) in zip(axes, results.items()):
            ax.set_title(name, fontsize=15)
            fig = self.plot_samples(samples=x, nbins=nbins, vmax=vmax, ax=ax)

        return {"plots/comparison": fig, "plots/groundtruth": self.plot_energies()}
