"""
Adapted from https://github.com/qsh-zh/pis
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas
import torch

from .base import DATA_DIR, Distribution


def read_points(file_path):
    df = pandas.read_csv(file_path)
    x_pos, y_pos = np.array(df["data_x"]), np.array(df["data_y"])
    pos = np.vstack([x_pos, y_pos]).T
    return pos


def get_bin_counts(array_in, num_bins_per_dim):
    scaled_array = array_in * num_bins_per_dim
    counts = np.zeros((num_bins_per_dim, num_bins_per_dim))
    for elem in scaled_array:
        flt_row, col_row = np.floor(elem)
        row = int(flt_row)
        col = int(col_row)
        # Deal with the case where the point lies exactly on upper/rightmost edge.
        if row == num_bins_per_dim:
            row -= 1
        if col == num_bins_per_dim:
            col -= 1
        counts[row, col] += 1
    return counts


def get_bin_vals(num_bins: int):
    grid_indices = np.arange(num_bins)
    bin_vals = np.array(
        [np.array(elem) for elem in itertools.product(grid_indices, grid_indices)]
    )

    return bin_vals


def th_batch_kernel_fn(x, y, signal_variance, num_grid_per_dim, raw_length_scale):
    x = x.view(-1, 1, x.shape[-1])
    y = y.view(1, -1, x.shape[-1])
    dist = torch.norm(x - y, dim=2) / (num_grid_per_dim * raw_length_scale)
    return signal_variance * torch.exp(-dist)


def get_latents_from_white(white, const_mean, cholesky_gram):
    """
    white: (B,D)
    const_mean: scalar
    cholesky_gram: (D,D)
    """
    return torch.einsum("ij,bj->bi", cholesky_gram, white) + const_mean


def get_white_from_latents(latents, const_mean, cholesky_gram):
    """
    latents: (B,D)
    const_mean: scalar
    cholesky_gram: (D,D)
    """
    white = torch.linalg.solve_triangular(
        cholesky_gram, latents.unsqueeze(-1) - const_mean, upper=False
    )
    return white.squeeze(dim=-1)


def poisson_process_log_likelihood(latent_function, bin_area, flat_bin_counts):
    """
    latent_function: (B,D)
    bin_area: Scalar
    flat_bin_counts: (D)
    """
    first_term = latent_function * flat_bin_counts.unsqueeze(0)  # (B,D)
    second_term = -bin_area * torch.exp(latent_function)
    return torch.sum(first_term + second_term, dim=1)  # (B,)


class Cox(Distribution):
    def __init__(
        self,
        dim: int = 1600,
        data_path: str | Path = DATA_DIR / "pines.csv",
        use_whitened: bool = False,
    ):
        # `log_norm_const` is taken from DDS paper, computed using long run SMC chain
        # (1000 temperatures, 30 seeds with 2000 samples each)
        super().__init__(dim=dim, log_norm_const=512.6)
        self.use_whitened = use_whitened
        self.path = Path(data_path)
        self.num_bins_per_dim = int(np.sqrt(dim))
        assert self.num_bins_per_dim**2 == self.dim

        self.signal_variance = 1.91
        self._poisson_a = 1.0 / self.dim
        self._beta = 1.0 / 33
        self.white_gaussian_log_normalizer = (
            -0.5 * self.dim * np.log(2.0 * np.pi)
        ).item()
        self.mu_zero = (np.log(126.0) - 0.5 * self.signal_variance).item()

        bin_counts = np.array(
            get_bin_counts(read_points(self.path), self.num_bins_per_dim)
        )
        bin_vals = torch.from_numpy(get_bin_vals(self.num_bins_per_dim)).float()
        gram_matrix = th_batch_kernel_fn(
            bin_vals, bin_vals, self.signal_variance, self.num_bins_per_dim, self._beta
        )

        self.register_buffer(
            "cholesky_gram", torch.linalg.cholesky(gram_matrix), persistent=False
        )
        self.register_buffer(
            "flat_bin_counts",
            torch.from_numpy(bin_counts.flatten()).float(),
            persistent=False,
        )

        half_log_det_gram = torch.sum(
            torch.log(torch.abs(torch.diag(self.cholesky_gram)))
        )
        self.unwhitened_gaussian_log_normalizer = (
            -0.5 * self.dim * np.log(2.0 * np.pi) - half_log_det_gram
        ).item()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_whitened:
            log_density = self.whitened_posterior_log_density(x)
        else:
            log_density = self.unwhitened_posterior_log_density(x)
        assert log_density.shape == (x.shape[0],)
        return log_density.unsqueeze(-1)

    def whitened_posterior_log_density(self, white):
        quadratic_term = -0.5 * torch.sum(white**2, dim=1)
        prior_log_density = self.white_gaussian_log_normalizer + quadratic_term
        latent_function = get_latents_from_white(
            white, self.mu_zero, self.cholesky_gram
        )
        log_likelihood = poisson_process_log_likelihood(
            latent_function, self._poisson_a, self.flat_bin_counts
        )
        return prior_log_density + log_likelihood

    def unwhitened_posterior_log_density(self, latents):
        white = get_white_from_latents(latents, self.mu_zero, self.cholesky_gram)
        prior_log_density = (
            -0.5 * torch.sum(white * white, dim=1)
            + self.unwhitened_gaussian_log_normalizer
        )
        log_likelihood = poisson_process_log_likelihood(
            latents, self._poisson_a, self.flat_bin_counts
        )
        return prior_log_density + log_likelihood
