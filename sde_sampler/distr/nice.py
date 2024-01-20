"""
Adapted from https://github.com/fmu2/NICE
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import make_grid

from .base import DATA_DIR, Distribution


class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super(StandardLogistic, self).__init__(validate_args=False)

    def log_prob(self, x):
        """Computes data log-likelihood.

        Args:
            x: input tensor.
        Returns:
            log-likelihood.
        """
        return -(nn.functional.softplus(x) + nn.functional.softplus(-x))

    def sample(self, size, eps=1e-20):
        """Samples from the distribution.

        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = torch.distributions.Uniform(eps, 1.0 - eps).sample(size)
        return torch.log(z) - torch.log(1.0 - z)


class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(nn.Linear(in_out_dim // 2, mid_dim), nn.ReLU())
        self.mid_block = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU())
                for _ in range(hidden - 1)
            ]
        )
        self.out_block = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W // 2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))


class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J


class NiceModel(nn.Module):
    def __init__(self, prior, coupling, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a NICE.

        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(NiceModel, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim

        self.coupling = nn.ModuleList(
            [
                Coupling(
                    in_out_dim=in_out_dim,
                    mid_dim=mid_dim,
                    hidden=hidden,
                    mask_config=(mask_config + i) % 2,
                )
                for i in range(coupling)
            ]
        )
        self.scaling = Scaling(in_out_dim)

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
        return self.scaling(x)

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        device = self.scaling.scale.device
        z = self.prior.sample((size, self.in_out_dim)).to(device)
        return self.g(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)


class Nice(Distribution):
    """NICE trained on resized MNIST."""

    def __init__(
        self,
        model: nn.Module | None = None,
        checkpoint: str = DATA_DIR / "nice.pt",
        mean_data_path: str = DATA_DIR / "mnist_mean.pt",
        sample_chunk_size: int = 10000,
        dim: int = 196,
        log_norm_const: float = 0.0,
        n_reference_samples=int(1e6),
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
        )
        self.shape = (14, 14)
        if not self.dim == math.prod(self.shape):
            raise ValueError(f"Dimension is {self.dim} but needs to be 196.")
        self.sample_chunk_size = sample_chunk_size
        mean = torch.load(mean_data_path).reshape((1, 28, 28))
        mean = Resize(size=self.shape, antialias=True)(mean).reshape((1, self.dim))
        self.register_buffer("mean", mean, persistent=False)

        # Model
        self.model = model
        if self.model is None:
            # Load checkpoint and NICE model
            ckpt = torch.load(checkpoint)
            self.model = NiceModel(
                prior=StandardLogistic(),
                coupling=ckpt["coupling"],
                in_out_dim=196,
                mid_dim=ckpt["mid_dim"],
                hidden=ckpt["hidden"],
                mask_config=ckpt["mask_config"],
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.log_prob(x).unsqueeze(-1) + self.log_norm_const

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = (1,)
        if len(shape) > 1:
            raise ValueError(f"Can only sample shapes (batch_size, dim).")

        # Chunk to avoid OOM
        size = shape[0]
        iterations, rem_size = divmod(size, self.sample_chunk_size)

        # Collect samples
        with torch.no_grad():
            samples = [
                self.model.sample(self.sample_chunk_size) for _ in range(iterations)
            ]
            if rem_size:
                samples.append(self.model.sample(rem_size))

        # Concatenate samples
        samples = torch.cat(samples)
        assert samples.shape == (size, self.dim)
        return samples

    def plots(self, samples: torch.Tensor, n_max=64) -> dict[str, Image.Image]:
        samples = samples + self.mean
        samples = samples.reshape(-1, 1, *self.shape)
        grid = make_grid(samples[:n_max], normalize=True)
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        im = Image.fromarray(ndarr)
        return {"plots/samples": im}
