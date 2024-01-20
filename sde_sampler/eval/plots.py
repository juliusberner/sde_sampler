from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Callable

import plotly.graph_objects as go
import torch
from matplotlib import pyplot as plt
from PIL.Image import Image
from scipy.ndimage import gaussian_filter

from sde_sampler.distr.base import Distribution


def bins_to_kwargs(
    binsx: torch.Tensor, binsy: torch.Tensor | None = None
) -> dict[str, float]:
    output = {
        "x0": (binsx[1] + binsx[0]) / 2,
        "dx": binsx[1] - binsx[0],
    }
    if binsy is not None:
        output.update({"y0": (binsy[1] + binsy[0]) / 2, "dy": binsy[1] - binsy[0]})
    return {k: v.item() for k, v in output.items()}


def plot_contours_2d(
    log_prob: Callable,
    domain: torch.Tensor,
    nbins: int = 200,
    levels: int = 50,
    thresh: float = -1000.0,
    ax: plt.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(1)

    x = torch.linspace(*domain[0], nbins, device=domain.device)
    y = torch.linspace(*domain[1], nbins, device=domain.device)
    x, y = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([x, y], dim=-1)
    log_p = log_prob(xy.flatten(end_dim=1)).clip(min=thresh).view_as(x)
    ax.contour(x.cpu(), y.cpu(), log_p.cpu(), levels=levels)
    ax.set_ylabel(r"$x_1$")
    ax.set_xlabel(r"$x_2$")
    return ax.get_figure()


def mpl_plot_marginal_2d(
    x: torch.Tensor,
    dim1: int = 0,
    dim2: int = 1,
    weights: torch.Tensor | None = None,
    nbins: int = 100,
    domain: torch.Tensor | None = None,
    smoothing: float = 0.1,
    ax: plt.Axes = None,
    scatter: bool = False,
    thresh: float = 0.0,
) -> plt.Figure:
    data = x[:, [dim1, dim2]].cpu()
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if domain is not None:
        domain = domain[[dim1, dim2]].flatten().tolist()
    if weights is not None:
        weights = weights.cpu()
    heatmap, (binsx, binsy) = torch.histogramdd(
        x=data, bins=nbins, range=domain, weight=weights, density=True
    )

    heatmap = gaussian_filter(heatmap, sigma=smoothing)
    extent = [binsx[0], binsx[-1], binsy[0], binsy[-1]]
    palette = plt.get_cmap("Blues")
    palette.set_under("white", 0.0)
    ax.imshow(heatmap.T, extent=extent, vmin=thresh, origin="lower", cmap=palette)

    if scatter:
        ax.scatter(x=data[:, 0], y=data[:, 1], s=0.001, c="w")

    ax.set_ylabel(rf"$x_{dim2+1:d}$")
    ax.set_xlabel(rf"$x_{dim1+1:d}$")
    return ax.get_figure()


def plot_marginal_2d(
    x: torch.Tensor,
    dim1: int = 0,
    dim2: int = 1,
    weights: torch.Tensor | None = None,
    nbins: int = 100,
    domain: torch.Tensor | None = None,
    decimals: int = 6,
) -> go.Figure:
    if domain is not None:
        domain = domain[[dim1, dim2]].flatten().tolist()
    if weights is not None:
        weights = weights.cpu()
    # TODO(jberner): Track torch histogram(dd) GPU support
    # See https://github.com/pytorch/pytorch/issues/69519
    heights, (binsx, binsy) = torch.histogramdd(
        x[:, [dim1, dim2]].cpu(),
        bins=nbins,
        range=domain,
        weight=weights,
        density=True,
    )
    fig = go.Figure(
        go.Heatmap(**bins_to_kwargs(binsx, binsy), z=heights.round(decimals=decimals))
    )
    fig.update_traces(showscale=False)
    return fig


def plot_marginal(
    x: torch.Tensor,
    weights: torch.Tensor | None = None,
    marginal: Callable | None = None,
    dim: int = 0,
    nbins: int = 100,
    domain: torch.Tensor | None = None,
    decimals: int = 6,
) -> go.Figure:
    data = x[:, dim].unsqueeze(-1).cpu()
    if domain is None:
        domain = (data.min().item(), data.max().item())
    else:
        domain = domain[dim].tolist()

    heights, bins = torch.histogram(data, bins=nbins, range=domain, density=True)
    fig = go.Figure(
        go.Bar(
            **bins_to_kwargs(bins),
            y=heights.round(decimals=decimals).cpu(),
            name="histogram",
        )
    )
    if weights is not None:
        heights, bins = torch.histogram(
            data, bins=nbins, range=domain, weight=weights.cpu(), density=True
        )
        fig = fig.add_trace(
            go.Bar(
                **bins_to_kwargs(bins),
                y=heights.round(decimals=decimals),
                name="histogram_is",
            )
        )
    if marginal is not None:
        xlin = torch.linspace(*domain, steps=nbins, device=x.device)
        marginal_eval = marginal(xlin.unsqueeze(-1), dim=dim).round(decimals=decimals)
        assert marginal_eval.shape == (nbins, 1)
        fig.add_trace(
            go.Scatter(
                x=xlin.cpu(),
                y=marginal_eval.squeeze(-1).cpu(),
                mode="lines",
                name="marginal",
            )
        )
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_traces(opacity=0.85)
    return fig


def plot_evolution(
    ts: torch.Tensor,
    xs: torch.Tensor,
    dim: int = 0,
    ntraj: int = 50,
    domain: torch.Tensor | None = None,
    decimals: int = 6,
) -> go.Figure:
    fig = go.Figure()
    if domain is not None:
        fig.update_layout(yaxis_range=domain[dim].cpu())
    trajs = xs[:, :, dim].T

    # filter non-finite
    mask = trajs.isfinite().all(dim=1)
    discard = mask.numel() - mask.sum()
    if discard > 0:
        logging.warning("Filtering %d trajectory with non-finite values.", discard)

    if discard < mask.numel():
        trajs = trajs[mask][:ntraj]

        hues = (
            100
            * (trajs[:, -1] - trajs[:, -1].min())
            / (1e-8 + trajs[:, -1].max() - trajs[:, -1].min())
        )
        for traj, hue in zip(trajs, hues):
            fig.add_trace(
                go.Scatter(
                    x=ts.round(decimals=decimals).cpu(),
                    y=traj.round(decimals=decimals).cpu(),
                    mode="lines",
                    line={
                        "color": f"hsv({hue.round(decimals=decimals):f}%,100%,50%)",
                        "width": 0.4,
                    },
                )
            )
    return fig


def get_plots(
    distr: Distribution,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    ts: torch.Tensor | None = None,
    xs: torch.Tensor | None = None,
    marginal_dims: list[int] | None = None,
    decimals: int = 6,
    nbins: int = 100,
    domain: torch.Tensor | None = None,
) -> dict[str, go.Figure]:
    plots = {}
    marginal_dims = marginal_dims or []
    if domain is None and distr.domain is not None:
        domain = distr.domain if distr.domain.isfinite().all() else None

    # Plots
    if not all(d < distr.dim for d in marginal_dims):
        logging.warning("Removing non-existent marginal dims for plotting.")
        marginal_dims = [d for d in marginal_dims if d < distr.dim]

    for d in marginal_dims:
        if ts is not None and xs is not None:
            plots[f"plots/traj_{d}"] = plot_evolution(
                ts=ts, xs=xs, dim=d, decimals=decimals, domain=domain
            )
        plots[f"plots/hist_{d}"] = plot_marginal(
            x=samples,
            weights=weights,
            marginal=getattr(distr, "marginal", None),
            dim=d,
            nbins=nbins,
            decimals=decimals,
            domain=domain,
        )

    for dim1, dim2 in itertools.combinations(marginal_dims, r=2):
        plots[f"plots/density_{dim1}_{dim2}"] = plot_marginal_2d(
            x=samples,
            dim1=dim1,
            dim2=dim2,
            nbins=nbins,
            decimals=decimals,
            domain=domain,
        )

    if hasattr(distr, "sample"):
        gt_samples = distr.sample((samples.shape[0],))
        for dim1, dim2 in itertools.combinations(marginal_dims, r=2):
            plots[f"plots/groundtruth_density_{dim1}_{dim2}"] = plot_marginal_2d(
                x=gt_samples,
                dim1=dim1,
                dim2=dim2,
                nbins=nbins,
                decimals=decimals,
                domain=domain,
            )

    return plots


def save_fig(fig: Image | go.Figure | plt.Figure, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig, Image):
        fig.save(path)
    elif isinstance(fig, go.Figure):
        fig.write_image(path)
    elif isinstance(fig, plt.Figure):
        fig.savefig(path)
    else:
        raise ValueError(f"Unknown figure type {type(fig)}.")
