from __future__ import annotations

import logging
from numbers import Number
from typing import Callable

import torch

from sde_sampler.distr.base import EXPECTATION_FNS, Distribution


def abs_and_rel_error(
    prediction: Number, target: Number, suffix: str = "", eps: float = 1e-8
) -> dict[str, float]:
    assert isinstance(prediction, Number)
    assert isinstance(target, Number)
    magnitude = abs(target) + eps
    error = abs(prediction - target)
    return {
        f"error{suffix}": error,
        f"rel_error{suffix}": error / magnitude,
    }


def compute_errors(
    prediction: torch.Tensor | float,
    target: torch.Tensor | float | None = None,
    name: str = "error",
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    # Prediction
    output = {}
    if isinstance(prediction, Number):
        output[f"eval/{name}"] = prediction
    else:
        assert isinstance(prediction, torch.Tensor)
        if prediction.ndim == 0:
            output[f"eval/{name}"] = prediction.item()
        else:
            assert prediction.ndim == 2 and prediction.shape[-1] == 1
            output[f"eval/{name}"] = prediction.mean().item()
            if weights is not None:
                assert weights.shape == prediction.shape
                weighted_pred = (prediction * weights).sum() / weights.sum()
                output[f"eval/{name}_is"] = weighted_pred.item()

    # Error
    if target is not None:
        if not isinstance(target, Number):
            assert target.ndim == 0
            target = target.item()

        for key_name, pred in output.copy().items():
            suffix = key_name.replace("eval", "")
            errors = abs_and_rel_error(
                prediction=pred, target=target, suffix=suffix, eps=eps
            )
            output.update(errors)

    return output


def frac_inside_domain(samples, domain):
    assert samples.shape[-1] == domain.shape[0]
    samples_inside = (domain[:, 0] <= samples) & (samples <= domain[:, 1])
    return samples_inside.all(dim=-1).float().mean().item()


def get_metrics(
    distr: Distribution,
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
    log_norm_const_preds: dict[str, torch.Tensor | float] | None = None,
    expectation_preds: dict[str, torch.Tensor | float] | None = None,
    marginal_dims: list[int] | None = None,
    sample_losses: dict[str, Callable] | None = None,
) -> dict[str, float]:
    # Filter
    if not all(d < distr.dim for d in marginal_dims):
        logging.warning("Removing non-existent marginal dims for metrics.")
        marginal_dims = [d for d in marginal_dims if d < distr.dim]

    metrics = {}
    marginal_dims = marginal_dims or []
    expectation_preds = expectation_preds or {}
    log_norm_const_preds = log_norm_const_preds or {}

    # Expectations
    for name, fn in EXPECTATION_FNS.items():
        target = distr.expectations.get(name)
        prediction = fn(samples)
        metrics.update(
            compute_errors(
                prediction=prediction, target=target, name=name, weights=weights
            )
        )

        if name in expectation_preds:
            direct_pred = expectation_preds[name]
            metrics.update(
                compute_errors(
                    prediction=direct_pred,
                    target=target,
                    name=name + "_direct",
                    weights=weights,
                )
            )

    # Log. normalization constant
    for name, log_norm_const_pred in log_norm_const_preds.items():
        metrics.update(
            compute_errors(
                prediction=log_norm_const_pred,
                target=distr.log_norm_const,
                name=name,
            )
        )

    # ESS
    if weights is not None:
        assert weights.shape == (samples.shape[0], 1)
        ess = weights.sum() ** 2 / (weights**2).sum()
        ess = ess.item()
        metrics["eval/effective_sample_size"] = ess
        metrics["eval/norm_effective_sample_size"] = ess / len(weights)

    # Stddevs
    stddevs = samples.std(dim=0)
    avg_stddev = stddevs.mean().item()
    means = samples.mean(dim=0)
    metrics["eval/avg_stddev"] = avg_stddev
    for dim in marginal_dims:
        metrics[f"eval/stddev_{dim}"] = stddevs[dim].item()
        metrics[f"eval/avg_{dim}"] = means[dim].item()

    if distr.stddevs is not None:
        assert distr.stddevs.shape == stddevs.shape
        metrics["error/avg_marginal_stddev"] = (
            (stddevs - distr.stddevs).abs().mean().item()
        )
        metrics.update(
            compute_errors(
                prediction=avg_stddev,
                target=distr.stddevs.mean(),
                name="avg_stddev",
            )
        )

    # Samples inside domain
    if distr.domain is not None:
        metrics["eval/frac_pred_in_domain"] = frac_inside_domain(samples, distr.domain)

    # Other Losses based on samples of the distr
    if sample_losses is not None:
        if hasattr(distr, "sample"):
            gt_samples = distr.sample((samples.shape[0],))
            assert gt_samples.shape == samples.shape
            if distr.domain is not None:
                metrics["eval/frac_groundtruth_in_domain"] = frac_inside_domain(
                    gt_samples, distr.domain
                )

            metrics.update(
                {
                    "error/" + name: loss(samples, gt_samples).item()
                    for name, loss in sample_losses.items()
                }
            )
        else:
            logging.warning(
                "Sampling not implemented for distribution %s.",
                distr.__class__.__name__,
            )

    # Objective
    if hasattr(distr, "objective"):
        metrics["eval/obj_avg"] = distr.objective(
            samples.mean(dim=0, keepdims=True)
        ).item()
        metrics["eval/avg_obj"] = distr.objective(samples).mean().item()
        metrics["eval/min_obj"] = distr.objective(samples).min().item()

    return metrics
