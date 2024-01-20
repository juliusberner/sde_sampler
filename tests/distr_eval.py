import unittest
from pathlib import Path

import numpy as np
import torch

from sde_sampler.distr.aladip import AlaDip
from sde_sampler.distr.base import Distribution, sample_uniform
from sde_sampler.distr.cox import Cox
from sde_sampler.distr.double_well import DoubleWell, MultiWell
from sde_sampler.distr.funnel import Funnel
from sde_sampler.distr.gauss import GMM, Gauss
from sde_sampler.distr.img import Img
from sde_sampler.distr.nice import Nice
from sde_sampler.distr.rings import Rings
from sde_sampler.distr.rosenbrock import Rosenbrock
from sde_sampler.eval import metrics, plots, sinkhorn

torch.manual_seed(42)
np.random.seed(42)


class TestDistr(unittest.TestCase):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setUpClass(cls):
        cls.distr = [
            DoubleWell(),
            Gauss(),
            GMM(name="grid"),
            MultiWell(),
            Funnel(),
            Img(),
            Rings(),
            AlaDip(),
            Cox(),
            Nice(),
            Rosenbrock(),
        ]
        for distr in cls.distr:
            distr.to(cls.device)
            distr.compute_stats()

    def test_score(self, batch_size=1000):
        for distr in TestDistr.distr:
            if not type(distr).score == Distribution.score:
                with self.subTest(distr=distr.__class__.__name__):
                    samples = sample_uniform(distr.domain, batch_size)
                    torch.testing.assert_close(
                        distr.score(samples),
                        Distribution.score(distr, samples),
                        rtol=1e-4,
                        atol=1e-4,
                    )

    @torch.no_grad()
    def test_metrics_and_plots(self, batch_size=int(5e6)):
        for distr in TestDistr.distr:
            if hasattr(distr, "sample"):
                distr_name = distr.__class__.__name__
                with self.subTest(distr=distr_name):
                    samples = distr.sample((batch_size,))
                    weights = torch.ones(samples.shape[0], 1, device=samples.device)
                    log_norm_const_preds = {}
                    if distr.log_norm_const is not None:
                        log_norm_const_preds = {"gt": distr.log_norm_const}
                    expectation_preds = distr.expectations
                    marginal_dims = list(range(distr.dim))
                    sample_losses = {"sinkhorn": sinkhorn.Sinkhorn(n_max=10000)}

                    outputs = plots.get_plots(
                        distr=distr,
                        samples=samples[:50000],
                        weights=None,
                        marginal_dims=marginal_dims[:3],
                    )
                    if hasattr(distr, "plots"):
                        outputs.update(distr.plots(samples[:50000]))

                    for name, fig in outputs.items():
                        name = name.split("/")[-1]
                        path = (
                            Path(__file__).parents[1]
                            / "logs"
                            / "test"
                            / distr_name
                            / f"{name}.png"
                        )
                        plots.save_fig(fig, path)

                    if hasattr(distr, "metrics"):
                        outputs.update(distr.metrics(samples))
                    outputs = metrics.get_metrics(
                        distr=distr,
                        samples=samples,
                        weights=weights,
                        log_norm_const_preds=log_norm_const_preds,
                        expectation_preds=expectation_preds,
                        marginal_dims=marginal_dims,
                        sample_losses=sample_losses,
                    )

                    for name, result in outputs.items():
                        if "error" in name:
                            if "direct" in name:
                                self.assertAlmostEqual(result, 0.0)

                            elif ("rel_error" in name) and abs(
                                outputs[name.replace("rel_error", "eval")]
                            ) > 0.1:
                                self.assertAlmostEqual(result, 0.0, delta=0.15)

                        elif "sinkhorn" in name:
                            self.assertAlmostEqual(result, 0.0, delta=0.1)

                        elif name.endswith("_is"):
                            self.assertAlmostEqual(
                                outputs[name.replace("_is", "")], result, delta=1e-4
                            )

                        elif "in_domain" in name:
                            self.assertGreater(outputs[name], 0.9)

                    for name, target in expectation_preds.items():
                        self.assertAlmostEqual(outputs[f"eval/{name}_direct"], target)


if __name__ == "__main__":
    unittest.main()
