"""
Adapted from https://github.com/lollcat/fab-torch/
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import boltzgen as bg
import matplotlib as mpl
import mdtraj as md
import numpy as np
import openmm
import torch
from matplotlib import pyplot as plt
from openmmtools import testsystems
from simtk import unit

from .base import DATA_DIR, Distribution


class AlaDip(Distribution):
    def __init__(
        self,
        dim: int = 60,
        data_path: Path | str | None = DATA_DIR / "position_min_energy.pt",
        temperature: int = 1000,
        energy_cut: float = 1.0e8,
        energy_max: float = 1.0e20,
        n_threads: int = 4,
        transform: str = "internal",
        ind_circ_dih: list = [],
        shift_dih: bool = False,
        shift_dih_params: dict = {"hist_bins": 100},
        default_std: dict = {"bond": 0.005, "angle": 0.15, "dih": 0.2},
        env: str = "vacuum",
        filter_chirality_train: bool = True,
        eval_data_path: Path | str | None = DATA_DIR / "aladip_val.pt",
        **kwargs,
    ):
        """
        Boltzmann distribution of Alanine dipeptide
        See https://github.com/lollcat/fab-torch/blob/master/fab/target_distributions/aldp.py
        :param data_path: Path to the trajectory file used to initialize the
            transformation, if None, a trajectory is generated
        :param temperature: Temperature of the system
        :param energy_cut: Value after which the energy is logarithmically scaled
        :param energy_max: Maximum energy allowed, higher energies are cut
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :param transform: Which transform to use, can be mixed or internal
        :param eval_data_path: Path to the test data
        """
        if dim != 60:
            raise ValueError(f"`dim` needs to be 60.")
        super().__init__(dim=dim, **kwargs)
        self.filter_chirality_train = filter_chirality_train
        self.transform = transform

        # Define molecule parameters
        if self.transform == "mixed":
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19]),
            ]
            cart_indices = [6, 8, 9, 10, 14]
        elif self.transform == "internal":
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (9, [8, 6, 4]),
                (10, [8, 6, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19]),
            ]
            cart_indices = [8, 6, 14]

        # System setup
        if env == "vacuum":
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == "implicit":
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError("This environment is not implemented.")
        sim = openmm.app.Simulation(
            system.topology,
            system.system,
            openmm.LangevinIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,
                1.0 * unit.femtosecond,
            ),
            openmm.Platform.getPlatformByName("Reference"),
        )

        if data_path is None:
            logging.info(
                "No data path is specified, Generating trajectory for coordinate transform."
            )
            sim = openmm.app.Simulation(
                system.topology,
                system.system,
                openmm.LangevinIntegrator(
                    temperature * unit.kelvin,
                    1.0 / unit.picosecond,
                    1.0 * unit.femtosecond,
                ),
                platform=openmm.Platform.getPlatformByName("Reference"),
            )
            sim.context.setPositions(system.positions)
            sim.minimizeEnergy()
            state = sim.context.getState(getPositions=True)
            position = state.getPositions(True).value_in_unit(unit.nanometer)
            tmp_dir = tempfile.gettempdir()
            data_path = tmp_dir + "/aldp.pt"
            torch.save(
                torch.tensor(position.reshape(1, 66).astype(np.float64)), data_path
            )

            del sim

        data_path = Path(data_path)
        if data_path.suffix == ".h5":
            # Load data for transform
            traj = md.load(data_path)
            traj.center_coordinates()

            # Superpose on the backbone
            ind = traj.top.select("backbone")
            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

            # Gather the training data into a pytorch Tensor with the right shape
            transform_data = traj.xyz
            n_atoms = transform_data.shape[1]
            n_dim = n_atoms * 3
            transform_data_npy = transform_data.reshape(-1, n_dim)
            transform_data = torch.from_numpy(transform_data_npy.astype("float64"))
        elif data_path.suffix == ".pt":
            transform_data = torch.load(data_path)
        else:
            raise NotImplementedError("Loading data or this format is not implemented.")

        # Set distribution
        self.coordinate_transform = bg.flows.CoordinateTransform(
            transform_data,
            self.dim + 6,
            z_matrix,
            cart_indices,
            mode=self.transform,
            ind_circ_dih=ind_circ_dih,
            shift_dih=shift_dih,
            shift_dih_params=shift_dih_params,
            default_std=default_std,
        )

        if n_threads > 1:
            self.distr = bg.distributions.TransformedBoltzmannParallel(
                system,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
                n_threads=n_threads,
            )
        else:
            self.distr = bg.distributions.TransformedBoltzmann(
                sim.context,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
            )

        # Evaluation data
        self.eval_data = None
        if eval_data_path is not None:
            self.eval_data = torch.load(eval_data_path).float().numpy()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.distr.log_prob(x).unsqueeze(-1)
        assert log_prob.shape == (x.shape[0], 1)
        return log_prob

    def filter(self, x: torch.Tensor) -> torch.Tensor:
        if self.filter_chirality_train:
            return filter_chirality(x)
        return True

    def plots(self, samples: torch.Tensor, batch_size=1000) -> dict[float, plt.Figure]:
        """
        Evaluate model of the Boltzmann distribution of the Alanine Dipeptide
        :param samples: Samples from the model
        :param batch_size: Batch size when processing the data
        """
        if self.eval_data is None:
            return {}

        # Transform data
        device = self.coordinate_transform.transform.permute_inv.device
        x_d_np = np.zeros((0, 66))
        n_batches = int(np.ceil(len(self.eval_data) / batch_size))
        for i in range(n_batches):
            if i == n_batches - 1:
                end = len(self.eval_data)
            else:
                end = (i + 1) * batch_size
            z = self.eval_data[(i * batch_size) : end, :]
            x, _ = self.coordinate_transform(torch.from_numpy(z).to(device).double())
            x_d_np = np.concatenate((x_d_np, x.cpu().numpy()))

        # Transform samples
        z_np = np.zeros((0, 60))
        x_np = np.zeros((0, 66))
        n_batches = int(np.ceil(len(samples) / batch_size))
        for i in range(n_batches):
            if i == n_batches - 1:
                end = len(samples)
            else:
                end = (i + 1) * batch_size
            z = samples[(i * batch_size) : end, :]
            x, _ = self.coordinate_transform(z.double())
            x_np = np.concatenate((x_np, x.cpu().numpy()))
            z, _ = self.coordinate_transform.inverse(x)
            z_np = np.concatenate((z_np, z.cpu().numpy()))

        # Estimate density of marginals
        nbins = 200
        hist_range = [-5, 5]
        ndims = z_np.shape[1]

        hists_test = np.zeros((nbins, ndims))
        hists_gen = np.zeros((nbins, ndims))

        for i in range(ndims):
            htest, _ = np.histogram(
                self.eval_data[:, i], nbins, range=hist_range, density=True
            )
            hgen, _ = np.histogram(z_np[:, i], nbins, range=hist_range, density=True)
            hists_test[:, i] = htest
            hists_gen[:, i] = hgen

        # Split into groups
        ncarts = self.coordinate_transform.transform.len_cart_inds
        permute_inv = self.coordinate_transform.transform.permute_inv.cpu().numpy()
        bond_ind = (
            self.coordinate_transform.transform.ic_transform.bond_indices.cpu().numpy()
        )
        angle_ind = (
            self.coordinate_transform.transform.ic_transform.angle_indices.cpu().numpy()
        )
        dih_ind = (
            self.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()
        )

        # Compute Ramachandran plot angles
        aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
        topology = md.Topology.from_openmm(aldp.topology)
        test_traj = md.Trajectory(x_d_np.reshape(-1, 22, 3), topology)
        sampled_traj = md.Trajectory(x_np.reshape(-1, 22, 3), topology)
        psi_d = md.compute_psi(test_traj)[1].reshape(-1)
        phi_d = md.compute_phi(test_traj)[1].reshape(-1)
        is_nan = np.logical_or(np.isnan(psi_d), np.isnan(phi_d))
        not_nan = np.logical_not(is_nan)
        psi_d = psi_d[not_nan]
        phi_d = phi_d[not_nan]
        psi = md.compute_psi(sampled_traj)[1].reshape(-1)
        phi = md.compute_phi(sampled_traj)[1].reshape(-1)
        is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
        not_nan = np.logical_not(is_nan)
        psi = psi[not_nan]
        phi = phi[not_nan]

        # Compute histograms
        htest_phi, _ = np.histogram(phi_d, nbins, range=[-np.pi, np.pi], density=True)
        hgen_phi, _ = np.histogram(phi, nbins, range=[-np.pi, np.pi], density=True)

        htest_psi, _ = np.histogram(psi_d, nbins, range=[-np.pi, np.pi], density=True)
        hgen_psi, _ = np.histogram(psi, nbins, range=[-np.pi, np.pi], density=True)

        hists_test_cart = hists_test[:, : (3 * ncarts - 6)]
        hists_test_ = np.concatenate(
            [
                hists_test[:, : (3 * ncarts - 6)],
                np.zeros((nbins, 6)),
                hists_test[:, (3 * ncarts - 6) :],
            ],
            axis=1,
        )
        hists_test_ = hists_test_[:, permute_inv]
        hists_test_bond = hists_test_[:, bond_ind]
        hists_test_angle = hists_test_[:, angle_ind]
        hists_test_dih = hists_test_[:, dih_ind]

        hists_gen_cart = hists_gen[:, : (3 * ncarts - 6)]
        hists_gen_ = np.concatenate(
            [
                hists_gen[:, : (3 * ncarts - 6)],
                np.zeros((nbins, 6)),
                hists_gen[:, (3 * ncarts - 6) :],
            ],
            axis=1,
        )
        hists_gen_ = hists_gen_[:, permute_inv]
        hists_gen_bond = hists_gen_[:, bond_ind]
        hists_gen_angle = hists_gen_[:, angle_ind]
        hists_gen_dih = hists_gen_[:, dih_ind]

        if self.transform == "internal":
            hists_test_bond = np.concatenate(
                (hists_test_cart[:, :2], hists_test_bond), 1
            )
            hists_gen_bond = np.concatenate((hists_gen_cart[:, :2], hists_gen_bond), 1)
            hists_test_angle = np.concatenate(
                (hists_test_cart[:, 2:], hists_test_angle), 1
            )
            hists_gen_angle = np.concatenate(
                (hists_gen_cart[:, 2:], hists_gen_angle), 1
            )

        label = ["bond", "angle", "dih"]
        hists_test_list = [hists_test_bond, hists_test_angle, hists_test_dih]
        hists_gen_list = [hists_gen_bond, hists_gen_angle, hists_gen_dih]
        if self.transform == "mixed":
            label = ["cart"] + label
            hists_test_list = [hists_test_cart] + hists_test_list
            hists_gen_list = [hists_gen_cart] + hists_gen_list
        x = np.linspace(*hist_range, nbins)

        figures = {}
        for i, name in enumerate(label):
            if self.transform == "mixed":
                ncol = 3
                if i == 0:
                    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
                else:
                    fig, ax = plt.subplots(6, 3, figsize=(10, 20))
                    ax[5, 2].set_axis_off()
            elif self.transform == "internal":
                ncol = 4
                if i == 0:
                    fig, ax = plt.subplots(6, 4, figsize=(15, 24))
                    for j in range(1, 4):
                        ax[5, j].set_axis_off()
                elif i == 2:
                    fig, ax = plt.subplots(5, 4, figsize=(15, 20))
                    ax[4, 3].set_axis_off()
                else:
                    fig, ax = plt.subplots(5, 4, figsize=(15, 20))
            for j in range(hists_test_list[i].shape[1]):
                ax[j // ncol, j % ncol].plot(x, hists_test_list[i][:, j])
                ax[j // ncol, j % ncol].plot(x, hists_gen_list[i][:, j])

            figures[f"plots/marginals_{name}"] = fig

        # Plot phi and psi
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        x = np.linspace(-np.pi, np.pi, nbins)
        ax[0].plot(x, htest_phi, linewidth=3)
        ax[0].plot(x, hgen_phi, linewidth=3)
        ax[0].set_xlabel("$\phi$")
        ax[1].plot(x, htest_psi, linewidth=3)
        ax[1].plot(x, hgen_psi, linewidth=3)
        ax[1].set_xlabel("$\psi$")
        figures["plots/phi_psi"] = fig

        # Ramachandran plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.hist2d(
            phi,
            psi,
            bins=64,
            norm=mpl.colors.LogNorm(),
            range=[[-np.pi, np.pi], [-np.pi, np.pi]],
        )
        ax.set_xlabel("$\phi$")
        ax.set_ylabel("$\psi$")
        figures["plots/ramachandran"] = fig

        return figures


def filter_chirality(
    x: torch.Tensor,
    ind: list = [17, 26],
    mean_diff: float = -0.043,
    threshold: float = 0.8,
) -> torch.Tensor:
    """
    Filters batch for the L-form
    See https://github.com/lollcat/fab-torch/blob/master/fab/utils/aldp.py
    :param x: Input batch
    :param ind: Indices to be used for determining the chirality
    :param mean_diff: Mean of the difference of the coordinates
    :param threshold: Threshold to be used for splitting
    :return: Returns indices of batch, where L-form is present
    """
    diff_ = torch.column_stack(
        (
            x[:, ind[0]] - x[:, ind[1]],
            x[:, ind[0]] - x[:, ind[1]] + 2 * np.pi,
            x[:, ind[0]] - x[:, ind[1]] - 2 * np.pi,
        )
    )
    min_diff_ind = torch.min(torch.abs(diff_), 1).indices
    diff = diff_[torch.arange(x.shape[0]), min_diff_ind]
    ind = torch.abs(diff - mean_diff) < threshold
    return ind.unsqueeze(-1)
