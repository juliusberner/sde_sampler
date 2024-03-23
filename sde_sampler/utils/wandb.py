from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from PIL.Image import Image

from sde_sampler.utils.common import CKPT_DIR


def format_fig(
    fig: Image | go.Figure | plt.Figure,
) -> go.Figure | plt.Figure | wandb.Image:
    if isinstance(fig, (Image, plt.Figure)):
        return wandb.Image(fig)
    return fig


def check_wandb(fun):
    def inner(*args, **kwargs):
        if (
            isinstance(wandb.run, wandb.sdk.wandb_run.Run)
            and wandb.run.settings.mode == "run"
        ):
            return fun(*args, **kwargs)
        elif wandb.run is None:
            mode = "none"
        elif isinstance(wandb.run.mode, wandb.sdk.lib.disabled.RunDisabled):
            mode = "disabled"
        else:
            mode = wandb.run.settings.mode
        logging.warning(
            "Wandb not available (mode=%s): Unable to call function %s.",
            mode,
            fun.__name__,
        )

    return inner


@check_wandb
def merge_wandb_cfg(cfg: DictConfig | dict) -> DictConfig:
    wandb_config = dict(wandb.run.config)
    wandb_config.pop("hydra", None)
    cfg = OmegaConf.merge(wandb_config, cfg)
    logging.info("Merged config with wandb config.")
    return cfg


@check_wandb
def upload_ckpt(path: Path | str, name: str = "ckpt"):
    name = f"{CKPT_DIR}/{name}"
    model_artifact = wandb.Artifact(
        wandb.run.id, type="model", metadata={"path": str(path), "name": name}
    )
    model_artifact.add_file(str(path), name=name)
    wandb.log_artifact(model_artifact)
    logging.info("Uploaded checkpoint %s to wandb.", name)


@check_wandb
def restore_ckpt(out_dir: Path | str):
    try:
        artifact = wandb.run.use_artifact(f"{wandb.run.id}:latest")
        ckpt = artifact.download(out_dir)
        logging.info(
            "Checkpoint %s restored from wandb.",
            artifact.metadata.get("name", ckpt),
        )
    except wandb.CommError as exception:
        logging.debug("Wandb raised exception %s", exception)
        logging.info("No previous checkpoints found for wandb id %s.", wandb.run.id)


@check_wandb
def delete_old_wandb_ckpts():
    try:
        run = wandb.Api().run(wandb.run.path)
        for artifact in run.logged_artifacts():
            if len(artifact.aliases) == 0:
                # Clean up versions that don't have an alias such as 'latest'
                artifact.delete()
                logging.info(
                    "Marked checkpoint %s for deletion on wandb.",
                    artifact.metadata["name"],
                )
    except wandb.CommError as exception:
        logging.debug("Wandb raised exception %s", exception)
        logging.warning("Unable to delete checkpoints on wandb.")
