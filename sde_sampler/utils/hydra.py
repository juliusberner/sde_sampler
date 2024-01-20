from __future__ import annotations

import logging
import subprocess as sp
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import OmegaConf
from submitit.slurm import slurm


def get_free_gpu(
    idx: int | None, exclude: list | None = None, remap: dict | None = None
):
    if idx is None:
        idx = 0

    COMMAND = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"]
    try:
        memory_info = sp.check_output(COMMAND).decode("ascii").split("\n")[1:-1]
    except sp.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    exclude = exclude or []
    memory_usage = [
        (i, int(x.split()[0])) for i, x in enumerate(memory_info) if i not in exclude
    ]
    memory_usage = sorted(memory_usage, key=lambda x: x[1])
    gpu_idx = memory_usage[idx % len(memory_usage)][0]
    logging.info("Choosing GPU %d for Job Nr. %d", gpu_idx, idx)
    remap = remap or {}
    return remap.get(gpu_idx, gpu_idx)


OmegaConf.register_new_resolver("get_free_gpu", get_free_gpu)
OmegaConf.register_new_resolver("eval", eval)


# The following fixes are only needed for specific clusters
@dataclass
class VSCSlurmQueueConf(SlurmQueueConf):
    """Configuration shared by all executors"""

    # Number of nodes to use for the jobpa
    nodes: tp.Optional[int] = None


ConfigStore.instance().store(
    group="hydra/launcher",
    name="submitit_slurm_vsc",
    node=VSCSlurmQueueConf(),
    provider="submitit_launcher",
)

slurm._make_sbatch_string_copy = slurm._make_sbatch_string


def _make_sbatch_string(
    command: str,
    folder: tp.Union[str, Path],
    job_name: str = "submitit",
    partition: tp.Optional[str] = None,
    time: int = 5,
    nodes: tp.Optional[int] = None,
    ntasks_per_node: tp.Optional[int] = None,
    cpus_per_task: tp.Optional[int] = None,
    cpus_per_gpu: tp.Optional[int] = None,
    num_gpus: tp.Optional[int] = None,  # Legacy
    gpus_per_node: tp.Optional[int] = None,
    gpus_per_task: tp.Optional[int] = None,
    qos: tp.Optional[str] = None,  # Quality of service
    setup: tp.Optional[tp.List[str]] = None,
    mem: tp.Optional[str] = None,
    mem_per_gpu: tp.Optional[str] = None,
    mem_per_cpu: tp.Optional[str] = None,
    signal_delay_s: int = 90,
    comment: tp.Optional[str] = None,
    constraint: tp.Optional[str] = None,
    exclude: tp.Optional[str] = None,
    account: tp.Optional[str] = None,
    gres: tp.Optional[str] = None,
    exclusive: tp.Optional[tp.Union[bool, str]] = None,
    array_parallelism: int = 256,
    wckey: str = "submitit",
    stderr_to_stdout: bool = False,
    map_count: tp.Optional[int] = None,  # Used internally
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
    srun_args: tp.Optional[tp.Iterable[str]] = None,
) -> str:
    return slurm._make_sbatch_string_copy(**locals())


slurm._make_sbatch_string = _make_sbatch_string
