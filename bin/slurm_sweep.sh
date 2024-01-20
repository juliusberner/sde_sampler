#!/bin/bash

# Run `SWEEP_ID=<wandb_entity>/<wandb_project>/<sweep_id> sbatch -a 0-<num agents> bin/slurm_sweep.sh`
# to start <num agents> wandb agents.

# Slurm parameters
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --job-name=sweep
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A-%a.out
#SBATCH --time=4320

# command
export PYTHONOPTIMIZE=1
wandb agent $SWEEP_ID --count 1
