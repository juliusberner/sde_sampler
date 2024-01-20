#!/bin/bash

# Run `sbatch bin/slurm_train_nice.sh` to train the NICE model.

# Slurm parameters
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --job-name=nice
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=4320

# command
python scripts/train_nice.py
