#!/bin/bash

# Test all solvers
python scripts/main.py -m +launcher=slurm solver=dis_subtraj,bridge_diff_init,bridge_subtraj,basic_bridge_subtraj,basic_pis,basic_dds,basic_dds_euler,basic_bridge,dis_no_score,langevin,pis,basic_dis,bridge,pis_no_score,dis,basic_dis_subtraj,dds,dds_euler wandb.project=test ++train_steps=2
