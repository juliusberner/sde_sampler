#!/bin/bash

# Test all targets
python scripts/main.py -m +launcher=slurm target=mw,cox,rings,nice,gmm,aladip,funnel,img,dw_shift,mw_50d,rosenbrock,gauss_shift wandb.project=test train_steps=2 eval_batch_size=2 train_batch_size=2
